"""Base entity for the Yandex AI integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity
from probatio import Schema as ProbatioSchema
from probatio import to_openapi

from .client import YandexGPTApiError, YandexGPTClient
from .const import (CONF_CHAT_MODEL, CONF_FOLDER_ID, CONF_MAX_TOKENS,
                    CONF_MAX_TOOL_ITERATIONS, CONF_MODEL_VERSION,
                    CONF_TEMPERATURE, DEFAULT_CHAT_MODEL,
                    DEFAULT_MAX_TOOL_ITERATIONS, DEFAULT_MODEL_VERSION,
                    DOMAIN, LOGGER, MULTIMODAL_MODELS, RECOMMENDED_MAX_TOKENS,
                    RECOMMENDED_TEMPERATURE)
from .mappers import ContentConverter, _tool_args_to_dict

type YandexGPTConfigEntry = ConfigEntry[YandexGPTClient]  # type: ignore[name-defined]


def _adjust_schema(schema: dict[str, Any]) -> None:
    """Adjust the schema to be compatible with the API."""

    if "enum" in schema and (not schema["enum"] or schema["enum"] == []):
        del schema["enum"]

    if schema.get("type") == "object":
        if "properties" not in schema:
            return
        if "required" not in schema:
            schema["required"] = []
        for prop, prop_info in schema["properties"].items():
            _adjust_schema(prop_info)
            if prop not in schema["required"]:
                prop_info["type"] = [prop_info["type"], "null"]
                schema["required"].append(prop)
    elif schema.get("type") == "array" and "items" in schema:
        _adjust_schema(schema["items"])


def _format_structured_output(
    name: str,
    structure: ProbatioSchema,
    llm_api: llm.APIInstance | None,
) -> dict[str, Any]:
    """Build the OpenAI json_schema response_format for structured output."""
    result_schema = to_openapi(
        structure,
        custom_serializer=llm_api.custom_serializer if llm_api else llm.selector_serializer,
    )
    _adjust_schema(result_schema)
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": result_schema},
    }


def _message_to_delta(message: dict[str, Any]) -> conversation.AssistantContentDeltaDict:
    """Convert a non-streaming assistant message into a chat log delta."""
    data: conversation.AssistantContentDeltaDict = {
        "role": message.get("role") or "assistant",
    }
    if content := message.get("content"):
        data["content"] = content
    if tool_calls := message.get("tool_calls"):
        data["tool_calls"] = [
            llm.ToolInput(
                tool_name=tool_call["function"]["name"],
                tool_args=_tool_args_to_dict(tool_call["function"].get("arguments")),
            )
            for tool_call in tool_calls
            if tool_call.get("type", "function") == "function"
        ]
    return data


class YandexGPTEntity(Entity):
    """Shared base entity."""

    _attr_has_entity_name = True

    def __init__(self, entry: YandexGPTConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the shared entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Yandex",
            model="Yandex AI",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @staticmethod
    def _model_id(settings: dict) -> str:
        """Build the model identifier for the OpenAI-compatible API."""
        model = settings.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        if "://" in model:
            return model
        model_version = settings.get(CONF_MODEL_VERSION, DEFAULT_MODEL_VERSION)
        return f"gpt://{settings[CONF_FOLDER_ID]}/{model}/{model_version}"

    @staticmethod
    def _model_name(settings: dict) -> str:
        """Extract the short model name from the configured chat model."""
        model = settings.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        if "://" in model:
            parts = model.split("/")
            return parts[3] if len(parts) > 3 else model
        return model.split("/")[0]

    @classmethod
    def _model_supports_attachments(cls, settings: dict) -> bool:
        """Return True if the configured model accepts image attachments."""
        return cls._model_name(settings) in MULTIMODAL_MODELS

    async def _async_attach_images(
        self,
        messages: list[dict[str, Any]],
        attachments: list[conversation.Attachment],
    ) -> None:
        """Append image attachments to the last user message as image_url parts."""
        last_message = messages[-1]
        if (
            last_message.get("role") != "user"
            or not isinstance(last_message.get("content"), str)
        ):
            raise HomeAssistantError(
                "Attachments can only be added to a user text message"
            )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": last_message["content"]}
        ]
        for attachment in attachments:
            if not attachment.mime_type.startswith("image/"):
                raise HomeAssistantError(
                    "Only image attachments are supported by Yandex AI Studio, "
                    f"`{attachment.media_content_id}` is not an image"
                )
            data = await asyncio.to_thread(attachment.path.read_bytes)
            b64 = base64.b64encode(data).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{attachment.mime_type};base64,{b64}"},
                }
            )
        last_message["content"] = content

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: ProbatioSchema | None = None,
    ) -> None:
        """Run a non-streaming chat completion (used by AI tasks)."""
        # Merge entry data (credentials) with subentry data (model options)
        options: dict[str, Any] = {
            **dict(self.entry.data),
            **dict(self.subentry.data),
        }
        client: YandexGPTClient = self.entry.runtime_data

        tools = None
        if chat_log.llm_api:
            tools = [
                ContentConverter.format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        response_format = None
        if structure:
            if structure_name is None:
                raise HomeAssistantError("Structured output requires a task name")
            response_format = _format_structured_output(
                structure_name, structure, chat_log.llm_api
            )

        content_converter = ContentConverter()
        messages = content_converter.to_openai_api(chat_log.content)

        last_content = chat_log.content[-1]
        if isinstance(last_content, conversation.UserContent) and last_content.attachments:
            await self._async_attach_images(messages, last_content.attachments)

        for _iteration in range(
            options.get(CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS)
        ):
            try:
                choice = await client.complete_chat(
                    self._model_id(options),
                    messages,
                    temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                    max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                    tools=tools,
                    response_format=response_format,
                )
            except YandexGPTApiError as err:
                LOGGER.exception("Error talking to Yandex AI Studio: %s", err)
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="yandex_cloud_error",
                    translation_placeholders={"details": err.message},
                ) from err
            except httpx.HTTPError as err:
                LOGGER.exception("Error talking to Yandex AI Studio: %s", err)
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="yandex_cloud_error",
                    translation_placeholders={"details": str(err)},
                ) from err

            message = choice.get("message") or {}
            if choice.get("finish_reason") == "content_filter":
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="ethics_filter",
                )
            if not message.get("content") and not message.get("tool_calls"):
                raise HomeAssistantError(
                    "YandexGPT returned an empty response "
                    f"(finish_reason: {choice.get('finish_reason') or 'unknown'})"
                )

            async def _transform() -> AsyncGenerator[
                conversation.AssistantContentDeltaDict, None
            ]:
                yield _message_to_delta(message)

            messages.extend(
                content_converter.to_openai_api(
                    [
                        content
                        async for content in chat_log.async_add_delta_content_stream(
                            self.entity_id, _transform()
                        )
                    ]
                )
            )

            if not chat_log.unresponded_tool_results:
                break
