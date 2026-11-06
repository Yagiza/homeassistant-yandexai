"""The conversation platform for the YandexGPT integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from typing import Any, Literal

import httpx
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import intent, template
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import YandexGPTConfigEntry
from .client import YandexGPTApiError, YandexGPTClient
from .const import (CONF_MAX_TOKENS, CONF_MAX_TOOL_ITERATIONS,
                    CONF_NO_HA_DEFAULT_PROMPT, CONF_PROMPT, CONF_TEMPERATURE,
                    DEFAULT_INSTRUCTIONS_PROMPT_RU,
                    DEFAULT_MAX_TOOL_ITERATIONS,
                    DEFAULT_NO_HA_DEFAULT_PROMPT, DOMAIN, LOGGER,
                    RECOMMENDED_MAX_TOKENS, RECOMMENDED_TEMPERATURE,
                    SUBENTRY_TYPE_CONVERSATION)
from .entity import YandexGPTEntity
from .mappers import ContentConverter, StreamTransformer


async def async_setup_entry(
    _hass: HomeAssistant,
    config_entry: YandexGPTConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities from subentries."""
    for subentry_id, subentry in config_entry.subentries.items():
        if subentry.subentry_type != SUBENTRY_TYPE_CONVERSATION:
            continue
        async_add_entities(
            [YandexGPTConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry_id,
        )


class YandexGPTConversationEntity(
    YandexGPTEntity,
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,  # type: ignore
):
    """YandexGPT conversation agent."""

    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, entry: YandexGPTConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    def _settings(self) -> dict[str, Any]:
        """Merge entry data (credentials) with subentry options."""
        return {**self.entry.data, **dict(self.subentry.data)}

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process a conversation with YandexGPT."""
        settings = self._settings()
        client: YandexGPTClient = self.entry.runtime_data

        system_prompt = settings.get(CONF_PROMPT, DEFAULT_INSTRUCTIONS_PROMPT_RU)

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                settings.get(CONF_LLM_HASS_API),
                system_prompt,
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        no_ha_default_prompt = settings.get(
            CONF_NO_HA_DEFAULT_PROMPT, DEFAULT_NO_HA_DEFAULT_PROMPT
        )
        system_prompt_override = (
            await self._async_expand_prompt_template(system_prompt, user_input)
            if no_ha_default_prompt
            else None
        )

        content_converter = ContentConverter(
            system_prompt_override=system_prompt_override
        )
        messages = content_converter.to_openai_api(chat_log.content)

        tools = None
        if chat_log.llm_api:
            tools = [
                ContentConverter.format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        try:
            max_tool_iterations = settings.get(
                CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS
            )
            for _iteration in range(max_tool_iterations):
                LOGGER.debug("Prompt: %s", messages)

                response_stream = client.stream_chat(
                    self._model_id(settings),
                    messages,
                    temperature=settings.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                    max_tokens=settings.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                    tools=tools,
                )

                stream_transformer = StreamTransformer(response_stream)
                content_converter = ContentConverter(
                    system_prompt_override=system_prompt_override
                )
                messages.extend(
                    content_converter.to_openai_api(
                        [
                            content
                            async for content in chat_log.async_add_delta_content_stream(
                                user_input.agent_id,
                                stream_transformer.to_chatlog_api(),
                            )
                        ]
                    )
                )

                if not chat_log.unresponded_tool_results:
                    break
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

        intent_response = intent.IntentResponse(language=user_input.language)
        assert type(chat_log.content[-1]) is conversation.AssistantContent
        intent_response.async_set_speech(chat_log.content[-1].content or "")
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_expand_prompt_template(
        self, prompt_template: str, user_input: conversation.ConversationInput
    ) -> str:
        """Render the prompt template."""
        try:
            system_prompt = template.Template(prompt_template, self.hass).async_render(parse_result=False)

            if user_input.extra_system_prompt:
                system_prompt += user_input.extra_system_prompt

            return system_prompt
        except TemplateError as err:
            raise HomeAssistantError("Error rendering prompt template") from err
