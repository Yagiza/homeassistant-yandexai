"""AI Task platform for the YandexGPT integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from json import JSONDecodeError
from typing import Any

import httpx

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
)
from homeassistant.util.json import json_loads

from . import YandexGPTConfigEntry
from .client import GeneratedImage, YandexGPTApiError, YandexGPTClient
from .const import (CAP_IMAGE_TO_TEXT, CAP_TEXT_TO_IMAGE, CAP_TEXT_TO_TEXT,
                    CONF_CHAT_MODEL, CONF_FOLDER_ID, DEFAULT_CHAT_MODEL, DOMAIN,
                    IMAGE_GENERATION_TIMEOUT, IMAGE_MODELS, IMAGE_MODEL_VERSION,
                    LOGGER, SUBENTRY_TYPE_AI_TASK_DATA, model_capabilities)
from .entity import YandexGPTEntity


async def async_setup_entry(
    _hass: HomeAssistant,
    config_entry: YandexGPTConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_AI_TASK_DATA:
            continue

        async_add_entities(
            [YandexGPTAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class YandexGPTAITaskEntity(ai_task.AITaskEntity, YandexGPTEntity):
    """YandexGPT AI Task entity."""

    _attr_name = None

    def __init__(self, entry: YandexGPTConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        options = {**dict(entry.data), **dict(subentry.data)}
        capabilities = model_capabilities(
            options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        )
        features = 0
        if CAP_TEXT_TO_TEXT in capabilities:
            features |= ai_task.AITaskEntityFeature.GENERATE_DATA
            if CAP_IMAGE_TO_TEXT in capabilities:
                features |= ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        if CAP_TEXT_TO_IMAGE in capabilities:
            features |= ai_task.AITaskEntityFeature.GENERATE_IMAGE
        self._attr_supported_features = features

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        await self._async_handle_chat_log(chat_log, task.name, task.structure)

        last_content = chat_log.content[-1]
        if not isinstance(last_content, conversation.AssistantContent):
            raise HomeAssistantError(
                "The model did not produce a final answer "
                f"(last content in chat log: {type(last_content).__name__})"
            )

        text = last_content.content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            raise HomeAssistantError("Error with YandexGPT structured response") from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle an image generation task."""
        options: dict[str, Any] = {
            **dict(self.entry.data),
            **dict(self.subentry.data),
        }
        client: YandexGPTClient = self.entry.runtime_data
        image: GeneratedImage | None = None
        selected = options.get(CONF_CHAT_MODEL) or IMAGE_MODELS[0]
        image_models = [selected] + [m for m in IMAGE_MODELS if m != selected]
        used_model = selected
        last_err: YandexGPTApiError | None = None
        for image_model in image_models:
            model_uri = (
                f"art://{options[CONF_FOLDER_ID]}/{image_model}/{IMAGE_MODEL_VERSION}"
            )
            try:
                image = await client.generate_image(
                    model_uri,
                    task.instructions,
                    timeout=IMAGE_GENERATION_TIMEOUT,
                )
                used_model = image_model
                break
            except YandexGPTApiError as err:
                if err.status_code not in (400, 403, 404):
                    LOGGER.exception(
                        "Error generating image with %s: %s", image_model, err
                    )
                    raise HomeAssistantError(
                        translation_domain=DOMAIN,
                        translation_key="yandex_cloud_error",
                        translation_placeholders={"details": err.message},
                    ) from err
                LOGGER.warning(
                    "Image model %s is not available, trying the next one: %s",
                    image_model,
                    err.message,
                )
                last_err = err
            except httpx.HTTPError as err:
                LOGGER.exception(
                    "Error generating image with %s: %s", image_model, err
                )
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="yandex_cloud_error",
                    translation_placeholders={"details": str(err)},
                ) from err
        if image is None:
            details = last_err.message if last_err else "Image generation failed"
            LOGGER.error("Could not generate image: %s", details)
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="yandex_cloud_error",
                translation_placeholders={"details": details},
            ) from last_err

        return ai_task.GenImageTaskResult(
            image_data=image.image_bytes,
            conversation_id=chat_log.conversation_id,
            mime_type=image.mime_type,
            model=f"{used_model}/{IMAGE_MODEL_VERSION}",
        )
