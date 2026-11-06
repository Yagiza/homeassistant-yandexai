"""AI Task platform for the YandexGPT integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from json import JSONDecodeError

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
)
from homeassistant.util.json import json_loads

from . import YandexGPTConfigEntry
from .const import SUBENTRY_TYPE_AI_TASK_DATA
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
    _attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA

    def __init__(self, entry: YandexGPTConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        if self._model_supports_attachments(
            {**dict(entry.data), **dict(subentry.data)}
        ):
            self._attr_supported_features |= ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS

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
