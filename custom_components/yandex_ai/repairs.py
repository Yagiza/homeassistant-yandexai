"""Repairs flow for Yandex AI task preference issues."""

from __future__ import annotations

from typing import Any, cast

import probatio
from homeassistant.components.ai_task.const import DATA_PREFERENCES
from homeassistant.components.repairs.models import RepairsFlow, RepairsFlowResult
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from . import (
    AI_TASK_PREFERENCES_ACKNOWLEDGED_DATA_KEY,
    AI_TASK_SETTINGS_URL,
    YandexAIConfigEntry,
    _ai_task_acknowledged_preference_keys,
    _ai_task_preference_feature,
    _async_apply_ai_task_preferences,
    _async_entity_supports_feature,
    _async_problematic_ai_task_preferences,
    _async_update_ai_task_preferences_issue,
)


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Create a repairs flow for an AI task preference issue."""
    return YandexAIAITaskPreferenceRepairFlow(data or {})


class YandexAIAITaskPreferenceRepairFlow(RepairsFlow):
    """Select system AI tasks for unset or invalid preferences."""

    def __init__(
        self,
        data: dict[str, str | int | float | None] | None = None,
    ) -> None:
        self.data = data or {}

    def _entry(self) -> YandexAIConfigEntry | None:
        entry_id = str(self.data.get("entry_id") or "")
        if not entry_id:
            return None
        return cast(
            YandexAIConfigEntry | None,
            self.hass.config_entries.async_get_entry(entry_id),
        )

    def _acknowledged_keys(self) -> set[str]:
        return _ai_task_acknowledged_preference_keys(
            self.data.get(AI_TASK_PREFERENCES_ACKNOWLEDGED_DATA_KEY)
        )

    def _problem_context(
        self,
    ) -> tuple[YandexAIConfigEntry | None, dict[str, list[dict[str, str | int]]]]:
        entry = self._entry()
        if entry is None or self.hass.data.get(DATA_PREFERENCES) is None:
            return None, {}

        _, candidates_by_key = _async_problematic_ai_task_preferences(
            self.hass, entry
        )
        acknowledged = self._acknowledged_keys()
        return entry, {
            preference_key: candidates
            for preference_key, candidates in candidates_by_key.items()
            if preference_key not in acknowledged
        }

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> RepairsFlowResult:
        entry, candidates_by_key = self._problem_context()
        if entry is None:
            return self.async_abort(reason="missing_data")
        if not candidates_by_key:
            return self.async_abort(reason="already_fixed")

        return await self.async_step_select()

    async def async_step_select(
        self, user_input: dict[str, Any] | None = None
    ) -> RepairsFlowResult:
        entry, candidates_by_key = self._problem_context()
        if entry is None:
            return self.async_abort(reason="missing_data")
        if not candidates_by_key:
            return self.async_abort(reason="already_fixed")

        valid_options_by_key: dict[str, set[str]] = {}
        suggested_values: dict[str, str] = {}
        schema: dict[Any, Any] = {}

        for preference_key, candidates in candidates_by_key.items():
            options = [
                SelectOptionDict(
                    value=str(candidate["entity_id"]),
                    label=str(candidate["name"]),
                )
                for candidate in candidates
            ]
            valid_options_by_key[preference_key] = {
                option["value"] for option in options
            }
            suggested_values[preference_key] = str(candidates[0]["entity_id"])
            schema[probatio.Optional(preference_key)] = SelectSelector(
                SelectSelectorConfig(
                    options=options,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )

        if user_input is None:
            return self.async_show_form(
                step_id="select",
                data_schema=self.add_suggested_values_to_schema(
                    probatio.Schema(schema),
                    suggested_values,
                ),
                description_placeholders={
                    "settings_url": AI_TASK_SETTINGS_URL,
                },
                last_step=True,
            )

        desired: dict[str, str] = {}
        skipped_keys: set[str] = set()
        for preference_key in candidates_by_key:
            selected = user_input.get(preference_key)
            if not selected:
                skipped_keys.add(preference_key)
                continue
            feature = _ai_task_preference_feature(preference_key)
            if feature == 0:
                continue
            if selected not in valid_options_by_key[preference_key]:
                continue
            if not _async_entity_supports_feature(self.hass, selected, feature):
                continue
            desired[preference_key] = str(selected)

        acknowledged = self._acknowledged_keys() | skipped_keys

        if not desired:
            remaining_problem, remaining_candidates = (
                _async_problematic_ai_task_preferences(self.hass, entry)
            )
            _async_update_ai_task_preferences_issue(
                self.hass,
                entry,
                remaining_problem,
                remaining_candidates,
                acknowledged,
            )
            return self.async_abort(reason="no_changes")

        applied = await _async_apply_ai_task_preferences(self.hass, desired)
        if not applied:
            remaining_problem, remaining_candidates = (
                _async_problematic_ai_task_preferences(self.hass, entry)
            )
            _async_update_ai_task_preferences_issue(
                self.hass,
                entry,
                remaining_problem,
                remaining_candidates,
                acknowledged,
            )
            return self.async_abort(reason="apply_failed")

        remaining_problem, remaining_candidates = (
            _async_problematic_ai_task_preferences(self.hass, entry)
        )
        if not remaining_problem:
            return self.async_create_entry(data={})

        _async_update_ai_task_preferences_issue(
            self.hass,
            entry,
            remaining_problem,
            remaining_candidates,
            acknowledged,
        )
        return self.async_abort(reason="partially_fixed")
