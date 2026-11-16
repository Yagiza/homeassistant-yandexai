"""Repairs flow for Yandex AI task preference issues."""

from __future__ import annotations

from homeassistant.components.ai_task.const import DATA_PREFERENCES
from homeassistant.components.repairs.models import RepairsFlow, RepairsFlowResult
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from . import (
    _ai_task_preference_feature,
    _async_apply_ai_task_preferences,
    _async_entity_supports_feature,
)


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Create a repairs flow for an AI task preference issue."""
    return YandexAIAITaskPreferenceRepairFlow(data or {})


class YandexAIAITaskPreferenceRepairFlow(RepairsFlow):
    """Offer to install the suggested task or ignore the issue."""

    def __init__(
        self,
        data: dict[str, str | int | float | None] | None = None,
    ) -> None:
        self.data = data

    def _preference_key(self) -> str:
        return str((self.data or {}).get("preference_key") or "")

    def _entity_id(self) -> str:
        return str((self.data or {}).get("entity_id") or "")

    def _validation_reason(self) -> str | None:
        preference_key = self._preference_key()
        entity_id = self._entity_id()
        if not preference_key or not entity_id:
            return "missing_data"

        feature = _ai_task_preference_feature(preference_key)
        if feature == 0 or not _async_entity_supports_feature(
            self.hass, entity_id, feature
        ):
            return "entity_unavailable"

        preferences = self.hass.data.get(DATA_PREFERENCES)
        current_entity_id = (
            getattr(preferences, preference_key, None) if preferences else None
        )
        if _async_entity_supports_feature(self.hass, current_entity_id, feature):
            return "already_fixed"

        return None

    async def async_step_init(
        self, user_input: dict[str, str] | None = None
    ) -> RepairsFlowResult:
        if (reason := self._validation_reason()) is not None:
            return self.async_abort(reason=reason)

        return await self.async_step_menu()

    async def async_step_menu(
        self, user_input: dict[str, str] | None = None
    ) -> RepairsFlowResult:
        if (reason := self._validation_reason()) is not None:
            return self.async_abort(reason=reason)

        if user_input is not None:
            if user_input.get("next_step_id") == "ignore":
                return await self.async_step_ignore()
            return await self.async_step_install()

        return self.async_show_menu(
            step_id="menu",
            menu_options=["install", "ignore"],
        )

    async def async_step_install(
        self, user_input: dict[str, str] | None = None
    ) -> RepairsFlowResult:
        if (reason := self._validation_reason()) is not None:
            return self.async_abort(reason=reason)

        applied = await _async_apply_ai_task_preferences(
            self.hass,
            {self._preference_key(): self._entity_id()},
        )
        if not applied:
            return self.async_abort(reason="apply_failed")

        return self.async_create_entry(data={})

    async def async_step_ignore(
        self, user_input: dict[str, str] | None = None
    ) -> RepairsFlowResult:
        issue_registry = ir.async_get(self.hass)
        if issue_registry.async_get_issue(self.handler, self.issue_id) is None:
            return self.async_abort(reason="missing_data")

        ir.async_ignore_issue(self.hass, self.handler, self.issue_id, True)
        return self.async_abort(reason="ignored")
