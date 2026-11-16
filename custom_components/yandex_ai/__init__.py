"""The Yandex AI integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import inspect
from types import MappingProxyType

import httpx
from homeassistant.components.ai_task import AITaskEntityFeature
from homeassistant.components.ai_task.const import DATA_PREFERENCES
from homeassistant.config_entries import (ConfigEntry, ConfigSubentry)
from homeassistant.const import (CONF_API_KEY, CONF_LLM_HASS_API, Platform)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry
from homeassistant.helpers.issue_registry import IssueSeverity
from homeassistant.helpers.typing import ConfigType

from .client import YandexAIStudioError, YandexAIClient
from .const import (AI_TASK_KIND_IMAGE, AI_TASK_KIND_TEXT,
                    CAP_IMAGE_TO_TEXT, CAP_TEXT_TO_IMAGE, CAP_TEXT_TO_TEXT,
                    CONF_AI_TASK_KIND, CONF_CHAT_MODEL,
                    CONF_ENABLE_SERVER_DATA_LOGGING, CONF_FOLDER_ID,
                    CONF_PENDING_AI_TASK_PREFERENCES,
                    DEFAULT_AI_TASK_MODEL,
                    DEFAULT_CHAT_MODEL,
                    DEFAULT_ENABLE_SERVER_DATA_LOGGING, DOMAIN, LOGGER,
                    RECOMMENDED_AI_TASK_OPTIONS,
                    RECOMMENDED_CONVERSATION_OPTIONS,
                    RECOMMENDED_TEXT_AI_TASK_OPTIONS,
                    SUBENTRY_TYPE_AI_TASK_DATA, SUBENTRY_TYPE_CONVERSATION,
                    friendly_model_label, model_capabilities)

PLATFORMS = (Platform.CONVERSATION, Platform.AI_TASK)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type YandexAIConfigEntry = ConfigEntry[YandexAIClient]  # type: ignore[name-defined]

AI_TASK_PREFERENCE_ORDER = (
    "gen_data_entity_id",
    "gen_image_entity_id",
    "evaluate_entity_id",
)
AI_TASK_PREFERENCE_MULTIMODAL_PRIORITY = {
    "gen_data_entity_id",
    "evaluate_entity_id",
}
AI_TASK_PREFERENCES_ISSUE_PREFIX = "ai_task_preferences_not_set"
AI_TASK_PREFERENCES_ACKNOWLEDGED_DATA_KEY = "acknowledged_preference_keys"
AI_TASK_SETTINGS_URL = "/config/ai"


def _ai_task_preferences_issue_id(entry: YandexAIConfigEntry) -> str:
    """Return the unified AI task preferences issue id."""
    return f"{AI_TASK_PREFERENCES_ISSUE_PREFIX}_{entry.entry_id}"


def _ai_task_preferences_issue_key(preference_count: int) -> str:
    """Return the issue translation key for the unified issue."""
    if preference_count == 1:
        return "ai_task_preferences_not_set_one"
    return "ai_task_preferences_not_set_many"


def _ai_task_acknowledged_preference_keys(value: object) -> set[str]:
    """Return acknowledged AI task preference keys from issue data."""
    if not isinstance(value, str) or not value:
        return set()
    return {key.strip() for key in value.split(",") if key.strip()}


def _ai_task_preferences_issue_data(
    entry: YandexAIConfigEntry,
    preference_keys: list[str],
    acknowledged_keys: set[str] | None = None,
) -> dict[str, str]:
    """Return issue data for the unified AI task preferences issue."""
    data = {
        "entry_id": entry.entry_id,
        "preference_keys": ",".join(preference_keys),
    }
    if acknowledged_keys:
        data[AI_TASK_PREFERENCES_ACKNOWLEDGED_DATA_KEY] = ",".join(
            sorted(acknowledged_keys)
        )
    return data


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Yandex AI component."""
    await async_migrate_integration(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: YandexAIConfigEntry) -> bool:
    """Set up Yandex AI from a config entry."""
    _async_ensure_default_subentries(hass, entry)

    enable_server_data_logging = entry.data.get(
        CONF_ENABLE_SERVER_DATA_LOGGING, DEFAULT_ENABLE_SERVER_DATA_LOGGING
    )

    client = YandexAIClient(
        folder_id=entry.data[CONF_FOLDER_ID],
        api_key=entry.data[CONF_API_KEY],
        enable_server_data_logging=enable_server_data_logging,
        timeout=10.0,
    )
    try:
        await client.list_models()
    except YandexAIStudioError as err:
        if err.status_code in (401, 403):
            LOGGER.error("Invalid API key or folder ID: %s", err)
            return False
        raise ConfigEntryNotReady(f"Cannot connect to Yandex AI Studio: {err}") from err
    except httpx.HTTPError as err:
        raise ConfigEntryNotReady(f"Cannot connect to Yandex AI Studio: {err}") from err
    finally:
        await client.close()

    entry.runtime_data = YandexAIClient(
        folder_id=entry.data[CONF_FOLDER_ID],
        api_key=entry.data[CONF_API_KEY],
        enable_server_data_logging=enable_server_data_logging,
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    pending_preferences = entry.data.get(CONF_PENDING_AI_TASK_PREFERENCES)
    if pending_preferences:
        await _async_install_ai_task_preferences(hass, entry, pending_preferences)
        data = dict(entry.data)
        data.pop(CONF_PENDING_AI_TASK_PREFERENCES, None)
        hass.config_entries.async_update_entry(entry, data=data)

    await _async_review_ai_task_preferences(hass, entry)

    # Reload when subentry options change
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    return True


async def _async_update_listener(hass: HomeAssistant, entry: YandexAIConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


def _ai_task_preference_kind(preference_key: str) -> str:
    """Return the created task kind required for a preference."""
    if preference_key == "gen_image_entity_id":
        return AI_TASK_KIND_IMAGE
    return AI_TASK_KIND_TEXT


def _ai_task_preference_feature(preference_key: str) -> AITaskEntityFeature:
    """Return the AI task feature required for a preference."""
    if preference_key == "gen_data_entity_id":
        return getattr(AITaskEntityFeature, "GENERATE_DATA", 0)
    if preference_key == "gen_image_entity_id":
        return getattr(AITaskEntityFeature, "GENERATE_IMAGE", 0)
    if preference_key == "evaluate_entity_id":
        return getattr(AITaskEntityFeature, "EVALUATE", 0)
    return 0


def _async_subentry_matches_ai_task_kind(
    subentry: ConfigSubentry, kind: str
) -> bool:
    """Return whether a subentry represents the requested AI task kind."""
    if subentry.subentry_type != SUBENTRY_TYPE_AI_TASK_DATA:
        return False

    stored_kind = subentry.data.get(CONF_AI_TASK_KIND)
    if stored_kind:
        return stored_kind == kind

    title = subentry.title or ""
    model = subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
    capabilities = model_capabilities(model)
    if kind == AI_TASK_KIND_TEXT:
        if "Text Task" in title:
            return True
        return CAP_TEXT_TO_TEXT in capabilities
    if "Image Task" in title:
        return True
    return CAP_TEXT_TO_IMAGE in capabilities


def _ai_task_preference_candidates(
    hass: HomeAssistant,
    entry: YandexAIConfigEntry,
    preference_key: str,
) -> list[dict[str, str | int]]:
    """Return usable Yandex AI task entities for a preference.

    The list is sorted so that multimodal text tasks come first for
    ``gen_data_entity_id`` and ``evaluate_entity_id``.
    """
    feature = _ai_task_preference_feature(preference_key)
    kind = _ai_task_preference_kind(preference_key)
    if feature == 0:
        return []

    entity_registry = er.async_get(hass)
    candidates: list[dict[str, str | int]] = []

    for subentry in entry.subentries.values():
        if not _async_subentry_matches_ai_task_kind(subentry, kind):
            continue
        entity_id = entity_registry.async_get_entity_id(
            Platform.AI_TASK, DOMAIN, subentry.subentry_id
        )
        if entity_id is None:
            continue
        registry_entity = entity_registry.async_get(entity_id)
        if registry_entity is None or registry_entity.disabled:
            continue
        state = hass.states.get(entity_id)
        if state is None:
            continue
        if not (
            state.attributes.get("supported_features", 0) & feature
        ):
            continue

        priority = 0
        if preference_key in AI_TASK_PREFERENCE_MULTIMODAL_PRIORITY:
            model = subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
            if CAP_IMAGE_TO_TEXT in model_capabilities(model):
                priority = 1

        candidates.append(
            {
                "entity_id": entity_id,
                "name": _ai_task_entity_display_name(
                    subentry, entity_id, state
                ),
                "priority": priority,
            }
        )

    candidates.sort(key=lambda candidate: candidate["priority"], reverse=True)
    return candidates


def _ai_task_preference_candidate_entity_ids(
    hass: HomeAssistant,
    entry: YandexAIConfigEntry,
    preference_key: str,
) -> list[str]:
    """Return the candidate entity ids for a preference."""
    return [
        str(candidate["entity_id"])
        for candidate in _ai_task_preference_candidates(
            hass, entry, preference_key
        )
    ]


def _find_ai_task_entity_for_preference(
    hass: HomeAssistant,
    entry: YandexAIConfigEntry,
    preference_key: str,
) -> str | None:
    """Find the created Yandex AI task entity that satisfies a preference."""
    candidates = _ai_task_preference_candidates(hass, entry, preference_key)
    if not candidates:
        return None
    return str(candidates[0]["entity_id"])


def _ai_task_entity_display_name(
    subentry: ConfigSubentry, entity_id: str, state
) -> str:
    """Return a human readable name for an AI task entity."""
    friendly_name = state.attributes.get("friendly_name")
    if (
        isinstance(friendly_name, str)
        and friendly_name
        and friendly_name != entity_id
    ):
        return friendly_name
    return subentry.title or entity_id


async def _async_apply_ai_task_preferences(
    hass: HomeAssistant, desired: dict[str, str]
) -> bool:
    """Apply supported AI task preferences and persist the result."""
    preferences = hass.data.get(DATA_PREFERENCES)
    if preferences is None or not desired:
        return False

    supported_keys = set(getattr(preferences, "KEYS", ()))
    set_preferences = getattr(preferences, "async_set_preferences", None)
    if callable(set_preferences):
        try:
            signature = inspect.signature(set_preferences)
            supported_keys.update(
                parameter for parameter in signature.parameters
            )
        except (TypeError, ValueError):
            pass

    applied: list[str] = []
    if callable(set_preferences):
        kwargs = {
            preference_key: entity_id
            for preference_key, entity_id in desired.items()
            if preference_key in supported_keys
        }
        if kwargs:
            set_preferences(**kwargs)
            applied.extend(kwargs.values())
    else:
        for preference_key, entity_id in desired.items():
            if preference_key in supported_keys:
                setattr(preferences, preference_key, entity_id)
                applied.append(entity_id)

    if not applied:
        return False

    store = getattr(preferences, "_store", None)
    as_dict = getattr(preferences, "as_dict", None)
    if store is not None and callable(as_dict) and hasattr(store, "async_save"):
        try:
            await store.async_save(as_dict())
        except Exception:
            LOGGER.warning(
                "Could not persist AI task preferences", exc_info=True
            )
            return False

    return True


async def _async_install_ai_task_preferences(
    hass: HomeAssistant,
    entry: YandexAIConfigEntry,
    requested_preferences: list[str],
) -> None:
    """Apply AI task preferences selected during the initial configuration."""
    preferences = hass.data.get(DATA_PREFERENCES)
    if preferences is None:
        return

    desired: dict[str, str] = {}
    for preference_key in requested_preferences:
        feature = _ai_task_preference_feature(preference_key)
        if feature == 0:
            continue
        entity_id = _find_ai_task_entity_for_preference(
            hass, entry, preference_key
        )
        if entity_id:
            desired[preference_key] = entity_id

    if desired:
        await _async_apply_ai_task_preferences(hass, desired)


def _async_entity_supports_feature(
    hass: HomeAssistant, entity_id: str | None, feature: AITaskEntityFeature
) -> bool:
    """Return whether an AI task entity exists and supports the feature."""
    if not entity_id or feature == 0:
        return False
    state = hass.states.get(entity_id)
    if state is None:
        return False
    return bool(state.attributes.get("supported_features", 0) & feature)


def _async_problematic_ai_task_preferences(
    hass: HomeAssistant,
    entry: YandexAIConfigEntry,
) -> tuple[list[str], dict[str, list[dict[str, str | int]]]]:
    """Return unset or invalid AI task preferences with candidates."""
    preferences = hass.data.get(DATA_PREFERENCES)
    if preferences is None:
        return [], {}

    problem: list[str] = []
    candidates_by_key: dict[str, list[dict[str, str | int]]] = {}

    for preference_key in AI_TASK_PREFERENCE_ORDER:
        feature = _ai_task_preference_feature(preference_key)
        if feature == 0:
            continue
        current_entity_id = getattr(preferences, preference_key, None)
        if _async_entity_supports_feature(
            hass, current_entity_id, feature
        ):
            continue

        candidates = _ai_task_preference_candidates(
            hass, entry, preference_key
        )
        if candidates:
            problem.append(preference_key)
            candidates_by_key[preference_key] = candidates

    return problem, candidates_by_key


def _async_cleanup_old_ai_task_preference_issues(
    hass: HomeAssistant, entry: YandexAIConfigEntry
) -> None:
    """Delete per-preference issues created by older integration versions."""
    for preference_key in AI_TASK_PREFERENCE_ORDER:
        issue_registry.async_delete_issue(
            hass, DOMAIN, f"ai_task_{preference_key}_{entry.entry_id}"
        )


def _async_update_ai_task_preferences_issue(
    hass: HomeAssistant,
    entry: YandexAIConfigEntry,
    problem: list[str],
    candidates_by_key: dict[str, list[dict[str, str | int]]],
    acknowledged_keys: set[str] | None = None,
) -> None:
    """Create or update the unified AI task preferences issue."""
    issue_id = _ai_task_preferences_issue_id(entry)
    if not problem:
        issue_registry.async_delete_issue(hass, DOMAIN, issue_id)
        return

    problem_set = set(problem)
    acknowledged = {
        key for key in (acknowledged_keys or set()) if key in problem_set
    }
    visible_problem = [key for key in problem if key not in acknowledged]
    issue_registry.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        config_entry_id=entry.entry_id,
        is_fixable=True,
        is_persistent=True,
        severity=IssueSeverity.WARNING,
        translation_key=_ai_task_preferences_issue_key(len(problem)),
        translation_placeholders={
            "settings_url": AI_TASK_SETTINGS_URL,
        },
        data=_ai_task_preferences_issue_data(
            entry,
            visible_problem or problem,
            acknowledged,
        ),
        learn_more_url="homeassistant://config/ai",
    )
    issue_registry.async_ignore_issue(
        hass, DOMAIN, issue_id, not visible_problem
    )


async def _async_review_ai_task_preferences(
    hass: HomeAssistant, entry: YandexAIConfigEntry
) -> None:
    """Check AI task preferences and suggest created Yandex AI tasks."""
    _async_cleanup_old_ai_task_preference_issues(hass, entry)
    problem, candidates_by_key = _async_problematic_ai_task_preferences(
        hass, entry
    )
    issue_id = _ai_task_preferences_issue_id(entry)
    existing_issue = issue_registry.async_get(hass).async_get_issue(DOMAIN, issue_id)
    acknowledged_data = (
        (existing_issue.data or {}).get(
            AI_TASK_PREFERENCES_ACKNOWLEDGED_DATA_KEY
        )
        if existing_issue
        else None
    )
    acknowledged = _ai_task_acknowledged_preference_keys(acknowledged_data)
    _async_update_ai_task_preferences_issue(
        hass,
        entry,
        problem,
        candidates_by_key,
        acknowledged,
    )


def _async_ensure_default_subentries(
    hass: HomeAssistant, entry: YandexAIConfigEntry
) -> None:
    """Create default subentries for entries from before they were automatic.

    Runs once per entry, marked by the minor version, so subentries the
    user later removes are not recreated.
    """
    if entry.minor_version >= 3:
        return

    existing_subentries = list(entry.subentries.values())
    subentry_types = {subentry.subentry_type for subentry in existing_subentries}

    if SUBENTRY_TYPE_CONVERSATION not in subentry_types:
        conversation_title = (
            f"{friendly_model_label(RECOMMENDED_CONVERSATION_OPTIONS.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL))} "
            "Conversation"
        )
        hass.config_entries.async_add_subentry(
            entry,
            ConfigSubentry(
                data=MappingProxyType(RECOMMENDED_CONVERSATION_OPTIONS),
                subentry_type=SUBENTRY_TYPE_CONVERSATION,
                title=conversation_title,
                unique_id=None,
            ),
        )

    ai_task_subentries = [
        subentry
        for subentry in existing_subentries
        if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK_DATA
    ]
    has_image_task = any(
        CAP_TEXT_TO_IMAGE
        in model_capabilities(subentry.data.get(CONF_CHAT_MODEL, DEFAULT_AI_TASK_MODEL))
        for subentry in ai_task_subentries
    )
    has_text_task = any(
        CAP_TEXT_TO_TEXT
        in model_capabilities(subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL))
        for subentry in ai_task_subentries
    )

    if not has_image_task:
        image_model = RECOMMENDED_AI_TASK_OPTIONS.get(
            CONF_CHAT_MODEL, DEFAULT_AI_TASK_MODEL
        )
        image_task_title = f"{friendly_model_label(image_model)} Image Task"
        image_task_data = {
            **RECOMMENDED_AI_TASK_OPTIONS,
            CONF_AI_TASK_KIND: AI_TASK_KIND_IMAGE,
        }
        hass.config_entries.async_add_subentry(
            entry,
            ConfigSubentry(
                data=MappingProxyType(image_task_data),
                subentry_type=SUBENTRY_TYPE_AI_TASK_DATA,
                title=image_task_title,
                unique_id=None,
            ),
        )
        LOGGER.info("Created default image AI task for entry %s", entry.entry_id)

    if not has_text_task:
        text_model = RECOMMENDED_TEXT_AI_TASK_OPTIONS.get(
            CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL
        )
        text_task_title = f"{friendly_model_label(text_model)} Text Task"
        text_task_data = {
            **RECOMMENDED_TEXT_AI_TASK_OPTIONS,
            CONF_AI_TASK_KIND: AI_TASK_KIND_TEXT,
        }
        hass.config_entries.async_add_subentry(
            entry,
            ConfigSubentry(
                data=MappingProxyType(text_task_data),
                subentry_type=SUBENTRY_TYPE_AI_TASK_DATA,
                title=text_task_title,
                unique_id=None,
            ),
        )
        LOGGER.info("Created default text AI task for entry %s", entry.entry_id)

    hass.config_entries.async_update_entry(entry, minor_version=3)


async def async_unload_entry(hass: HomeAssistant, entry: YandexAIConfigEntry) -> bool:
    """Unload Yandex AI."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        await entry.runtime_data.close()
    return unloaded


async def async_migrate_integration(hass: HomeAssistant) -> None:
    """Migrate old (v1) config entries to the new subentry structure."""
    entries = [entry for entry in hass.config_entries.async_entries(DOMAIN) if entry.version == 1]
    if not entries:
        return

    LOGGER.info("Migrating Yandex AI to subentries (v2)")

    api_keys_entries: dict[tuple[str, str], tuple[YandexAIConfigEntry, bool]] = {}
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)

    for entry in entries:
        use_existing = False
        key = (entry.data[CONF_FOLDER_ID], entry.data[CONF_API_KEY])

        # Old single string / "none" → new list format
        options = dict(entry.options)
        if CONF_LLM_HASS_API in options:
            value = options[CONF_LLM_HASS_API]
            if isinstance(value, str):
                options[CONF_LLM_HASS_API] = [] if value == "none" else [value]

        # Client-level setting is moved from subentry options to entry data
        enable_server_data_logging = options.pop(
            CONF_ENABLE_SERVER_DATA_LOGGING, DEFAULT_ENABLE_SERVER_DATA_LOGGING
        )

        subentry = ConfigSubentry(
            data=MappingProxyType(options),
            subentry_type=SUBENTRY_TYPE_CONVERSATION,
            title=f"{friendly_model_label(options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL))} Conversation",
            unique_id=None,
        )

        if key not in api_keys_entries:
            use_existing = True
            all_disabled = all(
                e.disabled_by is not None for e in entries if (e.data[CONF_FOLDER_ID], e.data[CONF_API_KEY]) == key
            )
            api_keys_entries[key] = (entry, all_disabled)

        parent_entry, all_disabled = api_keys_entries[key]

        hass.config_entries.async_add_subentry(parent_entry, subentry)

        # Migrate conversation entity
        conversation_entity_id = entity_registry.async_get_entity_id(
            "conversation", DOMAIN, entry.entry_id
        )
        device = device_registry.async_get_device(identifiers={(DOMAIN, entry.entry_id)})

        if conversation_entity_id:
            conversation_entity_entry = entity_registry.entities[conversation_entity_id]
            entity_disabled_by = conversation_entity_entry.disabled_by
            if (entity_disabled_by is er.RegistryEntryDisabler.CONFIG_ENTRY and not all_disabled):
                entity_disabled_by = (
                    er.RegistryEntryDisabler.DEVICE if device else er.RegistryEntryDisabler.USER
                )
            entity_registry.async_update_entity(
                conversation_entity_id,
                config_entry_id=parent_entry.entry_id,
                config_subentry_id=subentry.subentry_id,
                disabled_by=entity_disabled_by,
                new_unique_id=subentry.subentry_id,
            )

        # Migrate device
        if device:
            device_disabled_by = device.disabled_by
            if (device_disabled_by is dr.DeviceEntryDisabler.CONFIG_ENTRY and not all_disabled):
                device_disabled_by = dr.DeviceEntryDisabler.USER
            device_registry.async_update_device(
                device.id,
                disabled_by=device_disabled_by,
                new_identifiers={(DOMAIN, subentry.subentry_id)},
                add_config_subentry_id=subentry.subentry_id,
                add_config_entry_id=parent_entry.entry_id,
            )
            device_registry.async_update_device(
                device.id, remove_config_entry_id=entry.entry_id
            )

        if not use_existing:
            await hass.config_entries.async_remove(entry.entry_id)
        else:
            hass.config_entries.async_update_entry(
                entry,
                title="Yandex AI",
                data={**entry.data, CONF_ENABLE_SERVER_DATA_LOGGING: enable_server_data_logging},
                options={},
                version=2,
                minor_version=1,
            )

    LOGGER.info("Migration to subentries completed successfully")
