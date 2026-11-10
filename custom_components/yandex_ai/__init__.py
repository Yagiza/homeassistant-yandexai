"""The Yandex AI integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from types import MappingProxyType

import httpx
from homeassistant.config_entries import (ConfigEntry, ConfigSubentry)
from homeassistant.const import (CONF_API_KEY, CONF_LLM_HASS_API, Platform)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.typing import ConfigType

from .client import YandexGPTApiError, YandexGPTClient
from .const import (CONF_ENABLE_SERVER_DATA_LOGGING, CONF_FOLDER_ID,
                    DEFAULT_AI_TASK_NAME, DEFAULT_CONVERSATION_NAME,
                    DEFAULT_ENABLE_SERVER_DATA_LOGGING, DOMAIN, LOGGER,
                    RECOMMENDED_AI_TASK_OPTIONS,
                    RECOMMENDED_CONVERSATION_OPTIONS,
                    SUBENTRY_TYPE_AI_TASK_DATA, SUBENTRY_TYPE_CONVERSATION)

PLATFORMS = (Platform.CONVERSATION, Platform.AI_TASK)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type YandexGPTConfigEntry = ConfigEntry[YandexGPTClient]  # type: ignore[name-defined]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Yandex AI component."""
    await async_migrate_integration(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: YandexGPTConfigEntry) -> bool:
    """Set up Yandex AI from a config entry."""
    _async_ensure_default_subentries(hass, entry)

    enable_server_data_logging = entry.data.get(
        CONF_ENABLE_SERVER_DATA_LOGGING, DEFAULT_ENABLE_SERVER_DATA_LOGGING
    )

    client = YandexGPTClient(
        folder_id=entry.data[CONF_FOLDER_ID],
        api_key=entry.data[CONF_API_KEY],
        enable_server_data_logging=enable_server_data_logging,
        timeout=10.0,
    )
    try:
        await client.list_models()
    except YandexGPTApiError as err:
        if err.status_code in (401, 403):
            LOGGER.error("Invalid API key or folder ID: %s", err)
            return False
        raise ConfigEntryNotReady(f"Cannot connect to Yandex AI Studio: {err}") from err
    except httpx.HTTPError as err:
        raise ConfigEntryNotReady(f"Cannot connect to Yandex AI Studio: {err}") from err
    finally:
        await client.close()

    entry.runtime_data = YandexGPTClient(
        folder_id=entry.data[CONF_FOLDER_ID],
        api_key=entry.data[CONF_API_KEY],
        enable_server_data_logging=enable_server_data_logging,
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Reload when subentry options change
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    return True


async def _async_update_listener(hass: HomeAssistant, entry: YandexGPTConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


def _async_ensure_default_subentries(
    hass: HomeAssistant, entry: YandexGPTConfigEntry
) -> None:
    """Create default subentries for entries from before they were automatic.

    Runs once per entry, marked by the minor version, so subentries the
    user later removes are not recreated.
    """
    if entry.minor_version >= 2:
        return

    subentry_types = {
        subentry.subentry_type for subentry in entry.subentries.values()
    }

    if SUBENTRY_TYPE_CONVERSATION not in subentry_types:
        hass.config_entries.async_add_subentry(
            entry,
            ConfigSubentry(
                data=MappingProxyType(RECOMMENDED_CONVERSATION_OPTIONS),
                subentry_type=SUBENTRY_TYPE_CONVERSATION,
                title=DEFAULT_CONVERSATION_NAME,
                unique_id=None,
            ),
        )

    if SUBENTRY_TYPE_AI_TASK_DATA not in subentry_types:
        hass.config_entries.async_add_subentry(
            entry,
            ConfigSubentry(
                data=MappingProxyType(RECOMMENDED_AI_TASK_OPTIONS),
                subentry_type=SUBENTRY_TYPE_AI_TASK_DATA,
                title=DEFAULT_AI_TASK_NAME,
                unique_id=None,
            ),
        )
        LOGGER.info("Created default AI task subentry for entry %s", entry.entry_id)

    hass.config_entries.async_update_entry(entry, minor_version=2)


async def async_unload_entry(hass: HomeAssistant, entry: YandexGPTConfigEntry) -> bool:
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

    api_keys_entries: dict[tuple[str, str], tuple[YandexGPTConfigEntry, bool]] = {}
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
            title=entry.title,
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


async def async_migrate_entry(hass: HomeAssistant, entry: YandexGPTConfigEntry) -> bool:
    """Migrate config entry (for future versions)."""
    LOGGER.debug("Migrating from version %s.%s", entry.version, entry.minor_version)

    if entry.version > 2:
        return False

    if entry.version < 2:
        hass.config_entries.async_update_entry(entry, version=2, minor_version=1)

    return True
