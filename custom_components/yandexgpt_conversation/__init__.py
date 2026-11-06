"""The YandexGPT integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .client import YandexGPTClient
from .const import (CONF_ENABLE_SERVER_DATA_LOGGING, CONF_FOLDER_ID,
                    DEFAULT_ENABLE_SERVER_DATA_LOGGING, DOMAIN)

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, entry: ConfigType) -> bool:
    """Set up the YandexGPT component."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up YandexGPT from a config entry."""
    settings = {**entry.data, **entry.options}

    client = YandexGPTClient(
        folder_id=settings[CONF_FOLDER_ID],
        api_key=settings[CONF_API_KEY],
        enable_server_data_logging=settings.get(
            CONF_ENABLE_SERVER_DATA_LOGGING, DEFAULT_ENABLE_SERVER_DATA_LOGGING
        ),
    )

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload YandexGPT."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        await entry.runtime_data.close()
    return unloaded
