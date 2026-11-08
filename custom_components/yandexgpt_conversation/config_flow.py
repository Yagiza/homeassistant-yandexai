"""Config flow for YandexGPT integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import httpx
import voluptuous as vol
from homeassistant.config_entries import (SOURCE_USER, ConfigEntry, ConfigFlow,
                                          ConfigFlowResult, ConfigSubentryFlow,
                                          SubentryFlowResult)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (NumberSelector,
                                            NumberSelectorConfig,
                                            SelectOptionDict, SelectSelector,
                                            SelectSelectorConfig,
                                            SelectSelectorMode,
                                            TemplateSelector)

from .client import YandexGPTApiError, YandexGPTClient
from .const import (ASSIST_PARTIALLY_SUPPORTED_MODELS,
                    ASSIST_UNSUPPORTED_MODELS, CAP_TEXT_TO_TEXT, CHAT_MODELS,
                    CONF_CHAT_MODEL, CONF_ENABLE_SERVER_DATA_LOGGING,
                    CONF_FOLDER_ID, CONF_MAX_TOKENS, CONF_MAX_TOOL_ITERATIONS,
                    CONF_MODEL_VERSION, CONF_NO_HA_DEFAULT_PROMPT, CONF_PROMPT,
                    CONF_RECOMMENDED, CONF_TEMPERATURE, DEFAULT_AI_TASK_NAME,
                    DEFAULT_CHAT_MODEL, DEFAULT_CONVERSATION_NAME,
                    DEFAULT_ENABLE_SERVER_DATA_LOGGING,
                    DEFAULT_INSTRUCTIONS_PROMPT_RU,
                    DEFAULT_MAX_TOOL_ITERATIONS, DEFAULT_MODEL_VERSION,
                    DEFAULT_NO_HA_DEFAULT_PROMPT, DOC_CHAT_MODELS_URL,
                    DOC_PRICING_URL, DOC_PROMPT_TEMPLATES_URL, DOMAIN, IMAGE_MODELS,
                    IMAGE_MODEL_LABELS, LOGGER, RECOMMENDED_AI_TASK_OPTIONS,
                    RECOMMENDED_CONVERSATION_OPTIONS, RECOMMENDED_MAX_TOKENS,
                    RECOMMENDED_TEMPERATURE, SUBENTRY_TYPE_AI_TASK_DATA,
                    SUBENTRY_TYPE_CONVERSATION, model_capabilities,
                    model_capabilities_label)
STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_FOLDER_ID): str,
        vol.Required(CONF_API_KEY): str,
        vol.Required(
            CONF_ENABLE_SERVER_DATA_LOGGING,
            default=DEFAULT_ENABLE_SERVER_DATA_LOGGING,
        ): bool,
    }
)


def _model_name(model_id: str | None) -> str | None:
    """Extract the short model name from a model value (plain or gpt:// id)."""
    if not model_id:
        return None
    if "://" not in model_id:
        return model_id.split("/")[0]
    # gpt://<folder>/<name>/<version>
    parts = model_id.split("/")
    return parts[3] if len(parts) == 5 else None


def _model_label(model_id: str) -> str:
    """Build a human-readable label from a model id like gpt://folder/name/version."""
    parts = model_id.split("/")
    if len(parts) == 5:
        name, version = parts[3], parts[4]
        return name if version == "latest" else f"{name} ({version})"
    return model_id


class YandexGPTConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for YandexGPT."""

    VERSION = 2
    MINOR_VERSION = 2

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=STEP_USER_DATA_SCHEMA,
            )

        errors: dict[str, str] = {}
        self._async_abort_entries_match(user_input)

        client = YandexGPTClient(
            user_input[CONF_FOLDER_ID], user_input[CONF_API_KEY], timeout=10.0
        )
        try:
            await client.list_models()
        except YandexGPTApiError as err:
            if err.status_code in (401, 403):
                errors[CONF_API_KEY] = "invalid_credentials"
            else:
                LOGGER.warning("Could not validate credentials: %s", err)
        except httpx.HTTPError as err:
            LOGGER.warning("Could not validate credentials: %s", err)
        finally:
            await client.close()

        if errors:
            return self.async_show_form(
                step_id="user",
                data_schema=STEP_USER_DATA_SCHEMA,
                errors=errors,
            )

        return self.async_create_entry(
            title="YandexGPT",
            data=user_input,
            subentries=[
                {
                    "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                    "data": RECOMMENDED_CONVERSATION_OPTIONS,
                    "title": DEFAULT_CONVERSATION_NAME,
                    "unique_id": None,
                },
                {
                    "subentry_type": SUBENTRY_TYPE_AI_TASK_DATA,
                    "data": RECOMMENDED_AI_TASK_OPTIONS,
                    "title": DEFAULT_AI_TASK_NAME,
                    "unique_id": None,
                },
            ],
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """User initiated reconfiguration."""
        entry = self._get_reconfigure_entry()

        if user_input is not None:
            return self.async_update_reload_and_abort(entry, data_updates=user_input)

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=STEP_USER_DATA_SCHEMA,
        )

    @classmethod
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this handler."""
        return {
            SUBENTRY_TYPE_CONVERSATION: ConversationFlowHandler,
            SUBENTRY_TYPE_AI_TASK_DATA: AITaskDataFlowHandler,
        }


class ConversationFlowHandler(ConfigSubentryFlow):
    """Handle conversation subentry flow."""

    def __init__(self) -> None:
        """Initialize."""
        self.last_rendered_recommended = False
        self.options: dict[str, Any] | MappingProxyType[str, Any] = {}
        self.errors: dict[str, str] = {}
        self.last_selected_model: str | None = None
        self._chat_model_options: list[SelectOptionDict] | None = None

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Create a new conversation subentry."""
        self.options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
        self.last_rendered_recommended = self.options[CONF_RECOMMENDED]
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an existing conversation subentry."""
        self.options = self._get_reconfigure_subentry().data.copy()
        self.last_rendered_recommended = self.options.get(CONF_RECOMMENDED, False)
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage conversation subentry configuration."""
        if user_input is not None:
            recommended = user_input[CONF_RECOMMENDED]

            if recommended == self.last_rendered_recommended:
                if not user_input.get(CONF_LLM_HASS_API):
                    user_input.pop(CONF_LLM_HASS_API, None)

                self._validate_selected_model(user_input)

                if self.errors:
                    self.options = {**self.options, **user_input}
                elif self._is_new:
                    title = (
                        _model_name(user_input.get(CONF_CHAT_MODEL))
                        or DEFAULT_CHAT_MODEL
                    )
                    return self.async_create_entry(title=title, data=user_input)
                else:
                    return self.async_update_and_abort(
                        self._get_entry(),
                        self._get_reconfigure_subentry(),
                        data=user_input,
                    )
            else:
                # Re-render the form with the recommended options shown/hidden
                self.last_rendered_recommended = recommended
                self.options = dict(user_input)

        if self._chat_model_options is None:
            self._chat_model_options = await self._async_load_chat_models()

        suggested_values = dict(self.options)
        if not suggested_values.get(CONF_PROMPT):
            suggested_values[CONF_PROMPT] = DEFAULT_INSTRUCTIONS_PROMPT_RU

        # Migration: old single string / "none" → new list format
        if CONF_LLM_HASS_API in suggested_values:
            value = suggested_values[CONF_LLM_HASS_API]
            if isinstance(value, str):
                suggested_values[CONF_LLM_HASS_API] = [] if value == "none" else [value]

        if suggested_values.get(CONF_CHAT_MODEL):
            model_value = suggested_values[CONF_CHAT_MODEL]
            if "://" not in model_value:
                suggested_values[CONF_CHAT_MODEL] = model_value.split("/")[0]

        schema = self.add_suggested_values_to_schema(
            vol.Schema(
                yandexgpt_config_option_schema(
                    self.hass, suggested_values, self._chat_model_options
                )
            ),
            suggested_values,
        )

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=self.errors,
            description_placeholders={
                "chat_model_doc_url": DOC_CHAT_MODELS_URL,
                "pricing_doc_url": DOC_PRICING_URL,
                "prompt_templates_url": DOC_PROMPT_TEMPLATES_URL,
            },
        )

    def _validate_selected_model(self, user_input: dict[str, Any]) -> None:
        """Validate model when any LLM HASS API (Assist or MCP) is selected."""
        llm_apis: list[str] = user_input.get(CONF_LLM_HASS_API) or []
        if isinstance(llm_apis, str):  # safety for old data
            llm_apis = [llm_apis] if llm_apis != "none" else []

        if not llm_apis:
            self.errors.pop(CONF_CHAT_MODEL, None)
            return

        selected_model = user_input.get(CONF_CHAT_MODEL)
        model_name = _model_name(selected_model)
        if model_name in ASSIST_UNSUPPORTED_MODELS:
            self.errors[CONF_CHAT_MODEL] = "model_not_supported_for_tools"
        elif model_name in ASSIST_PARTIALLY_SUPPORTED_MODELS:
            if self.last_selected_model == selected_model:
                self.errors.pop(CONF_CHAT_MODEL, None)
            else:
                self.errors[CONF_CHAT_MODEL] = "model_partially_supported_for_tools"

        self.last_selected_model = selected_model

    async def _async_load_chat_models(self) -> list[SelectOptionDict]:
        """Load the available chat models, falling back to a static list."""
        try:
            client: YandexGPTClient = self._get_entry().runtime_data
            models = await client.list_models()
        except (httpx.HTTPError, YandexGPTApiError) as err:
            LOGGER.warning("Failed to load the model list: %s", err)
            models = []

        options = [
            SelectOptionDict(value=model["id"], label=_model_label(model["id"]))
            for model in models
        ]
        if not options:
            options = [
                SelectOptionDict(value=value, label=label)
                for value, label in CHAT_MODELS
            ]
        return options


class AITaskDataFlowHandler(ConfigSubentryFlow):
    """Handle AI task subentry flow."""

    def __init__(self) -> None:
        """Initialize."""
        self.options: dict[str, Any] = {}
        self.model_labels: dict[str, str] = {}
        self._rendered_model: str | None = None

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create an AI task."""
        self.options = {}
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of an AI task."""
        self.options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create / reconfigure an AI task."""
        if self._rendered_model is None:
            self._rendered_model = self.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)

        if user_input is not None:
            model_id = user_input[CONF_CHAT_MODEL]
            text_changed = (
                CAP_TEXT_TO_TEXT in model_capabilities(model_id)
            ) != (
                CAP_TEXT_TO_TEXT in model_capabilities(self._rendered_model)
            )
            if model_id == self._rendered_model or not text_changed:
                data: dict[str, Any] = {CONF_CHAT_MODEL: model_id}
                if CAP_TEXT_TO_TEXT in model_capabilities(model_id):
                    data[CONF_MAX_TOKENS] = (
                        user_input.get(CONF_MAX_TOKENS) or RECOMMENDED_MAX_TOKENS
                    )
                title = self.model_labels.get(model_id, _model_label(model_id))
                if self._is_new:
                    return self.async_create_entry(title=title, data=data)
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=data,
                )
            # Text<->image model: re-render to show/hide max tokens
            self.options = {**self.options, **user_input}
            self._rendered_model = model_id

        model_id = self._rendered_model

        model_options: list[SelectOptionDict] = []
        self.model_labels = {}

        def add_option(value: str, label: str) -> None:
            model_options.append(
                SelectOptionDict(
                    value=value,
                    label=f"{label} ({model_capabilities_label(value)})",
                )
            )
            self.model_labels[value] = label

        try:
            client: YandexGPTClient = self._get_entry().runtime_data
            models = await client.list_models()
        except (httpx.HTTPError, YandexGPTApiError) as err:
            LOGGER.warning("Failed to fetch models for AI Task: %s", err)
            models = []

        for model in models:
            add_option(model["id"], _model_label(model["id"]))
        if not models:
            for value, label in CHAT_MODELS:
                add_option(value, label)

        for image_model in IMAGE_MODELS:
            add_option(
                image_model, IMAGE_MODEL_LABELS.get(image_model, image_model)
            )

        schema: dict = {
            vol.Required(
                CONF_CHAT_MODEL,
                default=self.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=model_options,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
        }
        if CAP_TEXT_TO_TEXT in model_capabilities(model_id):
            schema[
                vol.Optional(
                    CONF_MAX_TOKENS,
                    description={"suggested_value": self.options.get(CONF_MAX_TOKENS)},
                    default=RECOMMENDED_MAX_TOKENS,
                )
            ] = int

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
            description_placeholders={
                "chat_model_doc_url": DOC_CHAT_MODELS_URL,
                "pricing_doc_url": DOC_PRICING_URL,
            },
        )


def yandexgpt_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
    chat_model_options: list[SelectOptionDict],
) -> dict:
    """Return a schema for YandexGPT completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    ]

    schema = {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT)},
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
        ): SelectSelector(
            SelectSelectorConfig(
                options=hass_apis,
                multiple=True,
                translation_key=CONF_LLM_HASS_API,
            )
        ),
        vol.Optional(
            CONF_CHAT_MODEL,
            description={"suggested_value": options.get(CONF_CHAT_MODEL)},
            default=DEFAULT_CHAT_MODEL,
        ): SelectSelector(
            SelectSelectorConfig(
                mode=SelectSelectorMode.DROPDOWN,
                options=chat_model_options,
                custom_value=True,
            )
        ),
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

    if options.get(CONF_RECOMMENDED):
        return schema

    model_versions = [
        SelectOptionDict(label="Deprecated", value="deprecated"),
        SelectOptionDict(label="Latest", value="latest"),
        SelectOptionDict(label="Release Candidate", value="rc"),
    ]

    schema.update(
        {
            vol.Optional(
                CONF_MODEL_VERSION,
                description={"suggested_value": options.get(CONF_MODEL_VERSION)},
                default=DEFAULT_MODEL_VERSION,
            ): SelectSelector(
                SelectSelectorConfig(mode=SelectSelectorMode.DROPDOWN, options=model_versions)
            ),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_NO_HA_DEFAULT_PROMPT,
                description={"suggested_value": options.get(CONF_NO_HA_DEFAULT_PROMPT, DEFAULT_NO_HA_DEFAULT_PROMPT)},
                default=options.get(CONF_NO_HA_DEFAULT_PROMPT, DEFAULT_NO_HA_DEFAULT_PROMPT),
            ): bool,
            vol.Optional(
                CONF_MAX_TOOL_ITERATIONS,
                description={"suggested_value": options.get(CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS)},
                default=options.get(CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS),
            ): int,
        }
    )
    return schema
