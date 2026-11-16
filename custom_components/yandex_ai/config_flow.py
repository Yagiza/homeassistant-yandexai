"""Config flow for Yandex AI integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import httpx
import voluptuous as vol
from homeassistant.components.ai_task import AITaskEntityFeature
from homeassistant.components.ai_task.const import DATA_PREFERENCES
from homeassistant.config_entries import (SOURCE_USER, ConfigEntry, ConfigFlow,
                                          ConfigFlowResult, ConfigSubentryFlow,
                                          SubentryFlowResult)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (BooleanSelector, NumberSelector,
                                             NumberSelectorConfig,
                                             SelectOptionDict, SelectSelector,
                                             SelectSelectorConfig,
                                             SelectSelectorMode,
                                             TemplateSelector, TextSelector,
                                             TextSelectorConfig)

from .client import YandexAIStudioError, YandexAIClient
from .const import (AI_TASK_KIND_IMAGE, AI_TASK_KIND_TEXT,
                    ASSIST_PARTIALLY_SUPPORTED_MODELS,
                    ASSIST_UNSUPPORTED_MODELS, CAP_TEXT_TO_IMAGE,
                    CAP_TEXT_TO_TEXT, CHAT_MODELS, CONF_AI_TASK_KIND,
                    CONF_CHAT_MODEL, CONF_ENABLE_SERVER_DATA_LOGGING,
                    CONF_FOLDER_ID, CONF_FREQUENCY_PENALTY, CONF_IMAGE_N,
                    CONF_IMAGE_NEGATIVE_PROMPT, CONF_IMAGE_SEED,
                    CONF_IMAGE_SIZE, CONF_MAX_TOKENS,
                    CONF_MAX_TOOL_ITERATIONS,
                    CONF_MODEL_VERSION, CONF_NO_HA_DEFAULT_PROMPT,
                    CONF_PENDING_AI_TASK_PREFERENCES,
                    CONF_PRESENCE_PENALTY, CONF_PROMPT,
                    CONF_REASONING_EFFORT, CONF_RECOMMENDED, CONF_SEED,
                    CONF_STOP, CONF_TEMPERATURE, CONF_TOP_K, CONF_TOP_P,
                    DEFAULT_AI_TASK_MODEL, DEFAULT_CHAT_MODEL,
                    DEFAULT_ENABLE_SERVER_DATA_LOGGING,
                    DEFAULT_INSTRUCTIONS_PROMPT_RU,
                    DEFAULT_MAX_TOOL_ITERATIONS,
                    DEFAULT_MODEL_VERSION, DOC_CHAT_MODELS_URL,
                    DOC_PRICING_URL, DOC_PROMPT_TEMPLATES_URL, DOMAIN,
                    IMAGE_MODELS, IMAGE_MODEL_LABELS, IMAGE_MODEL_VERSION,
                    LOGGER, MAX_IMAGE_N,
                    MIN_IMAGE_N, REASONING_EFFORT_DEFAULT,
                    REASONING_EFFORT_VALUES,
                    RECOMMENDED_CONVERSATION_OPTIONS,
                    RECOMMENDED_MAX_TOKENS, RECOMMENDED_TEMPERATURE,
                    SUBENTRY_TYPE_AI_TASK_DATA,
                    SUBENTRY_TYPE_CONVERSATION, friendly_model_label,
                    model_capabilities, model_capabilities_label,
                    model_image_options, model_reasoning_effort_kind,
                    normalize_stop_text)
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


def _model_version_of(model_id: str | None) -> str | None:
    """Return the version segment of a full gpt:// id, else None."""
    if not model_id or "://" not in model_id:
        return None
    parts = model_id.split("/")
    return parts[4] if len(parts) == 5 else None


def _base_or_value(value: str | None) -> str | None:
    """Return the base model name for display, falling back to the raw value."""
    if not value:
        return value
    return _model_name(value) or value


def _versions_by_base(
    chat_options: list[SelectOptionDict],
) -> dict[str, list[str]]:
    """Map each base model name to the versions available in the catalog."""
    versions: dict[str, list[str]] = {}
    for option in chat_options:
        value = option.get("value", "")
        base = _model_name(value)
        version = _model_version_of(value)
        if not base or not version:
            continue
        bucket = versions.setdefault(base, [])
        if version not in bucket:
            bucket.append(version)
    return versions


def _version_select_options(
    model_id: str | None, versions_by_base: dict[str, list[str]]
) -> list[SelectOptionDict]:
    """Return the version options available for the selected base model."""
    base = _model_name(model_id)
    available = list(versions_by_base.get(base, [])) if base else []
    ordered = [
        version for version in ("latest", "rc", "deprecated") if version in available
    ]
    ordered.extend(version for version in available if version not in ordered)
    if not ordered:
        ordered = [DEFAULT_MODEL_VERSION]
    labels = {
        "latest": "Latest",
        "rc": "Release Candidate",
        "deprecated": "Deprecated",
    }
    return [
        SelectOptionDict(label=labels.get(version, version), value=version)
        for version in ordered
    ]


def _version_select_default(
    options: dict[str, Any] | MappingProxyType[str, Any],
    version_options: list[SelectOptionDict],
) -> str:
    """Return a valid non-empty default version for the select."""
    current = str(options.get(CONF_MODEL_VERSION) or "").strip()
    for option in version_options:
        if option.get("value") == current:
            return current
    return version_options[0].get("value") or DEFAULT_MODEL_VERSION


def _effective_version(
    model_id: str | None,
    stored_version: str | None,
    versions_by_base: dict[str, list[str]],
) -> str:
    """Resolve the suggested version, keeping any version baked into a full id."""
    base = _model_name(model_id)
    embedded = _model_version_of(model_id)
    if base and embedded and embedded in versions_by_base.get(base, []):
        return embedded
    if stored_version:
        return stored_version
    return DEFAULT_MODEL_VERSION


def _normalize_model_choice(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a selected/stored model value to its base name."""
    raw_model = data.get(CONF_CHAT_MODEL)
    if not raw_model:
        return data
    data[CONF_CHAT_MODEL] = _base_or_value(raw_model) or raw_model
    embedded_version = _model_version_of(raw_model)
    if embedded_version and not data.get(CONF_MODEL_VERSION):
        data[CONF_MODEL_VERSION] = embedded_version
    return data


def _image_settings_submission(
    base: dict[str, Any],
    user_input: dict[str, Any],
    include_version: bool,
) -> dict[str, Any]:
    """Build image settings data without resurrecting stale optional values."""
    data = dict(base)
    for key in (
        CONF_IMAGE_SIZE,
        CONF_IMAGE_N,
        CONF_IMAGE_SEED,
        CONF_IMAGE_NEGATIVE_PROMPT,
    ):
        data.pop(key, None)
    if not include_version:
        data.pop(CONF_MODEL_VERSION, None)
    return _normalize_model_choice({**data, **dict(user_input)})


def _build_model_options(
    chat_options: list[SelectOptionDict], capability: str | None = None
) -> list[SelectOptionDict]:
    """Build model options, using base names for gpt models."""
    model_options: list[SelectOptionDict] = []
    seen: set[str] = set()

    def add_option(value: str, label: str) -> None:
        if value in seen:
            return
        if capability is not None and capability not in model_capabilities(value):
            return
        seen.add(value)
        model_options.append(
            SelectOptionDict(
                value=value,
                label=f"{label} ({model_capabilities_label(value)})",
            )
        )

    for option in chat_options:
        base = _model_name(option.get("value", ""))
        if base is None:
            continue
        add_option(base, friendly_model_label(base))
    for image_model in IMAGE_MODELS:
        add_option(image_model, IMAGE_MODEL_LABELS.get(image_model, image_model))
    return model_options


GENERATION_OPTION_KEYS = (
    CONF_REASONING_EFFORT,
    CONF_TOP_P,
    CONF_STOP,
    CONF_SEED,
    CONF_FREQUENCY_PENALTY,
    CONF_PRESENCE_PENALTY,
    CONF_TOP_K,
)
_REASONING_EFFORT_QWEN_FIELD = "reasoning_effort_qwen"
_REASONING_EFFORT_DEEPSEEK_FIELD = "reasoning_effort_deepseek"


def _is_default_advanced_value(key: str, value: Any) -> bool:
    """Return whether an advanced option value is still the default."""
    if value is None or value == "":
        return True
    if key == CONF_REASONING_EFFORT:
        return value in (REASONING_EFFORT_DEFAULT, True)
    if key == CONF_NO_HA_DEFAULT_PROMPT:
        return not value
    if key == CONF_MAX_TOOL_ITERATIONS:
        try:
            return int(value) == DEFAULT_MAX_TOOL_ITERATIONS
        except (TypeError, ValueError):
            return False
    if key == CONF_MAX_TOKENS:
        try:
            return int(value) == RECOMMENDED_MAX_TOKENS
        except (TypeError, ValueError):
            return False
    if key == CONF_TEMPERATURE:
        try:
            return float(value) == RECOMMENDED_TEMPERATURE
        except (TypeError, ValueError):
            return False
    if key == CONF_STOP:
        if isinstance(value, (list, tuple)):
            return not any(str(part).strip() for part in value)
        return not str(value).strip()
    return False


def _advanced_options_are_defaults(
    options: dict[str, Any], *, conversation: bool = False
) -> bool:
    """Return whether text-generation advanced options are all defaults."""
    version = str(options.get(CONF_MODEL_VERSION) or "").strip()
    if version and version != DEFAULT_MODEL_VERSION:
        return False
    keys = set(GENERATION_OPTION_KEYS)
    keys.update({CONF_TEMPERATURE, CONF_MAX_TOKENS})
    if conversation:
        keys.update({CONF_MAX_TOOL_ITERATIONS, CONF_NO_HA_DEFAULT_PROMPT})
    return all(
        _is_default_advanced_value(key, options.get(key)) for key in keys
    )


def _image_options_are_defaults(
    options: dict[str, Any], model_id: str
) -> bool:
    """Return whether image-generation options are all defaults."""
    version = str(options.get(CONF_MODEL_VERSION) or IMAGE_MODEL_VERSION).strip()
    if version != IMAGE_MODEL_VERSION:
        return False
    return all(
        _is_default_advanced_value(key, options.get(key))
        for key in model_image_options(model_id)
    )


def _map_reasoning_field(
    user_input: dict[str, Any]
) -> dict[str, Any]:
    """Map the model-specific reasoning field back to the stored key."""
    data = dict(user_input)
    for field in (
        _REASONING_EFFORT_QWEN_FIELD,
        _REASONING_EFFORT_DEEPSEEK_FIELD,
    ):
        if field in data:
            data[CONF_REASONING_EFFORT] = data.pop(field)
            break
    return data


def _clean_generation_options(
    model_id: str | None, user_input: dict[str, Any]
) -> dict[str, Any]:
    """Normalize generation options before saving them."""
    data: dict[str, Any] = {}
    kind = model_reasoning_effort_kind(model_id)
    user_input = _map_reasoning_field(user_input)
    raw_reasoning = user_input.get(CONF_REASONING_EFFORT)
    if kind == "qwen":
        if raw_reasoning in (False, "none"):
            data[CONF_REASONING_EFFORT] = "none"
    elif kind == "deepseek":
        if raw_reasoning in REASONING_EFFORT_VALUES:
            data[CONF_REASONING_EFFORT] = raw_reasoning

    numeric_options = {
        CONF_TOP_P: float,
        CONF_SEED: int,
        CONF_FREQUENCY_PENALTY: float,
        CONF_PRESENCE_PENALTY: float,
        CONF_TOP_K: int,
    }
    for key, caster in numeric_options.items():
        value = user_input.get(key)
        if value not in (None, ""):
            data[key] = caster(value)

    stop = normalize_stop_text(user_input.get(CONF_STOP))
    if stop is not None:
        data[CONF_STOP] = stop

    return data


def _clean_conversation_data(user_input: dict[str, Any]) -> dict[str, Any]:
    """Normalize conversation options before saving them."""
    if user_input.get(CONF_RECOMMENDED):
        data = RECOMMENDED_CONVERSATION_OPTIONS.copy()
        raw_model = user_input.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        data[CONF_CHAT_MODEL] = _base_or_value(raw_model) or DEFAULT_CHAT_MODEL
        if embedded_version := _model_version_of(raw_model):
            data[CONF_MODEL_VERSION] = embedded_version
        prompt = user_input.get(CONF_PROMPT)
        data[CONF_PROMPT] = (
            prompt if prompt else RECOMMENDED_CONVERSATION_OPTIONS[CONF_PROMPT]
        )
        if CONF_LLM_HASS_API in user_input:
            llm_apis = user_input[CONF_LLM_HASS_API]
            if isinstance(llm_apis, str):
                llm_apis = [] if llm_apis == "none" else [llm_apis]
            if llm_apis:
                data[CONF_LLM_HASS_API] = llm_apis
            else:
                data.pop(CONF_LLM_HASS_API, None)
        data[CONF_RECOMMENDED] = True
        return data

    raw_model = user_input.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
    model_id = _base_or_value(raw_model) or DEFAULT_CHAT_MODEL
    data: dict[str, Any] = {
        CONF_CHAT_MODEL: model_id,
        CONF_PROMPT: user_input.get(CONF_PROMPT, ""),
        CONF_RECOMMENDED: False,
    }
    if user_input.get(CONF_LLM_HASS_API):
        llm_apis = user_input[CONF_LLM_HASS_API]
        if isinstance(llm_apis, str):
            llm_apis = [] if llm_apis == "none" else [llm_apis]
        if llm_apis:
            data[CONF_LLM_HASS_API] = llm_apis
    model_version = user_input.get(CONF_MODEL_VERSION) or _model_version_of(raw_model)
    if model_version:
        data[CONF_MODEL_VERSION] = model_version
    if user_input.get(CONF_TEMPERATURE) not in (None, ""):
        data[CONF_TEMPERATURE] = float(user_input[CONF_TEMPERATURE])
    if user_input.get(CONF_MAX_TOKENS) not in (None, ""):
        data[CONF_MAX_TOKENS] = int(user_input[CONF_MAX_TOKENS])
    if user_input.get(CONF_NO_HA_DEFAULT_PROMPT):
        data[CONF_NO_HA_DEFAULT_PROMPT] = True
    if user_input.get(CONF_MAX_TOOL_ITERATIONS) not in (None, ""):
        data[CONF_MAX_TOOL_ITERATIONS] = int(user_input[CONF_MAX_TOOL_ITERATIONS])
    data.update(_clean_generation_options(model_id, user_input))
    return data


def _clean_ai_task_data(
    model_id: str, user_input: dict[str, Any]
) -> dict[str, Any]:
    """Normalize AI task options before saving them."""
    recommended = bool(user_input.get(CONF_RECOMMENDED, True))
    base_model = _base_or_value(model_id) or DEFAULT_CHAT_MODEL
    model_version = (
        user_input.get(CONF_MODEL_VERSION)
        or _model_version_of(model_id)
        or DEFAULT_MODEL_VERSION
    )
    data: dict[str, Any] = {
        CONF_CHAT_MODEL: base_model,
        CONF_RECOMMENDED: recommended,
        CONF_MODEL_VERSION: model_version,
    }
    if user_input.get(CONF_MAX_TOKENS) not in (None, ""):
        data[CONF_MAX_TOKENS] = int(user_input[CONF_MAX_TOKENS])
    if user_input.get(CONF_TEMPERATURE) not in (None, ""):
        data[CONF_TEMPERATURE] = float(user_input[CONF_TEMPERATURE])
    if not recommended:
        data.update(_clean_generation_options(model_id, user_input))
    return data


def _clean_image_task_data(
    model_id: str, user_input: dict[str, Any]
) -> dict[str, Any]:
    """Normalize image generation options before saving them."""
    base_model = _base_or_value(model_id) or DEFAULT_AI_TASK_MODEL
    data: dict[str, Any] = {CONF_CHAT_MODEL: base_model}
    raw_model = user_input.get(CONF_CHAT_MODEL) or model_id
    version = str(
        user_input.get(CONF_MODEL_VERSION) or _model_version_of(raw_model) or ""
    ).strip()
    if version:
        data[CONF_MODEL_VERSION] = version
    supported = model_image_options(base_model)

    if CONF_IMAGE_SIZE in supported:
        size = str(user_input.get(CONF_IMAGE_SIZE) or "").strip()
        if size:
            data[CONF_IMAGE_SIZE] = size

    if CONF_IMAGE_N in supported:
        value = user_input.get(CONF_IMAGE_N)
        if value not in (None, ""):
            try:
                image_n = int(value)
            except (TypeError, ValueError):
                image_n = None
            if image_n is not None and MIN_IMAGE_N <= image_n <= MAX_IMAGE_N:
                data[CONF_IMAGE_N] = image_n

    if CONF_IMAGE_SEED in supported:
        value = user_input.get(CONF_IMAGE_SEED)
        if value not in (None, ""):
            try:
                data[CONF_IMAGE_SEED] = int(value)
            except (TypeError, ValueError):
                pass

    if CONF_IMAGE_NEGATIVE_PROMPT in supported:
        negative = str(user_input.get(CONF_IMAGE_NEGATIVE_PROMPT) or "").strip()
        if negative:
            data[CONF_IMAGE_NEGATIVE_PROMPT] = negative

    return data


def _ai_task_preference_keys_to_offer(
    image_task_data: dict[str, Any], text_task_data: dict[str, Any]
) -> dict[str, AITaskEntityFeature]:
    """Return AI task preferences that can be filled by created tasks."""
    text_capabilities = model_capabilities(
        text_task_data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
    )
    image_capabilities = model_capabilities(
        image_task_data.get(CONF_CHAT_MODEL, DEFAULT_AI_TASK_MODEL)
    )
    generate_data_feature = getattr(AITaskEntityFeature, "GENERATE_DATA", 0)
    generate_image_feature = getattr(AITaskEntityFeature, "GENERATE_IMAGE", 0)
    evaluate_feature = getattr(AITaskEntityFeature, "EVALUATE", 0)
    offered: dict[str, AITaskEntityFeature] = {}
    text_supported = CAP_TEXT_TO_TEXT in text_capabilities
    if text_supported and generate_data_feature:
        offered["gen_data_entity_id"] = generate_data_feature
    if CAP_TEXT_TO_IMAGE in image_capabilities and generate_image_feature:
        offered["gen_image_entity_id"] = generate_image_feature
    if text_supported and evaluate_feature:
        offered["evaluate_entity_id"] = evaluate_feature
    return offered


def _async_ai_task_preference_is_missing(
    hass: HomeAssistant, preference_key: str, feature: AITaskEntityFeature
) -> bool:
    """Return True when the current AI task preference needs a replacement."""
    if feature == 0:
        return False
    preferences = hass.data.get(DATA_PREFERENCES)
    if preferences is None:
        return True
    if preference_key not in getattr(preferences, "KEYS", ()):
        return False
    current_entity_id = getattr(preferences, preference_key, None)
    if not current_entity_id:
        return True
    state = hass.states.get(current_entity_id)
    if state is None:
        return True
    supported_features = state.attributes.get("supported_features", 0)
    return not (supported_features & feature)


def _async_missing_ai_task_preferences(
    hass: HomeAssistant,
    image_task_data: dict[str, Any],
    text_task_data: dict[str, Any],
) -> dict[str, AITaskEntityFeature]:
    """Return missing preferences that created tasks can satisfy."""
    missing: dict[str, AITaskEntityFeature] = {}
    for preference_key, feature in _ai_task_preference_keys_to_offer(
        image_task_data, text_task_data
    ).items():
        if _async_ai_task_preference_is_missing(hass, preference_key, feature):
            missing[preference_key] = feature
    return missing


def _async_with_ai_task_kind(
    data: dict[str, Any], model_id: str | None = None
) -> dict[str, Any]:
    """Tag an AI task subentry with the task kind it represents."""
    model = model_id or data.get(CONF_CHAT_MODEL) or DEFAULT_AI_TASK_MODEL
    result = dict(data)
    if CAP_TEXT_TO_IMAGE in model_capabilities(model):
        result[CONF_AI_TASK_KIND] = AI_TASK_KIND_IMAGE
    else:
        result[CONF_AI_TASK_KIND] = AI_TASK_KIND_TEXT
    return result


def _ai_task_subentry_title(task_data: dict[str, Any], kind: str) -> str:
    """Return the default title for an AI task subentry."""
    suffix = "Image Task" if kind == AI_TASK_KIND_IMAGE else "Text Task"
    return f"{friendly_model_label(task_data.get(CONF_CHAT_MODEL))} {suffix}"


def _image_task_settings_schema(
    options: dict[str, Any], include_version: bool = True
) -> dict:
    """Return schema entries for the supported image generation options."""
    schema: dict = {}
    if include_version:
        schema[
            vol.Required(
                CONF_MODEL_VERSION,
                description={
                    "suggested_value": options.get(
                        CONF_MODEL_VERSION, IMAGE_MODEL_VERSION
                    )
                },
                default=IMAGE_MODEL_VERSION,
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                mode=SelectSelectorMode.DROPDOWN,
                options=[
                    SelectOptionDict(label="Latest", value=IMAGE_MODEL_VERSION)
                ],
                custom_value=False,
            )
        )
    supported = model_image_options(options.get(CONF_CHAT_MODEL))

    if CONF_IMAGE_SIZE in supported:
        schema[
            vol.Optional(
                CONF_IMAGE_SIZE,
                description={"suggested_value": options.get(CONF_IMAGE_SIZE)},
            )
        ] = TextSelector()

    if CONF_IMAGE_N in supported:
        schema[
            vol.Optional(
                CONF_IMAGE_N,
                description={"suggested_value": options.get(CONF_IMAGE_N)},
            )
        ] = NumberSelector(
            NumberSelectorConfig(min=MIN_IMAGE_N, max=MAX_IMAGE_N, step=1)
        )

    if CONF_IMAGE_SEED in supported:
        schema[
            vol.Optional(
                CONF_IMAGE_SEED,
                description={"suggested_value": options.get(CONF_IMAGE_SEED)},
            )
        ] = NumberSelector(NumberSelectorConfig(min=0, step=1))

    if CONF_IMAGE_NEGATIVE_PROMPT in supported:
        schema[
            vol.Optional(
                CONF_IMAGE_NEGATIVE_PROMPT,
                description={
                    "suggested_value": options.get(CONF_IMAGE_NEGATIVE_PROMPT)
                },
            )
        ] = TextSelector(TextSelectorConfig(multiline=True))

    return schema


def _reasoning_effort_schema_entry(
    kind: str, suggested: Any | None
) -> dict:
    """Return the schema entry for the model-specific reasoning control."""
    if kind == "qwen":
        suggested_bool = suggested not in (False, "none")
        return {
            vol.Optional(
                _REASONING_EFFORT_QWEN_FIELD,
                description={"suggested_value": suggested_bool},
                default=True,
            ): BooleanSelector()
        }
    suggested_value = (
        suggested if suggested in REASONING_EFFORT_VALUES else ""
    )
    marker = (
        vol.Optional(
            _REASONING_EFFORT_DEEPSEEK_FIELD,
            description={"suggested_value": suggested_value},
            default=suggested_value,
        )
        if suggested_value
        else vol.Optional(
            _REASONING_EFFORT_DEEPSEEK_FIELD,
            description={"suggested_value": suggested_value},
        )
    )
    return {
        marker: SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(label=option, value=option)
                    for option in REASONING_EFFORT_VALUES
                ],
                mode=SelectSelectorMode.DROPDOWN,
                custom_value=False,
                translation_key=CONF_REASONING_EFFORT,
            )
        )
    }


def _generation_option_schema_entries(options: dict[str, Any]) -> dict:
    """Return schema entries for the advanced generation options."""
    return {
        vol.Optional(
            CONF_TOP_P,
            description={"suggested_value": options.get(CONF_TOP_P)},
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        vol.Optional(
            CONF_STOP,
            description={"suggested_value": options.get(CONF_STOP)},
        ): TextSelector(TextSelectorConfig(multiline=True)),
        vol.Optional(
            CONF_SEED,
            description={"suggested_value": options.get(CONF_SEED)},
        ): int,
        vol.Optional(
            CONF_FREQUENCY_PENALTY,
            description={"suggested_value": options.get(CONF_FREQUENCY_PENALTY)},
        ): NumberSelector(NumberSelectorConfig(min=-2, max=2, step=0.1)),
        vol.Optional(
            CONF_PRESENCE_PENALTY,
            description={"suggested_value": options.get(CONF_PRESENCE_PENALTY)},
        ): NumberSelector(NumberSelectorConfig(min=-2, max=2, step=0.1)),
        vol.Optional(
            CONF_TOP_K,
            description={"suggested_value": options.get(CONF_TOP_K)},
        ): NumberSelector(NumberSelectorConfig(min=0, max=1000, step=1)),
    }


def _normalize_generation_suggestions(
    suggested_values: dict[str, Any]
) -> None:
    """Normalize stored generation values for the rendered form."""
    kind = model_reasoning_effort_kind(suggested_values.get(CONF_CHAT_MODEL))
    if kind == "qwen":
        suggested_values[CONF_REASONING_EFFORT] = (
            suggested_values.get(CONF_REASONING_EFFORT) not in (False, "none")
        )
    elif kind == "deepseek":
        if suggested_values.get(CONF_REASONING_EFFORT) not in REASONING_EFFORT_VALUES:
            suggested_values.pop(CONF_REASONING_EFFORT, None)
    else:
        suggested_values.pop(CONF_REASONING_EFFORT, None)

    stop = suggested_values.get(CONF_STOP)
    if isinstance(stop, (list, tuple)):
        suggested_values[CONF_STOP] = "\n".join(
            str(part) for part in stop if str(part).strip()
        )
    for key in GENERATION_OPTION_KEYS:
        if suggested_values.get(key) in (None, ""):
            suggested_values.pop(key, None)


class YandexAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Yandex AI."""

    VERSION = 2
    MINOR_VERSION = 3

    def __init__(self) -> None:
        """Initialize."""
        self._credentials: dict[str, Any] | None = None
        self._chat_model_options: list[SelectOptionDict] = []
        self._conversation_options: dict[str, Any] = {}
        self._conversation_data: dict[str, Any] = {}
        self._conversation_errors: dict[str, str] = {}
        self._last_selected_model: str | None = None
        self._conversation_reasoning_kind: str | None = None
        self._ai_task_image_data: dict[str, Any] = {}
        self._ai_task_text_options: dict[str, Any] = {}
        self._ai_task_text_model: str | None = None
        self._ai_task_text_reasoning_kind: str | None = None
        self._ai_task_image_final_data: dict[str, Any] = {}
        self._ai_task_text_final_data: dict[str, Any] = {}
        self._pending_ai_task_preferences: list[str] = []
        self._conversation_model_version_hidden = False
        self._ai_task_image_model_version_hidden = False
        self._ai_task_text_model_version_hidden = False

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

        client = YandexAIClient(
            user_input[CONF_FOLDER_ID], user_input[CONF_API_KEY], timeout=10.0
        )
        try:
            await client.list_models()
        except YandexAIStudioError as err:
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

        self._credentials = user_input
        self._chat_model_options = await self._async_load_initial_chat_models(
            user_input
        )
        self._conversation_options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
        self._conversation_data = {}
        self._conversation_errors = {}
        self._last_selected_model = None
        self._conversation_reasoning_kind = model_reasoning_effort_kind(
            self._conversation_options.get(CONF_CHAT_MODEL)
        )
        self._ai_task_image_data = {CONF_CHAT_MODEL: DEFAULT_AI_TASK_MODEL}
        self._ai_task_text_options = {
            CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
            CONF_RECOMMENDED: True,
        }
        self._ai_task_text_model = DEFAULT_CHAT_MODEL
        self._ai_task_text_reasoning_kind = model_reasoning_effort_kind(
            self._ai_task_text_model
        )
        return await self.async_step_conversation()

    async def async_step_conversation(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure the model and basic options for the conversation subentry."""
        if user_input is not None:
            recommended = user_input[CONF_RECOMMENDED]
            raw_model = (
                user_input.get(CONF_CHAT_MODEL) or DEFAULT_CHAT_MODEL
            )
            self._conversation_model_version_hidden = _model_version_of(
                raw_model
            ) is not None
            submission = dict(user_input)
            submission[CONF_CHAT_MODEL] = raw_model
            combined = _normalize_model_choice(
                {**self._conversation_options, **submission}
            )
            combined[CONF_RECOMMENDED] = recommended
            self._conversation_options = combined
            self._conversation_reasoning_kind = model_reasoning_effort_kind(
                combined.get(CONF_CHAT_MODEL)
            )

            if not combined.get(CONF_LLM_HASS_API):
                combined.pop(CONF_LLM_HASS_API, None)

            self._validate_conversation_model(combined)

            if not self._conversation_errors:
                if recommended:
                    self._conversation_data = _clean_conversation_data(combined)
                    return await self.async_step_ai_task_image_model()
                return await self.async_step_conversation_advanced()

        suggested_values = _normalize_model_choice(dict(self._conversation_options))
        if not suggested_values.get(CONF_PROMPT):
            suggested_values[CONF_PROMPT] = DEFAULT_INSTRUCTIONS_PROMPT_RU

        if CONF_LLM_HASS_API in suggested_values:
            value = suggested_values[CONF_LLM_HASS_API]
            if isinstance(value, str):
                suggested_values[CONF_LLM_HASS_API] = [] if value == "none" else [value]

        model_value = suggested_values.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        base_schema = base_config_option_schema(
            self.hass,
            suggested_values,
        )
        schema = {
            vol.Required(CONF_CHAT_MODEL, default=model_value): SelectSelector(
                SelectSelectorConfig(
                    mode=SelectSelectorMode.DROPDOWN,
                    options=_build_model_options(
                        self._chat_model_options, CAP_TEXT_TO_TEXT
                    ),
                    custom_value=False,
                )
            ),
            **base_schema,
        }
        suggested_values[CONF_RECOMMENDED] = bool(
            suggested_values.get(CONF_RECOMMENDED, False)
            or _advanced_options_are_defaults(
                suggested_values, conversation=True
            )
        )
        schema = self.add_suggested_values_to_schema(
            vol.Schema(schema),
            {k: v for k, v in suggested_values.items() if k != CONF_CHAT_MODEL},
        )

        return self.async_show_form(
            step_id="conversation",
            data_schema=schema,
            errors=self._conversation_errors,
            description_placeholders={
                "chat_model_doc_url": DOC_CHAT_MODELS_URL,
                "pricing_doc_url": DOC_PRICING_URL,
                "prompt_templates_url": DOC_PROMPT_TEMPLATES_URL,
            },
            last_step=False,
        )

    async def async_step_conversation_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure advanced options for the default conversation subentry."""
        if user_input is not None:
            base = dict(self._conversation_options)
            base.pop(CONF_MAX_TOOL_ITERATIONS, None)
            combined = {**base, **user_input}
            self._conversation_data = _clean_conversation_data(combined)
            return await self.async_step_ai_task_image_model()

        suggested_values = _normalize_model_choice(dict(self._conversation_options))
        model_id = suggested_values.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        versions_by_base = _versions_by_base(self._chat_model_options)
        version_options = None
        if not self._conversation_model_version_hidden:
            version_options = _version_select_options(model_id, versions_by_base)
            suggested_values[CONF_MODEL_VERSION] = _effective_version(
                model_id, suggested_values.get(CONF_MODEL_VERSION), versions_by_base
            )
        _normalize_generation_suggestions(suggested_values)

        schema = self.add_suggested_values_to_schema(
            vol.Schema(
                _conversation_advanced_schema(suggested_values, version_options)
            ),
            suggested_values,
        )

        return self.async_show_form(
            step_id="conversation_advanced",
            data_schema=schema,
            description_placeholders={
                "chat_model_doc_url": DOC_CHAT_MODELS_URL,
                "pricing_doc_url": DOC_PRICING_URL,
                "prompt_templates_url": DOC_PROMPT_TEMPLATES_URL,
            },
            last_step=False,
        )

    async def async_step_ai_task_image_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Select the model and advanced options for the image task."""
        if user_input is not None:
            raw_model = (
                user_input.get(CONF_CHAT_MODEL) or DEFAULT_AI_TASK_MODEL
            )
            self._ai_task_image_model_version_hidden = _model_version_of(
                raw_model
            ) is not None
            normalized = _normalize_model_choice(
                {**self._ai_task_image_data, CONF_CHAT_MODEL: raw_model}
            )
            model_id = normalized.get(CONF_CHAT_MODEL) or DEFAULT_AI_TASK_MODEL
            self._ai_task_image_data = {CONF_CHAT_MODEL: model_id}
            if normalized.get(CONF_MODEL_VERSION):
                self._ai_task_image_data[CONF_MODEL_VERSION] = (
                    normalized[CONF_MODEL_VERSION]
                )
            use_defaults = user_input[CONF_RECOMMENDED]
            if not use_defaults and model_image_options(model_id):
                return await self.async_step_ai_task_image_settings()
            return await self.async_step_ai_task()

        model_options = self._build_model_options_for_capability(CAP_TEXT_TO_IMAGE)
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_CHAT_MODEL,
                    default=_base_or_value(
                        self._ai_task_image_data.get(CONF_CHAT_MODEL)
                    )
                    or DEFAULT_AI_TASK_MODEL,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=model_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        custom_value=False,
                    )
                ),
                vol.Required(
                    CONF_RECOMMENDED,
                    default=self._ai_task_image_data.get(
                        CONF_RECOMMENDED, True
                    ),
                ): bool,
            }
        )
        return self.async_show_form(
            step_id="ai_task_image_model",
            data_schema=schema,
            description_placeholders={
                "chat_model_doc_url": DOC_CHAT_MODELS_URL,
                "pricing_doc_url": DOC_PRICING_URL,
            },
            last_step=False,
        )

    async def async_step_ai_task_image_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure supported options for the image generation AI task."""
        if user_input is not None:
            combined = _image_settings_submission(
                self._ai_task_image_data,
                user_input,
                not self._ai_task_image_model_version_hidden,
            )
            self._ai_task_image_data = _clean_image_task_data(
                combined.get(CONF_CHAT_MODEL, DEFAULT_AI_TASK_MODEL),
                combined,
            )
            return await self.async_step_ai_task()

        model_id = self._ai_task_image_data.get(
            CONF_CHAT_MODEL, DEFAULT_AI_TASK_MODEL
        )
        suggested_values = dict(self._ai_task_image_data)
        schema = self.add_suggested_values_to_schema(
            vol.Schema(
                _image_task_settings_schema(
                    suggested_values,
                    include_version=not self._ai_task_image_model_version_hidden,
                )
            ),
            suggested_values,
        )
        return self.async_show_form(
            step_id="ai_task_image_settings",
            data_schema=schema,
            description_placeholders={
                "chat_model_doc_url": DOC_CHAT_MODELS_URL,
                "pricing_doc_url": DOC_PRICING_URL,
            },
            last_step=False,
        )

    async def async_step_ai_task(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure the model and basic options for the text AI task."""
        if self._ai_task_text_model is None:
            self._ai_task_text_model = self._ai_task_text_options.get(
                CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL
            )
        if self._ai_task_text_reasoning_kind is None:
            self._ai_task_text_reasoning_kind = model_reasoning_effort_kind(
                self._ai_task_text_model
            )

        if user_input is not None:
            raw_model = (
                user_input.get(CONF_CHAT_MODEL) or DEFAULT_CHAT_MODEL
            )
            self._ai_task_text_model_version_hidden = _model_version_of(
                raw_model
            ) is not None
            self._ai_task_text_options = _normalize_model_choice(
                {
                    **self._ai_task_text_options,
                    CONF_CHAT_MODEL: raw_model,
                }
            )
            self._ai_task_text_model = _base_or_value(raw_model) or DEFAULT_CHAT_MODEL
            self._ai_task_text_reasoning_kind = model_reasoning_effort_kind(
                self._ai_task_text_model
            )
            recommended = user_input[CONF_RECOMMENDED]
            combined = _normalize_model_choice(
                {**self._ai_task_text_options, **user_input}
            )
            combined[CONF_RECOMMENDED] = recommended
            if recommended:
                ai_task_data = _clean_ai_task_data(self._ai_task_text_model, combined)
                image_task_data = self._ai_task_image_data or {
                    CONF_CHAT_MODEL: DEFAULT_AI_TASK_MODEL
                }
                self._ai_task_image_final_data = _async_with_ai_task_kind(
                    image_task_data
                )
                self._ai_task_text_final_data = _async_with_ai_task_kind(
                    ai_task_data
                )
                return await self.async_step_ai_task_preferences()
            self._ai_task_text_options = combined
            return await self.async_step_ai_task_advanced()

        model_options = self._build_model_options_for_capability(CAP_TEXT_TO_TEXT)
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_CHAT_MODEL,
                    default=_base_or_value(
                        self._ai_task_text_options.get(CONF_CHAT_MODEL)
                    )
                    or DEFAULT_CHAT_MODEL,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=model_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        custom_value=False,
                    )
                ),
                vol.Required(
                    CONF_RECOMMENDED,
                    default=self._ai_task_text_options.get(
                        CONF_RECOMMENDED, True
                    ),
                ): bool,
            }
        )
        return self.async_show_form(
            step_id="ai_task",
            data_schema=schema,
            description_placeholders={
                "chat_model_doc_url": DOC_CHAT_MODELS_URL,
                "pricing_doc_url": DOC_PRICING_URL,
            },
            last_step=False,
        )

    async def async_step_ai_task_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure advanced options for the text generation AI task."""
        if self._ai_task_text_model is None:
            self._ai_task_text_model = self._ai_task_text_options.get(
                CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL
            )
        if self._ai_task_text_reasoning_kind is None:
            self._ai_task_text_reasoning_kind = model_reasoning_effort_kind(
                self._ai_task_text_model
            )

        model_id = _base_or_value(self._ai_task_text_model) or DEFAULT_CHAT_MODEL

        if user_input is not None:
            combined = {**self._ai_task_text_options, **user_input}
            ai_task_data = _clean_ai_task_data(model_id, combined)
            image_task_data = self._ai_task_image_data or {
                CONF_CHAT_MODEL: DEFAULT_AI_TASK_MODEL
            }
            self._ai_task_image_final_data = _async_with_ai_task_kind(
                image_task_data
            )
            self._ai_task_text_final_data = _async_with_ai_task_kind(ai_task_data)
            return await self.async_step_ai_task_preferences()

        suggested_values = _normalize_model_choice(dict(self._ai_task_text_options))
        model_id = suggested_values.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        versions_by_base = _versions_by_base(self._chat_model_options)
        version_options = None
        if not self._ai_task_text_model_version_hidden:
            version_options = _version_select_options(model_id, versions_by_base)
            suggested_values[CONF_MODEL_VERSION] = _effective_version(
                model_id, suggested_values.get(CONF_MODEL_VERSION), versions_by_base
            )
        _normalize_generation_suggestions(suggested_values)

        schema = self.add_suggested_values_to_schema(
            vol.Schema(_ai_task_advanced_schema(suggested_values, version_options)),
            suggested_values,
        )

        return self.async_show_form(
            step_id="ai_task_advanced",
            data_schema=schema,
            description_placeholders={
                "chat_model_doc_url": DOC_CHAT_MODELS_URL,
                "pricing_doc_url": DOC_PRICING_URL,
            },
            last_step=False,
        )

    async def async_step_ai_task_preferences(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Offer to use the created tasks as system AI task preferences."""
        image_task_data = self._ai_task_image_final_data or _async_with_ai_task_kind(
            self._ai_task_image_data or {CONF_CHAT_MODEL: DEFAULT_AI_TASK_MODEL}
        )
        text_task_data = self._ai_task_text_final_data or _async_with_ai_task_kind(
            {
                **self._ai_task_text_options,
                CONF_CHAT_MODEL: self._ai_task_text_model or DEFAULT_CHAT_MODEL,
            }
        )

        if user_input is not None:
            self._pending_ai_task_preferences = [
                preference_key
                for preference_key, selected in user_input.items()
                if selected
            ]
            return self._async_create_entry(image_task_data, text_task_data)

        missing = _async_missing_ai_task_preferences(
            self.hass, image_task_data, text_task_data
        )
        if not missing:
            self._pending_ai_task_preferences = []
            return self._async_create_entry(image_task_data, text_task_data)

        schema = vol.Schema(
            {
                vol.Required(preference_key, default=True): BooleanSelector()
                for preference_key in missing
            }
        )
        return self.async_show_form(
            step_id="ai_task_preferences",
            data_schema=schema,
            description_placeholders={
                "gen_data_task_name": _ai_task_subentry_title(
                    text_task_data, AI_TASK_KIND_TEXT
                ),
                "gen_image_task_name": _ai_task_subentry_title(
                    image_task_data, AI_TASK_KIND_IMAGE
                ),
                "evaluate_task_name": _ai_task_subentry_title(
                    text_task_data, AI_TASK_KIND_TEXT
                ),
            },
            last_step=True,
        )

    def _build_model_options_for_capability(
        self, capability: str
    ) -> list[SelectOptionDict]:
        """Build model options filtered by the required capability."""
        return _build_model_options(self._chat_model_options, capability)

    def _validate_conversation_model(self, user_input: dict[str, Any]) -> None:
        """Validate model when any LLM HASS API (Assist or MCP) is selected."""
        llm_apis: list[str] = user_input.get(CONF_LLM_HASS_API) or []
        if isinstance(llm_apis, str):
            llm_apis = [llm_apis] if llm_apis != "none" else []

        if not llm_apis:
            self._conversation_errors.pop("base", None)
            return

        selected_model = user_input.get(CONF_CHAT_MODEL)
        model_name = _model_name(selected_model)
        if model_name in ASSIST_UNSUPPORTED_MODELS:
            self._conversation_errors["base"] = "model_not_supported_for_tools"
        elif model_name in ASSIST_PARTIALLY_SUPPORTED_MODELS:
            if self._last_selected_model == selected_model:
                self._conversation_errors.pop("base", None)
            else:
                self._conversation_errors["base"] = (
                    "model_partially_supported_for_tools"
                )

        self._last_selected_model = selected_model

    def _async_create_entry(
        self,
        image_task_data: dict[str, Any],
        text_task_data: dict[str, Any],
    ) -> ConfigFlowResult:
        """Create the config entry with the default subentries."""
        conversation_data = (
            self._conversation_data or RECOMMENDED_CONVERSATION_OPTIONS.copy()
        )
        conversation_title = (
            f"{friendly_model_label(conversation_data.get(CONF_CHAT_MODEL))} "
            "Conversation"
        )
        image_task_title = _ai_task_subentry_title(
            image_task_data, AI_TASK_KIND_IMAGE
        )
        text_task_title = _ai_task_subentry_title(
            text_task_data, AI_TASK_KIND_TEXT
        )
        data = dict(self._credentials or {})
        if self._pending_ai_task_preferences:
            data[CONF_PENDING_AI_TASK_PREFERENCES] = list(
                self._pending_ai_task_preferences
            )

        return self.async_create_entry(
            title="Yandex AI",
            data=data,
            subentries=[
                {
                    "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                    "data": conversation_data,
                    "title": conversation_title,
                    "unique_id": None,
                },
                {
                    "subentry_type": SUBENTRY_TYPE_AI_TASK_DATA,
                    "data": _async_with_ai_task_kind(image_task_data),
                    "title": image_task_title,
                    "unique_id": None,
                },
                {
                    "subentry_type": SUBENTRY_TYPE_AI_TASK_DATA,
                    "data": _async_with_ai_task_kind(text_task_data),
                    "title": text_task_title,
                    "unique_id": None,
                },
            ],
        )

    async def _async_load_initial_chat_models(
        self, user_input: dict[str, Any]
    ) -> list[SelectOptionDict]:
        """Load available chat models from the provided credentials."""
        client = YandexAIClient(
            user_input[CONF_FOLDER_ID],
            user_input[CONF_API_KEY],
            enable_server_data_logging=user_input.get(
                CONF_ENABLE_SERVER_DATA_LOGGING,
                DEFAULT_ENABLE_SERVER_DATA_LOGGING,
            ),
            timeout=10.0,
        )
        try:
            models = await client.list_models()
        except (httpx.HTTPError, YandexAIStudioError) as err:
            LOGGER.warning("Failed to load the model list: %s", err)
            models = []
        finally:
            await client.close()

        options = [
            SelectOptionDict(
                value=model["id"],
                label=friendly_model_label(model["id"]),
            )
            for model in models
        ]
        if not options:
            options = [
                SelectOptionDict(value=value, label=label)
                for value, label in CHAT_MODELS
            ]
        return options

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
        self.options: dict[str, Any] | MappingProxyType[str, Any] = {}
        self.errors: dict[str, str] = {}
        self.last_selected_model: str | None = None
        self.reasoning_kind: str | None = None
        self._chat_model_options: list[SelectOptionDict] | None = None
        self._model_version_hidden = False

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Create a new conversation subentry."""
        self.options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
        self.reasoning_kind = model_reasoning_effort_kind(
            self.options.get(CONF_CHAT_MODEL)
        )
        self._model_version_hidden = False
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an existing conversation subentry."""
        self.options = self._get_reconfigure_subentry().data.copy()
        self.reasoning_kind = model_reasoning_effort_kind(
            self.options.get(CONF_CHAT_MODEL)
        )
        self._model_version_hidden = _model_version_of(
            self.options.get(CONF_CHAT_MODEL)
        ) is not None
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage the model and basic options for a conversation subentry."""
        if self._chat_model_options is None:
            self._chat_model_options = await self._async_load_chat_models()

        self.options = _normalize_model_choice(dict(self.options))

        if self.reasoning_kind is None:
            self.reasoning_kind = model_reasoning_effort_kind(
                self.options.get(CONF_CHAT_MODEL)
            )

        if user_input is not None:
            recommended = user_input[CONF_RECOMMENDED]
            raw_model = (
                user_input.get(CONF_CHAT_MODEL) or DEFAULT_CHAT_MODEL
            )
            self._model_version_hidden = _model_version_of(
                raw_model
            ) is not None
            submission = dict(user_input)
            submission[CONF_CHAT_MODEL] = raw_model
            combined = _normalize_model_choice(
                {**dict(self.options), **submission}
            )
            combined[CONF_RECOMMENDED] = recommended
            self.options = combined
            self.reasoning_kind = model_reasoning_effort_kind(
                combined.get(CONF_CHAT_MODEL)
            )

            if not combined.get(CONF_LLM_HASS_API):
                combined.pop(CONF_LLM_HASS_API, None)

            self._validate_selected_model(combined)

            if not self.errors:
                if recommended:
                    data = _clean_conversation_data(combined)
                    if self._is_new:
                        title = (
                            f"{friendly_model_label(data.get(CONF_CHAT_MODEL))} "
                            "Conversation"
                        )
                        return self.async_create_entry(title=title, data=data)
                    return self.async_update_and_abort(
                        self._get_entry(),
                        self._get_reconfigure_subentry(),
                        data=data,
                    )
                return await self.async_step_advanced()

        suggested_values = _normalize_model_choice(dict(self.options))
        if not suggested_values.get(CONF_PROMPT):
            suggested_values[CONF_PROMPT] = DEFAULT_INSTRUCTIONS_PROMPT_RU

        # Migration: old single string / "none" → new list format
        if CONF_LLM_HASS_API in suggested_values:
            value = suggested_values[CONF_LLM_HASS_API]
            if isinstance(value, str):
                suggested_values[CONF_LLM_HASS_API] = [] if value == "none" else [value]

        model_value = suggested_values.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        base_schema = base_config_option_schema(
            self.hass,
            suggested_values,
        )
        schema = {
            vol.Required(CONF_CHAT_MODEL, default=model_value): SelectSelector(
                SelectSelectorConfig(
                    mode=SelectSelectorMode.DROPDOWN,
                    options=_build_model_options(
                        self._chat_model_options, CAP_TEXT_TO_TEXT
                    ),
                    custom_value=False,
                )
            ),
            **base_schema,
        }
        suggested_values[CONF_RECOMMENDED] = bool(
            suggested_values.get(CONF_RECOMMENDED, False)
            or _advanced_options_are_defaults(
                suggested_values, conversation=True
            )
        )
        schema = self.add_suggested_values_to_schema(
            vol.Schema(schema),
            {k: v for k, v in suggested_values.items() if k != CONF_CHAT_MODEL},
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
            last_step=False,
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure advanced options for the conversation subentry."""
        if user_input is not None:
            base = dict(self.options)
            base.pop(CONF_MAX_TOOL_ITERATIONS, None)
            combined = _normalize_model_choice({**base, **user_input})
            data = _clean_conversation_data(combined)
            if self._is_new:
                title = (
                    f"{friendly_model_label(data.get(CONF_CHAT_MODEL))} "
                    "Conversation"
                )
                return self.async_create_entry(title=title, data=data)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=data,
            )

        suggested_values = _normalize_model_choice(dict(self.options))
        model_id = suggested_values.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        versions_by_base = _versions_by_base(self._chat_model_options or [])
        version_options = None
        if not self._model_version_hidden:
            version_options = _version_select_options(model_id, versions_by_base)
            suggested_values[CONF_MODEL_VERSION] = _effective_version(
                model_id, suggested_values.get(CONF_MODEL_VERSION), versions_by_base
            )
        _normalize_generation_suggestions(suggested_values)

        schema = self.add_suggested_values_to_schema(
            vol.Schema(
                _conversation_advanced_schema(suggested_values, version_options)
            ),
            suggested_values,
        )

        return self.async_show_form(
            step_id="advanced",
            data_schema=schema,
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
            self.errors.pop("base", None)
            return

        selected_model = user_input.get(CONF_CHAT_MODEL)
        model_name = _model_name(selected_model)
        if model_name in ASSIST_UNSUPPORTED_MODELS:
            self.errors["base"] = "model_not_supported_for_tools"
        elif model_name in ASSIST_PARTIALLY_SUPPORTED_MODELS:
            if self.last_selected_model == selected_model:
                self.errors.pop("base", None)
            else:
                self.errors["base"] = "model_partially_supported_for_tools"

        self.last_selected_model = selected_model

    async def _async_load_chat_models(self) -> list[SelectOptionDict]:
        """Load the available chat models, falling back to a static list."""
        try:
            client: YandexAIClient = self._get_entry().runtime_data
            models = await client.list_models()
        except (httpx.HTTPError, YandexAIStudioError) as err:
            LOGGER.warning("Failed to load the model list: %s", err)
            models = []

        options = [
            SelectOptionDict(
                value=model["id"],
                label=friendly_model_label(model["id"]),
            )
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
        self._raw_chat_model_options: list[SelectOptionDict] | None = None
        self._rendered_model: str | None = None
        self._rendered_reasoning_kind: str | None = None
        self._model_version_hidden = False

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create an AI task."""
        self.options = {CONF_RECOMMENDED: True}
        self._model_version_hidden = False
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of an AI task."""
        self.options = self._get_reconfigure_subentry().data.copy()
        self._model_version_hidden = _model_version_of(
            self.options.get(CONF_CHAT_MODEL)
        ) is not None
        if CONF_RECOMMENDED not in self.options:
            has_advanced_options = any(
                key in self.options for key in GENERATION_OPTION_KEYS
            )
            self.options[CONF_RECOMMENDED] = not has_advanced_options
        return await self.async_step_init(user_input)

    async def async_step_image_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure supported options for an image generation AI task."""
        model_id = (
            _base_or_value(
                self._rendered_model or self.options.get(CONF_CHAT_MODEL)
            )
            or DEFAULT_AI_TASK_MODEL
        )

        if user_input is not None:
            combined = _image_settings_submission(
                self.options,
                user_input,
                not self._model_version_hidden,
            )
            return self._async_finish(_clean_image_task_data(model_id, combined))

        suggested_values = _normalize_model_choice(dict(self.options))
        schema = self.add_suggested_values_to_schema(
            vol.Schema(
                _image_task_settings_schema(
                    suggested_values,
                    include_version=not self._model_version_hidden,
                )
            ),
            suggested_values,
        )
        return self.async_show_form(
            step_id="image_settings",
            data_schema=schema,
            description_placeholders={
                "chat_model_doc_url": DOC_CHAT_MODELS_URL,
                "pricing_doc_url": DOC_PRICING_URL,
            },
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure the model and basic options for an AI task subentry."""
        self.options = _normalize_model_choice(dict(self.options))
        if self._rendered_model is None:
            self._rendered_model = (
                _base_or_value(self.options.get(CONF_CHAT_MODEL))
                or DEFAULT_AI_TASK_MODEL
            )
        if self._rendered_reasoning_kind is None:
            self._rendered_reasoning_kind = model_reasoning_effort_kind(
                self._rendered_model
            )
        model_options = await self._async_build_model_options()

        if user_input is not None:
            raw_model = (
                user_input.get(CONF_CHAT_MODEL) or DEFAULT_AI_TASK_MODEL
            )
            self._model_version_hidden = _model_version_of(raw_model) is not None
            model_id = _base_or_value(raw_model) or DEFAULT_AI_TASK_MODEL
            self.options = _normalize_model_choice(
                {**dict(self.options), CONF_CHAT_MODEL: raw_model}
            )
            self._rendered_model = model_id
            self._rendered_reasoning_kind = model_reasoning_effort_kind(model_id)

            if CAP_TEXT_TO_TEXT not in model_capabilities(model_id):
                image_data = _normalize_model_choice(
                    {**dict(self.options), **user_input}
                )
                if not user_input[CONF_RECOMMENDED] and model_image_options(model_id):
                    return await self.async_step_image_settings()
                return self._async_finish(_clean_image_task_data(model_id, image_data))

            recommended = user_input[CONF_RECOMMENDED]
            combined = _normalize_model_choice(
                {**dict(self.options), **user_input}
            )
            combined[CONF_RECOMMENDED] = recommended
            if recommended:
                return self._async_finish(
                    _clean_ai_task_data(model_id, combined)
                )
            self.options = combined
            return await self.async_step_advanced()

        current_model = (
            _base_or_value(self.options.get(CONF_CHAT_MODEL)) or DEFAULT_AI_TASK_MODEL
        )
        if CAP_TEXT_TO_TEXT in model_capabilities(current_model):
            use_defaults_default = bool(
                self.options.get(CONF_RECOMMENDED, False)
                or _advanced_options_are_defaults(self.options)
            )
        else:
            use_defaults_default = _image_options_are_defaults(
                self.options, current_model
            )
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_CHAT_MODEL,
                    default=current_model,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=model_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        custom_value=False,
                    )
                ),
                vol.Required(
                    CONF_RECOMMENDED,
                    default=use_defaults_default,
                ): bool,
            }
        )
        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            description_placeholders={
                "chat_model_doc_url": DOC_CHAT_MODELS_URL,
                "pricing_doc_url": DOC_PRICING_URL,
            },
            last_step=False,
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure advanced options for an AI task subentry."""
        self.options = _normalize_model_choice(dict(self.options))
        if self._rendered_model is None:
            self._rendered_model = (
                _base_or_value(self.options.get(CONF_CHAT_MODEL))
                or DEFAULT_AI_TASK_MODEL
            )
        if self._rendered_reasoning_kind is None:
            self._rendered_reasoning_kind = model_reasoning_effort_kind(
                self._rendered_model
            )

        model_id = _base_or_value(self._rendered_model) or DEFAULT_CHAT_MODEL

        if user_input is not None:
            base = dict(self.options)
            base.pop(CONF_MAX_TOKENS, None)
            combined = _normalize_model_choice({**base, **user_input})
            return self._async_finish(_clean_ai_task_data(model_id, combined))

        suggested_values = _normalize_model_choice(dict(self.options))
        model_id = suggested_values.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        versions_by_base = _versions_by_base(self._raw_chat_model_options or [])
        version_options = None
        if not self._model_version_hidden:
            version_options = _version_select_options(model_id, versions_by_base)
            suggested_values[CONF_MODEL_VERSION] = _effective_version(
                model_id, suggested_values.get(CONF_MODEL_VERSION), versions_by_base
            )
        _normalize_generation_suggestions(suggested_values)

        schema = self.add_suggested_values_to_schema(
            vol.Schema(_ai_task_advanced_schema(suggested_values, version_options)),
            suggested_values,
        )

        return self.async_show_form(
            step_id="advanced",
            data_schema=schema,
            description_placeholders={
                "chat_model_doc_url": DOC_CHAT_MODELS_URL,
                "pricing_doc_url": DOC_PRICING_URL,
            },
        )

    def _async_finish(
        self, data: dict[str, Any]
    ) -> SubentryFlowResult:
        """Create or update the AI task subentry."""
        model_id = data.get(CONF_CHAT_MODEL) or DEFAULT_AI_TASK_MODEL
        label = self.model_labels.get(model_id, friendly_model_label(model_id))
        if CAP_TEXT_TO_IMAGE in model_capabilities(model_id):
            suffix = "Image Task"
        else:
            suffix = "Text Task"
        data = _async_with_ai_task_kind(data, model_id)
        if self._is_new:
            return self.async_create_entry(title=f"{label} {suffix}", data=data)
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            data=data,
        )

    async def _async_build_model_options(self) -> list[SelectOptionDict]:
        """Load and build the model options for AI task flows."""
        try:
            client: YandexAIClient = self._get_entry().runtime_data
            models = await client.list_models()
        except (httpx.HTTPError, YandexAIStudioError) as err:
            LOGGER.warning("Failed to fetch models for AI Task: %s", err)
            models = []

        raw_options: list[SelectOptionDict] = [
            SelectOptionDict(
                value=model["id"],
                label=friendly_model_label(model["id"]),
            )
            for model in models
        ]
        if not raw_options:
            raw_options = [
                SelectOptionDict(value=value, label=label)
                for value, label in CHAT_MODELS
            ]
        self._raw_chat_model_options = raw_options

        model_options = _build_model_options(raw_options)
        self.model_labels = {
            option["value"]: friendly_model_label(option["value"])
            for option in model_options
        }
        return model_options


def base_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
) -> dict:
    """Return the basic schema for Yandex AI completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    ]

    schema: dict = {
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
    }
    schema[
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, True)
        )
    ] = bool
    return schema


def _conversation_advanced_schema(
    options: dict[str, Any] | MappingProxyType[str, Any],
    version_options: list[SelectOptionDict] | None,
) -> dict:
    """Return the advanced schema for conversation options."""
    schema: dict = {}
    if version_options is not None:
        default_version = _version_select_default(options, version_options)
        schema[
            vol.Required(
                CONF_MODEL_VERSION,
                description={"suggested_value": default_version},
                default=default_version,
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                mode=SelectSelectorMode.DROPDOWN, options=version_options
            )
        )
    schema.update(
        {
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            ): int,
            vol.Optional(
                CONF_NO_HA_DEFAULT_PROMPT,
                description={
                    "suggested_value": options.get(CONF_NO_HA_DEFAULT_PROMPT)
                },
            ): bool,
            vol.Optional(
                CONF_MAX_TOOL_ITERATIONS,
                description={
                    "suggested_value": options.get(CONF_MAX_TOOL_ITERATIONS)
                },
            ): int,
        }
    )

    reasoning_kind = model_reasoning_effort_kind(options.get(CONF_CHAT_MODEL))
    if reasoning_kind:
        schema.update(
            _reasoning_effort_schema_entry(
                reasoning_kind, options.get(CONF_REASONING_EFFORT)
            )
        )

    schema.update(_generation_option_schema_entries(options))
    return schema


def _ai_task_advanced_schema(
    options: dict[str, Any] | MappingProxyType[str, Any],
    version_options: list[SelectOptionDict] | None,
) -> dict:
    """Return the advanced schema for AI task options."""
    schema: dict = {}
    if version_options is not None:
        default_version = _version_select_default(options, version_options)
        schema[
            vol.Required(
                CONF_MODEL_VERSION,
                description={"suggested_value": default_version},
                default=default_version,
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                mode=SelectSelectorMode.DROPDOWN, options=version_options
            )
        )
    schema[
        vol.Optional(
            CONF_TEMPERATURE,
            description={"suggested_value": options.get(CONF_TEMPERATURE)},
        )
    ] = NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05))
    schema[
        vol.Optional(
            CONF_MAX_TOKENS,
            description={"suggested_value": options.get(CONF_MAX_TOKENS)},
        )
    ] = int

    reasoning_kind = model_reasoning_effort_kind(options.get(CONF_CHAT_MODEL))
    if reasoning_kind:
        schema.update(
            _reasoning_effort_schema_entry(
                reasoning_kind, options.get(CONF_REASONING_EFFORT)
            )
        )

    schema.update(_generation_option_schema_entries(options))
    return schema
