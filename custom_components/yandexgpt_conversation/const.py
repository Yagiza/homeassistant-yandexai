"""Constants for the YandexGPT Conversation integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import logging

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

DOMAIN = "yandexgpt_conversation"
LOGGER = logging.getLogger(__package__)

SUBENTRY_TYPE_CONVERSATION = "conversation"
SUBENTRY_TYPE_AI_TASK_DATA = "ai_task_data"

CONF_FOLDER_ID = "folder_id"
CONF_PROMPT = "prompt"
CONF_RECOMMENDED = "recommended"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_CHAT_MODEL = "chat_model"
CONF_MODEL_VERSION = "model_version"
CONF_ENABLE_SERVER_DATA_LOGGING = "enable_server_data_logging"
CONF_MAX_TOOL_ITERATIONS = "max_tool_iterations"
CONF_NO_HA_DEFAULT_PROMPT = "no_ha_default_prompt"
DEFAULT_CHAT_MODEL = "yandexgpt-lite"
DEFAULT_MODEL_VERSION = "latest"
DEFAULT_NO_HA_DEFAULT_PROMPT = False
DEFAULT_ENABLE_SERVER_DATA_LOGGING = True
DEFAULT_MAX_TOOL_ITERATIONS = 10
RECOMMENDED_MAX_TOKENS = 1024
RECOMMENDED_TEMPERATURE = 0.6

DEFAULT_BASE_URL = "https://llm.api.cloud.yandex.net/v1"

DEFAULT_INSTRUCTIONS_PROMPT_RU = """Ты — голосовой ассистент для Home Assistant.
Отвечай на вопросы правдиво. Отвечай кратко, чётко и на русском языке.
"""

RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_PROMPT: DEFAULT_INSTRUCTIONS_PROMPT_RU,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
    CONF_MAX_TOOL_ITERATIONS: DEFAULT_MAX_TOOL_ITERATIONS,
}

RECOMMENDED_AI_TASK_OPTIONS = {
    CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
}

DEFAULT_CONVERSATION_NAME = "YandexGPT"
DEFAULT_AI_TASK_NAME = "YandexGPT AI Task"

# Fallback list of known chat models, used when GET /models is unavailable
CHAT_MODELS = (
    ("yandexgpt-lite", "YandexGPT Lite"),
    ("yandexgpt", "YandexGPT Pro"),
    ("aliceai-llm", "Alice AI LLM"),
)

ASSIST_UNSUPPORTED_MODELS = []
ASSIST_PARTIALLY_SUPPORTED_MODELS = ["yandexgpt-lite"]

# Yandex AI Studio models that accept images in the chat (OpenAI image_url)
MULTIMODAL_MODELS = (
    "deepseek-v4.1-flash",
    "qwen3.6-35b-a3b",
)

DOC_CHAT_MODELS_URL = "https://aistudio.yandex.ru/docs/ai-studio/concepts/generation/models"
DOC_PRICING_URL = "https://aistudio.yandex.ru/docs/ai-studio/pricing#common-instance-sync"
DOC_PROMPT_TEMPLATES_URL = "https://github.com/black-roland/homeassistant-yandexgpt/wiki/%D0%98%D1%81%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5-%D1%88%D0%B0%D0%B1%D0%BB%D0%BE%D0%BD%D0%BE%D0%B2-%D0%B2-%D1%81%D0%B8%D1%81%D1%82%D0%B5%D0%BC%D0%BD%D0%BE%D0%BC-%D0%BF%D1%80%D0%BE%D0%BC%D0%BF%D1%82%D0%B5"  # noqa: E501
