"""AI Task platform for the YandexGPT integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any

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
                    CONF_CHAT_MODEL, CONF_FOLDER_ID, CONF_MAX_TOKENS,
                    CONF_TEMPERATURE, DEFAULT_CHAT_MODEL, DOMAIN,
                    IMAGE_GENERATION_TIMEOUT, IMAGE_MODELS, IMAGE_MODEL_VERSION,
                    LOGGER, RECOMMENDED_MAX_TOKENS, RECOMMENDED_TEMPERATURE,
                    SUBENTRY_TYPE_AI_TASK_DATA, model_capabilities)
from .entity import YandexGPTEntity

if TYPE_CHECKING:
    from homeassistant.components.ai_task.evaluation import (
        EvaluationTask,
        EvaluationTaskResult,
    )


_EVALUATE_SYSTEM_PROMPT = (
    "You are an evaluation engine for Home Assistant. "
    "You are given a state to evaluate and a list of questions. "
    "Answer every question based only on the provided state. "
    "Answer formats: "
    'a "noul" question estimates the probability that the answer is yes - '
    'respond with {"noul": p} where p is a number between 0 and 1; '
    'a "choice" question selects exactly one alternative by its id - '
    'respond with {"choice": "<id>", "probabilities": {"<id>": p, ...}} '
    "with one probability for every alternative, each between 0 and 1 "
    "and summing to 1; "
    'a "score" question assesses ordered levels numbered 0 to N-1 - '
    'respond with {"score": s, "probabilities": [p0, ...]} where s is the '
    "expected level index between 0 and N-1 and probabilities has one entry "
    "per level, each between 0 and 1 and summing to 1. "
    "Respond with a single JSON object mapping each question id to its "
    "answer and no text outside the JSON object."
)


def _build_evaluate_prompt(state: Any, questions: dict[str, Any]) -> str:
    """Build the user message with the state and the questions."""
    if state is None:
        state_text = "No state was provided."
    else:
        state_text = json.dumps(state, ensure_ascii=False, default=str)
    lines = [f"State: {state_text}", "", "Questions:"]
    for question_id, question in questions.items():
        lines.append(
            f"- {question_id} ({question['type']}): {question['instructions']}"
        )
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            lines.extend(
                f"  {key}: {description}" for key, description in criteria.items()
            )
        elif isinstance(criteria, list):
            lines.extend(
                f"  {index}: {level}" for index, level in enumerate(criteria)
            )
    return "\n".join(lines)


def _clamp_probability(value: Any) -> float:
    """Convert a model value to a probability clamped to the 0..1 range."""
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(probability, 0.0), 1.0)


def _parse_evaluate_payload(text: str) -> dict[str, Any]:
    """Parse the JSON answers object from the model response."""
    payload: Any
    try:
        payload = json_loads(text)
    except JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise HomeAssistantError(
                "The model did not return a JSON object"
            ) from None
        try:
            payload = json_loads(text[start : end + 1])
        except JSONDecodeError as err:
            raise HomeAssistantError(
                "The model did not return a JSON object"
            ) from err
    if not isinstance(payload, dict):
        raise HomeAssistantError("The model did not return a JSON object")
    return payload


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
            features |= getattr(ai_task.AITaskEntityFeature, "EVALUATE", 0)
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

    async def _async_evaluate(
        self,
        task: EvaluationTask,
    ) -> EvaluationTaskResult:
        """Handle an evaluation task."""
        from homeassistant.components.ai_task.evaluation import (
            ChoiceAnswer,
            EvaluationTaskResult,
            NoulAnswer,
            ScoreAnswer,
        )

        options: dict[str, Any] = {
            **dict(self.entry.data),
            **dict(self.subentry.data),
        }
        client: YandexGPTClient = self.entry.runtime_data

        if task.attachments and not self._model_supports_attachments(options):
            raise HomeAssistantError(
                f"Model {self._model_name(options)} does not support "
                "image attachments"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _EVALUATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_evaluate_prompt(task.state, task.questions),
            },
        ]
        if task.attachments:
            await self._async_attach_images(messages, task.attachments)

        try:
            text = await self._async_evaluate_completion(
                client, options, messages, {"type": "json_object"}
            )
        except YandexGPTApiError as err:
            if err.status_code != 400:
                LOGGER.exception("Error talking to Yandex AI Studio: %s", err)
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="yandex_cloud_error",
                    translation_placeholders={"details": err.message},
                ) from err
            LOGGER.warning(
                "The model rejected the json_object response format, "
                "retrying without it: %s",
                err.message,
            )
            try:
                text = await self._async_evaluate_completion(
                    client, options, messages, None
                )
            except YandexGPTApiError as err:
                LOGGER.exception(
                    "Error talking to Yandex AI Studio: %s", err
                )
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="yandex_cloud_error",
                    translation_placeholders={"details": err.message},
                ) from err

        payload = _parse_evaluate_payload(text)
        answers: dict[str, Any] = {}
        for question_id, question in task.questions.items():
            raw = payload.get(question_id)
            if not isinstance(raw, dict):
                raise HomeAssistantError(
                    f"The model did not answer question '{question_id}'"
                )
            question_type = question.get("type")
            if question_type == "noul":
                answers[question_id] = NoulAnswer(
                    noul=_clamp_probability(raw.get("noul"))
                )
            elif question_type == "choice":
                criteria = question.get("criteria") or {}
                probabilities_raw = raw.get("probabilities")
                probabilities = {
                    key: _clamp_probability(
                        probabilities_raw.get(key)
                        if isinstance(probabilities_raw, dict)
                        else None
                    )
                    for key in criteria
                }
                choice = raw.get("choice")
                if not isinstance(choice, str) or choice not in criteria:
                    if any(probabilities.values()):
                        choice = max(probabilities, key=probabilities.get)
                    else:
                        raise HomeAssistantError(
                            f"The model returned an invalid choice for "
                            f"question '{question_id}'"
                        )
                answers[question_id] = ChoiceAnswer(
                    choice=choice, probabilities=probabilities
                )
            elif question_type == "score":
                levels = len(question.get("criteria") or [])
                probabilities_raw = raw.get("probabilities")
                probabilities = [
                    _clamp_probability(
                        probabilities_raw[index]
                        if isinstance(probabilities_raw, list)
                        and index < len(probabilities_raw)
                        else None
                    )
                    for index in range(levels)
                ]
                try:
                    score = float(raw.get("score"))
                except (TypeError, ValueError):
                    score = 0.0
                if levels:
                    score = min(max(score, 0.0), float(levels - 1))
                else:
                    score = 0.0
                answers[question_id] = ScoreAnswer(
                    score=score, probabilities=probabilities
                )
            else:
                raise HomeAssistantError(
                    f"Unsupported question type '{question_type}' "
                    f"for question '{question_id}'"
                )
        return EvaluationTaskResult(answers=answers)

    async def _async_evaluate_completion(
        self,
        client: YandexGPTClient,
        options: dict[str, Any],
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
    ) -> str:
        """Run a non-streaming evaluation completion and return its text."""
        try:
            choice = await client.complete_chat(
                self._model_id(options),
                messages,
                temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                response_format=response_format,
            )
        except httpx.HTTPError as err:
            LOGGER.exception("Error talking to Yandex AI Studio: %s", err)
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="yandex_cloud_error",
                translation_placeholders={"details": str(err)},
            ) from err

        message = choice.get("message") or {}
        if choice.get("finish_reason") == "content_filter":
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="ethics_filter",
            )
        if not message.get("content"):
            raise HomeAssistantError(
                "YandexGPT returned an empty response "
                f"(finish_reason: {choice.get('finish_reason') or 'unknown'})"
            )
        return message["content"]
