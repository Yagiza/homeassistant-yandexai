"""Async client for the Yandex AI Studio OpenAI-compatible API."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from httpx_sse import EventSource

from .const import (DEFAULT_BASE_URL, DEFAULT_IMAGE_BASE_URL, LOGGER)


class YandexAIStudioError(Exception):
    """Error returned by the Yandex AI Studio API."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Initialize the exception."""
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class ToolCall:
    """A single tool call requested by the model."""

    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class StreamChunk:
    """A piece of the streamed model response."""

    role: str | None = None
    content: str | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class GeneratedImage:
    """A single generated image from the Yandex AI API."""

    image_bytes: bytes
    mime_type: str


def _error_message_from_body(body: Any) -> str | None:
    """Extract a human-readable message from an API error body."""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            if message := error.get("error") or error.get("message"):
                return str(message)
        elif isinstance(error, str):
            return error
        if message := body.get("message"):
            return str(message)
    return None


def _detect_image_mime(image_bytes: bytes) -> str | None:
    """Detect the mime type of an image from its magic bytes."""
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return None


class YandexAIClient:
    """Client for the OpenAI-compatible Yandex AI Studio chat API."""

    def __init__(
        self,
        folder_id: str,
        api_key: str,
        *,
        enable_server_data_logging: bool = True,
        timeout: float = 300.0,
    ) -> None:
        """Initialize the client."""
        self._folder_id = folder_id
        self._client = httpx.AsyncClient(
            base_url=DEFAULT_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Project": folder_id,
                "User-Agent": "homeassistant-yandexai",
                "x-data-logging-enabled": (
                    "true" if enable_server_data_logging else "false"
                ),
            },
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise a YandexAIStudioError with a readable message on HTTP errors."""
        if response.status_code < 400:
            return
        message: str | None = None
        try:
            message = _error_message_from_body(response.json())
        except ValueError:
            pass
        if not message:
            message = response.reason_phrase or response.text[:200]
        raise YandexAIStudioError(message, status_code=response.status_code)

    async def list_models(self) -> list[dict[str, str]]:
        """Return the chat models available to this account."""
        response = await self._client.get("/models")
        self._raise_for_status(response)
        data = response.json().get("data", [])
        return [model for model in data if model.get("id", "").startswith("gpt://")]

    def _build_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        reasoning_effort: str | None,
        top_p: float | None,
        stop: str | list[str] | None,
        seed: int | None,
        frequency_penalty: float | None,
        presence_penalty: float | None,
        top_k: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build the chat completions request body."""
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if tools:
            request["tools"] = tools
        if response_format:
            request["response_format"] = response_format
        if reasoning_effort:
            request["reasoning_effort"] = reasoning_effort
        if top_p is not None:
            request["top_p"] = top_p
        if stop:
            if isinstance(stop, str):
                stop_items = [part.strip() for part in stop.splitlines()]
            else:
                stop_items = [str(part).strip() for part in stop]
            stop_items = [part for part in stop_items if part]
            if stop_items:
                request["stop"] = stop_items
        if seed is not None:
            request["seed"] = seed
        if frequency_penalty is not None:
            request["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            request["presence_penalty"] = presence_penalty
        if top_k is not None:
            request["top_k"] = top_k
        return request

    async def complete_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Run a non-streaming chat completion and return the choice dict.

        The result contains ``message`` and ``finish_reason``.
        """
        payload = self._build_request(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            top_p=top_p,
            stop=stop,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_k=top_k,
            stream=False,
        )
        LOGGER.debug("Sending chat request: %s", json.dumps(payload, ensure_ascii=False))

        response = await self._client.post("/chat/completions", json=payload)
        self._raise_for_status(response)
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise YandexAIStudioError("API returned an empty response")

        return choices[0]

    async def generate_image(
        self,
        model_uri: str,
        prompt: str,
        *,
        size: str | None = None,
        n: int | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        timeout: float = 120.0,
    ) -> GeneratedImage:
        """Generate an image via the OpenAI-compatible images endpoint.

        The request is synchronous: the response already contains the
        image as base64, no operation polling is needed. When ``n`` is
        greater than one, only the first generated image is returned.
        ``size``, ``n``, ``seed`` and ``negative_prompt`` are sent only
        when provided.
        """
        payload: dict[str, Any] = {
            "model": model_uri,
            "prompt": prompt,
            "response_format": "b64_json",
        }
        if size is not None and str(size).strip():
            payload["size"] = str(size).strip()
        if n is not None:
            payload["n"] = int(n)
        if seed is not None:
            payload["seed"] = seed
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        LOGGER.debug(
            "Sending image generation request: %s (size=%s, n=%s)",
            model_uri,
            payload.get("size"),
            payload.get("n"),
        )

        response = await self._client.post(
            f"{DEFAULT_IMAGE_BASE_URL}/images/generations",
            json=payload,
            timeout=timeout,
        )
        self._raise_for_status(response)
        data = response.json()
        items = data.get("data") or []
        if not items or not (image_b64 := items[0].get("b64_json")):
            raise YandexAIStudioError("API returned no image data")

        image_bytes = base64.b64decode(image_b64)
        return GeneratedImage(
            image_bytes=image_bytes,
            mime_type=_detect_image_mime(image_bytes) or "image/png",
        )

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion from the model."""
        payload = self._build_request(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            response_format=None,
            reasoning_effort=reasoning_effort,
            top_p=top_p,
            stop=stop,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_k=top_k,
            stream=True,
        )
        LOGGER.debug("Sending chat request: %s", json.dumps(payload, ensure_ascii=False))

        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            self._raise_for_status(response)
            tool_calls: dict[int, ToolCall] = {}
            event_source = EventSource(response)
            async for sse in event_source.aiter_sse():
                if sse.data.startswith("[DONE]"):
                    break
                try:
                    data = json.loads(sse.data)
                except ValueError:
                    raise YandexAIStudioError("Malformed stream response") from None

                if error := data.get("error"):
                    raise YandexAIStudioError(
                        _error_message_from_body({"error": error})
                        or "streaming error"
                    )

                choices = data.get("choices") or [{}]
                delta = choices[0].get("delta") or {}

                if tool_calls_delta := delta.get("tool_calls"):
                    for tool_call_delta in tool_calls_delta:
                        index = tool_call_delta.get("index", 0)
                        call = tool_calls.setdefault(index, ToolCall())
                        if tool_call_delta.get("id"):
                            call.id = tool_call_delta["id"]
                        if function := tool_call_delta.get("function"):
                            if name := function.get("name"):
                                call.name += name
                            if arguments := function.get("arguments"):
                                call.arguments += arguments

                finish_reason = choices[0].get("finish_reason")
                content = delta.get("content")
                if content is None and not finish_reason:
                    continue

                if finish_reason == "tool_calls" and tool_calls:
                    yield StreamChunk(
                        finish_reason=finish_reason,
                        tool_calls=[tool_calls[index] for index in sorted(tool_calls)],
                    )
                    tool_calls.clear()
                else:
                    yield StreamChunk(
                        role=delta.get("role"),
                        content=content,
                        finish_reason=finish_reason,
                    )
