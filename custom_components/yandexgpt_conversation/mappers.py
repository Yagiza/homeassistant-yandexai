"""Mappers for converting between the OpenAI API and Home Assistant formats."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
from collections import deque
from collections.abc import (AsyncGenerator, AsyncIterator, Callable,
                              Iterable)
from typing import Any

from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from probatio import to_openapi

from .client import StreamChunk
from .const import DOMAIN, LOGGER


class StreamTransformer:
    """Transform a streamed model response into Home Assistant chat deltas."""

    def __init__(self, stream: AsyncIterator[StreamChunk]) -> None:
        """Initialize the transformer."""
        self.stream = stream
        self._assistant_role_sent = False

    async def to_chatlog_api(
        self,
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
        """Yield Home Assistant content deltas from the model stream."""
        async for chunk in self.stream:
            LOGGER.debug("Received stream chunk: %s", chunk)

            if chunk.tool_calls:
                yield {
                    "tool_calls": [
                        llm.ToolInput(tool_name=call.name, tool_args=call.arguments)
                        for call in chunk.tool_calls
                    ]
                }
                continue

            if chunk.role and not self._assistant_role_sent:
                self._assistant_role_sent = True
                yield {"role": chunk.role}

            if chunk.content:
                if not self._assistant_role_sent:
                    self._assistant_role_sent = True
                    yield {"role": "assistant"}
                yield {"content": chunk.content}

            if chunk.finish_reason == "length":
                LOGGER.warning("Response was truncated by the model")
            elif chunk.finish_reason == "content_filter":
                LOGGER.warning("The message got blocked by the content filter")
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="ethics_filter",
                )


def _tool_args_to_json(tool_args: Any) -> str:
    """Normalize tool call arguments to a JSON string."""
    if isinstance(tool_args, str):
        return tool_args
    return json.dumps(tool_args, ensure_ascii=False)


class ContentConverter:
    """Convert Home Assistant conversation content to OpenAI message format."""

    def __init__(self, system_prompt_override: str | None = None) -> None:
        """Initialize the converter."""
        self._system_prompt_override = system_prompt_override

    def to_openai_api(
        self, chat_logs: Iterable[conversation.Content]
    ) -> list[dict[str, Any]]:
        """Convert Home Assistant conversation content to OpenAI message format.

        Tool call ids are synthetic and assigned while scanning the content,
        so the assistant ``tool_calls`` and the following ``tool`` messages
        always reference matching ids within a single request.
        """
        messages: list[dict[str, Any]] = []
        pending_tool_call_ids: deque[str] = deque()
        tool_call_counter = 0

        for content in chat_logs:
            if isinstance(content, conversation.ToolResultContent):
                tool_call_id = pending_tool_call_ids.popleft() if pending_tool_call_ids else None
                if tool_call_id is None:
                    LOGGER.warning(
                        "Tool result without a matching tool call: %s", content.tool_name
                    )
                    continue
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(content.tool_result, ensure_ascii=False),
                })

            elif isinstance(content, conversation.AssistantContent) and content.tool_calls:
                tool_calls = []
                for index, tool_call in enumerate(content.tool_calls):
                    tool_call_id = f"call_{tool_call_counter + index}"
                    tool_calls.append({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": _tool_args_to_json(tool_call.tool_args),
                        },
                    })
                    pending_tool_call_ids.append(tool_call_id)
                tool_call_counter += len(tool_calls)

                message: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
                if content.content:
                    message["content"] = content.content
                messages.append(message)

            elif isinstance(content, conversation.SystemContent):
                text = self._system_prompt_override or content.content
                if text:
                    messages.append({"role": "system", "content": text})

            elif content.content:
                messages.append({"role": content.role, "content": content.content})

            else:
                raise TypeError(f"Unexpected content type: {type(content)}")

        return messages

    @staticmethod
    def format_tool(
        tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
    ) -> dict[str, Any]:
        """Convert a Home Assistant tool to the OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": to_openapi(tool.parameters, custom_serializer=custom_serializer),
            },
        }
