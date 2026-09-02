from __future__ import annotations

import json
from typing import TYPE_CHECKING

from capycode.llm.types import Message, ToolDefinition

from .profile import Profile

if TYPE_CHECKING:
    from capycode.core.state import SessionState


class ContextBuilder:
    _MAX_TOOL_RESULT_CHARS = 12_000
    _ESTIMATED_CHARS_PER_TOKEN = 3

    @classmethod
    def _compact_tool_message(cls, message: Message) -> Message:
        if (
            message.role != "tool"
            or not message.content
            or len(message.content) <= cls._MAX_TOOL_RESULT_CHARS
        ):
            return message
        try:
            payload = json.loads(message.content)
            content = payload.get("content") if isinstance(payload, dict) else None
            if isinstance(content, str) and len(content) > cls._MAX_TOOL_RESULT_CHARS:
                payload["content"] = (
                    content[: cls._MAX_TOOL_RESULT_CHARS // 2]
                    + "\n... older tool output compacted; use a focused read ...\n"
                    + content[-cls._MAX_TOOL_RESULT_CHARS // 2 :]
                )
                return message.model_copy(
                    update={"content": json.dumps(payload, ensure_ascii=False)}
                )
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        return message

    def build(
        self, state: SessionState, profile: Profile, tools: list[ToolDefinition]
    ) -> tuple[list[Message], list[ToolDefinition]]:
        selected_tools = tools
        if profile.tools:
            selected_tools = [tool for tool in tools if tool.name in profile.tools]
        messages = [self._compact_tool_message(item) for item in state.history]
        if profile.instruction:
            insert_at = 1 if messages and messages[0].role == "system" else 0
            messages.insert(insert_at, Message(role="system", content=profile.instruction))
        if profile.context_policy == "retrieval":
            messages = self._select_recent_protocol_safe(messages, 6)
        elif profile.context_policy == "diagnosis":
            messages = self._select_recent_protocol_safe(messages, 12)
        return messages, selected_tools

    @classmethod
    def _conversation_groups(cls, messages: list[Message]) -> list[list[Message]]:
        """Return complete turns and discard orphaned OpenAI tool messages."""
        groups: list[list[Message]] = []
        position = 0
        while position < len(messages):
            current = messages[position]
            position += 1
            if current.role == "tool":
                # A tool result is only valid immediately after the assistant
                # message that requested it.
                continue
            if current.role != "assistant" or not current.tool_calls:
                groups.append([current])
                continue

            expected_ids = {call.id for call in current.tool_calls}
            seen_ids: set[str] = set()
            group = [current]
            while position < len(messages) and messages[position].role == "tool":
                tool_message = messages[position]
                position += 1
                tool_call_id = tool_message.tool_call_id
                if tool_call_id in expected_ids and tool_call_id not in seen_ids:
                    group.append(tool_message)
                    seen_ids.add(tool_call_id)
            if seen_ids == expected_ids:
                groups.append(group)
        return groups

    @classmethod
    def _select_recent_protocol_safe(
        cls, messages: list[Message], recent_message_target: int
    ) -> list[Message]:
        system_messages = [item for item in messages if item.role == "system"]
        conversation = [item for item in messages if item.role != "system"]
        groups = cls._conversation_groups(conversation)
        task_group_index = next(
            (
                index
                for index, group in enumerate(groups)
                if any(message.role == "user" for message in group)
            ),
            None,
        )
        retained_indices: list[int] = []
        retained_messages = 0
        for index in range(len(groups) - 1, -1, -1):
            if index == task_group_index:
                continue
            if retained_messages >= recent_message_target:
                break
            retained_indices.append(index)
            retained_messages += len(groups[index])
        retained_indices.sort()
        task_group = groups[task_group_index] if task_group_index is not None else []
        return (
            system_messages
            + task_group
            + [message for index in retained_indices for message in groups[index]]
        )

    @classmethod
    def estimate_tokens(cls, messages: list[Message], tools: list[ToolDefinition]) -> int:
        characters = sum(
            len(message.model_dump_json(exclude_none=True)) + 12 for message in messages
        )
        characters += sum(len(tool.model_dump_json(exclude_none=True)) + 12 for tool in tools)
        return max(
            1,
            (characters + cls._ESTIMATED_CHARS_PER_TOKEN - 1) // cls._ESTIMATED_CHARS_PER_TOKEN,
        )

    @classmethod
    def fit_to_budget(
        cls,
        messages: list[Message],
        tools: list[ToolDefinition],
        max_input_tokens: int | None,
    ) -> list[Message]:
        """Trim old turns while preserving system prompts and tool-call pairs."""
        compacted = [cls._compact_tool_message(item) for item in messages]
        if max_input_tokens is None or cls.estimate_tokens(compacted, tools) <= max_input_tokens:
            return compacted

        system_messages = [item for item in compacted if item.role == "system"]
        conversation = [item for item in compacted if item.role != "system"]
        groups = cls._conversation_groups(conversation)

        task_group_index = next(
            (
                index
                for index, group in enumerate(groups)
                if any(message.role == "user" for message in group)
            ),
            None,
        )
        task_group = groups[task_group_index] if task_group_index is not None else []
        notice = Message(
            role="system",
            content=(
                "Earlier interaction was compacted to fit the model context window. "
                "Use the retained task and recent tool results; do not repeat completed work."
            ),
        )
        retained_indices: list[int] = []
        for index in range(len(groups) - 1, -1, -1):
            if index == task_group_index:
                continue
            candidate_indices = sorted([index, *retained_indices])
            candidate = (
                system_messages
                + [notice]
                + task_group
                + [
                    message
                    for candidate_index in candidate_indices
                    for message in groups[candidate_index]
                ]
            )
            if cls.estimate_tokens(candidate, tools) > max_input_tokens:
                continue
            retained_indices = candidate_indices

        result = (
            system_messages
            + [notice]
            + task_group
            + [message for index in retained_indices for message in groups[index]]
        )
        return result
