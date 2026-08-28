"""Provider-neutral model gateway and OpenAI-compatible adapters."""

from .base import LLMClient, LLMError, LLMErrorKind, TextDeltaHandler
from .fake import ScriptedLLM
from .openai_compatible import OpenAICompatibleLLM
from .types import LLMRequest, LLMResponse, Message, ToolCall, ToolDefinition, Usage

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMErrorKind",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "OpenAICompatibleLLM",
    "ScriptedLLM",
    "TextDeltaHandler",
    "ToolCall",
    "ToolDefinition",
    "Usage",
]
