from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"
SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)$",
    re.IGNORECASE,
)
BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


class TraceRedactor:
    def __init__(self, sensitive_values: Sequence[str] = ()) -> None:
        self.sensitive_values = tuple(
            sorted({value for value in sensitive_values if value}, key=len, reverse=True)
        )

    def redact(self, value: Any, *, key: str | None = None) -> Any:
        if key is not None and SENSITIVE_KEY.search(key):
            return REDACTED
        if isinstance(value, str):
            cleaned = BEARER_VALUE.sub(f"Bearer {REDACTED}", value)
            for sensitive in self.sensitive_values:
                cleaned = cleaned.replace(sensitive, REDACTED)
            return cleaned
        if isinstance(value, Mapping):
            return {
                str(item_key): self.redact(item, key=str(item_key))
                for item_key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self.redact(item) for item in value]
        return value
