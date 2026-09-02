class RequestCache:
    _values: dict[str, str] = {}

    def put(self, key: str, value: str) -> None:
        self._values[key] = value

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def clear(self) -> None:
        self._values.clear()
