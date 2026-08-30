from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class UserEndpointSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    available_models: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def normalize_models(self) -> UserEndpointSettings:
        self.available_models = sorted(
            {item.strip() for item in self.available_models if item.strip()}
        )
        if not self.available_models:
            raise ValueError("endpoint must expose at least one model")
        return self


class UserModelPricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_million: float = Field(default=0, ge=0)
    cached_input_per_million: float | None = Field(default=None, ge=0)
    output_per_million: float = Field(default=0, ge=0)
    currency: str = Field(default="USD", min_length=1, max_length=12)
    snapshot_date: date = Field(default_factory=date.today)


class UserModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_window: int = Field(default=128_000, gt=0)
    pricing: UserModelPricing = Field(default_factory=UserModelPricing)


class UserSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    default_model: str | None = None
    default_endpoint: str | None = None
    endpoint: UserEndpointSettings | None = None
    endpoints: dict[str, UserEndpointSettings] = Field(default_factory=dict)
    models: dict[str, UserModelSettings] = Field(default_factory=dict)
    endpoint_models: dict[str, dict[str, UserModelSettings]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selected_model(self) -> UserSettings:
        if not self.endpoints and self.endpoint is not None:
            self.endpoints = {"default": self.endpoint}
        if self.endpoints and self.default_endpoint is None:
            self.default_endpoint = (
                "default" if "default" in self.endpoints else next(iter(self.endpoints))
            )
        if self.default_endpoint is not None and self.default_endpoint not in self.endpoints:
            raise ValueError("default_endpoint must reference a configured endpoint")
        if self.default_endpoint is not None:
            self.endpoint = self.endpoints[self.default_endpoint]
        if self.endpoint is None:
            if self.default_model is not None:
                raise ValueError("default_model requires a configured endpoint")
            return self
        if self.default_model is None:
            self.default_model = self.endpoint.available_models[0]
        for endpoint_id, configured in self.endpoints.items():
            endpoint_pricing = self.endpoint_models.setdefault(endpoint_id, {})
            for model_id in configured.available_models:
                endpoint_pricing.setdefault(
                    model_id, self.models.get(model_id, UserModelSettings())
                )
                self.models.setdefault(model_id, endpoint_pricing[model_id])
        if self.default_model not in self.endpoint.available_models:
            raise ValueError("default_model must be returned by the configured endpoint")
        return self


class ResolvedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    endpoint_id: str
    base_url: str
    api_key: str
    context_window: int
    input_per_million: float
    cached_input_per_million: float | None
    output_per_million: float
    currency: str
    pricing_snapshot_date: date


class UserSettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        capycode_home = os.getenv("CAPYCODE_HOME")
        self.path = path or (
            Path(capycode_home) / "settings.json"
            if capycode_home
            else Path.home() / ".capycode" / "settings.json"
        )

    def load(self) -> UserSettings:
        if not self.path.exists():
            return UserSettings()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            migrated_payload, migrated = self._migrate_legacy_payload(payload)
            if (
                isinstance(migrated_payload, dict)
                and migrated_payload.get("endpoint") is not None
                and not migrated_payload.get("endpoints")
            ):
                migrated_payload = dict(migrated_payload)
                migrated_payload["default_endpoint"] = "default"
                migrated_payload["endpoints"] = {"default": migrated_payload["endpoint"]}
                migrated = True
            settings = UserSettings.model_validate(migrated_payload)
            if migrated:
                self.save(settings)
            return settings
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(f"invalid local settings file {self.path}: {exc}") from exc

    def save(self, settings: UserSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="settings-", suffix=".tmp", dir=self.path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(settings.model_dump_json(indent=2))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.chmod(0o600)
            os.replace(temporary_path, self.path)
            self.path.chmod(0o600)
        finally:
            temporary_path.unlink(missing_ok=True)

    def configure_endpoint(
        self,
        *,
        endpoint_id: str = "default",
        model: str,
        base_url: str,
        api_key: str,
        available_models: list[str],
    ) -> UserSettings:
        settings = self.load()
        normalized_model = model.strip()
        endpoint_id = endpoint_id.strip()
        allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if not endpoint_id or any(char not in allowed_chars for char in endpoint_id):
            raise ValueError("endpoint_id must contain only letters, numbers, '_' or '-'")
        endpoint = UserEndpointSettings(
            base_url=base_url.strip(),
            api_key=api_key.strip(),
            available_models=available_models,
        )
        if normalized_model not in endpoint.available_models:
            raise ValueError(f"model {normalized_model!r} is not returned by the endpoint")
        settings.endpoints[endpoint_id] = endpoint
        settings.default_endpoint = endpoint_id
        settings.endpoint = endpoint
        settings.default_model = normalized_model
        endpoint_pricing = settings.endpoint_models.setdefault(endpoint_id, {})
        for model_id in endpoint.available_models:
            endpoint_pricing.setdefault(
                model_id, settings.models.get(model_id, UserModelSettings())
            )
            settings.models.setdefault(model_id, UserModelSettings())
        self.save(settings)
        return settings

    def select_endpoint(self, endpoint_id: str) -> UserSettings:
        settings = self.load()
        if endpoint_id not in settings.endpoints:
            raise ValueError(f"unknown endpoint: {endpoint_id}")
        settings.default_endpoint = endpoint_id
        settings.endpoint = settings.endpoints[endpoint_id]
        if settings.default_model not in settings.endpoint.available_models:
            settings.default_model = settings.endpoint.available_models[0]
        self.save(settings)
        return settings

    def delete_endpoint(self, endpoint_id: str) -> UserSettings:
        settings = self.load()
        if endpoint_id not in settings.endpoints:
            raise ValueError(f"unknown endpoint: {endpoint_id}")
        del settings.endpoints[endpoint_id]
        settings.endpoint_models.pop(endpoint_id, None)
        if settings.default_endpoint == endpoint_id:
            settings.default_endpoint = next(iter(settings.endpoints), None)
            settings.endpoint = (
                settings.endpoints[settings.default_endpoint]
                if settings.default_endpoint is not None
                else None
            )
            settings.default_model = (
                settings.endpoint.available_models[0] if settings.endpoint is not None else None
            )
        self.save(settings)
        return settings

    def select_model(self, model: str, endpoint_id: str | None = None) -> UserSettings:
        settings = self.load()
        if endpoint_id is not None:
            settings = self.select_endpoint(endpoint_id)
        if settings.endpoint is None:
            raise ValueError("model endpoint is not configured")
        if model not in settings.endpoint.available_models:
            raise ValueError(f"model {model!r} is not returned by the configured endpoint")
        settings.default_model = model
        settings.models.setdefault(model, UserModelSettings())
        settings.endpoint_models.setdefault(settings.default_endpoint or "default", {}).setdefault(
            model, settings.models[model]
        )
        self.save(settings)
        return settings

    def configure_pricing(
        self,
        model: str,
        *,
        input_per_million: float,
        cached_input_per_million: float | None = None,
        output_per_million: float,
        currency: str,
        snapshot_date: date,
        context_window: int,
    ) -> UserSettings:
        settings = self.load()
        if settings.endpoint is None or model not in settings.endpoint.available_models:
            raise ValueError(f"model {model!r} is not returned by the configured endpoint")
        settings.models[model] = UserModelSettings(
            context_window=context_window,
            pricing=UserModelPricing(
                input_per_million=input_per_million,
                cached_input_per_million=cached_input_per_million,
                output_per_million=output_per_million,
                currency=currency.strip().upper(),
                snapshot_date=snapshot_date,
            ),
        )
        endpoint_id = settings.default_endpoint or "default"
        settings.endpoint_models.setdefault(endpoint_id, {})[model] = settings.models[model]
        self.save(settings)
        return settings

    @staticmethod
    def _migrate_legacy_payload(payload: Any) -> tuple[Any, bool]:
        if not isinstance(payload, dict) or payload.get("schema_version") == 2:
            return payload, False
        legacy_models = payload.get("models")
        if not isinstance(legacy_models, dict) or not legacy_models:
            return payload, False
        alias = payload.get("default_model")
        legacy = legacy_models.get(alias) if isinstance(alias, str) else None
        if not isinstance(legacy, dict):
            legacy = next(
                (item for item in legacy_models.values() if isinstance(item, dict)),
                None,
            )
        if not isinstance(legacy, dict) or not {"model", "base_url", "api_key"} <= legacy.keys():
            return payload, False
        selected = str(legacy["model"])
        available = legacy.get("available_models") or [selected]
        model_ids = sorted({str(item) for item in available} | {selected})
        return (
            {
                "schema_version": 2,
                "default_model": selected,
                "default_endpoint": "default",
                "endpoint": {
                    "base_url": legacy["base_url"],
                    "api_key": legacy["api_key"],
                    "available_models": model_ids,
                },
                "endpoints": {
                    "default": {
                        "base_url": legacy["base_url"],
                        "api_key": legacy["api_key"],
                        "available_models": model_ids,
                    }
                },
                "models": {model_id: {} for model_id in model_ids},
            },
            True,
        )


def resolve_model(
    model_id: str | None,
    settings: UserSettings,
    endpoint_id: str | None = None,
) -> ResolvedModel:
    selected_endpoint_id = endpoint_id or settings.default_endpoint
    if selected_endpoint_id is None and settings.endpoint is not None:
        selected_endpoint_id = "default"
    configured_endpoint = (
        settings.endpoints.get(selected_endpoint_id) if selected_endpoint_id else settings.endpoint
    )
    if configured_endpoint is None or selected_endpoint_id is None:
        raise ValueError("model endpoint is not configured; run /config first")
    selected = model_id or settings.default_model
    if selected not in configured_endpoint.available_models:
        selected = configured_endpoint.available_models[0] if model_id is None else model_id
    if selected is None or selected not in configured_endpoint.available_models:
        raise ValueError(f"model {selected!r} is not returned by the configured endpoint")
    metadata = settings.endpoint_models.get(selected_endpoint_id, {}).get(
        selected, settings.models.get(selected, UserModelSettings())
    )
    return ResolvedModel(
        model=selected,
        endpoint_id=selected_endpoint_id,
        base_url=configured_endpoint.base_url,
        api_key=configured_endpoint.api_key,
        context_window=metadata.context_window,
        input_per_million=metadata.pricing.input_per_million,
        cached_input_per_million=metadata.pricing.cached_input_per_million,
        output_per_million=metadata.pricing.output_per_million,
        currency=metadata.pricing.currency,
        pricing_snapshot_date=metadata.pricing.snapshot_date,
    )
