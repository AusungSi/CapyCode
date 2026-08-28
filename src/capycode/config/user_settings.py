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
    endpoint: UserEndpointSettings | None = None
    models: dict[str, UserModelSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selected_model(self) -> UserSettings:
        if self.endpoint is None:
            if self.default_model is not None:
                raise ValueError("default_model requires a configured endpoint")
            return self
        if self.default_model not in self.endpoint.available_models:
            raise ValueError("default_model must be returned by the configured endpoint")
        for model_id in self.endpoint.available_models:
            self.models.setdefault(model_id, UserModelSettings())
        return self


class ResolvedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    base_url: str
    api_key: str
    context_window: int
    input_per_million: float
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
        model: str,
        base_url: str,
        api_key: str,
        available_models: list[str],
    ) -> UserSettings:
        settings = self.load()
        normalized_model = model.strip()
        settings.endpoint = UserEndpointSettings(
            base_url=base_url.strip(),
            api_key=api_key.strip(),
            available_models=available_models,
        )
        if normalized_model not in settings.endpoint.available_models:
            raise ValueError(f"model {normalized_model!r} is not returned by the endpoint")
        settings.default_model = normalized_model
        for model_id in settings.endpoint.available_models:
            settings.models.setdefault(model_id, UserModelSettings())
        self.save(settings)
        return settings

    def select_model(self, model: str) -> UserSettings:
        settings = self.load()
        if settings.endpoint is None:
            raise ValueError("model endpoint is not configured")
        if model not in settings.endpoint.available_models:
            raise ValueError(f"model {model!r} is not returned by the configured endpoint")
        settings.default_model = model
        settings.models.setdefault(model, UserModelSettings())
        self.save(settings)
        return settings

    def configure_pricing(
        self,
        model: str,
        *,
        input_per_million: float,
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
                output_per_million=output_per_million,
                currency=currency.strip().upper(),
                snapshot_date=snapshot_date,
            ),
        )
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
                "endpoint": {
                    "base_url": legacy["base_url"],
                    "api_key": legacy["api_key"],
                    "available_models": model_ids,
                },
                "models": {model_id: {} for model_id in model_ids},
            },
            True,
        )


def resolve_model(model_id: str | None, settings: UserSettings) -> ResolvedModel:
    if settings.endpoint is None:
        raise ValueError("model endpoint is not configured; run /config first")
    selected = model_id or settings.default_model
    if selected is None or selected not in settings.endpoint.available_models:
        raise ValueError(f"model {selected!r} is not returned by the configured endpoint")
    metadata = settings.models.get(selected, UserModelSettings())
    return ResolvedModel(
        model=selected,
        base_url=settings.endpoint.base_url,
        api_key=settings.endpoint.api_key,
        context_window=metadata.context_window,
        input_per_million=metadata.pricing.input_per_million,
        output_per_million=metadata.pricing.output_per_million,
        currency=metadata.pricing.currency,
        pricing_snapshot_date=metadata.pricing.snapshot_date,
    )
