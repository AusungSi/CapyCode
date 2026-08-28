from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .models import ModelConfig


class UserModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    available_models: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def include_selected_model(self) -> UserModelSettings:
        self.available_models = sorted({*self.available_models, self.model})
        return self


class UserSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_model: str = "small"
    models: dict[str, UserModelSettings] = Field(default_factory=dict)


class ResolvedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    base_url: str
    api_key: str


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
            return UserSettings.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
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

    def configure_model(
        self,
        alias: str,
        *,
        model: str,
        base_url: str,
        api_key: str,
        available_models: list[str] | None = None,
    ) -> UserSettings:
        settings = self.load()
        settings.models[alias] = UserModelSettings(
            model=model.strip(),
            base_url=base_url.strip(),
            api_key=api_key.strip(),
            available_models=available_models or [model.strip()],
        )
        settings.default_model = alias
        self.save(settings)
        return settings

    def select_model(self, alias: str, model: str) -> UserSettings:
        settings = self.load()
        try:
            configured = settings.models[alias]
        except KeyError as exc:
            raise ValueError(f"model endpoint {alias!r} is not configured") from exc
        if model not in configured.available_models:
            raise ValueError(f"model {model!r} is not available from the configured endpoint")
        configured.model = model
        settings.default_model = alias
        self.save(settings)
        return settings


def resolve_model(alias: str, model_config: ModelConfig, settings: UserSettings) -> ResolvedModel:
    local = settings.models.get(alias)
    if local is not None:
        return ResolvedModel(
            model=local.model,
            base_url=local.base_url,
            api_key=local.api_key,
        )
    return ResolvedModel(
        model=model_config.model,
        base_url=model_config.resolve_base_url(),
        api_key=model_config.resolve_api_key(),
    )
