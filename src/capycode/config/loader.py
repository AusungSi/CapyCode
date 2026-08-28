from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .models import ModelRegistryConfig, ProfileRegistryConfig

ConfigT = TypeVar("ConfigT", bound=BaseModel)


@dataclass(frozen=True)
class ConfigurationBundle:
    models: ModelRegistryConfig
    profiles: ProfileRegistryConfig


def _load_yaml(path: Path, schema: type[ConfigT]) -> ConfigT:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    try:
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        return schema.model_validate(payload)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    except ValidationError as exc:
        raise ValueError(f"invalid configuration in {path}: {exc}") from exc


def load_models(path: str | Path) -> ModelRegistryConfig:
    return _load_yaml(Path(path), ModelRegistryConfig)


def load_profiles(path: str | Path) -> ProfileRegistryConfig:
    return _load_yaml(Path(path), ProfileRegistryConfig)


def load_configuration(models_path: str | Path, profiles_path: str | Path) -> ConfigurationBundle:
    models = load_models(models_path)
    profiles = load_profiles(profiles_path)
    unknown_models = {
        profile.model
        for profile in profiles.profiles.values()
        if profile.model not in models.models
    }
    if unknown_models:
        values = ", ".join(sorted(unknown_models))
        raise ValueError(f"profiles reference unknown models: {values}")
    return ConfigurationBundle(models=models, profiles=profiles)
