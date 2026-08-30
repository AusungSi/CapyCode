from __future__ import annotations

import os
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PricingConfig(StrictConfigModel):
    input_per_million: float = Field(ge=0)
    cached_input_per_million: float | None = Field(default=None, ge=0)
    output_per_million: float = Field(ge=0)
    snapshot_date: date


class ModelConfig(StrictConfigModel):
    provider: Literal["openai-compatible"]
    model: str = Field(min_length=1)
    base_url_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    context_window: int = Field(gt=0)
    pricing: PricingConfig

    def resolve_base_url(self) -> str:
        value = os.getenv(self.base_url_env)
        if not value:
            raise ValueError(f"environment variable {self.base_url_env!r} is not set")
        return value

    def resolve_api_key(self) -> str:
        value = os.getenv(self.api_key_env)
        if not value:
            raise ValueError(f"environment variable {self.api_key_env!r} is not set")
        return value


class ModelRegistryConfig(StrictConfigModel):
    models: dict[str, ModelConfig]

    @model_validator(mode="after")
    def require_models(self) -> ModelRegistryConfig:
        if not self.models:
            raise ValueError("at least one model must be configured")
        return self


class BudgetConfig(StrictConfigModel):
    max_output_tokens: int = Field(gt=0)
    max_steps: int = Field(gt=0)


class ProfileConfig(StrictConfigModel):
    capability: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    model: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    tools: list[str] = Field(min_length=1)
    context_policy: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    budget: BudgetConfig


class ProfileRegistryConfig(StrictConfigModel):
    profiles: dict[str, ProfileConfig]

    @model_validator(mode="after")
    def require_profiles(self) -> ProfileRegistryConfig:
        if not self.profiles:
            raise ValueError("at least one profile must be configured")
        return self
