from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from capycode.config.models import ProfileConfig, ProfileRegistryConfig
from capycode.llm.types import DEFAULT_MAX_OUTPUT_TOKENS

from .models import Capability


@dataclass(frozen=True)
class Profile:
    profile_id: str
    capability: Capability
    model_ref: str
    instruction: str
    tools: frozenset[str]
    context_policy: str
    max_output_tokens: int
    max_steps: int

    @classmethod
    def from_config(cls, profile_id: str, config: ProfileConfig) -> Profile:
        return cls(
            profile_id=profile_id,
            capability=Capability(config.capability),
            model_ref=config.model,
            instruction=config.instruction,
            tools=frozenset(config.tools),
            context_policy=config.context_policy,
            max_output_tokens=config.budget.max_output_tokens,
            max_steps=config.budget.max_steps,
        )


class ProfileRegistry:
    def __init__(self, profiles: dict[str, Profile]) -> None:
        if not profiles:
            raise ValueError("at least one profile is required")
        self._profiles = dict(profiles)

    @classmethod
    def from_config(
        cls,
        config: ProfileRegistryConfig,
        *,
        model_resolver: Callable[[str], str] | None = None,
        instruction_resolver: Callable[[str], str] | None = None,
    ) -> ProfileRegistry:
        profiles: dict[str, Profile] = {}
        for profile_id, profile_config in config.profiles.items():
            try:
                profile = Profile.from_config(profile_id, profile_config)
            except ValueError as exc:
                raise ValueError(f"invalid capability in profile {profile_id}: {exc}") from exc
            if model_resolver is not None or instruction_resolver is not None:
                profile = replace(
                    profile,
                    model_ref=(
                        model_resolver(profile.model_ref)
                        if model_resolver is not None
                        else profile.model_ref
                    ),
                    instruction=(
                        instruction_resolver(profile.instruction)
                        if instruction_resolver is not None
                        else profile.instruction
                    ),
                )
            profiles[profile_id] = profile
        return cls(profiles)

    def get(self, profile_id: str) -> Profile | None:
        return self._profiles.get(profile_id)

    def for_capability(self, capability: Capability) -> list[Profile]:
        return [profile for profile in self._profiles.values() if profile.capability == capability]

    def all(self) -> tuple[Profile, ...]:
        return tuple(self._profiles.values())

    def with_model_overrides(self, overrides: dict[str, str]) -> ProfileRegistry:
        """Return a registry whose measured profile choices use their recorded model IDs."""
        return ProfileRegistry(
            {
                profile_id: replace(profile, model_ref=overrides.get(profile_id, profile.model_ref))
                for profile_id, profile in self._profiles.items()
            }
        )


def build_default_profile_registry() -> ProfileRegistry:
    """Return a safe built-in profile set for the currently selected real model."""
    tools = frozenset(
        {
            "git_diff",
            "list_files",
            "process_status",
            "read_file",
            "replace_text",
            "run_command",
            "run_tests",
            "search_code",
            "stop_process",
            "write_file",
        }
    )
    profiles = {
        f"{capability.value}_default": Profile(
            profile_id=f"{capability.value}_default",
            capability=capability,
            model_ref="current",
            instruction=f"Focus on {capability.value.replace('_', ' ')} for this step.",
            tools=tools,
            context_policy="diagnosis" if capability == Capability.DIAGNOSIS else "full",
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            max_steps=10,
        )
        for capability in Capability
    }
    return ProfileRegistry(profiles)
