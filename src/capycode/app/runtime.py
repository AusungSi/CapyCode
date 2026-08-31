from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from capycode.capability import (
    CapabilityDetector,
    EscalationPolicy,
    ProfiledRoutingArtifact,
    ProfileRegistry,
    ProfileRouter,
    build_default_profile_registry,
)
from capycode.config.loader import load_configuration
from capycode.config.user_settings import UserSettingsStore, resolve_model
from capycode.core import AgentRuntime, RuntimeObserver, SessionState
from capycode.llm import OpenAICompatibleLLM
from capycode.tools import build_p0_runtime_tools
from capycode.trace import RunTrackingConfig


def load_profile_instruction(instruction: str, profiles_path: Path) -> str:
    """Load a referenced instruction file, or preserve inline instruction text."""
    instruction_path = Path(instruction)
    candidates = (
        profiles_path.parent / instruction_path,
        profiles_path.parent.parent / instruction_path,
        instruction_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    if instruction_path.suffix.lower() in {".md", ".txt"}:
        raise ValueError(
            "profile instruction file was not found: "
            f"{instruction}. Use an existing path or inline instruction text."
        )
    return instruction


async def discover_models(base_url: str, api_key: str) -> list[str]:
    client = OpenAICompatibleLLM(base_url, api_key)
    try:
        return await client.list_models()
    finally:
        await client.aclose()


async def execute_task(
    task: str,
    workspace: Path,
    model_id: str | None,
    models_path: Path,
    max_steps: int,
    settings_store: UserSettingsStore | None = None,
    observer: RuntimeObserver | None = None,
    session_state: SessionState | None = None,
    checkpoint: Callable[[SessionState], None] | None = None,
    profiles_path: Path | None = None,
    profiled_routing_path: Path | None = None,
    force_profile_model: str | None = None,
    endpoint_id: str | None = None,
    profile_step_limit: int | None = None,
    container_image: str | None = None,
) -> SessionState:
    store = settings_store or UserSettingsStore()
    resolved = resolve_model(model_id, store.load(), endpoint_id)
    client = OpenAICompatibleLLM(resolved.base_url, resolved.api_key)
    try:
        tools = build_p0_runtime_tools(container_image=container_image)
        registry = build_default_profile_registry()
        configured_profiles = profiles_path or Path("config/profiles.yaml")
        if await asyncio.to_thread(configured_profiles.is_file) and await asyncio.to_thread(
            models_path.is_file
        ):
            bundle = await asyncio.to_thread(load_configuration, models_path, configured_profiles)

            def resolve_profile_model(model_ref: str) -> str:
                if model_ref in {"current", resolved.model}:
                    return resolved.model
                configured = bundle.models.models.get(model_ref)
                if configured is None:
                    raise ValueError(f"profile references unknown model: {model_ref}")
                candidate = configured.model
                if candidate.startswith(("replace-with-", "configure-with-")):
                    return resolved.model
                return candidate

            registry = await asyncio.to_thread(
                ProfileRegistry.from_config,
                bundle.profiles,
                model_resolver=resolve_profile_model,
                instruction_resolver=lambda instruction: load_profile_instruction(
                    instruction, configured_profiles
                ),
            )
        artifact_path = profiled_routing_path or workspace / ".capy" / "profiles.json"
        profiled_routing = None
        if await asyncio.to_thread(artifact_path.is_file):
            profiled_routing = await asyncio.to_thread(ProfiledRoutingArtifact.load, artifact_path)
            settings = store.load()
            if not settings.endpoints:
                raise ValueError("model endpoint is not configured; run /config first")
            selected_models = {
                selection.model_id for selection in profiled_routing.selected_by_capability.values()
            }
            available = set()
            for endpoint in settings.endpoints.values():
                available.update(endpoint.available_models)
            unavailable = sorted(selected_models - available)
            if unavailable:
                raise ValueError(
                    "profiled routing artifact references models not exposed by the current "
                    "endpoint: " + ", ".join(unavailable)
                )
            registry = registry.with_routing_overrides(
                {
                    selection.profile_id: (
                        selection.model_id,
                        selection.reasoning_effort,
                    )
                    for selection in profiled_routing.selected_by_capability.values()
                }
            )
        if force_profile_model is not None:
            registry = registry.with_model_overrides(
                {profile.profile_id: force_profile_model for profile in registry.all()}
            )
        try:
            runtime = AgentRuntime(
                client,
                tools,
                max_steps=max_steps,
                context_window=resolved.context_window,
                profile_step_limit=profile_step_limit,
                capability_detector=CapabilityDetector(),
                profile_router=ProfileRouter(registry, profiled_routing=profiled_routing),
                escalation_policy=EscalationPolicy(registry),
            )
            tracking = RunTrackingConfig(
                provider="openai-compatible",
                model_id=resolved.model,
                input_per_million=resolved.input_per_million,
                output_per_million=resolved.output_per_million,
                currency=resolved.currency,
                pricing_snapshot_date=resolved.pricing_snapshot_date.isoformat(),
                cached_input_per_million=resolved.cached_input_per_million,
                model_pricing={
                    candidate: (
                        metadata.pricing.input_per_million,
                        metadata.pricing.output_per_million,
                        metadata.pricing.cached_input_per_million,
                    )
                    for candidate, metadata in store.load()
                    .endpoint_models.get(resolved.endpoint_id, store.load().models)
                    .items()
                },
                sensitive_values=(resolved.api_key,),
                event_sink=(getattr(observer, "on_run_event", None) if observer else None),
            )
            return await runtime.run(
                task,
                workspace,
                resolved.model,
                observer,
                session_state,
                checkpoint,
                tracking,
            )
        finally:
            await tools.aclose()
    finally:
        await client.aclose()
