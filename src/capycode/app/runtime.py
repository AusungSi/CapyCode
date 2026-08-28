from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from capycode.config.loader import load_models
from capycode.config.user_settings import UserSettingsStore, resolve_model
from capycode.core import AgentRuntime, RuntimeObserver, SessionState
from capycode.llm import OpenAICompatibleLLM
from capycode.tools import build_p0_runtime_tools


async def discover_models(base_url: str, api_key: str) -> list[str]:
    client = OpenAICompatibleLLM(base_url, api_key)
    try:
        return await client.list_models()
    finally:
        await client.aclose()


async def execute_task(
    task: str,
    workspace: Path,
    model_alias: str,
    models_path: Path,
    max_steps: int,
    settings_store: UserSettingsStore | None = None,
    observer: RuntimeObserver | None = None,
    session_state: SessionState | None = None,
    checkpoint: Callable[[SessionState], None] | None = None,
) -> SessionState:
    registry = load_models(models_path)
    try:
        model_config = registry.models[model_alias]
    except KeyError as exc:
        choices = ", ".join(sorted(registry.models))
        raise ValueError(f"unknown model alias {model_alias!r}; available: {choices}") from exc

    store = settings_store or UserSettingsStore()
    resolved = resolve_model(model_alias, model_config, store.load())
    client = OpenAICompatibleLLM(resolved.base_url, resolved.api_key)
    try:
        tools = build_p0_runtime_tools()
        try:
            runtime = AgentRuntime(client, tools, max_steps=max_steps)
            return await runtime.run(
                task,
                workspace,
                resolved.model,
                observer,
                session_state,
                checkpoint,
            )
        finally:
            await tools.aclose()
    finally:
        await client.aclose()
