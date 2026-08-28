from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from capycode.config.user_settings import UserSettingsStore, resolve_model
from capycode.core import AgentRuntime, RuntimeObserver, SessionState
from capycode.llm import OpenAICompatibleLLM
from capycode.tools import build_p0_runtime_tools
from capycode.trace import RunTrackingConfig


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
) -> SessionState:
    store = settings_store or UserSettingsStore()
    resolved = resolve_model(model_id, store.load())
    client = OpenAICompatibleLLM(resolved.base_url, resolved.api_key)
    try:
        tools = build_p0_runtime_tools()
        try:
            runtime = AgentRuntime(client, tools, max_steps=max_steps)
            tracking = RunTrackingConfig(
                provider="openai-compatible",
                model_id=resolved.model,
                input_per_million=resolved.input_per_million,
                output_per_million=resolved.output_per_million,
                currency=resolved.currency,
                pricing_snapshot_date=resolved.pricing_snapshot_date.isoformat(),
                sensitive_values=(resolved.api_key,),
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
