from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Input, OptionList, Select, Static

from capycode.app.tui import CapyCodeApp, ChatMessage, ThinkingIndicator, ToolActivity
from capycode.config import UserSettingsStore
from capycode.core import RuntimeObserver, SessionState
from capycode.tools import ToolResult


@pytest.mark.asyncio
async def test_slash_opens_command_menu(tmp_path: Path) -> None:
    app = CapyCodeApp(
        workspace=tmp_path,
        models_path=Path("config/models.example.yaml"),
        settings_store=UserSettingsStore(tmp_path / "settings.json"),
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/"
        await pilot.pause()

        menu = app.query_one("#command-menu", OptionList)
        assert menu.has_class("visible")
        option_ids = [option.id for option in menu.options]
        assert "/help" in option_ids
        assert "/config" in option_ids


@pytest.mark.asyncio
async def test_startup_splash_dismisses_without_using_conversation_space(
    tmp_path: Path,
) -> None:
    app = CapyCodeApp(
        workspace=tmp_path,
        models_path=Path("config/models.example.yaml"),
        settings_store=UserSettingsStore(tmp_path / "settings.json"),
    )

    async with app.run_test() as pilot:
        splash = app.query_one("#splash")
        assert app.splash_visible
        assert app.query_one("#splash-animation", Static).is_mounted

        app._dismiss_splash()
        await pilot.pause()

        assert splash.has_class("hidden")
        assert not app.splash_visible
        assert app.query_one("#prompt", Input).has_focus


@pytest.mark.asyncio
async def test_slash_model_picker_switches_real_model(tmp_path: Path) -> None:
    store = UserSettingsStore(tmp_path / "settings.json")
    store.configure_model(
        "small",
        model="model-a",
        base_url="https://example.test/v1",
        api_key="local-secret",
        available_models=["model-a"],
    )

    async def fake_model_fetcher(base_url: str, api_key: str) -> list[str]:
        return ["model-a", "model-b", "model-c", "model-d"]

    app = CapyCodeApp(
        workspace=tmp_path,
        models_path=Path("config/models.example.yaml"),
        settings_store=store,
        model_fetcher=fake_model_fetcher,
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/model"
        await pilot.press("enter")
        await pilot.pause()
        picker = app.screen.query_one("#model-picker-list", OptionList)
        assert [option.id for option in picker.options] == [
            "model-a",
            "model-b",
            "model-c",
            "model-d",
        ]

        await pilot.press("down", "down", "enter")
        await pilot.pause()

        assert app.model_id == "model-c"
        persisted = store.load().models["small"]
        assert persisted.model == "model-c"
        assert persisted.available_models == ["model-a", "model-b", "model-c", "model-d"]
        status = app.query_one("#status-line", Static)
        assert "model-c" in str(status.render())
        assert "small" not in str(status.render())


@pytest.mark.asyncio
async def test_config_dialog_saves_local_model_settings(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    store = UserSettingsStore(settings_path)

    async def fake_model_fetcher(base_url: str, api_key: str) -> list[str]:
        assert base_url == "https://example.test/v1"
        assert api_key == "local-secret"
        return ["demo-model", "other-model"]

    app = CapyCodeApp(
        workspace=tmp_path,
        models_path=Path("config/models.example.yaml"),
        settings_store=store,
        model_fetcher=fake_model_fetcher,
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/config"
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one("#config-base-url", Input).value = "https://example.test/v1"
        app.screen.query_one("#config-api-key", Input).value = "local-secret"
        app.screen.query_one("#config-discover", Button).press()
        await pilot.pause()
        await app.screen.workers.wait_for_complete()

        model_select = app.screen.query_one("#config-model", Select)
        assert model_select.value == "demo-model"
        model_select.value = "demo-model"
        app.screen.query_one("#config-save", Button).press()
        await pilot.pause()

        configured = store.load().models["small"]
        assert configured.model == "demo-model"
        assert configured.available_models == ["demo-model", "other-model"]
        assert configured.base_url == "https://example.test/v1"
        assert configured.api_key == "local-secret"


@pytest.mark.asyncio
async def test_slash_menu_supports_arrow_and_tab_completion(tmp_path: Path) -> None:
    app = CapyCodeApp(
        workspace=tmp_path,
        models_path=Path("config/models.example.yaml"),
        settings_store=UserSettingsStore(tmp_path / "settings.json"),
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/mo"
        await pilot.pause()
        await pilot.press("down", "tab")
        await pilot.pause()

        assert prompt.value == "/model"


@pytest.mark.asyncio
async def test_normal_prompt_runs_agent_and_keeps_session(tmp_path: Path) -> None:
    calls: list[tuple[str, Path, str]] = []

    async def fake_runner(
        task: str,
        workspace: Path,
        model_alias: str,
        models_path: Path,
        max_steps: int,
        settings_store: UserSettingsStore | None,
        observer: RuntimeObserver | None,
    ) -> SessionState:
        calls.append((task, workspace, model_alias))
        return SessionState(
            workspace=str(workspace),
            task=task,
            status="completed",
            step=1,
            final_answer="done",
            current_model=model_alias,
        )

    app = CapyCodeApp(
        workspace=tmp_path,
        models_path=Path("config/models.example.yaml"),
        settings_store=UserSettingsStore(tmp_path / "settings.json"),
        task_runner=fake_runner,
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "Read the repository"
        await pilot.press("enter")
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert calls == [("Read the repository", tmp_path, "small")]
        assert app.last_session is not None
        assert app.last_session.final_answer == "done"


@pytest.mark.asyncio
async def test_tui_updates_assistant_message_during_stream(tmp_path: Path) -> None:
    async def streaming_runner(
        task: str,
        workspace: Path,
        model_alias: str,
        models_path: Path,
        max_steps: int,
        settings_store: UserSettingsStore | None,
        observer: RuntimeObserver | None,
    ) -> SessionState:
        assert observer is not None
        await observer.on_model_start(1)
        await observer.on_tool_start("read_file", {"path": "README.md"})
        await observer.on_tool_result(
            "read_file",
            ToolResult(status="success", content="README contents"),
        )
        await observer.on_model_start(2)
        await observer.on_text_delta("Hello")
        await observer.on_text_delta(" world")
        return SessionState(
            workspace=str(workspace),
            task=task,
            status="completed",
            step=1,
            final_answer="Hello world",
            current_model="demo-model",
        )

    app = CapyCodeApp(
        workspace=tmp_path,
        models_path=Path("config/models.example.yaml"),
        settings_store=UserSettingsStore(tmp_path / "settings.json"),
        task_runner=streaming_runner,
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "Say hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assistant_messages = app.query("ChatMessage.assistant-message")
        assert len(assistant_messages) == 1
        message = assistant_messages.first(ChatMessage)
        assert message.content == "Hello world"
        assert len(app.query(ThinkingIndicator)) == 0
        tool = app.query_one(ToolActivity)
        assert tool.has_class("tool-success")
