from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest
from textual.widgets import Button, Input, OptionList, Select, Static

from capycode.app.tui import (
    CapyCodeApp,
    ChatMessage,
    RunDetailScreen,
    RunPickerScreen,
    SessionPickerScreen,
    ThinkingIndicator,
    ToolActivity,
)
from capycode.config import UserSettingsStore
from capycode.core import RuntimeObserver, SessionState
from capycode.llm import Message
from capycode.tools import ToolResult
from capycode.trace import (
    AssistantTextEvent,
    RunStatusEvent,
    RunTracker,
    RunTrackingConfig,
    StepTraceEvent,
    ToolRequestEvent,
    ToolResultEvent,
)


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
    store.configure_endpoint(
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
        persisted = store.load()
        assert persisted.default_model == "model-c"
        assert persisted.endpoint is not None
        assert persisted.endpoint.available_models == [
            "model-a",
            "model-b",
            "model-c",
            "model-d",
        ]
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

        configured = store.load()
        assert configured.default_model == "demo-model"
        assert configured.endpoint is not None
        assert configured.endpoint.available_models == ["demo-model", "other-model"]
        assert configured.endpoint.base_url == "https://example.test/v1"
        assert configured.endpoint.api_key == "local-secret"


@pytest.mark.asyncio
async def test_pricing_dialog_saves_price_for_real_model_id(tmp_path: Path) -> None:
    store = UserSettingsStore(tmp_path / "settings.json")
    store.configure_endpoint(
        model="model-a",
        base_url="https://example.test/v1",
        api_key="local-secret",
        available_models=["model-a", "model-b"],
    )
    app = CapyCodeApp(workspace=tmp_path, settings_store=store)

    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "/pricing"
        await pilot.press("enter")
        await pilot.pause()

        app.screen.query_one("#pricing-input", Input).value = "2.5"
        app.screen.query_one("#pricing-output", Input).value = "10"
        app.screen.query_one("#pricing-currency", Input).value = "cny"
        app.screen.query_one("#pricing-context", Input).value = "200000"
        app.screen.query_one("#pricing-date", Input).value = "2026-08-28"
        app.screen.query_one("#pricing-save", Button).press()
        await pilot.pause()

        metadata = store.load().models["model-a"]
        assert metadata.pricing.input_per_million == 2.5
        assert metadata.pricing.output_per_million == 10
        assert metadata.pricing.currency == "CNY"
        assert metadata.pricing.snapshot_date == date(2026, 8, 28)
        assert metadata.context_window == 200_000


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
    calls: list[tuple[str, Path, str | None]] = []

    async def fake_runner(
        task: str,
        workspace: Path,
        model_id: str | None,
        models_path: Path,
        max_steps: int,
        settings_store: UserSettingsStore | None,
        observer: RuntimeObserver | None,
        session_state: SessionState | None,
        checkpoint: object,
    ) -> SessionState:
        calls.append((task, workspace, model_id))
        return SessionState(
            workspace=str(workspace),
            task=task,
            status="completed",
            step=1,
            final_answer="done",
            current_model=model_id,
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

        assert calls == [("Read the repository", tmp_path, None)]
        assert app.last_session is not None
        assert app.last_session.final_answer == "done"


@pytest.mark.asyncio
async def test_tui_updates_assistant_message_during_stream(tmp_path: Path) -> None:
    async def streaming_runner(
        task: str,
        workspace: Path,
        model_id: str | None,
        models_path: Path,
        max_steps: int,
        settings_store: UserSettingsStore | None,
        observer: RuntimeObserver | None,
        session_state: SessionState | None,
        checkpoint: object,
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


def test_tool_activity_compacts_long_failure_output() -> None:
    activity = ToolActivity("run_tests", "python -m pytest -q")

    activity.finish(
        ToolResult(
            status="error",
            content="Tests exited with code 1.\n" + "failure details " * 100,
        )
    )

    assert activity.has_class("tool-error")
    assert len(str(activity.render())) < 220
    assert "Tests exited with code 1." in str(activity.render())


@pytest.mark.asyncio
async def test_session_can_resume_after_reopening_tui(tmp_path: Path) -> None:
    settings = UserSettingsStore(tmp_path / "local" / "settings.json")
    received_states: list[SessionState | None] = []

    async def session_runner(
        task: str,
        workspace: Path,
        model_id: str | None,
        models_path: Path,
        max_steps: int,
        settings_store: UserSettingsStore | None,
        observer: RuntimeObserver | None,
        session_state: SessionState | None,
        checkpoint: Callable[[SessionState], None] | None,
    ) -> SessionState:
        received_states.append(session_state)
        if session_state is None:
            state = SessionState(
                workspace=str(workspace),
                task=task,
                status="completed",
                final_answer="first answer",
                history=[
                    Message(role="system", content="system"),
                    Message(role="user", content=task),
                    Message(role="assistant", content="first answer"),
                ],
            )
        else:
            state = session_state
            state.task = task
            state.status = "completed"
            state.final_answer = "follow-up answer"
            state.history.extend(
                [
                    Message(role="user", content=task),
                    Message(role="assistant", content="follow-up answer"),
                ]
            )
        if checkpoint is not None:
            checkpoint(state)
        return state

    first_app = CapyCodeApp(
        workspace=tmp_path,
        models_path=Path("config/models.example.yaml"),
        settings_store=settings,
        task_runner=session_runner,
    )
    async with first_app.run_test() as pilot:
        first_app.query_one("#prompt", Input).value = "first task"
        await pilot.press("enter")
        await first_app.workers.wait_for_complete()

    second_app = CapyCodeApp(
        workspace=tmp_path,
        models_path=Path("config/models.example.yaml"),
        settings_store=settings,
        task_runner=session_runner,
    )
    async with second_app.run_test() as pilot:
        second_app.query_one("#prompt", Input).value = "/resume"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(second_app.screen, SessionPickerScreen)
        await pilot.press("enter")
        await pilot.pause()

        restored_id = second_app.last_session.session_id if second_app.last_session else None
        assert restored_id is not None
        restored_messages = second_app.query(ChatMessage)
        assert len(restored_messages) == 2

        second_app.query_one("#prompt", Input).value = "follow up"
        await pilot.press("enter")
        await second_app.workers.wait_for_complete()

        assert received_states[-1] is not None
        assert received_states[-1].session_id == restored_id
        assert second_app.last_session is not None
        assert second_app.last_session.final_answer == "follow-up answer"


@pytest.mark.asyncio
async def test_run_events_project_multiple_tools_by_call_id(tmp_path: Path) -> None:
    async def event_runner(
        task: str,
        workspace: Path,
        model_id: str | None,
        models_path: Path,
        max_steps: int,
        settings_store: UserSettingsStore | None,
        observer: RuntimeObserver | None,
        session_state: SessionState | None,
        checkpoint: object,
    ) -> SessionState:
        assert observer is not None
        run = "run-event-test"
        session = "session-event-test"
        observer.on_run_event(
            RunStatusEvent(run_id=run, session_id=session, sequence=1, status="started")
        )
        observer.on_run_event(
            ToolRequestEvent(
                run_id=run,
                session_id=session,
                sequence=2,
                step=1,
                tool_call_id="read-one",
                tool_name="read_file",
                arguments={"path": "README.md"},
            )
        )
        observer.on_run_event(
            ToolRequestEvent(
                run_id=run,
                session_id=session,
                sequence=3,
                step=1,
                tool_call_id="tests-one",
                tool_name="run_tests",
                arguments={"argv": ["pytest", "-q"], "cwd": "."},
            )
        )
        observer.on_run_event(
            ToolResultEvent(
                run_id=run,
                session_id=session,
                sequence=4,
                step=1,
                tool_call_id="tests-one",
                tool_name="run_tests",
                status="error",
                content="1 failed",
                data={"exit_code": 1},
                latency_seconds=0.4,
            )
        )
        observer.on_run_event(
            ToolResultEvent(
                run_id=run,
                session_id=session,
                sequence=5,
                step=1,
                tool_call_id="read-one",
                tool_name="read_file",
                status="success",
                content="contents",
                data={"path": "README.md"},
                latency_seconds=0.1,
            )
        )
        observer.on_run_event(
            StepTraceEvent(
                run_id=run,
                session_id=session,
                sequence=6,
                step=1,
                provider="fake",
                model_id="model-a",
                latency_seconds=1,
                input_tokens=10,
                output_tokens=5,
                cost=0.002,
                currency="CNY",
            )
        )
        observer.on_run_event(
            AssistantTextEvent(
                run_id=run,
                session_id=session,
                sequence=7,
                step=2,
                text="Finished",
            )
        )
        return SessionState(
            workspace=str(workspace),
            task=task,
            status="completed",
            step=2,
            final_answer="Finished",
            current_model="model-a",
            current_run_id=run,
        )

    app = CapyCodeApp(
        workspace=tmp_path,
        settings_store=UserSettingsStore(tmp_path / "settings.json"),
        task_runner=event_runner,
    )
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "inspect"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        activities = list(app.query(ToolActivity))
        assert [item.tool_call_id for item in activities] == ["read-one", "tests-one"]
        assert activities[0].has_class("tool-success")
        assert activities[1].has_class("tool-error")
        activities[1].toggle_details()
        detail = str(activities[1].render())
        assert '"cwd": "."' in detail
        assert "exit_code" in detail
        assert "1 failed" in detail
        assert "15 tok" in app._status_text(90)


@pytest.mark.asyncio
async def test_runs_command_opens_picker_and_detail(tmp_path: Path) -> None:
    tracker = RunTracker(
        tmp_path,
        "session-one",
        RunTrackingConfig(
            provider="fake",
            model_id="model-a",
            input_per_million=1,
            output_per_million=2,
            currency="CNY",
            pricing_snapshot_date="2026-08-28",
        ),
        run_id="run-one",
    )
    tracker.start("inspect repository")
    tracker.finish(
        SessionState(
            workspace=str(tmp_path),
            task="inspect repository",
            status="completed",
            termination_reason="completed",
        )
    )
    app = CapyCodeApp(
        workspace=tmp_path,
        settings_store=UserSettingsStore(tmp_path / "settings.json"),
    )

    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "/runs"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, RunPickerScreen)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, RunDetailScreen)
        detail = app.screen.query_one("#run-detail-content", Static)
        assert "inspect repository" in str(detail.render())


@pytest.mark.asyncio
async def test_cancel_keeps_busy_until_worker_cleanup_and_restores_focus(tmp_path: Path) -> None:
    started = asyncio.Event()

    async def waiting_runner(
        task: str,
        workspace: Path,
        model_id: str | None,
        models_path: Path,
        max_steps: int,
        settings_store: UserSettingsStore | None,
        observer: RuntimeObserver | None,
        session_state: SessionState | None,
        checkpoint: object,
    ) -> SessionState:
        started.set()
        await asyncio.sleep(60)
        raise AssertionError("worker should have been cancelled")

    app = CapyCodeApp(
        workspace=tmp_path,
        settings_store=UserSettingsStore(tmp_path / "settings.json"),
        task_runner=waiting_runner,
    )
    async with app.run_test(size=(60, 24)) as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "wait"
        await pilot.press("enter")
        await started.wait()
        assert app.busy
        await pilot.press("ctrl+c")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert not app.busy
        assert prompt.has_focus
        assert len(app._status_text(60)) < 60
