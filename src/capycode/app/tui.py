from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Label,
    Markdown,
    OptionList,
    Select,
    Static,
)
from textual.widgets.option_list import Option

from capycode.config.loader import DEFAULT_MODELS_PATH, load_models
from capycode.config.user_settings import UserModelSettings, UserSettingsStore
from capycode.core import RuntimeObserver, SessionState
from capycode.tools import ToolResult

from .runtime import discover_models, execute_task

TaskRunner = Callable[
    [
        str,
        Path,
        str,
        Path,
        int,
        UserSettingsStore | None,
        RuntimeObserver | None,
    ],
    Awaitable[SessionState],
]
ModelFetcher = Callable[[str, str], Awaitable[list[str]]]


@dataclass(frozen=True)
class SlashCommand:
    name: str
    usage: str
    description: str


COMMANDS = (
    SlashCommand("/help", "/help", "显示可用命令"),
    SlashCommand("/config", "/config", "配置当前模型和本地凭据"),
    SlashCommand("/models", "/models", "列出服务端返回的可用模型"),
    SlashCommand("/model", "/model [model-id]", "打开模型选择器或直接切换"),
    SlashCommand("/workspace", "/workspace [path]", "查看或切换工作区"),
    SlashCommand("/status", "/status", "显示当前会话状态"),
    SlashCommand("/clear", "/clear", "清空会话显示"),
    SlashCommand("/quit", "/quit", "退出 CapyCode"),
)


class ChatMessage(Vertical):
    DEFAULT_CSS = """
    ChatMessage {
        height: auto;
        margin: 1 1 0 1;
        padding: 0 1;
    }

    ChatMessage.user-message {
        background: $panel;
        border-left: thick $accent;
        padding: 1 2;
    }

    ChatMessage.assistant-message {
        border-left: thick $success;
    }

    ChatMessage .message-role {
        height: 1;
        text-style: bold;
        color: $text-muted;
    }

    ChatMessage.user-message .message-role {
        color: $accent;
    }

    ChatMessage.assistant-message .message-role {
        color: $success;
    }

    ChatMessage Markdown {
        height: auto;
        margin: 0;
        padding: 0;
    }
    """

    def __init__(self, role: str, content: str, *, assistant: bool = False) -> None:
        classes = "assistant-message" if assistant else "user-message"
        super().__init__(classes=classes)
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static(self.role, classes="message-role")
        yield Markdown(self.content, classes="message-content")

    async def append_delta(self, delta: str) -> None:
        self.content += delta
        await self.query_one(Markdown).update(self.content)


class NoticeMessage(Static):
    DEFAULT_CSS = """
    NoticeMessage {
        height: auto;
        margin: 1 2 0 2;
        color: $text-muted;
    }

    NoticeMessage.error-notice {
        color: $error;
        border-left: thick $error;
        padding-left: 1;
    }
    """


class ToolActivity(Static):
    DEFAULT_CSS = """
    ToolActivity {
        height: auto;
        margin: 1 2 0 2;
        color: $text-muted;
    }

    ToolActivity.tool-success {
        color: $success;
    }

    ToolActivity.tool-error {
        color: $error;
    }
    """

    def __init__(self, name: str, target: str) -> None:
        super().__init__(f"◌  {name}  {target}")
        self.tool_name = name
        self.target = target

    def finish(self, result: ToolResult) -> None:
        self.remove_class("tool-success", "tool-error")
        if result.status == "success":
            self.add_class("tool-success")
            self.update(f"✓  {self.tool_name}  {self.target}")
        else:
            self.add_class("tool-error")
            self.update(f"×  {self.tool_name}  {result.content}")


class ThinkingIndicator(Static):
    FRAMES: ClassVar[tuple[str, ...]] = ("·", "✢", "✳", "✶", "✻", "✽")

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: 1;
        margin: 1 2 0 2;
        color: $accent;
    }
    """

    def __init__(self, label: str = "正在思考") -> None:
        super().__init__()
        self.label = label
        self.frame = 0

    def on_mount(self) -> None:
        self._advance()
        self.set_interval(0.12, self._advance)

    def _advance(self) -> None:
        glyph = self.FRAMES[self.frame % len(self.FRAMES)]
        self.frame += 1
        self.update(f"{glyph}  {self.label}")


class SplashAnimation(Static):
    FRAMES: ClassVar[tuple[str, ...]] = ("·", "✢", "✳", "✶", "✻", "✽")

    def __init__(self) -> None:
        super().__init__("·  正在准备工作区", id="splash-animation")
        self.frame = 0

    def on_mount(self) -> None:
        self._advance()
        self.set_interval(0.12, self._advance)

    def _advance(self) -> None:
        glyph = self.FRAMES[self.frame % len(self.FRAMES)]
        self.frame += 1
        self.update(f"{glyph}  正在准备工作区")


class ModelConfigScreen(ModalScreen[bool]):
    CSS = """
    ModelConfigScreen {
        align: center middle;
        background: $background 70%;
    }

    #config-dialog {
        width: 72;
        height: auto;
        max-height: 90%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }

    #config-dialog Input {
        margin-bottom: 1;
    }

    #config-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #config-actions {
        align-horizontal: right;
        height: auto;
        margin-top: 1;
    }

    #config-actions Button {
        margin-left: 1;
    }

    #config-error {
        color: $error;
        height: auto;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        alias: str,
        store: UserSettingsStore,
        model_fetcher: ModelFetcher,
    ) -> None:
        super().__init__()
        self.alias = alias
        self.store = store
        self.model_fetcher = model_fetcher
        local = self.store.load().models.get(self.alias)
        self.available_models = list(local.available_models) if local else []

    def compose(self) -> ComposeResult:
        local = self.store.load().models.get(self.alias)
        with Vertical(id="config-dialog"):
            yield Label("配置模型服务", id="config-title")
            yield Label("Base URL")
            yield Input(
                value=local.base_url if local else "",
                placeholder="https://example.com/v1",
                id="config-base-url",
            )
            yield Label("API Key（仅保存在本机用户目录）")
            yield Input(
                value=local.api_key if local else "",
                placeholder="输入 API Key",
                password=True,
                id="config-api-key",
            )
            yield Button("获取可用模型", id="config-discover")
            yield Static("", id="config-error")
            yield Label("模型 ID")
            initial_options = [(model, model) for model in self.available_models]
            yield Select[str](
                initial_options,
                value=local.model if local else Select.NULL,
                prompt="请先获取模型列表",
                allow_blank=True,
                id="config-model",
            )
            with Horizontal(id="config-actions"):
                yield Button("取消", id="config-cancel")
                yield Button("保存", variant="primary", id="config-save")

    @on(Button.Pressed, "#config-cancel")
    def cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#config-save")
    def save(self) -> None:
        base_url = self.query_one("#config-base-url", Input).value.strip()
        api_key = self.query_one("#config-api-key", Input).value.strip()
        selected_model = self.query_one("#config-model", Select).value
        if selected_model is Select.NULL or not base_url or not api_key:
            self.query_one("#config-error", Static).update(
                "请先填写 URL 和 API Key、获取模型列表并选择模型。"
            )
            return
        if not base_url.startswith(("http://", "https://")):
            self.query_one("#config-error", Static).update(
                "Base URL 必须以 http:// 或 https:// 开头。"
            )
            return
        try:
            self.store.configure_model(
                self.alias,
                model=str(selected_model),
                base_url=base_url,
                api_key=api_key,
                available_models=self.available_models,
            )
        except (OSError, ValueError) as exc:
            self.query_one("#config-error", Static).update(str(exc))
            return
        self.dismiss(True)

    @on(Button.Pressed, "#config-discover")
    def start_model_discovery(self) -> None:
        self.fetch_models()

    @work(exclusive=True, group="model-discovery")
    async def fetch_models(self) -> None:
        base_url = self.query_one("#config-base-url", Input).value.strip()
        api_key = self.query_one("#config-api-key", Input).value.strip()
        error = self.query_one("#config-error", Static)
        button = self.query_one("#config-discover", Button)
        if not base_url or not api_key:
            error.update("请先填写 Base URL 和 API Key。")
            return
        if not base_url.startswith(("http://", "https://")):
            error.update("Base URL 必须以 http:// 或 https:// 开头。")
            return
        button.disabled = True
        button.label = "正在获取…"
        error.update("正在访问模型列表…")
        try:
            models = await self.model_fetcher(base_url, api_key)
            self.available_models = models
            select = self.query_one("#config-model", Select)
            select.set_options((model, model) for model in models)
            select.value = models[0]
            error.update(f"已获取 {len(models)} 个模型，请选择后保存。")
        except Exception as exc:
            error.update(f"获取模型失败：{exc}")
        finally:
            button.disabled = False
            button.label = "重新获取模型"


class ModelPickerScreen(ModalScreen[str | None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "cancel", "取消"),
    ]

    CSS = """
    ModelPickerScreen {
        align: center middle;
        background: $background 70%;
    }

    #model-picker-dialog {
        width: 76;
        height: auto;
        max-height: 80%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }

    #model-picker-title {
        text-style: bold;
    }

    #model-picker-list {
        height: auto;
        max-height: 14;
        margin: 1 0;
    }

    #model-picker-hint {
        color: $text-muted;
    }
    """

    def __init__(self, models: list[str], current: str) -> None:
        super().__init__()
        self.models = models
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker-dialog"):
            yield Label("选择模型", id="model-picker-title")
            yield OptionList(
                *[
                    Option(
                        f"{'● ' if model == self.current else '  '}{model}",
                        id=model,
                    )
                    for model in self.models
                ],
                id="model-picker-list",
            )
            yield Static("↑/↓ 选择  ·  Enter 确认  ·  Esc 取消", id="model-picker-hint")

    def on_mount(self) -> None:
        picker = self.query_one("#model-picker-list", OptionList)
        picker.highlighted = self.models.index(self.current)
        picker.focus()

    @on(OptionList.OptionSelected, "#model-picker-list")
    def select_model(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class CapyCodeApp(App[None]):
    TITLE = "CapyCode"
    SUB_TITLE = "Local Coding Agent"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen {
        background: $background;
    }

    #shell {
        height: 1fr;
        padding: 0 1;
    }

    #splash {
        layer: splash;
        dock: top;
        width: 100%;
        height: 100%;
        align: center middle;
        background: $background;
    }

    #splash.hidden {
        display: none;
    }

    #splash-content {
        width: 66;
        height: auto;
        align-horizontal: center;
    }

    #splash-logo {
        width: 66;
        height: 7;
        color: $accent;
        text-style: bold;
        text-align: center;
    }

    #splash-subtitle {
        height: 1;
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }

    #splash-animation {
        height: 1;
        color: $accent;
        text-align: center;
        margin-top: 2;
    }

    #session-bar {
        height: 1;
        margin: 0 1;
        color: $text-muted;
    }

    #transcript {
        height: 1fr;
        padding: 0 1 1 1;
        scrollbar-size: 1 1;
    }

    #command-menu {
        display: none;
        height: auto;
        max-height: 12;
        margin: 0 2;
        padding: 0 1;
        border: round $accent;
        background: $panel;
    }

    #command-menu.visible {
        display: block;
    }

    #prompt {
        margin: 0 2;
        border: none;
        border-top: solid $accent;
        background: $background;
    }

    #status-line {
        height: 1;
        margin: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c", "cancel_or_quit", "取消/退出"),
        ("ctrl+d", "quit", "退出"),
        ("ctrl+l", "clear_transcript", "清屏"),
    ]

    def __init__(
        self,
        *,
        workspace: Path | None = None,
        model_alias: str | None = None,
        models_path: Path = DEFAULT_MODELS_PATH,
        max_steps: int = 10,
        settings_store: UserSettingsStore | None = None,
        task_runner: TaskRunner = execute_task,
        model_fetcher: ModelFetcher = discover_models,
    ) -> None:
        super().__init__()
        self.workspace = (workspace or Path.cwd()).resolve()
        self.models_path = models_path
        self.max_steps = max_steps
        self.settings_store = settings_store or UserSettingsStore()
        settings = self.settings_store.load()
        self.model_alias = model_alias or settings.default_model
        configured = settings.models.get(self.model_alias)
        self.model_id = configured.model if configured else None
        self.task_runner = task_runner
        self.model_fetcher = model_fetcher
        self.busy = False
        self.last_session: SessionState | None = None
        self.prompt_history: list[str] = []
        self.history_index: int | None = None
        self.streaming_message: ChatMessage | None = None
        self.thinking_indicator: ThinkingIndicator | None = None
        self.active_tool: ToolActivity | None = None
        self.received_stream_delta = False
        self.splash_visible = True

    def compose(self) -> ComposeResult:
        with Container(id="splash"):
            with Vertical(id="splash-content"):
                yield Static(
                    "  ____                    ____          _      \n"
                    " / ___|__ _ _ __  _   _ / ___|___   __| | ___ \n"
                    "| |   / _` | '_ \\| | | | |   / _ \\ / _` |/ _ \\\n"
                    "| |__| (_| | |_) | |_| | |__| (_) | (_| |  __/\n"
                    " \\____\\__,_| .__/ \\__, |\\____\\___/ \\__,_|\\___|\n"
                    "           |_|    |___/",
                    id="splash-logo",
                    markup=False,
                )
                yield Static(
                    "Local coding agent  ·  工作在你的代码目录中",
                    id="splash-subtitle",
                )
                yield SplashAnimation()
        with Container(id="shell"):
            yield Static(id="session-bar")
            yield VerticalScroll(id="transcript")
            yield OptionList(id="command-menu")
            yield Input(placeholder="输入任务，或输入 / 查看命令", id="prompt")
            yield Static(id="status-line")

    def on_mount(self) -> None:
        self._refresh_session_bar()
        self._write_system("输入任务开始工作，输入 / 查看命令。")
        self._refresh_status()
        self.query_one("#prompt", Input).focus()
        self.set_timer(1.15, self._dismiss_splash)

    @on(Input.Changed, "#prompt")
    def show_slash_commands(self, event: Input.Changed) -> None:
        value = event.value.strip().lower()
        menu = self.query_one("#command-menu", OptionList)
        if not value.startswith("/") or " " in value:
            menu.remove_class("visible")
            return
        matches = [command for command in COMMANDS if command.name.startswith(value)]
        menu.set_options(
            [
                Option(
                    f"[b]{command.usage:<22}[/b] [dim]{command.description}[/dim]",
                    id=command.name,
                )
                for command in matches
            ]
        )
        if matches:
            menu.highlighted = 0
            menu.add_class("visible")
        else:
            menu.remove_class("visible")

    @on(OptionList.OptionSelected, "#command-menu")
    def choose_slash_command(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is None:
            return
        prompt = self.query_one("#prompt", Input)
        prompt.value = event.option_id
        prompt.cursor_position = len(prompt.value)
        self.query_one("#command-menu", OptionList).remove_class("visible")
        prompt.focus()

    def on_key(self, event: events.Key) -> None:
        if self.splash_visible:
            self._dismiss_splash()
        prompt = self.query_one("#prompt", Input)
        if not prompt.has_focus:
            return
        menu = self.query_one("#command-menu", OptionList)
        if menu.has_class("visible") and menu.option_count:
            if event.key in {"up", "down"}:
                current = menu.highlighted or 0
                delta = -1 if event.key == "up" else 1
                menu.highlighted = (current + delta) % menu.option_count
                event.prevent_default()
                event.stop()
                return
            if event.key == "tab":
                option = menu.highlighted_option
                if option is not None and option.id is not None:
                    prompt.value = option.id
                    prompt.cursor_position = len(prompt.value)
                    menu.remove_class("visible")
                event.prevent_default()
                event.stop()
                return
            if event.key == "escape":
                menu.remove_class("visible")
                event.prevent_default()
                event.stop()
                return
        if event.key in {"up", "down"} and self.prompt_history:
            if event.key == "up":
                self.history_index = (
                    len(self.prompt_history) - 1
                    if self.history_index is None
                    else max(0, self.history_index - 1)
                )
                prompt.value = self.prompt_history[self.history_index]
            elif self.history_index is not None:
                self.history_index += 1
                if self.history_index >= len(self.prompt_history):
                    self.history_index = None
                    prompt.value = ""
                else:
                    prompt.value = self.prompt_history[self.history_index]
            prompt.cursor_position = len(prompt.value)
            event.prevent_default()
            event.stop()

    def _dismiss_splash(self) -> None:
        if not self.splash_visible:
            return
        self.splash_visible = False
        self.query_one("#splash", Container).add_class("hidden")
        self.query_one("#prompt", Input).focus()

    @on(Input.Submitted, "#prompt")
    async def submit_prompt(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        event.input.clear()
        self.query_one("#command-menu", OptionList).remove_class("visible")
        if value.startswith("/"):
            await self._run_command(value)
            return
        if self.busy:
            self._write_error("当前任务仍在执行，请等待完成后再提交。")
            return
        self.prompt_history.append(value)
        self.history_index = None
        self._write_user(value)
        self.run_task(value)

    async def _run_command(self, raw: str) -> None:
        command, _, argument = raw.partition(" ")
        command = command.lower()
        argument = argument.strip()
        if command == "/help":
            lines = [f"{item.usage:<24} {item.description}" for item in COMMANDS]
            self._write_system("可用命令：\n" + "\n".join(lines))
        elif command == "/clear":
            self.action_clear_transcript()
        elif command == "/quit":
            self.exit()
        elif command == "/status":
            self._write_system(self._status_text())
        elif command == "/models":
            await self._show_models()
        elif command == "/model":
            await self._select_model(argument)
        elif command == "/workspace":
            self._select_workspace(argument)
        elif command == "/config":
            self._open_config()
        else:
            self._write_error(f"未知命令：{command}。输入 /help 查看可用命令。")

    async def _show_models(self) -> None:
        configured = await self._refresh_models()
        if configured is None:
            return
        lines = [
            f"{'●' if model == configured.model else ' '} {model}"
            for model in configured.available_models
        ]
        self._write_system("可用模型：\n" + "\n".join(lines))

    async def _select_model(self, model_id: str) -> None:
        if not model_id:
            configured = await self._refresh_models()
            if configured is None:
                return
            self.push_screen(
                ModelPickerScreen(configured.available_models, configured.model),
                self._model_picker_closed,
            )
            return
        self._apply_model_selection(model_id)

    async def _refresh_models(self) -> UserModelSettings | None:
        try:
            configured = self.settings_store.load().models.get(self.model_alias)
        except (OSError, ValueError) as exc:
            self._write_error(str(exc))
            return None
        if configured is None:
            self._write_error("尚未配置模型。请先运行 /config。")
            return None

        self._write_system("正在刷新服务端模型列表…")
        try:
            models = await self.model_fetcher(configured.base_url, configured.api_key)
            selected = configured.model if configured.model in models else models[0]
            settings = self.settings_store.configure_model(
                self.model_alias,
                model=selected,
                base_url=configured.base_url,
                api_key=configured.api_key,
                available_models=models,
            )
            self.model_id = selected
            self._refresh_status()
            return settings.models[self.model_alias]
        except Exception as exc:
            self._write_error(f"刷新模型列表失败，使用本地缓存：{exc}")
            return configured

    def _model_picker_closed(self, model_id: str | None) -> None:
        if model_id is not None:
            self._apply_model_selection(model_id)

    def _apply_model_selection(self, model_id: str) -> None:
        try:
            self.settings_store.select_model(self.model_alias, model_id)
        except (OSError, ValueError) as exc:
            self._write_error(str(exc))
            return
        self.model_id = model_id
        self._write_system(f"已切换模型：{model_id}")
        self._refresh_status()

    def _select_workspace(self, value: str) -> None:
        if not value:
            self._write_system(f"当前工作区：{self.workspace}")
            return
        candidate = Path(value).expanduser().resolve()
        if not candidate.is_dir():
            self._write_error(f"工作区不存在或不是目录：{candidate}")
            return
        self.workspace = candidate
        self._write_system(f"已切换工作区：{candidate}")
        self._refresh_status()

    def _open_config(self) -> None:
        try:
            registry = load_models(self.models_path)
        except (OSError, ValueError) as exc:
            self._write_error(str(exc))
            return
        if self.model_alias not in registry.models:
            self._write_error(f"模型别名不存在：{self.model_alias}")
            return
        self.push_screen(
            ModelConfigScreen(
                self.model_alias,
                self.settings_store,
                self.model_fetcher,
            ),
            self._config_closed,
        )

    def _config_closed(self, saved: bool | None) -> None:
        if saved:
            configured = self.settings_store.load().models[self.model_alias]
            self.model_id = configured.model
            self._write_system(f"模型配置已保存到 {self.settings_store.path}")
            self._refresh_status()

    @work(exclusive=True)
    async def run_task(self, task: str) -> None:
        self.busy = True
        self.received_stream_delta = False
        self.streaming_message = None
        self.active_tool = None
        self._refresh_status()
        try:
            state = await self.task_runner(
                task,
                self.workspace,
                self.model_alias,
                self.models_path,
                self.max_steps,
                self.settings_store,
                self,
            )
            self.last_session = state
            if state.final_answer and not self.received_stream_delta:
                self._write_assistant(state.final_answer)
            if state.status != "completed":
                self._write_error(state.last_error or f"任务结束：{state.status}")
        except Exception as exc:
            self._write_error(str(exc))
        finally:
            await self._stop_thinking()
            self.streaming_message = None
            self.busy = False
            self._refresh_status()
            self.query_one("#prompt", Input).focus()

    async def on_model_start(self, step: int) -> None:
        self.streaming_message = None
        await self._stop_thinking()
        indicator = ThinkingIndicator("正在思考" if step == 1 else "正在继续")
        self.thinking_indicator = indicator
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(indicator)
        transcript.scroll_end(animate=False)

    async def on_text_delta(self, delta: str) -> None:
        await self._stop_thinking()
        transcript = self.query_one("#transcript", VerticalScroll)
        if self.streaming_message is None:
            self.streaming_message = ChatMessage("CapyCode", "", assistant=True)
            await transcript.mount(self.streaming_message)
        self.received_stream_delta = True
        await self.streaming_message.append_delta(delta)
        transcript.scroll_end(animate=False)

    async def on_tool_start(self, name: str, arguments: dict[str, object]) -> None:
        await self._stop_thinking()
        target = str(arguments.get("path") or arguments.get("cwd") or "")
        self.active_tool = ToolActivity(name, target)
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(self.active_tool)
        transcript.scroll_end(animate=False)

    async def on_tool_result(self, name: str, result: ToolResult) -> None:
        if self.active_tool is not None and self.active_tool.tool_name == name:
            self.active_tool.finish(result)
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    async def _stop_thinking(self) -> None:
        if self.thinking_indicator is not None:
            indicator = self.thinking_indicator
            self.thinking_indicator = None
            if indicator.is_mounted:
                await indicator.remove()

    def action_cancel_or_quit(self) -> None:
        workers = list(self.workers)
        if self.busy and workers:
            for worker in workers:
                worker.cancel()
            self.busy = False
            self._write_system("已取消当前任务。")
            self._refresh_status()
        else:
            self.exit()

    def action_clear_transcript(self) -> None:
        self.query_one("#transcript", VerticalScroll).remove_children()

    def _write_user(self, content: str) -> None:
        self._mount_transcript(ChatMessage("You", content))

    def _write_assistant(self, content: str) -> None:
        self._mount_transcript(ChatMessage("CapyCode", content, assistant=True))

    def _write_system(self, content: str) -> None:
        self._mount_transcript(NoticeMessage(f"·  {content}", markup=False))

    def _write_error(self, content: str) -> None:
        self._mount_transcript(NoticeMessage(f"!  {content}", classes="error-notice", markup=False))

    def _mount_transcript(self, widget: Static | Vertical) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(widget)
        transcript.scroll_end(animate=False)

    def _status_text(self) -> str:
        state = "运行中" if self.busy else "就绪"
        return f"{state}  ·  model: {self.model_id or '未配置'}  ·  workspace: {self.workspace}"

    def _refresh_status(self) -> None:
        self.query_one("#status-line", Static).update(self._status_text())
        self._refresh_session_bar()

    def _refresh_session_bar(self) -> None:
        workspace_name = self.workspace.name or str(self.workspace)
        self.query_one("#session-bar", Static).update(
            f"CapyCode  ·  {self.model_id or '未配置模型'}  ·  {workspace_name}"
        )


def launch_tui(
    *,
    workspace: Path | None = None,
    model_alias: str | None = None,
    models_path: Path = DEFAULT_MODELS_PATH,
) -> None:
    CapyCodeApp(
        workspace=workspace,
        model_alias=model_alias,
        models_path=models_path,
    ).run()
