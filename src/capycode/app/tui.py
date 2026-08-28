from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import ClassVar

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message as TextualMessage
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

from capycode.config.loader import DEFAULT_MODELS_PATH
from capycode.config.user_settings import UserEndpointSettings, UserSettingsStore
from capycode.core import RuntimeObserver, SessionState, SessionStore, SessionSummary
from capycode.tools import ToolResult
from capycode.trace import (
    AssistantTextEvent,
    ContextSnapshotEvent,
    RunCatalog,
    RunEvent,
    RunStatusEvent,
    RunSummary,
    StepTraceEvent,
    ToolRequestEvent,
    ToolResultEvent,
)

from .runtime import discover_models, execute_task

TaskRunner = Callable[
    [
        str,
        Path,
        str | None,
        Path,
        int,
        UserSettingsStore | None,
        RuntimeObserver | None,
        SessionState | None,
        Callable[[SessionState], None] | None,
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
    SlashCommand("/pricing", "/pricing", "配置当前真实模型的价格和上下文窗口"),
    SlashCommand("/workspace", "/workspace [path]", "查看或切换工作区"),
    SlashCommand("/resume", "/resume [会话 ID]", "选择并恢复当前工作区的会话"),
    SlashCommand("/continue", "/continue", "继续当前工作区最近的会话"),
    SlashCommand("/sessions", "/sessions", "列出当前工作区的历史会话"),
    SlashCommand("/runs", "/runs", "列出当前工作区最近的运行记录"),
    SlashCommand("/new", "/new", "开始一个新会话"),
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

    async def set_content(self, content: str) -> None:
        self.content = content
        await self.query_one(Markdown).update(content)


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


class RunEventMessage(TextualMessage):
    """Transfers a persisted, redacted run event onto Textual's message pump."""

    def __init__(self, event: RunEvent) -> None:
        super().__init__()
        self.event = event


class ToolActivity(Static, can_focus=True):
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

    ToolActivity:focus {
        background: $panel;
    }
    """

    def __init__(
        self,
        name: str,
        target: str,
        *,
        tool_call_id: str | None = None,
        arguments: dict[str, object] | None = None,
    ) -> None:
        super().__init__(markup=False)
        self.tool_name = name
        self.target = target
        self.tool_call_id = tool_call_id
        self.arguments = arguments or {}
        self.result: ToolResult | None = None
        self.latency_seconds: float | None = None
        self.expanded = False
        self._refresh_content()

    def finish(self, result: ToolResult, latency_seconds: float | None = None) -> None:
        self.result = result
        self.latency_seconds = latency_seconds
        self.remove_class("tool-success", "tool-error")
        if result.status == "success":
            self.add_class("tool-success")
        else:
            self.add_class("tool-error")
        self._refresh_content()

    def toggle_details(self) -> None:
        self.expanded = not self.expanded
        self._refresh_content()

    def on_click(self) -> None:
        self.toggle_details()

    def on_key(self, event: events.Key) -> None:
        if event.key in {"enter", "space"}:
            self.toggle_details()
            event.stop()

    def _refresh_content(self) -> None:
        glyph = "◌"
        suffix = self.target
        if self.result is not None:
            glyph = "✓" if self.result.status == "success" else "×"
            if self.result.status == "error":
                suffix = next(
                    (line.strip() for line in self.result.content.splitlines() if line.strip()),
                    "failed",
                )
                if len(suffix) > 160:
                    suffix = suffix[:157] + "..."
        disclosure = "▾" if self.expanded else "▸"
        header = f"{glyph}  {self.tool_name}  {suffix}  {disclosure}"
        if not self.expanded:
            self.update(header)
            return
        self.update(f"{header}\n{self._details_text()}")

    def _details_text(self) -> str:
        lines = [f"   call: {self.tool_call_id or '-'}"]
        if self.arguments:
            rendered = json.dumps(self.arguments, ensure_ascii=False, indent=2, sort_keys=True)
            lines.append("   arguments:\n" + self._indent(_bounded_text(rendered)))
        if self.result is not None:
            latency = f"{self.latency_seconds:.3f}s" if self.latency_seconds is not None else "-"
            lines.append(f"   status: {self.result.status}  ·  duration: {latency}")
            metadata = {
                key: self.result.data[key]
                for key in ("cwd", "exit_code", "task_id", "stdout_truncated", "stderr_truncated")
                if key in self.result.data
            }
            if metadata:
                lines.append(
                    "   metadata: " + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                )
            if self.result.content:
                lines.append("   output:\n" + self._indent(_bounded_text(self.result.content)))
        return "\n".join(lines)

    @staticmethod
    def _indent(value: str) -> str:
        return "\n".join(f"      {line}" for line in value.splitlines())


def _bounded_text(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    half = (limit - 40) // 2
    return f"{value[:half]}\n... [界面已截断] ...\n{value[-half:]}"


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
        store: UserSettingsStore,
        model_fetcher: ModelFetcher,
    ) -> None:
        super().__init__()
        self.store = store
        self.model_fetcher = model_fetcher
        endpoint = self.store.load().endpoint
        self.available_models = list(endpoint.available_models) if endpoint else []

    def compose(self) -> ComposeResult:
        settings = self.store.load()
        endpoint = settings.endpoint
        with Vertical(id="config-dialog"):
            yield Label("配置模型服务", id="config-title")
            yield Label("Base URL")
            yield Input(
                value=endpoint.base_url if endpoint else "",
                placeholder="https://example.com/v1",
                id="config-base-url",
            )
            yield Label("API Key（仅保存在本机用户目录）")
            yield Input(
                value=endpoint.api_key if endpoint else "",
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
                value=settings.default_model or Select.NULL,
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
            self.store.configure_endpoint(
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


class PricingConfigScreen(ModalScreen[bool]):
    CSS = """
    PricingConfigScreen {
        align: center middle;
        background: $background 70%;
    }

    #pricing-dialog {
        width: 72;
        height: auto;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }

    #pricing-dialog Input {
        margin-bottom: 1;
    }

    #pricing-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #pricing-actions {
        align-horizontal: right;
        height: auto;
    }

    #pricing-error {
        color: $error;
        height: auto;
    }
    """

    def __init__(self, model_id: str, store: UserSettingsStore) -> None:
        super().__init__()
        self.model_id = model_id
        self.store = store

    def compose(self) -> ComposeResult:
        metadata = self.store.load().models[self.model_id]
        pricing = metadata.pricing
        with Vertical(id="pricing-dialog"):
            yield Label(f"模型费用 · {self.model_id}", id="pricing-title")
            yield Label("输入价格（每 100 万 Token）")
            yield Input(value=str(pricing.input_per_million), id="pricing-input")
            yield Label("输出价格（每 100 万 Token）")
            yield Input(value=str(pricing.output_per_million), id="pricing-output")
            yield Label("币种（例如 CNY、USD）")
            yield Input(value=pricing.currency, id="pricing-currency")
            yield Label("上下文窗口 Token 数")
            yield Input(value=str(metadata.context_window), id="pricing-context")
            yield Label("价格日期（YYYY-MM-DD）")
            yield Input(value=pricing.snapshot_date.isoformat(), id="pricing-date")
            yield Static("", id="pricing-error")
            with Horizontal(id="pricing-actions"):
                yield Button("取消", id="pricing-cancel")
                yield Button("保存", variant="primary", id="pricing-save")

    @on(Button.Pressed, "#pricing-cancel")
    def cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#pricing-save")
    def save(self) -> None:
        try:
            input_price = float(self.query_one("#pricing-input", Input).value)
            output_price = float(self.query_one("#pricing-output", Input).value)
            context_window = int(self.query_one("#pricing-context", Input).value)
            currency = self.query_one("#pricing-currency", Input).value.strip()
            snapshot_date = date.fromisoformat(self.query_one("#pricing-date", Input).value)
            self.store.configure_pricing(
                self.model_id,
                input_per_million=input_price,
                output_per_million=output_price,
                currency=currency,
                snapshot_date=snapshot_date,
                context_window=context_window,
            )
        except (OSError, ValueError) as exc:
            self.query_one("#pricing-error", Static).update(str(exc))
            return
        self.dismiss(True)


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


class SessionPickerScreen(ModalScreen[str | None]):
    BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "取消")]

    CSS = """
    SessionPickerScreen {
        align: center middle;
        background: $background 70%;
    }

    #session-picker-dialog {
        width: 88;
        height: auto;
        max-height: 80%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }

    #session-picker-title {
        text-style: bold;
    }

    #session-picker-list {
        height: auto;
        max-height: 16;
        margin: 1 0;
    }

    #session-picker-hint {
        color: $text-muted;
    }
    """

    def __init__(self, sessions: list[SessionSummary], current_id: str | None) -> None:
        super().__init__()
        self.sessions = sessions
        self.current_id = current_id

    def compose(self) -> ComposeResult:
        with Vertical(id="session-picker-dialog"):
            yield Label("恢复会话", id="session-picker-title")
            yield OptionList(
                *[
                    Option(
                        (
                            f"{'●' if item.session_id == self.current_id else ' '}  "
                            f"{item.title}  [dim]{item.updated_at.astimezone():%m-%d %H:%M}"
                            f"  {item.session_id[:8]}[/dim]"
                        ),
                        id=item.session_id,
                    )
                    for item in self.sessions
                ],
                id="session-picker-list",
            )
            yield Static("↑/↓ 选择  ·  Enter 恢复  ·  Esc 取消", id="session-picker-hint")

    def on_mount(self) -> None:
        picker = self.query_one("#session-picker-list", OptionList)
        picker.highlighted = 0
        picker.focus()

    @on(OptionList.OptionSelected, "#session-picker-list")
    def select_session(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class RunPickerScreen(ModalScreen[str | None]):
    BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "取消")]

    CSS = """
    RunPickerScreen {
        align: center middle;
        background: $background 70%;
    }

    #run-picker-dialog {
        width: 95%;
        max-width: 96;
        height: auto;
        max-height: 85%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }

    #run-picker-title { text-style: bold; }
    #run-picker-list { height: auto; max-height: 18; margin: 1 0; }
    #run-picker-hint { color: $text-muted; }
    """

    def __init__(self, summaries: list[RunSummary]) -> None:
        super().__init__()
        self.summaries = summaries

    def compose(self) -> ComposeResult:
        with Vertical(id="run-picker-dialog"):
            yield Label("运行记录", id="run-picker-title")
            yield OptionList(
                *[
                    Option(
                        (
                            f"{item.run_id[:8]}  {item.status:<9}  {item.model_id}  "
                            f"{item.steps} steps  {item.latency_seconds:.2f}s  "
                            f"[dim]{item.finished_at.astimezone():%m-%d %H:%M}[/dim]"
                        ),
                        id=item.run_id,
                    )
                    for item in self.summaries
                ],
                id="run-picker-list",
            )
            yield Static("↑/↓ 选择  ·  Enter 查看  ·  Esc 取消", id="run-picker-hint")

    def on_mount(self) -> None:
        picker = self.query_one("#run-picker-list", OptionList)
        picker.highlighted = 0
        picker.focus()

    @on(OptionList.OptionSelected, "#run-picker-list")
    def select_run(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class RunDetailScreen(ModalScreen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "close", "关闭"),
        ("q", "close", "关闭"),
    ]

    CSS = """
    RunDetailScreen {
        align: center middle;
        background: $background 70%;
    }

    #run-detail-dialog {
        width: 95%;
        max-width: 100;
        height: 88%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }

    #run-detail-title { height: 1; text-style: bold; }
    #run-detail-body { height: 1fr; margin: 1 0; scrollbar-size: 1 1; }
    #run-detail-content { height: auto; }
    #run-detail-hint { height: 1; color: $text-muted; }
    """

    def __init__(self, summary: RunSummary, events_: list[RunEvent]) -> None:
        super().__init__()
        self.summary = summary
        self.events_ = events_

    def compose(self) -> ComposeResult:
        with Vertical(id="run-detail-dialog"):
            yield Label(f"Run {self.summary.run_id[:8]}", id="run-detail-title")
            with VerticalScroll(id="run-detail-body"):
                yield Static(self._render_detail(), id="run-detail-content", markup=False)
            yield Static("Esc/Q 关闭  ·  ↑/↓ 滚动", id="run-detail-hint")

    def _render_detail(self) -> str:
        summary = self.summary
        tokens = summary.input_tokens + summary.output_tokens
        price = f"{summary.cost:.6f} {summary.currency}"
        lines = [
            f"状态        {summary.status} / {summary.termination_reason}",
            f"模型        {summary.model_id} ({summary.provider})",
            f"时间        {summary.finished_at.astimezone():%Y-%m-%d %H:%M:%S}",
            f"统计        {summary.steps} steps · {tokens} tokens · {price}",
            f"耗时        {summary.latency_seconds:.3f}s",
            f"工具        {summary.tool_successes} 成功 / {summary.tool_failures} 失败",
            f"任务        {summary.task}",
        ]
        if summary.modified_files:
            lines.append("修改文件    " + ", ".join(summary.modified_files))
        if summary.error:
            lines.append("错误        " + summary.error)
        lines.append("\n步骤")
        for event in self.events_:
            if isinstance(event, StepTraceEvent):
                step_tokens = event.input_tokens + event.output_tokens
                first = (
                    f" · first {event.first_token_latency_seconds:.3f}s"
                    if event.first_token_latency_seconds is not None
                    else ""
                )
                lines.append(
                    f"  #{event.step}  {event.latency_seconds:.3f}s{first} · "
                    f"{step_tokens} tokens · {event.cost:.6f} {event.currency}"
                )
            elif isinstance(event, ToolRequestEvent):
                lines.append(f"    ◌ {event.tool_name}  {_tool_target(event.arguments)}")
            elif isinstance(event, ToolResultEvent):
                glyph = "✓" if event.status == "success" else "×"
                excerpt = next(
                    (line.strip() for line in event.content.splitlines() if line.strip()), ""
                )
                lines.append(
                    f"    {glyph} {event.tool_name}  {event.latency_seconds:.3f}s  "
                    f"{excerpt[:140]}"
                )
        if summary.final_diff:
            lines.append("\n最终 diff\n" + _bounded_text(summary.final_diff, 8000))
        return "\n".join(lines)

    def action_close(self) -> None:
        self.dismiss(None)


def _tool_target(arguments: dict[str, object]) -> str:
    argv = arguments.get("argv")
    if isinstance(argv, list):
        return " ".join(str(part) for part in argv[:6])
    return str(arguments.get("path") or arguments.get("task_id") or arguments.get("cwd") or "")


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
        model_id: str | None = None,
        models_path: Path = DEFAULT_MODELS_PATH,
        max_steps: int = 10,
        settings_store: UserSettingsStore | None = None,
        session_store: SessionStore | None = None,
        initial_resume: str | None = None,
        task_runner: TaskRunner = execute_task,
        model_fetcher: ModelFetcher = discover_models,
    ) -> None:
        super().__init__()
        self.workspace = (workspace or Path.cwd()).resolve()
        self.models_path = models_path
        self.max_steps = max_steps
        self.settings_store = settings_store or UserSettingsStore()
        self.session_store = session_store or SessionStore(
            self.settings_store.path.parent / "sessions"
        )
        settings = self.settings_store.load()
        self.model_id = model_id or settings.default_model
        self.task_runner = task_runner
        self.model_fetcher = model_fetcher
        self.initial_resume = initial_resume
        self.busy = False
        self.last_session: SessionState | None = None
        self.prompt_history: list[str] = []
        self.history_index: int | None = None
        self.streaming_message: ChatMessage | None = None
        self.thinking_indicator: ThinkingIndicator | None = None
        self.tool_activities: dict[str, ToolActivity] = {}
        self.fallback_active_tool: ToolActivity | None = None
        self.event_stream_active = False
        self.received_stream_delta = False
        self.splash_visible = True
        self.live_run_id: str | None = None
        self.live_input_tokens = 0
        self.live_output_tokens = 0
        self.live_cost = 0.0
        self.live_currency = ""
        self.run_started_counter: float | None = None
        self.cancel_requested = False

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
        sessions = self.session_store.list(self.workspace)
        resume_hint = (
            f" 检测到 {len(sessions)} 个历史会话，可输入 /resume 恢复。" if sessions else ""
        )
        self._write_system(f"输入任务开始工作，输入 / 查看命令。{resume_hint}")
        self._refresh_status()
        self.query_one("#prompt", Input).focus()
        self.set_interval(0.5, self._refresh_status)
        self.set_timer(1.15, self._dismiss_splash)
        if self.initial_resume is not None:
            self.set_timer(0.05, self._resume_initial_session)

    def _resume_initial_session(self) -> None:
        assert self.initial_resume is not None
        self._resume_session(self.initial_resume)

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
        elif command == "/pricing":
            self._open_pricing()
        elif command == "/workspace":
            self._select_workspace(argument)
        elif command == "/sessions":
            self._show_sessions()
        elif command == "/runs":
            self._show_runs()
        elif command == "/resume":
            self._resume_session(argument)
        elif command == "/continue":
            self._resume_session("latest")
        elif command == "/new":
            self._new_session()
        elif command == "/config":
            self._open_config()
        else:
            self._write_error(f"未知命令：{command}。输入 /help 查看可用命令。")

    async def _show_models(self) -> None:
        configured = await self._refresh_models()
        if configured is None:
            return
        lines = [
            f"{'●' if model == self.model_id else ' '} {model}"
            for model in configured.available_models
        ]
        self._write_system("可用模型：\n" + "\n".join(lines))

    async def _select_model(self, model_id: str) -> None:
        if not model_id:
            configured = await self._refresh_models()
            if configured is None:
                return
            self.push_screen(
                ModelPickerScreen(configured.available_models, self.model_id or ""),
                self._model_picker_closed,
            )
            return
        self._apply_model_selection(model_id)

    async def _refresh_models(self) -> UserEndpointSettings | None:
        try:
            settings = self.settings_store.load()
            configured = settings.endpoint
        except (OSError, ValueError) as exc:
            self._write_error(str(exc))
            return None
        if configured is None:
            self._write_error("尚未配置模型。请先运行 /config。")
            return None

        self._write_system("正在刷新服务端模型列表…")
        try:
            models = await self.model_fetcher(configured.base_url, configured.api_key)
            selected = self.model_id if self.model_id in models else models[0]
            settings = self.settings_store.configure_endpoint(
                model=selected,
                base_url=configured.base_url,
                api_key=configured.api_key,
                available_models=models,
            )
            self.model_id = selected
            self._refresh_status()
            return settings.endpoint
        except Exception as exc:
            self._write_error(f"刷新模型列表失败，使用本地缓存：{exc}")
            return configured

    def _model_picker_closed(self, model_id: str | None) -> None:
        if model_id is not None:
            self._apply_model_selection(model_id)

    def _apply_model_selection(self, model_id: str) -> None:
        try:
            self.settings_store.select_model(model_id)
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
        self.last_session = None
        count = len(self.session_store.list(candidate))
        suffix = f"，发现 {count} 个历史会话，可用 /resume 恢复" if count else ""
        self._write_system(f"已切换工作区：{candidate}{suffix}")
        self._refresh_status()

    def _show_sessions(self) -> None:
        sessions = self.session_store.list(self.workspace)
        if not sessions:
            self._write_system("当前工作区还没有历史会话。")
            return
        current_id = self.last_session.session_id if self.last_session else None
        lines = [
            (
                f"{'●' if item.session_id == current_id else ' '} "
                f"{item.session_id[:8]}  {item.updated_at.astimezone():%Y-%m-%d %H:%M}  "
                f"{item.title}"
            )
            for item in sessions
        ]
        self._write_system("当前工作区的会话：\n" + "\n".join(lines))

    def _show_runs(self) -> None:
        summaries = RunCatalog(self.workspace).list()[:10]
        if not summaries:
            self._write_system("当前工作区还没有运行记录。")
            return
        self.push_screen(RunPickerScreen(summaries), self._run_picker_closed)

    def _run_picker_closed(self, run_id: str | None) -> None:
        if run_id is None:
            return
        catalog = RunCatalog(self.workspace)
        try:
            summary = catalog.resolve(run_id)
            events_ = catalog.events(run_id)
        except ValueError as exc:
            self._write_error(str(exc))
            return
        self.push_screen(RunDetailScreen(summary, events_))

    def _resume_session(self, value: str) -> None:
        if self.busy:
            self._write_error("当前任务仍在执行，完成或取消后才能切换会话。")
            return
        if not value:
            sessions = self.session_store.list(self.workspace)
            if not sessions:
                self._write_system("当前工作区还没有历史会话。")
                return
            current_id = self.last_session.session_id if self.last_session else None
            self.push_screen(
                SessionPickerScreen(sessions, current_id),
                self._session_picker_closed,
            )
            return
        try:
            record = self.session_store.resolve(value, self.workspace)
        except (OSError, ValueError) as exc:
            self._write_error(str(exc))
            return
        self._apply_resumed_session(record.state, record.title)

    def _session_picker_closed(self, session_id: str | None) -> None:
        if session_id is not None:
            self._resume_session(session_id)

    def _apply_resumed_session(self, state: SessionState, title: str) -> None:
        self.last_session = state
        self.action_clear_transcript()
        for message in state.history:
            if message.role == "user" and message.content:
                self._write_user(message.content)
            elif message.role == "assistant" and message.content:
                self._write_assistant(message.content)
        self._write_system(
            f"已恢复会话 {state.session_id[:8]}：{title}。文件状态可能已变化，后续会重新检查。"
        )
        self._refresh_status()

    def _new_session(self) -> None:
        if self.busy:
            self._write_error("当前任务仍在执行，完成或取消后才能新建会话。")
            return
        self.last_session = None
        self.action_clear_transcript()
        self._write_system("已开始新会话。历史会话仍保存在本机，可随时用 /resume 恢复。")
        self._refresh_status()

    def _open_config(self) -> None:
        self.push_screen(
            ModelConfigScreen(
                self.settings_store,
                self.model_fetcher,
            ),
            self._config_closed,
        )

    def _config_closed(self, saved: bool | None) -> None:
        if saved:
            self.model_id = self.settings_store.load().default_model
            self._write_system(f"模型配置已保存到 {self.settings_store.path}")
            self._refresh_status()

    def _open_pricing(self) -> None:
        if self.model_id is None:
            self._write_error("尚未选择真实模型。请先运行 /config。")
            return
        self.push_screen(
            PricingConfigScreen(self.model_id, self.settings_store),
            self._pricing_closed,
        )

    def _pricing_closed(self, saved: bool | None) -> None:
        if saved and self.model_id is not None:
            pricing = self.settings_store.load().models[self.model_id].pricing
            self._write_system(
                f"已保存 {self.model_id} 的费用：输入 {pricing.input_per_million:g} / "
                f"输出 {pricing.output_per_million:g} {pricing.currency} / 1M tokens"
            )

    @work(exclusive=True, group="agent-run")
    async def run_task(self, task: str) -> None:
        self.busy = True
        self.cancel_requested = False
        self.received_stream_delta = False
        self.streaming_message = None
        self.tool_activities = {}
        self.fallback_active_tool = None
        self.event_stream_active = False
        self.live_run_id = None
        self.live_input_tokens = 0
        self.live_output_tokens = 0
        self.live_cost = 0.0
        self.live_currency = ""
        self.run_started_counter = time.perf_counter()
        self._refresh_status()
        try:
            state = await self.task_runner(
                task,
                self.workspace,
                self.model_id,
                self.models_path,
                self.max_steps,
                self.settings_store,
                self,
                self.last_session,
                self._checkpoint_session,
            )
            self.last_session = state
            if state.final_answer and not self.received_stream_delta:
                self._write_assistant(state.final_answer)
            if state.current_run_id:
                tokens = state.last_run_input_tokens + state.last_run_output_tokens
                self._write_system(
                    f"run: {state.current_run_id[:8]} · {state.step} steps · "
                    f"{tokens} tokens · cost {state.last_run_cost:.6f} "
                    f"{state.last_run_currency} · "
                    f"{state.last_run_latency:.2f}s"
                )
            if state.status != "completed":
                self._write_error(state.last_error or f"任务结束：{state.status}")
        except Exception as exc:
            self._write_error(str(exc))
        finally:
            cancelled = self.cancel_requested
            await self._stop_thinking()
            self.streaming_message = None
            self.busy = False
            self.cancel_requested = False
            self.run_started_counter = None
            if cancelled:
                self._write_system("当前任务已取消，已完成运行记录收尾。")
            self._refresh_status()
            self.query_one("#prompt", Input).focus()

    def _checkpoint_session(self, state: SessionState) -> None:
        self.last_session = state
        self.session_store.save(state, model_id=self.model_id or state.current_model or "unknown")

    def on_run_event(self, event: RunEvent) -> None:
        self.event_stream_active = True
        self.post_message(RunEventMessage(event))

    @on(RunEventMessage)
    async def project_run_event(self, message: RunEventMessage) -> None:
        event = message.event
        transcript = self.query_one("#transcript", VerticalScroll)
        if isinstance(event, RunStatusEvent):
            self.live_run_id = event.run_id
            if event.status == "started":
                self.run_started_counter = time.perf_counter()
            self._refresh_status()
        elif isinstance(event, ToolRequestEvent):
            await self._stop_thinking()
            requested_activity = ToolActivity(
                event.tool_name,
                _tool_target(event.arguments),
                tool_call_id=event.tool_call_id,
                arguments=event.arguments,
            )
            self.tool_activities[event.tool_call_id] = requested_activity
            await transcript.mount(requested_activity)
            transcript.scroll_end(animate=False)
        elif isinstance(event, ToolResultEvent):
            result_activity = self.tool_activities.get(event.tool_call_id)
            if result_activity is not None:
                result_activity.finish(
                    ToolResult(status=event.status, content=event.content, data=event.data),
                    event.latency_seconds,
                )
            transcript.scroll_end(animate=False)
        elif isinstance(event, AssistantTextEvent) and event.text:
            await self._stop_thinking()
            if self.streaming_message is None:
                self.streaming_message = ChatMessage("CapyCode", event.text, assistant=True)
                await transcript.mount(self.streaming_message)
            elif self.streaming_message.content != event.text:
                await self.streaming_message.set_content(event.text)
            self.received_stream_delta = True
            transcript.scroll_end(animate=False)
        elif isinstance(event, StepTraceEvent):
            self.live_input_tokens += event.input_tokens
            self.live_output_tokens += event.output_tokens
            self.live_cost += event.cost
            self.live_currency = event.currency
            self._refresh_status()
        elif isinstance(event, ContextSnapshotEvent):
            notice = NoticeMessage(
                f"·  上下文已压缩：{event.estimated_tokens_before} → "
                f"{event.estimated_tokens_after} tokens",
                markup=False,
            )
            await transcript.mount(notice)
            transcript.scroll_end(animate=False)

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
        if self.event_stream_active:
            return
        self.fallback_active_tool = ToolActivity(name, _tool_target(arguments))
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(self.fallback_active_tool)
        transcript.scroll_end(animate=False)

    async def on_tool_result(self, name: str, result: ToolResult) -> None:
        if self.event_stream_active:
            return
        if self.fallback_active_tool is not None and self.fallback_active_tool.tool_name == name:
            self.fallback_active_tool.finish(result)
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    async def _stop_thinking(self) -> None:
        if self.thinking_indicator is not None:
            indicator = self.thinking_indicator
            self.thinking_indicator = None
            if indicator.is_mounted:
                await indicator.remove()

    def action_cancel_or_quit(self) -> None:
        run_workers = [worker for worker in self.workers if worker.group == "agent-run"]
        if self.busy:
            if run_workers and not self.cancel_requested:
                self.cancel_requested = True
                for worker in run_workers:
                    worker.cancel()
                self._write_system("正在取消当前任务并保存运行记录…")
                self._refresh_status()
            return
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

    def _status_text(self, width: int | None = None) -> str:
        state = "取消中" if self.cancel_requested else ("运行中" if self.busy else "就绪")
        session = self.last_session.session_id[:8] if self.last_session else "new"
        saved_run = self.last_session.current_run_id if self.last_session else None
        run_id = (self.live_run_id or saved_run or "-")[:8]
        terminal_width = width or self.size.width
        model = self.model_id or "未配置"
        tokens = self.live_input_tokens + self.live_output_tokens
        elapsed = (
            time.perf_counter() - self.run_started_counter
            if self.busy and self.run_started_counter is not None
            else (self.last_session.last_run_latency if self.last_session else 0)
        )
        core = f"{state} · {model} · run {run_id}"
        if terminal_width < 72:
            return core
        metrics = (
            f" · {tokens} tok · {self.live_cost:.4f} "
            f"{self.live_currency or '-'} · {elapsed:.1f}s"
        )
        if terminal_width < 105:
            return core + metrics
        workspace_name = self.workspace.name or str(self.workspace)
        return f"{core} · session {session}{metrics} · {workspace_name}"

    def _refresh_status(self) -> None:
        try:
            self.query_one("#status-line", Static).update(self._status_text())
            self._refresh_session_bar()
        except NoMatches:
            # A final interval tick may race with Textual removing the screen tree.
            return

    def on_resize(self) -> None:
        self._refresh_status()

    def _refresh_session_bar(self) -> None:
        workspace_name = self.workspace.name or str(self.workspace)
        self.query_one("#session-bar", Static).update(
            f"CapyCode  ·  {self.model_id or '未配置模型'}  ·  {workspace_name}"
        )


def launch_tui(
    *,
    workspace: Path | None = None,
    model_id: str | None = None,
    models_path: Path = DEFAULT_MODELS_PATH,
    initial_resume: str | None = None,
) -> None:
    CapyCodeApp(
        workspace=workspace,
        model_id=model_id,
        models_path=models_path,
        initial_resume=initial_resume,
    ).run()
