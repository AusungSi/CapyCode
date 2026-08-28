from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

from capycode import __version__
from capycode.config.loader import DEFAULT_MODELS_PATH, load_configuration
from capycode.llm import LLMError
from capycode.trace import RunCatalog, RunSummary

from .runtime import execute_task
from .tui import launch_tui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capycode",
        description="Profiled capability routing coding agent",
    )
    parser.add_argument("--version", action="version", version=__version__)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="continue the most recent conversation in the current workspace",
    )
    resume_group.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="resume a conversation in the current workspace",
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="validate local project configuration")
    doctor.add_argument("--models", type=Path, default=Path("config/models.yaml"))
    doctor.add_argument("--profiles", type=Path, default=Path("config/profiles.yaml"))
    doctor.add_argument(
        "--strict-secrets",
        action="store_true",
        help="fail when a referenced API or base URL environment variable is missing",
    )

    run = subparsers.add_parser("run", help="run the P0 runtime against a workspace")
    run.add_argument("task", help="natural-language repository task")
    run.add_argument("--workspace", type=Path, default=Path.cwd())
    run.add_argument("--model", default=None, help="real model ID returned by the endpoint")
    run.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    run.add_argument("--max-steps", type=int, default=10)

    tui = subparsers.add_parser("tui", help="open the interactive terminal interface")
    tui.add_argument("--workspace", type=Path, default=Path.cwd())
    tui.add_argument("--model", default=None, help="initial real model ID")
    tui.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    tui_resume_group = tui.add_mutually_exclusive_group()
    tui_resume_group.add_argument("--continue", dest="continue_session", action="store_true")
    tui_resume_group.add_argument("--resume", metavar="SESSION_ID")

    runs = subparsers.add_parser("runs", help="list recent runs for a workspace")
    runs.add_argument("--workspace", type=Path, default=Path.cwd())
    runs.add_argument("--limit", type=int, default=20)

    inspect_run = subparsers.add_parser("inspect-run", help="inspect one local run summary")
    inspect_run.add_argument("run_id", help="full or unique run ID prefix, or latest")
    inspect_run.add_argument("--workspace", type=Path, default=Path.cwd())
    return parser


def show_welcome(workspace: Path | None = None) -> int:
    current_workspace = (workspace or Path.cwd()).resolve()
    print(
        r"""
   ______                  ______          __
  / ____/___ _____  __  __/ ____/___  ____/ /__
 / /   / __ `/ __ \/ / / / /   / __ \/ __  / _ \
/ /___/ /_/ / /_/ / /_/ / /___/ /_/ / /_/ /  __/
\____/\__,_/ .___/\__, /\____/\____/\__,_/\___/
          /_/    /____/
""".strip("\n")
    )
    print(f"\nCapyCode {__version__} - Profiled Capability Routing Coding Agent")
    print(f"workspace: {current_workspace}")
    print("stage: P0-1 interactive runtime")
    print("\nAvailable now:")
    print('  capycode run "<task>" --workspace <path> --model <model-id>')
    print("  capycode doctor --models <models.yaml> --profiles <profiles.yaml>")
    print("  capycode --help")
    print("\nRun capycode without arguments to open the interactive terminal interface.")
    return 0


def run_doctor(models_path: Path, profiles_path: Path, *, strict_secrets: bool) -> int:
    bundle = load_configuration(models_path, profiles_path)
    print(f"models: {len(bundle.models.models)} ({models_path})")
    print(f"profiles: {len(bundle.profiles.profiles)} ({profiles_path})")

    missing: list[str] = []
    for model_id, model in sorted(bundle.models.models.items()):
        base_url_ready = bool(os.getenv(model.base_url_env))
        api_key_ready = bool(os.getenv(model.api_key_env))
        print(
            f"- {model_id}: provider={model.provider} model={model.model} "
            f"base_url={'ready' if base_url_ready else 'missing'} "
            f"api_key={'ready' if api_key_ready else 'missing'}"
        )
        if not base_url_ready:
            missing.append(model.base_url_env)
        if not api_key_ready:
            missing.append(model.api_key_env)

    if missing:
        names = ", ".join(sorted(set(missing)))
        print(f"warning: missing local environment variables: {names}")
        if strict_secrets:
            return 1
    print("configuration: valid")
    return 0


async def run_agent(
    task: str,
    workspace: Path,
    model_id: str | None,
    models_path: Path,
    *,
    max_steps: int,
) -> int:
    state = await execute_task(task, workspace, model_id, models_path, max_steps)

    if state.final_answer:
        print(state.final_answer)
    print(f"session_id: {state.session_id}")
    print(f"status: {state.status}")
    print(f"steps: {state.step}")
    if state.current_run_id:
        print(f"run_id: {state.current_run_id}")
    if state.last_trace_path:
        print(f"trace: {state.last_trace_path}")
    print(f"relevant_files: {', '.join(state.relevant_files) or '-'}")
    if state.last_error:
        print(f"error: {state.last_error}")
    return 0 if state.status == "completed" else 1


def format_run_summary(summary: RunSummary) -> str:
    return "\n".join(
        [
            f"run_id: {summary.run_id}",
            f"session_id: {summary.session_id}",
            f"status: {summary.status}",
            f"termination: {summary.termination_reason}",
            f"model: {summary.model_id} ({summary.provider})",
            f"steps: {summary.steps}",
            f"tokens: {summary.input_tokens} input, {summary.output_tokens} output",
            f"cost: {summary.cost:.6f} {summary.currency}",
            f"latency: {summary.latency_seconds:.3f}s",
            (
                "tools: "
                f"{summary.tool_requests} requested, {summary.tool_successes} succeeded, "
                f"{summary.tool_failures} failed"
            ),
            f"tests_passed: {summary.tests_passed}",
            f"modified_files: {', '.join(summary.modified_files) or '-'}",
            f"trace: {summary.trace_path}",
        ]
    )


def show_runs(workspace: Path, *, limit: int) -> int:
    if limit <= 0:
        raise ValueError("limit must be positive")
    summaries = RunCatalog(workspace).list()[:limit]
    if not summaries:
        print("No runs found for this workspace.")
        return 0
    for summary in summaries:
        finished = summary.finished_at.astimezone().strftime("%m-%d %H:%M")
        print(
            f"{summary.run_id[:8]}  {summary.status:<9}  {finished}  "
            f"{summary.model_id}  {summary.steps} steps  {summary.latency_seconds:.2f}s"
        )
    return 0


def inspect_run(workspace: Path, run_id: str) -> int:
    print(format_run_summary(RunCatalog(workspace).resolve(run_id)))
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            initial_resume = args.resume or ("latest" if args.continue_session else None)
            if initial_resume is None:
                launch_tui()
            else:
                launch_tui(initial_resume=initial_resume)
            code = 0
        elif args.command == "doctor":
            code = run_doctor(args.models, args.profiles, strict_secrets=args.strict_secrets)
        elif args.command == "run":
            code = asyncio.run(
                run_agent(
                    args.task,
                    args.workspace,
                    args.model,
                    args.models,
                    max_steps=args.max_steps,
                )
            )
        elif args.command == "tui":
            launch_tui(
                workspace=args.workspace,
                model_id=args.model,
                models_path=args.models,
                initial_resume=args.resume or ("latest" if args.continue_session else None),
            )
            code = 0
        elif args.command == "runs":
            code = show_runs(args.workspace, limit=args.limit)
        elif args.command == "inspect-run":
            code = inspect_run(args.workspace, args.run_id)
        else:
            parser.error(f"unsupported command: {args.command}")
            return
    except (LLMError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return
    raise SystemExit(code)
