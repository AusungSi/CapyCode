from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

from capycode import __version__
from capycode.config.loader import DEFAULT_MODELS_PATH, load_configuration
from capycode.llm import LLMError

from .runtime import execute_task
from .tui import launch_tui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capycode",
        description="Profiled capability routing coding agent",
    )
    parser.add_argument("--version", action="version", version=__version__)
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
    run.add_argument("--model", default="small", help="model alias from models.yaml")
    run.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    run.add_argument("--max-steps", type=int, default=10)

    tui = subparsers.add_parser("tui", help="open the interactive terminal interface")
    tui.add_argument("--workspace", type=Path, default=Path.cwd())
    tui.add_argument("--model", default=None, help="initial model alias")
    tui.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
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
    print('  capycode run "<task>" --workspace <path> --model <alias>')
    print("  capycode doctor --models <models.yaml> --profiles <profiles.yaml>")
    print("  capycode --help")
    print("\nRun capycode without arguments to open the interactive terminal interface.")
    return 0


def run_doctor(models_path: Path, profiles_path: Path, *, strict_secrets: bool) -> int:
    bundle = load_configuration(models_path, profiles_path)
    print(f"models: {len(bundle.models.models)} ({models_path})")
    print(f"profiles: {len(bundle.profiles.profiles)} ({profiles_path})")

    missing: list[str] = []
    for alias, model in sorted(bundle.models.models.items()):
        base_url_ready = bool(os.getenv(model.base_url_env))
        api_key_ready = bool(os.getenv(model.api_key_env))
        print(
            f"- {alias}: provider={model.provider} model={model.model} "
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
    model_alias: str,
    models_path: Path,
    *,
    max_steps: int,
) -> int:
    state = await execute_task(task, workspace, model_alias, models_path, max_steps)

    if state.final_answer:
        print(state.final_answer)
    print(f"session_id: {state.session_id}")
    print(f"status: {state.status}")
    print(f"steps: {state.step}")
    print(f"relevant_files: {', '.join(state.relevant_files) or '-'}")
    if state.last_error:
        print(f"error: {state.last_error}")
    return 0 if state.status == "completed" else 1


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            launch_tui()
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
                model_alias=args.model,
                models_path=args.models,
            )
            code = 0
        else:
            parser.error(f"unsupported command: {args.command}")
            return
    except (LLMError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return
    raise SystemExit(code)
