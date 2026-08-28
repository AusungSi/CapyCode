from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from capycode import __version__
from capycode.config.loader import load_configuration


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
    print("stage: P0-0 project scaffold")
    print("\nAvailable now:")
    print("  capycode doctor --models <models.yaml> --profiles <profiles.yaml>")
    print("  capycode --help")
    print("\nInteractive Agent runtime will be enabled in the P0 Runtime/TUI milestones.")
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            code = show_welcome()
        elif args.command == "doctor":
            code = run_doctor(args.models, args.profiles, strict_secrets=args.strict_secrets)
        else:
            parser.error(f"unsupported command: {args.command}")
            return
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
        return
    raise SystemExit(code)
