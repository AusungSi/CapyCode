from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

from capycode import __version__
from capycode.config.loader import DEFAULT_MODELS_PATH, load_configuration
from capycode.config.user_settings import UserSettingsStore, resolve_model
from capycode.core import SessionState
from capycode.llm import LLMError
from capycode.profiling import (
    EvaluationStrategy,
    GateTaskCatalog,
    P0GateRunner,
    P2ProfilingRunner,
    RoutingEvaluationRunner,
    TaskExecutor,
)
from capycode.trace import RunCatalog, RunSummary

from .runtime import execute_task
from .tui import launch_tui

DEFAULT_SWEBENCH_CONTAINER_IMAGE = "capycode/swebench-python:3.11"


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
    run.add_argument("--endpoint", default=None, help="named endpoint ID")
    run.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    run.add_argument("--profiles", type=Path, default=Path("config/profiles.yaml"))
    run.add_argument("--max-steps", type=int, default=10)

    tui = subparsers.add_parser("tui", help="open the interactive terminal interface")
    tui.add_argument("--workspace", type=Path, default=Path.cwd())
    tui.add_argument("--model", default=None, help="initial real model ID")
    tui.add_argument("--endpoint", default=None, help="initial named endpoint ID")
    tui.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    tui.add_argument("--profiles", type=Path, default=Path("config/profiles.yaml"))
    tui_resume_group = tui.add_mutually_exclusive_group()
    tui_resume_group.add_argument("--continue", dest="continue_session", action="store_true")
    tui_resume_group.add_argument("--resume", metavar="SESSION_ID")

    runs = subparsers.add_parser("runs", help="list recent runs for a workspace")
    runs.add_argument("--workspace", type=Path, default=Path.cwd())
    runs.add_argument("--limit", type=int, default=20)

    inspect_run = subparsers.add_parser("inspect-run", help="inspect one local run summary")
    inspect_run.add_argument("run_id", help="full or unique run ID prefix, or latest")
    inspect_run.add_argument("--workspace", type=Path, default=Path.cwd())

    subparsers.add_parser("endpoints", help="list configured model endpoints")
    endpoint = subparsers.add_parser("endpoint", help="select a configured model endpoint")
    endpoint_commands = endpoint.add_subparsers(dest="endpoint_command", required=True)
    endpoint_select = endpoint_commands.add_parser("select", help="select endpoint by ID")
    endpoint_select.add_argument("endpoint_id")

    benchmark = subparsers.add_parser("benchmark", help="run reproducible local benchmarks")
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    p0_benchmark = benchmark_commands.add_parser("p0", help="run the P0 baseline gate")
    p0_benchmark.add_argument("--model", default=None, help="configured real model ID")
    p0_benchmark.add_argument("--endpoint", default=None, help="named endpoint ID")
    p0_benchmark.add_argument("--repeats", type=int, default=2)
    p0_benchmark.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="run one task ID; may be provided multiple times",
    )
    swebench = benchmark_commands.add_parser(
        "swebench", help="run Agent on prepared SWE-bench instances and write predictions"
    )
    swebench.add_argument("--instances", type=Path, required=True, help="JSONL task manifest")
    swebench.add_argument("--model", default=None, help="configured real model ID")
    swebench.add_argument("--endpoint", default=None, help="named endpoint ID")
    swebench.add_argument("--max-steps", type=int, default=200)
    swebench.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="number of SWE-bench instances to run concurrently",
    )
    swebench.add_argument("--output", type=Path, default=None)
    swebench.add_argument("--container-image", default=DEFAULT_SWEBENCH_CONTAINER_IMAGE)
    swebench.add_argument("--profiles", type=Path, default=Path("config/profiles.yaml"))
    swebench.add_argument("--profiled-artifact", type=Path, default=None)
    p0_benchmark.add_argument("--output", type=Path, default=None)
    p0_benchmark.add_argument(
        "--validate-only",
        action="store_true",
        help="verify that selected fixtures start with failing tests without calling a model",
    )

    profile = subparsers.add_parser(
        "profile", help="measure models and generate P2 routing profiles"
    )
    profile_commands = profile.add_subparsers(dest="profile_command", required=True)
    p0_profile = profile_commands.add_parser("p0", help="profile real models on frozen P0 tasks")
    p0_profile.add_argument("--model", action="append", dest="models", required=True)
    p0_profile.add_argument("--endpoint", default=None, help="named endpoint ID")
    p0_profile.add_argument("--repeats", type=int, default=1)
    p0_profile.add_argument("--task", action="append", dest="tasks")
    p0_profile.add_argument("--output", type=Path, default=None)
    p0_profile.add_argument("--minimum-samples", type=int, default=2)
    p0_profile.add_argument("--reliability-threshold", type=float, default=0.6)
    p0_profile.add_argument("--quality-tolerance", type=float, default=0.05)
    p0_profile.add_argument("--profiles", type=Path, default=Path("config/profiles.yaml"))
    p0_profile.add_argument(
        "--install",
        action="store_true",
        help="install the generated profiles.json into the current workspace",
    )

    evaluate = subparsers.add_parser("evaluate", help="compare fixed and profiled routing policies")
    evaluate_commands = evaluate.add_subparsers(dest="evaluation_command", required=True)
    p0_evaluate = evaluate_commands.add_parser("p0", help="compare strategies on frozen P0 tasks")
    p0_evaluate.add_argument("--fixed-model", action="append", dest="fixed_models", default=[])
    p0_evaluate.add_argument("--endpoint", default=None, help="named endpoint ID")
    p0_evaluate.add_argument("--profiled-artifact", type=Path)
    p0_evaluate.add_argument("--repeats", type=int, default=1)
    p0_evaluate.add_argument("--task", action="append", dest="tasks")
    p0_evaluate.add_argument("--output", type=Path, default=None)
    p0_evaluate.add_argument("--profiles", type=Path, default=Path("config/profiles.yaml"))
    p0_evaluate.add_argument(
        "--allow-overlap",
        action="store_true",
        help="allow evaluation tasks that also produced the routing profile (debugging only)",
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
    print("stage: P0 baseline gate")
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
    profiles_path: Path,
    endpoint_id: str | None = None,
) -> int:
    state = await execute_task(
        task,
        workspace,
        model_id,
        models_path,
        max_steps,
        profiles_path=profiles_path,
        endpoint_id=endpoint_id,
    )

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
            f"tokens: {summary.input_tokens} input, "
            f"{summary.cached_input_tokens} cached input, "
            f"{summary.output_tokens} output",
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


async def run_p0_benchmark(
    *,
    model_id: str | None,
    repeats: int,
    task_ids: list[str] | None,
    output: Path | None,
    validate_only: bool,
    endpoint_id: str | None = None,
) -> int:
    catalog = GateTaskCatalog()
    runner = P0GateRunner(catalog=catalog, output_root=output, progress=print)
    if validate_only:
        validated = await runner.validate_fixtures(task_ids)
        print(f"validated {len(validated)} P0 fixture(s): {', '.join(validated)}")
        for task in catalog.select(task_ids):
            print(f"- {task.manifest.task_id}: {task.fingerprint}")
        return 0

    settings_store = UserSettingsStore()
    selected_model = resolve_model(model_id, settings_store.load(), endpoint_id).model

    async def execute_benchmark_task(
        task: str,
        workspace: Path,
        selected: str | None,
        max_steps: int,
    ) -> SessionState:
        return await execute_task(
            task,
            workspace,
            selected,
            DEFAULT_MODELS_PATH,
            max_steps,
            settings_store=settings_store,
            endpoint_id=endpoint_id,
        )

    report, report_root = await runner.run(
        execute_benchmark_task,
        model_id=selected_model,
        repeats=repeats,
        task_ids=task_ids,
    )
    print(f"campaign: {report.campaign_id}")
    print(f"model: {report.model_id}")
    smoke_passed = (
        not report.gate_eligible
        and report.passed_runs == report.total_runs
        and report.task_coverage == 1
    )
    result_label = (
        "PASS"
        if report.gate_passed
        else ("SMOKE PASS (full gate not evaluated)" if smoke_passed else "FAIL")
    )
    print(f"result: {result_label}")
    print(f"pass rate: {report.passed_runs}/{report.total_runs} ({report.pass_rate:.1%})")
    print(f"pass@1: {report.pass_at_1:.1%}")
    print(f"task coverage: {report.task_coverage:.1%}")
    print(f"report: {report_root / 'report.md'}")
    return 0 if report.gate_passed or smoke_passed else 1


async def run_swebench(
    *,
    instances: Path,
    model_id: str | None,
    endpoint_id: str | None,
    max_steps: int,
    max_concurrency: int,
    output: Path | None,
    container_image: str = DEFAULT_SWEBENCH_CONTAINER_IMAGE,
    profiles_path: Path = Path("config/profiles.yaml"),
    profiled_artifact: Path | None = None,
) -> int:
    settings_store = UserSettingsStore()
    selected = resolve_model(model_id, settings_store.load(), endpoint_id)
    from capycode.capability import ProfiledRoutingArtifact
    from capycode.profiling import SWEbenchRunner

    if profiled_artifact is not None:
        if not await asyncio.to_thread(profiled_artifact.is_file):
            raise ValueError(f"profiled routing artifact does not exist: {profiled_artifact}")
        routing = await asyncio.to_thread(ProfiledRoutingArtifact.load, profiled_artifact)
        _require_configured_models(
            [selection.model_id for selection in routing.selected_by_capability.values()],
            settings_store,
            endpoint_id,
        )

    runner = SWEbenchRunner(output_root=output, progress=print)
    tasks = SWEbenchRunner.load_tasks(instances)

    async def executor(task: str, workspace: Path, model: str | None, steps: int) -> SessionState:
        return await execute_task(
            task,
            workspace,
            model,
            DEFAULT_MODELS_PATH,
            steps,
            settings_store=settings_store,
            endpoint_id=endpoint_id,
            profiles_path=profiles_path,
            profiled_routing_path=profiled_artifact,
            profile_step_limit=steps,
            container_image=container_image,
        )

    report, root = await runner.run(
        executor,
        tasks=tasks,
        model_id=selected.model,
        max_steps=max_steps,
        max_concurrency=max_concurrency,
    )
    print(f"campaign: {report.campaign_id}")
    print(f"completed: {report.completed_tasks}/{report.total_tasks}")
    print(
        f"tokens: {report.total_input_tokens} input, "
        f"{report.total_cached_input_tokens} cached input, "
        f"{report.total_output_tokens} output"
    )
    print(f"cost: {report.total_cost:.6f} {report.currency}")
    print(f"predictions: {root / 'predictions.jsonl'}")
    print(f"report: {root / 'report.md'}")
    return (
        0
        if report.infrastructure_errors == 0
        and report.model_errors == 0
        and report.failed_tasks == 0
        else 1
    )


def _require_configured_models(
    model_ids: Sequence[str], settings_store: UserSettingsStore, endpoint_id: str | None = None
) -> None:
    settings = settings_store.load()
    configured = settings.endpoints.get(endpoint_id or settings.default_endpoint or "")
    if configured is None:
        configured = settings.endpoint
    if configured is None:
        raise ValueError("model endpoint is not configured; run capycode and use /config first")
    unavailable = sorted(set(model_ids) - set(configured.available_models))
    if unavailable:
        raise ValueError(
            "models are not returned by the configured endpoint: " + ", ".join(unavailable)
        )


def show_endpoints(settings_store: UserSettingsStore | None = None) -> int:
    settings = (settings_store or UserSettingsStore()).load()
    if not settings.endpoints:
        print("No endpoints configured. Start capycode and use /config.")
        return 0
    for endpoint_id, endpoint in settings.endpoints.items():
        marker = "*" if endpoint_id == settings.default_endpoint else " "
        print(
            f"{marker} {endpoint_id}: {endpoint.base_url} ({len(endpoint.available_models)} models)"
        )
    return 0


async def run_p2_profile(
    *,
    model_ids: list[str],
    repeats: int,
    task_ids: list[str] | None,
    output: Path | None,
    minimum_samples: int,
    reliability_threshold: float,
    quality_tolerance: float,
    profiles_path: Path,
    install: bool,
    endpoint_id: str | None = None,
) -> int:
    settings_store = UserSettingsStore()
    _require_configured_models(model_ids, settings_store, endpoint_id)
    runner = P2ProfilingRunner(catalog=GateTaskCatalog(), output_root=output, progress=print)

    def executor_factory(candidate_model: str) -> TaskExecutor:
        async def execute_profiled_task(
            task: str,
            workspace: Path,
            _model: str | None,
            max_steps: int,
        ) -> SessionState:
            return await execute_task(
                task,
                workspace,
                candidate_model,
                DEFAULT_MODELS_PATH,
                max_steps,
                settings_store=settings_store,
                endpoint_id=endpoint_id,
                profiles_path=profiles_path,
                force_profile_model=candidate_model,
            )

        return execute_profiled_task

    report, artifact, root = await runner.run(
        executor_factory,
        model_ids=model_ids,
        repeats=repeats,
        task_ids=task_ids,
        minimum_samples=minimum_samples,
        reliability_threshold=reliability_threshold,
        quality_tolerance=quality_tolerance,
    )
    print(f"campaign: {report.campaign_id}")
    print(f"step measurements: {report.measurements}")
    print(f"selected capabilities: {report.selected_capabilities}")
    print(f"profiles: {root / 'profiles.json'}")
    print(f"leaderboard: {root / 'leaderboard.md'}")
    print(f"report: {root / 'report.md'}")
    print(f"manifest: {root / 'manifest.json'}")
    if install:
        installed_path = Path.cwd() / ".capy" / "profiles.json"
        artifact.write(installed_path)
        print(f"installed routing profile: {installed_path}")
    if not artifact.selected_by_capability:
        print("no capability met the reliability threshold; runtime will retain fallback")
    return 0


async def run_p2_evaluation(
    *,
    fixed_models: list[str],
    profiled_artifact: Path | None,
    repeats: int,
    task_ids: list[str] | None,
    output: Path | None,
    profiles_path: Path,
    allow_overlap: bool,
    endpoint_id: str | None = None,
) -> int:
    settings_store = UserSettingsStore()
    strategies: list[EvaluationStrategy] = []
    for position, model_id in enumerate(dict.fromkeys(fixed_models), start=1):
        strategies.append(
            EvaluationStrategy(
                strategy_id=f"fixed-{position}",
                label=f"fixed:{model_id}",
                configuration={"routing": "fixed", "model_id": model_id},
            )
        )
    if profiled_artifact is not None:
        if not await asyncio.to_thread(profiled_artifact.is_file):
            raise ValueError(f"profiled routing artifact does not exist: {profiled_artifact}")
        strategies.append(
            EvaluationStrategy(
                strategy_id="profiled",
                label="profiled",
                configuration={
                    "routing": "profiled",
                    "artifact": str(profiled_artifact),
                },
            )
        )
    if len(strategies) < 2:
        raise ValueError(
            "provide at least two --fixed-model values, or one --fixed-model plus "
            "--profiled-artifact"
        )
    _require_configured_models(fixed_models, settings_store, endpoint_id)
    artifact = None
    if profiled_artifact is not None:
        from capycode.capability import ProfiledRoutingArtifact

        artifact = ProfiledRoutingArtifact.load(profiled_artifact)
        _require_configured_models(
            [item.model_id for item in artifact.selected_by_capability.values()],
            settings_store,
            endpoint_id,
        )
        artifact.validate_holdout(
            {
                task.manifest.task_id: task.fingerprint
                for task in GateTaskCatalog().select(task_ids)
            },
            allow_overlap=allow_overlap,
        )
    # A profiled run must create its client from the selected endpoint. The
    # user's global default may belong to a different endpoint and cause 401s
    # before the capability-level model overrides are applied.
    initial_model = fixed_models[0] if fixed_models else settings_store.load().default_model

    def executor_factory(strategy: EvaluationStrategy) -> TaskExecutor:
        async def execute_evaluation_task(
            task: str,
            workspace: Path,
            _model: str | None,
            max_steps: int,
        ) -> SessionState:
            if strategy.strategy_id.startswith("fixed-"):
                model_id = strategy.label.removeprefix("fixed:")
                return await execute_task(
                    task,
                    workspace,
                    model_id,
                    DEFAULT_MODELS_PATH,
                    max_steps,
                    settings_store=settings_store,
                    profiles_path=profiles_path,
                    force_profile_model=model_id,
                    endpoint_id=endpoint_id,
                )
            return await execute_task(
                task,
                workspace,
                initial_model,
                DEFAULT_MODELS_PATH,
                max_steps,
                settings_store=settings_store,
                profiles_path=profiles_path,
                profiled_routing_path=profiled_artifact,
                endpoint_id=endpoint_id,
            )

        return execute_evaluation_task

    runner = RoutingEvaluationRunner(catalog=GateTaskCatalog(), output_root=output, progress=print)
    report, root = await runner.run(
        strategies, executor_factory, repeats=repeats, task_ids=task_ids
    )
    print(f"campaign: {report.campaign_id}")
    for item in report.strategies:
        print(
            f"{item.label}: pass_rate={item.pass_rate:.1%} pass@1={item.pass_at_1:.1%} "
            f"cost={item.total_cost:.6f} latency={item.total_latency_seconds:.2f}s"
        )
    print(f"report: {root / 'report.md'}")
    print(f"comparison: {root / 'comparison.csv'}")
    print(f"manifest: {root / 'manifest.json'}")
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
                    profiles_path=args.profiles,
                    endpoint_id=args.endpoint,
                )
            )
        elif args.command == "tui":
            launch_tui(
                workspace=args.workspace,
                model_id=args.model,
                endpoint_id=args.endpoint,
                models_path=args.models,
                profiles_path=args.profiles,
                initial_resume=args.resume or ("latest" if args.continue_session else None),
            )
            code = 0
        elif args.command == "runs":
            code = show_runs(args.workspace, limit=args.limit)
        elif args.command == "inspect-run":
            code = inspect_run(args.workspace, args.run_id)
        elif args.command == "endpoints":
            code = show_endpoints()
        elif args.command == "endpoint" and args.endpoint_command == "select":
            UserSettingsStore().select_endpoint(args.endpoint_id)
            print(f"selected endpoint: {args.endpoint_id}")
            code = 0
        elif args.command == "benchmark" and args.benchmark_command == "p0":
            code = asyncio.run(
                run_p0_benchmark(
                    model_id=args.model,
                    repeats=args.repeats,
                    task_ids=args.tasks,
                    output=args.output,
                    validate_only=args.validate_only,
                    endpoint_id=args.endpoint,
                )
            )
        elif args.command == "benchmark" and args.benchmark_command == "swebench":
            code = asyncio.run(
                run_swebench(
                    instances=args.instances,
                    model_id=args.model,
                    endpoint_id=args.endpoint,
                    max_steps=args.max_steps,
                    max_concurrency=args.max_concurrency,
                    output=args.output,
                    container_image=args.container_image,
                    profiles_path=args.profiles,
                    profiled_artifact=args.profiled_artifact,
                )
            )
        elif args.command == "profile" and args.profile_command == "p0":
            code = asyncio.run(
                run_p2_profile(
                    model_ids=args.models,
                    repeats=args.repeats,
                    task_ids=args.tasks,
                    output=args.output,
                    minimum_samples=args.minimum_samples,
                    reliability_threshold=args.reliability_threshold,
                    quality_tolerance=args.quality_tolerance,
                    profiles_path=args.profiles,
                    install=args.install,
                    endpoint_id=args.endpoint,
                )
            )
        elif args.command == "evaluate" and args.evaluation_command == "p0":
            code = asyncio.run(
                run_p2_evaluation(
                    fixed_models=args.fixed_models,
                    profiled_artifact=args.profiled_artifact,
                    repeats=args.repeats,
                    task_ids=args.tasks,
                    output=args.output,
                    profiles_path=args.profiles,
                    allow_overlap=args.allow_overlap,
                    endpoint_id=args.endpoint,
                )
            )
        else:
            parser.error(f"unsupported command: {args.command}")
            return
    except (LLMError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return
    raise SystemExit(code)
