from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from capycode.core import SessionState
from capycode.trace import RunCatalog, RunSummary, ToolRequestEvent, ToolResultEvent

TaskExecutor = Callable[[str, Path, str | None, int], Awaitable[SessionState]]
ProgressSink = Callable[[str], None]
GateTaskList = list["GateTask"]

IGNORED_PARTS = {".capy", ".git", ".pytest_cache", "__pycache__"}
TEST_PATTERN = "test_*.py"


class GateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GateTaskManifest(GateModel):
    schema_version: Literal[1] = 1
    task_id: str = Field(pattern=r"^p0-[0-9]{2}$")
    title: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    allowed_modified_files: list[str] = Field(min_length=1)
    max_steps: int = Field(default=12, ge=1, le=30)

    @model_validator(mode="after")
    def normalize_paths(self) -> GateTaskManifest:
        normalized = [Path(item).as_posix() for item in self.allowed_modified_files]
        if any(item.startswith("../") or item.startswith("/") for item in normalized):
            raise ValueError("allowed_modified_files must stay inside the task workspace")
        if len(set(normalized)) != len(normalized):
            raise ValueError("allowed_modified_files contains duplicates")
        self.allowed_modified_files = normalized
        return self


class GateCatalogLock(GateModel):
    schema_version: Literal[1] = 1
    tasks: dict[str, str] = Field(min_length=1)


class ProcessResult(GateModel):
    argv: list[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0


class GateRunResult(GateModel):
    task_id: str
    task_title: str
    capability: str
    repeat_index: int = Field(ge=1)
    model_id: str
    task_fingerprint: str
    status: Literal["passed", "failed", "infrastructure_error"]
    reasons: list[str] = Field(default_factory=list)
    initial_tests_failed: bool = False
    public_tests_passed: bool = False
    hidden_tests_passed: bool = False
    diff_scope_passed: bool = False
    trace_integrity_passed: bool = False
    agent_status: str | None = None
    termination_reason: str | None = None
    run_id: str | None = None
    modified_files: list[str] = Field(default_factory=list)
    unexpected_modified_files: list[str] = Field(default_factory=list)
    security_violations: list[str] = Field(default_factory=list)
    steps: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0, ge=0)
    currency: str = ""
    latency_seconds: float = Field(default=0, ge=0)
    tool_requests: int = Field(default=0, ge=0)
    tool_failures: int = Field(default=0, ge=0)
    workspace: str
    trace_path: str | None = None
    public_test_output: str = ""
    hidden_test_output: str = ""


class GateCampaignReport(GateModel):
    schema_version: Literal[1] = 1
    campaign_id: str
    model_id: str
    started_at: datetime
    finished_at: datetime
    repeats: int = Field(ge=1)
    task_count: int = Field(ge=1)
    total_runs: int = Field(ge=1)
    passed_runs: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    pass_at_1: float = Field(ge=0, le=1)
    task_coverage: float = Field(ge=0, le=1)
    gate_eligible: bool
    gate_passed: bool
    total_input_tokens: int = Field(ge=0)
    total_cached_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(ge=0)
    total_cost: float = Field(ge=0)
    currency: str
    total_latency_seconds: float = Field(ge=0)
    average_steps: float = Field(ge=0)
    average_tokens: float = Field(ge=0)
    average_cost: float = Field(ge=0)
    average_latency_seconds: float = Field(ge=0)
    total_tool_requests: int = Field(ge=0)
    total_tool_failures: int = Field(ge=0)
    tool_failure_rate: float = Field(ge=0, le=1)
    results: list[GateRunResult]


class GateTask:
    def __init__(self, root: Path, manifest: GateTaskManifest) -> None:
        self.root = root.resolve()
        self.manifest = manifest
        self.workspace_source = self.root / "workspace"
        self.oracle_source = self.root / "oracle"

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(
            item
            for item in self.root.rglob("*")
            if item.is_file()
            and not any(part in IGNORED_PARTS for part in item.relative_to(self.root).parts)
            and item.suffix != ".pyc"
        ):
            digest.update(path.relative_to(self.root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()


class GateTaskCatalog:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or Path(__file__).with_name("fixtures") / "p0").resolve()

    def list(self) -> GateTaskList:
        if not self.root.is_dir():
            raise ValueError(f"P0 benchmark fixture directory does not exist: {self.root}")
        tasks = [self._load(path) for path in sorted(self.root.glob("p0-*")) if path.is_dir()]
        if not tasks:
            raise ValueError(f"P0 benchmark fixture directory is empty: {self.root}")
        identifiers = [task.manifest.task_id for task in tasks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("P0 benchmark contains duplicate task IDs")
        self._verify_lock(tasks)
        return tasks

    def select(self, task_ids: Sequence[str] | None = None) -> GateTaskList:
        tasks = self.list()
        if not task_ids:
            return tasks
        requested = list(dict.fromkeys(task_ids))
        indexed = {task.manifest.task_id: task for task in tasks}
        missing = [task_id for task_id in requested if task_id not in indexed]
        if missing:
            raise ValueError(f"unknown P0 benchmark task(s): {', '.join(missing)}")
        return [indexed[task_id] for task_id in requested]

    @staticmethod
    def _load(root: Path) -> GateTask:
        manifest_path = root / "manifest.json"
        try:
            manifest = GateTaskManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise ValueError(f"invalid P0 benchmark manifest {manifest_path}: {exc}") from exc
        if manifest.task_id != root.name:
            raise ValueError(f"task directory {root.name!r} does not match {manifest.task_id!r}")
        task = GateTask(root, manifest)
        if not task.workspace_source.is_dir() or not task.oracle_source.is_dir():
            raise ValueError(f"task {manifest.task_id} requires workspace/ and oracle/")
        if not any(task.workspace_source.glob(TEST_PATTERN)):
            raise ValueError(f"task {manifest.task_id} has no public unittest file")
        if not any(task.oracle_source.glob(TEST_PATTERN)):
            raise ValueError(f"task {manifest.task_id} has no hidden unittest file")
        for allowed in manifest.allowed_modified_files:
            if not (task.workspace_source / allowed).is_file():
                raise ValueError(f"task {manifest.task_id} allows missing file: {allowed}")
        return task

    def _verify_lock(self, tasks: GateTaskList) -> None:
        lock_path = self.root / "index.json"
        try:
            lock = GateCatalogLock.model_validate_json(lock_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise ValueError(f"invalid P0 benchmark fingerprint lock {lock_path}: {exc}") from exc
        actual = {task.manifest.task_id: task.fingerprint for task in tasks}
        if lock.tasks.keys() != actual.keys():
            raise ValueError("P0 benchmark fingerprint lock does not match the task catalog")
        changed = [task_id for task_id, digest in actual.items() if lock.tasks[task_id] != digest]
        if changed:
            raise ValueError(
                "P0 benchmark fixture fingerprint changed without updating index.json: "
                + ", ".join(changed)
            )


class P0GateRunner:
    def __init__(
        self,
        *,
        catalog: GateTaskCatalog | None = None,
        output_root: Path | None = None,
        test_timeout_seconds: float = 60,
        agent_timeout_seconds: float = 900,
        progress: ProgressSink | None = None,
    ) -> None:
        self.catalog = catalog or GateTaskCatalog()
        self.output_root = (output_root or Path.cwd() / ".capy" / "benchmarks" / "p0").resolve()
        self.test_timeout_seconds = test_timeout_seconds
        self.agent_timeout_seconds = agent_timeout_seconds
        if self.test_timeout_seconds <= 0 or self.agent_timeout_seconds <= 0:
            raise ValueError("benchmark timeouts must be positive")
        self.progress = progress or (lambda _message: None)

    async def validate_fixtures(self, task_ids: Sequence[str] | None = None) -> list[str]:
        validated: list[str] = []
        with tempfile.TemporaryDirectory(prefix="capycode-p0-validate-") as temporary:
            temporary_root = Path(temporary)
            for task in self.catalog.select(task_ids):
                workspace = temporary_root / task.manifest.task_id
                shutil.copytree(task.workspace_source, workspace)
                initial = await self.run_unittests(workspace)
                if initial.timed_out or initial.exit_code < 0:
                    raise ValueError(
                        f"task {task.manifest.task_id} could not execute its public tests: "
                        + self._output(initial)
                    )
                if initial.passed:
                    raise ValueError(
                        f"task {task.manifest.task_id} is invalid: public tests already pass"
                    )
                validated.append(task.manifest.task_id)
        return validated

    async def run(
        self,
        executor: TaskExecutor,
        *,
        model_id: str,
        repeats: int = 2,
        task_ids: Sequence[str] | None = None,
    ) -> tuple[GateCampaignReport, Path]:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        tasks = self.catalog.select(task_ids)
        campaign_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        campaign_root = self.output_root / campaign_id
        campaign_root.mkdir(parents=True, exist_ok=False)
        self.progress(f"campaign {campaign_id} -> {campaign_root}")
        started_at = datetime.now(UTC)
        results: list[GateRunResult] = []
        total_runs = repeats * len(tasks)
        for repeat_index in range(1, repeats + 1):
            for task in tasks:
                position = len(results) + 1
                self.progress(
                    f"[{position}/{total_runs}] {task.manifest.task_id} "
                    f"repeat {repeat_index}: {task.manifest.title}"
                )
                run_root = campaign_root / "runs" / f"{task.manifest.task_id}-r{repeat_index}"
                result = await self._run_one(
                    task,
                    executor,
                    model_id=model_id,
                    repeat_index=repeat_index,
                    run_root=run_root,
                )
                results.append(result)
                self.progress(f"[{position}/{total_runs}] {result.status.upper()}")
                self._write_json(campaign_root / "partial-results.json", results)
        report = self.build_report(
            campaign_id=campaign_id,
            model_id=model_id,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            repeats=repeats,
            tasks=tasks,
            results=results,
        )
        self._write_report(campaign_root, report)
        return report, campaign_root

    async def _run_one(
        self,
        task: GateTask,
        executor: TaskExecutor,
        *,
        model_id: str,
        repeat_index: int,
        run_root: Path,
    ) -> GateRunResult:
        workspace = run_root / "workspace"
        workspace.parent.mkdir(parents=True, exist_ok=False)
        shutil.copytree(task.workspace_source, workspace)
        initial_snapshot = self._snapshot(workspace)
        initial = await self.run_unittests(workspace)
        if initial.timed_out or initial.exit_code < 0:
            return self._infrastructure_result(
                task,
                model_id,
                repeat_index,
                workspace,
                "unable to execute initial public tests: " + self._output(initial),
            )
        if initial.passed:
            return self._infrastructure_result(
                task, model_id, repeat_index, workspace, "public tests pass before the agent runs"
            )
        git_setup = await self._initialize_git(workspace, sorted(initial_snapshot))
        if not git_setup.passed:
            return self._infrastructure_result(
                task,
                model_id,
                repeat_index,
                workspace,
                "unable to initialize benchmark Git repository: " + self._output(git_setup),
            )

        state: SessionState | None = None
        execution_error: str | None = None
        try:
            state = await asyncio.wait_for(
                executor(
                    task.manifest.prompt,
                    workspace,
                    model_id,
                    task.manifest.max_steps,
                ),
                timeout=self.agent_timeout_seconds,
            )
        except TimeoutError:
            execution_error = f"agent execution exceeded {self.agent_timeout_seconds:.0f} seconds"
        except Exception as exc:
            detail = str(exc).strip() or "<no error message>"
            execution_error = f"agent execution raised {type(exc).__name__}: {detail}"

        final_snapshot = self._snapshot(workspace)
        modified_files = sorted(
            path
            for path in set(initial_snapshot) | set(final_snapshot)
            if initial_snapshot.get(path) != final_snapshot.get(path)
        )
        allowed = set(task.manifest.allowed_modified_files)
        unexpected = sorted(set(modified_files) - allowed)
        security_violations = self._find_symlinks(workspace)
        if security_violations:
            public = self._failed_process("evaluation skipped because the workspace has symlinks")
            hidden = self._failed_process("evaluation skipped because the workspace has symlinks")
        else:
            public, hidden = await self._evaluate_final(task, workspace, run_root)
        trace_ok, trace_reason, summary = self._validate_trace(workspace, state)

        reasons: list[str] = []
        if execution_error:
            reasons.append(execution_error)
        if state is None or state.status != "completed":
            reasons.append(
                f"agent status is {state.status if state is not None else 'unavailable'}"
            )
        if not public.passed:
            reasons.append("public tests failed after the agent run")
        if not hidden.passed:
            reasons.append("hidden tests failed after the agent run")
        if unexpected:
            reasons.append("modified files outside the allowed scope: " + ", ".join(unexpected))
        if not modified_files:
            reasons.append("agent did not modify any task file")
        if security_violations:
            reasons.append("workspace contains symbolic links: " + ", ".join(security_violations))
        if not trace_ok:
            reasons.append(trace_reason or "trace integrity check failed")

        passed = not reasons
        return GateRunResult(
            task_id=task.manifest.task_id,
            task_title=task.manifest.title,
            capability=task.manifest.capability,
            repeat_index=repeat_index,
            model_id=model_id,
            task_fingerprint=task.fingerprint,
            status="passed" if passed else "failed",
            reasons=reasons,
            initial_tests_failed=True,
            public_tests_passed=public.passed,
            hidden_tests_passed=hidden.passed,
            diff_scope_passed=not unexpected and bool(modified_files),
            trace_integrity_passed=trace_ok,
            agent_status=state.status if state is not None else None,
            termination_reason=state.termination_reason if state is not None else None,
            run_id=state.current_run_id if state is not None else None,
            modified_files=modified_files,
            unexpected_modified_files=unexpected,
            security_violations=security_violations,
            steps=summary.steps
            if summary is not None
            else (state.step if state is not None else 0),
            input_tokens=summary.input_tokens if summary is not None else 0,
            cached_input_tokens=(summary.cached_input_tokens if summary is not None else 0),
            output_tokens=summary.output_tokens if summary is not None else 0,
            cost=summary.cost if summary is not None else 0,
            currency=summary.currency if summary is not None else "",
            latency_seconds=summary.latency_seconds if summary is not None else 0,
            tool_requests=summary.tool_requests if summary is not None else 0,
            tool_failures=summary.tool_failures if summary is not None else 0,
            workspace=str(workspace),
            trace_path=state.last_trace_path if state is not None else None,
            public_test_output=self._output(public),
            hidden_test_output=self._output(hidden),
        )

    async def _evaluate_final(
        self, task: GateTask, workspace: Path, run_root: Path
    ) -> tuple[ProcessResult, ProcessResult]:
        evaluation = run_root / "evaluation"
        shutil.copytree(workspace, evaluation, ignore=self._copy_ignore)
        for source in task.workspace_source.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(task.workspace_source)
            if relative.as_posix() in task.manifest.allowed_modified_files:
                continue
            target = evaluation / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        public = await self.run_unittests(evaluation)
        for source in task.oracle_source.rglob("*"):
            if source.is_file():
                target = evaluation / source.relative_to(task.oracle_source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        hidden = await self.run_unittests(evaluation)
        return public, hidden

    def _validate_trace(
        self, workspace: Path, state: SessionState | None
    ) -> tuple[bool, str | None, RunSummary | None]:
        if state is None or not state.current_run_id:
            return False, "agent did not produce a run ID", None
        try:
            catalog = RunCatalog(workspace)
            summary = catalog.resolve(state.current_run_id)
            events = catalog.events(state.current_run_id)
        except ValueError as exc:
            return False, str(exc), None
        requests = {event.tool_call_id for event in events if isinstance(event, ToolRequestEvent)}
        results = {event.tool_call_id for event in events if isinstance(event, ToolResultEvent)}
        if requests != results:
            return False, "trace contains orphaned or unexpected tool results", summary
        if not requests:
            return False, "trace contains no tool calls", summary
        test_results = [
            event
            for event in events
            if isinstance(event, ToolResultEvent) and self._is_test_result(event)
        ]
        first_failure = next(
            (index for index, event in enumerate(test_results) if event.status == "error"), None
        )
        later_success = first_failure is not None and any(
            event.status == "success" for event in test_results[first_failure + 1 :]
        )
        if not later_success:
            return False, "trace does not show a failing test followed by a passing test", summary
        return True, None, summary

    @staticmethod
    def _is_test_result(event: ToolResultEvent) -> bool:
        if event.tool_name == "run_tests":
            return True
        argv = event.data.get("argv")
        if event.tool_name != "run_command" or not isinstance(argv, list):
            return False
        normalized = {str(part).casefold() for part in argv}
        return bool(normalized & {"pytest", "unittest"})

    async def _initialize_git(self, workspace: Path, tracked_files: list[str]) -> ProcessResult:
        initialized = await self._run_process(["git", "init", "-q"], workspace)
        if not initialized.passed:
            return initialized
        staged = await self._run_process(["git", "add", "--force", "--", *tracked_files], workspace)
        if not staged.passed:
            return staged
        staged_files = await self._run_process(
            ["git", "diff", "--cached", "--name-only", "--"], workspace
        )
        if not staged_files.passed:
            return staged_files
        actual = sorted(line.strip() for line in staged_files.stdout.splitlines() if line.strip())
        # Some Windows Git configurations ignore explicit pathspecs even with
        # --force (notably when a global excludes file is inherited by a shim).
        # Retry from the isolated workspace root before declaring infrastructure
        # failure; the subsequent exact-file comparison still prevents leakage.
        if actual != tracked_files:
            fallback = await self._run_process(["git", "add", "--force", "--", "."], workspace)
            if not fallback.passed:
                return fallback
            staged_files = await self._run_process(
                ["git", "diff", "--cached", "--name-only", "--"], workspace
            )
            if not staged_files.passed:
                return staged_files
            actual = sorted(
                line.strip() for line in staged_files.stdout.splitlines() if line.strip()
            )
        if actual != tracked_files:
            return ProcessResult(
                argv=staged_files.argv,
                exit_code=-1,
                stderr=(
                    f"benchmark staging mismatch; expected {tracked_files!r}, received {actual!r}"
                ),
            )
        return await self._run_process(
            [
                "git",
                "-c",
                "user.name=CapyCode Gate",
                "-c",
                "user.email=gate@example.invalid",
                "commit",
                "-qm",
                "baseline fixture",
            ],
            workspace,
        )

    async def run_unittests(self, workspace: Path) -> ProcessResult:
        return await self._run_process(
            [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", TEST_PATTERN, "-q"],
            workspace,
        )

    async def _run_process(self, argv: list[str], cwd: Path) -> ProcessResult:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {
                "APPDATA",
                "HOME",
                "HOMEDRIVE",
                "HOMEPATH",
                "LANG",
                "LC_ALL",
                "LOCALAPPDATA",
                "PATH",
                "PATHEXT",
                "PROGRAMDATA",
                "SYSTEMDRIVE",
                "SYSTEMROOT",
                "TEMP",
                "TMP",
                "USERPROFILE",
                "WINDIR",
            }
        }
        if argv and argv[0] == "git":
            environment["GIT_CONFIG_GLOBAL"] = os.devnull
            environment["GIT_CONFIG_NOSYSTEM"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return ProcessResult(argv=argv, exit_code=-1, stderr=str(exc))
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.test_timeout_seconds
            )
        except TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return ProcessResult(
                argv=argv,
                exit_code=process.returncode or -1,
                stdout=self._decode(stdout),
                stderr=self._decode(stderr),
                timed_out=True,
            )
        return ProcessResult(
            argv=argv,
            exit_code=process.returncode or 0,
            stdout=self._decode(stdout),
            stderr=self._decode(stderr),
        )

    @staticmethod
    def _snapshot(root: Path) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            # Check ignore markers relative to the fixture workspace. The
            # absolute path commonly contains `.capy` because campaigns are
            # stored under the user's workspace, and must not hide all files.
            if any(part in IGNORED_PARTS for part in Path(relative).parts):
                continue
            if path.is_symlink():
                snapshot[relative] = "symlink:" + str(path.readlink())
                continue
            if not path.is_file():
                continue
            if path.suffix == ".pyc":
                continue
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return snapshot

    @staticmethod
    def _find_symlinks(root: Path) -> list[str]:
        return sorted(
            path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()
        )

    @staticmethod
    def _failed_process(message: str) -> ProcessResult:
        return ProcessResult(argv=[], exit_code=-1, stderr=message)

    @staticmethod
    def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in IGNORED_PARTS or name.endswith(".pyc")}

    @staticmethod
    def _decode(value: bytes) -> str:
        return value.decode("utf-8", errors="replace")[-8000:]

    @staticmethod
    def _output(result: ProcessResult) -> str:
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        return output[-8000:]

    def _infrastructure_result(
        self,
        task: GateTask,
        model_id: str,
        repeat_index: int,
        workspace: Path,
        reason: str,
    ) -> GateRunResult:
        return GateRunResult(
            task_id=task.manifest.task_id,
            task_title=task.manifest.title,
            capability=task.manifest.capability,
            repeat_index=repeat_index,
            model_id=model_id,
            task_fingerprint=task.fingerprint,
            status="infrastructure_error",
            reasons=[reason],
            workspace=str(workspace),
        )

    @staticmethod
    def build_report(
        *,
        campaign_id: str,
        model_id: str,
        started_at: datetime,
        finished_at: datetime,
        repeats: int,
        tasks: list[GateTask],
        results: list[GateRunResult],
    ) -> GateCampaignReport:
        passed = [result for result in results if result.status == "passed"]
        first_runs = [result for result in results if result.repeat_index == 1]
        covered = {result.task_id for result in passed}
        currencies = {result.currency for result in results if result.currency}
        currency = currencies.pop() if len(currencies) == 1 else ("MIXED" if currencies else "")
        pass_rate = len(passed) / len(results)
        pass_at_1 = sum(result.status == "passed" for result in first_runs) / len(tasks)
        coverage = len(covered) / len(tasks)
        gate_eligible = len(tasks) >= 5 and repeats >= 2
        gate_passed = (
            gate_eligible
            and coverage == 1
            and pass_rate >= 0.8
            and not any(result.status == "infrastructure_error" for result in results)
        )
        total_tokens = sum(result.input_tokens + result.output_tokens for result in results)
        total_tool_requests = sum(result.tool_requests for result in results)
        total_tool_failures = sum(result.tool_failures for result in results)
        return GateCampaignReport(
            campaign_id=campaign_id,
            model_id=model_id,
            started_at=started_at,
            finished_at=finished_at,
            repeats=repeats,
            task_count=len(tasks),
            total_runs=len(results),
            passed_runs=len(passed),
            pass_rate=pass_rate,
            pass_at_1=pass_at_1,
            task_coverage=coverage,
            gate_eligible=gate_eligible,
            gate_passed=gate_passed,
            total_input_tokens=sum(result.input_tokens for result in results),
            total_cached_input_tokens=sum(result.cached_input_tokens for result in results),
            total_output_tokens=sum(result.output_tokens for result in results),
            total_cost=sum(result.cost for result in results),
            currency=currency,
            total_latency_seconds=sum(result.latency_seconds for result in results),
            average_steps=sum(result.steps for result in results) / len(results),
            average_tokens=total_tokens / len(results),
            average_cost=sum(result.cost for result in results) / len(results),
            average_latency_seconds=(
                sum(result.latency_seconds for result in results) / len(results)
            ),
            total_tool_requests=total_tool_requests,
            total_tool_failures=total_tool_failures,
            tool_failure_rate=(
                total_tool_failures / total_tool_requests if total_tool_requests else 0
            ),
            results=results,
        )

    def _write_report(self, root: Path, report: GateCampaignReport) -> None:
        self._atomic_write(
            root / "report.json",
            report.model_dump_json(indent=2) + "\n",
        )
        gate_label = (
            "PASS" if report.gate_passed else ("FAIL" if report.gate_eligible else "NOT EVALUATED")
        )
        lines = [
            "# CapyCode P0 Baseline Gate",
            "",
            f"- Campaign: `{report.campaign_id}`",
            f"- Model: `{report.model_id}`",
            f"- Gate: **{gate_label}**",
            f"- Pass rate: {report.passed_runs}/{report.total_runs} ({report.pass_rate:.1%})",
            f"- Pass@1: {report.pass_at_1:.1%}",
            f"- Task coverage: {report.task_coverage:.1%}",
            f"- Tokens: {report.total_input_tokens} input / "
            f"{report.total_cached_input_tokens} cached input / "
            f"{report.total_output_tokens} output",
            f"- Cost: {report.total_cost:.6f} {report.currency or 'UNSPECIFIED'}",
            f"- Model/tool latency: {report.total_latency_seconds:.3f}s",
            f"- Average steps/tokens: {report.average_steps:.2f} / {report.average_tokens:.1f}",
            f"- Average cost/latency: {report.average_cost:.6f} / "
            f"{report.average_latency_seconds:.3f}s",
            f"- Tool failure rate: {report.total_tool_failures}/{report.total_tool_requests} "
            f"({report.tool_failure_rate:.1%})",
            "",
            "| Task | Repeat | Status | Steps | Tokens | Cost | Reasons |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
        for result in report.results:
            reasons = "; ".join(result.reasons).replace("|", "\\|") or "-"
            lines.append(
                f"| {result.task_id} | {result.repeat_index} | {result.status} | "
                f"{result.steps} | {result.input_tokens + result.output_tokens} | "
                f"{result.cost:.6f} | {reasons} |"
            )
        self._atomic_write(root / "report.md", "\n".join(lines) + "\n")

    def _write_json(self, path: Path, values: list[GateRunResult]) -> None:
        payload = json.dumps(
            [value.model_dump(mode="json") for value in values],
            ensure_ascii=False,
            indent=2,
        )
        self._atomic_write(path, payload + "\n")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=path.name + "-", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
