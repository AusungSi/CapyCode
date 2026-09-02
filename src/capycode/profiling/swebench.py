from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from capycode.core import SessionState
from capycode.llm import LLMError
from capycode.trace import RunCatalog

TaskExecutor = Callable[[str, Path, str | None, int], Awaitable[SessionState]]


class SWEbenchTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    instance_id: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    workspace: Path | None = None
    repo: str | None = None
    base_commit: str | None = None


class SWEbenchResult(BaseModel):
    instance_id: str
    status: Literal["completed", "failed", "model_error", "infrastructure_error"]
    model_patch: str = ""
    model: str
    workspace: str
    run_id: str | None = None
    steps: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0
    currency: str = ""
    latency_seconds: float = 0
    error: str | None = None


class SWEbenchReport(BaseModel):
    schema_version: Literal[1] = 1
    campaign_id: str
    model: str
    started_at: datetime
    finished_at: datetime
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    model_errors: int = 0
    infrastructure_errors: int
    total_input_tokens: int
    total_cached_input_tokens: int = 0
    total_output_tokens: int
    total_cost: float
    currency: str
    total_latency_seconds: float
    predictions_path: str
    results: list[SWEbenchResult]


class SWEbenchRunner:
    """Run agent patches on prepared SWE-bench workspaces.

    The runner deliberately does not execute hidden SWE-bench tests. The generated
    predictions file is consumed by the official Docker harness, which supplies
    the repository-specific test environment and grading policy.
    """

    def __init__(
        self,
        *,
        output_root: Path | None = None,
        progress: Callable[[str], None] = print,
    ) -> None:
        self.output_root = (
            output_root or Path.cwd() / ".capy" / "benchmarks" / "swebench"
        ).resolve()
        self.progress = progress

    @staticmethod
    def load_tasks(path: Path) -> list[SWEbenchTask]:
        tasks: list[SWEbenchTask] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                task = SWEbenchTask.model_validate(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid SWE-bench task manifest line {line_number}: {exc}"
                ) from exc
            if any(item.instance_id == task.instance_id for item in tasks):
                raise ValueError(f"duplicate SWE-bench instance_id: {task.instance_id}")
            tasks.append(task)
        if not tasks:
            raise ValueError(f"SWE-bench task manifest is empty: {path}")
        return tasks

    @staticmethod
    def agent_task(problem_statement: str) -> str:
        """Add SWE-bench-specific constraints without leaking hidden tests."""
        return (
            "SWE-bench repository repair task. Modify implementation files only; do not "
            "edit, add, or delete tests. Inspect existing validation and exception conventions "
            "before choosing behavior. Run focused existing tests, inspect the final diff, and "
            "finish with the smallest production-code patch that satisfies the report.\n\n"
            f"Issue:\n{problem_statement}"
        )

    async def run(
        self,
        executor: TaskExecutor,
        *,
        tasks: Sequence[SWEbenchTask],
        model_id: str,
        max_steps: int = 200,
        max_concurrency: int = 1,
    ) -> tuple[SWEbenchReport, Path]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        started = datetime.now(UTC)
        campaign_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        root = self.output_root / campaign_id
        root.mkdir(parents=True, exist_ok=False)
        repo_cache = self.output_root / "repo-cache"
        repo_cache.mkdir(parents=True, exist_ok=True)
        cache_lock = asyncio.Lock()
        checkpoint_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max_concurrency)
        partial_results_path = root / "partial-results.jsonl"
        partial_predictions_path = root / "predictions.partial.jsonl"

        async def run_task(
            position: int, task: SWEbenchTask
        ) -> tuple[SWEbenchResult, dict[str, object]]:
            async with semaphore:
                self.progress(f"[{position}/{len(tasks)}] {task.instance_id}")
                run_workspace = root / "workspaces" / task.instance_id.replace("/", "__")
                try:
                    await self._prepare_workspace(
                        task, run_workspace, repo_cache=repo_cache, cache_lock=cache_lock
                    )
                    state = await executor(
                        self.agent_task(task.problem_statement),
                        run_workspace,
                        model_id,
                        max_steps,
                    )
                    patch = await self._git_diff(run_workspace)
                    execution_completed = state.status == "completed" or (
                        bool(patch)
                        and state.termination_reason
                        in {"empty_model_response", "loop_detected", "max_steps"}
                    )
                    summary = (
                        RunCatalog(run_workspace).resolve(state.current_run_id)
                        if state.current_run_id
                        else None
                    )
                    result = SWEbenchResult(
                        instance_id=task.instance_id,
                        status="completed" if execution_completed else "failed",
                        model_patch=patch,
                        model=model_id,
                        workspace=str(run_workspace),
                        run_id=state.current_run_id,
                        steps=summary.steps if summary else state.step,
                        input_tokens=(
                            summary.input_tokens if summary else state.last_run_input_tokens
                        ),
                        cached_input_tokens=(
                            summary.cached_input_tokens
                            if summary
                            else state.last_run_cached_input_tokens
                        ),
                        output_tokens=(
                            summary.output_tokens if summary else state.last_run_output_tokens
                        ),
                        cost=summary.cost if summary else state.last_run_cost,
                        currency=summary.currency if summary else state.last_run_currency,
                        latency_seconds=(
                            summary.latency_seconds if summary else state.last_run_latency
                        ),
                        error=state.last_error,
                    )
                except LLMError as exc:
                    result = SWEbenchResult(
                        instance_id=task.instance_id,
                        status="model_error",
                        model=model_id,
                        workspace=str(run_workspace),
                        error=f"{exc.kind}: {exc}",
                    )
                except Exception as exc:
                    result = SWEbenchResult(
                        instance_id=task.instance_id,
                        status="infrastructure_error",
                        model=model_id,
                        workspace=str(run_workspace),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                self.progress(f"[{position}/{len(tasks)}] {result.status}")
                prediction: dict[str, object] = {
                    "instance_id": task.instance_id,
                    "model_name_or_path": model_id,
                    "model_patch": result.model_patch,
                }
                async with checkpoint_lock:
                    await asyncio.to_thread(
                        self._append_jsonl,
                        partial_results_path,
                        result.model_dump(mode="json"),
                    )
                    await asyncio.to_thread(
                        self._append_jsonl,
                        partial_predictions_path,
                        prediction,
                    )
                return result, prediction

        outcomes = await asyncio.gather(
            *(run_task(position, task) for position, task in enumerate(tasks, 1))
        )
        results = [result for result, _prediction in outcomes]
        predictions = [prediction for _result, prediction in outcomes]
        predictions_path = root / "predictions.jsonl"
        predictions_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
            encoding="utf-8",
        )
        finished = datetime.now(UTC)
        currency = next((item.currency for item in results if item.currency), "")
        report = SWEbenchReport(
            campaign_id=campaign_id,
            model=model_id,
            started_at=started,
            finished_at=finished,
            total_tasks=len(results),
            completed_tasks=sum(item.status == "completed" for item in results),
            failed_tasks=sum(item.status == "failed" for item in results),
            model_errors=sum(item.status == "model_error" for item in results),
            infrastructure_errors=sum(item.status == "infrastructure_error" for item in results),
            total_input_tokens=sum(item.input_tokens for item in results),
            total_cached_input_tokens=sum(item.cached_input_tokens for item in results),
            total_output_tokens=sum(item.output_tokens for item in results),
            total_cost=sum(item.cost for item in results),
            currency=currency,
            total_latency_seconds=sum(item.latency_seconds for item in results),
            predictions_path=str(predictions_path),
            results=results,
        )
        (root / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        (root / "report.md").write_text(self._markdown(report), encoding="utf-8")
        return report, root

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()

    @staticmethod
    async def _prepare_workspace(
        task: SWEbenchTask,
        destination: Path,
        *,
        repo_cache: Path,
        cache_lock: asyncio.Lock,
    ) -> None:
        if task.workspace is not None:
            source = task.workspace.expanduser().resolve()
            if not source.is_dir():
                raise ValueError(f"workspace does not exist: {source}")
            await asyncio.to_thread(
                shutil.copytree,
                source,
                destination,
                ignore=shutil.ignore_patterns(".capy", ".venv", "__pycache__"),
            )
            return
        if not task.repo or not task.base_commit:
            raise ValueError("SWE-bench task requires workspace or both repo and base_commit")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", task.repo):
            raise ValueError(f"invalid GitHub repository identifier: {task.repo}")

        # Keep one pristine clone per repository across campaigns. Instance
        # workspaces use a shared local clone, so no network download is needed
        # and Git objects are not duplicated on disk.
        cache_repo = repo_cache / task.repo.replace("/", "__")
        async with cache_lock:
            if not (cache_repo / ".git").is_dir():
                if cache_repo.exists():
                    await asyncio.to_thread(shutil.rmtree, cache_repo, ignore_errors=True)
                cache_repo.parent.mkdir(parents=True, exist_ok=True)
                clone_cache = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "git",
                        "clone",
                        "--quiet",
                        f"https://github.com/{task.repo}.git",
                        str(cache_repo),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                if clone_cache.returncode != 0:
                    await asyncio.to_thread(shutil.rmtree, cache_repo, ignore_errors=True)
                    raise RuntimeError(clone_cache.stderr.strip() or f"could not clone {task.repo}")

        clone = await asyncio.to_thread(
            subprocess.run,
            [
                "git",
                "clone",
                "--quiet",
                "--reference",
                str(cache_repo),
                str(cache_repo),
                str(destination),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if clone.returncode != 0:
            raise RuntimeError(
                clone.stderr.strip() or f"could not create workspace for {task.repo}"
            )
        checkout = await asyncio.to_thread(
            subprocess.run,
            ["git", "checkout", "--quiet", task.base_commit],
            cwd=destination,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if checkout.returncode != 0:
            raise RuntimeError(checkout.stderr.strip() or f"could not checkout {task.base_commit}")

    @staticmethod
    async def _git_diff(workspace: Path) -> str:
        process = await asyncio.to_thread(
            subprocess.run,
            ["git", "diff", "--binary", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or "git diff failed")
        return process.stdout

    @staticmethod
    def _markdown(report: SWEbenchReport) -> str:
        lines = [
            f"# SWE-bench campaign `{report.campaign_id}`",
            "",
            f"- Model: `{report.model}`",
            f"- Completed: {report.completed_tasks}/{report.total_tasks}",
            f"- Failed: {report.failed_tasks}",
            f"- Model request errors: {report.model_errors}",
            f"- Infrastructure errors: {report.infrastructure_errors}",
            f"- Tokens: {report.total_input_tokens} input / "
            f"{report.total_cached_input_tokens} cached input / "
            f"{report.total_output_tokens} output",
            f"- Cost: {report.total_cost:.6f} {report.currency}",
            f"- Latency: {report.total_latency_seconds:.2f}s",
            f"- Predictions: `{report.predictions_path}`",
            "",
            "Official SWE-bench Docker evaluation must be run separately on predictions.jsonl.",
        ]
        return "\n".join(lines) + "\n"
