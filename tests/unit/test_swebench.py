import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from capycode.core import SessionState
from capycode.llm import LLMError, LLMErrorKind
from capycode.profiling import SWEbenchRunner, SWEbenchTask


def test_swebench_manifest_accepts_prepared_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = tmp_path / "instances.jsonl"
    manifest.write_text(
        json.dumps(
            {"instance_id": "demo-1", "problem_statement": "fix it", "workspace": str(workspace)}
        )
        + "\n",
        encoding="utf-8",
    )
    tasks = SWEbenchRunner.load_tasks(manifest)
    assert tasks[0].instance_id == "demo-1"
    assert tasks[0].workspace == workspace


@pytest.mark.asyncio
async def test_swebench_runner_writes_predictions_and_metrics(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    await asyncio.to_thread(subprocess.run, ["git", "init", "-q"], cwd=source, check=True)
    (source / "README.md").write_text("baseline\n", encoding="utf-8")
    await asyncio.to_thread(subprocess.run, ["git", "add", "."], cwd=source, check=True)
    await asyncio.to_thread(
        subprocess.run,
        [
            "git",
            "-c",
            "user.name=CapyCode Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=source,
        check=True,
    )
    manifest = tmp_path / "instances.jsonl"
    manifest.write_text(
        json.dumps(
            {"instance_id": "demo-1", "problem_statement": "fix it", "workspace": str(source)}
        )
        + "\n",
        encoding="utf-8",
    )
    tasks = SWEbenchRunner.load_tasks(manifest)

    received_steps: list[int] = []

    async def executor(
        task: str, workspace: Path, model: str | None, max_steps: int
    ) -> SessionState:
        received_steps.append(max_steps)
        return SessionState(workspace=str(workspace), task=task, status="completed", step=1)

    report, root = await SWEbenchRunner(output_root=tmp_path / "out").run(
        executor, tasks=tasks, model_id="demo-model"
    )
    assert report.completed_tasks == 1
    assert received_steps == [200]
    assert (root / "predictions.jsonl").is_file()
    assert '"instance_id": "demo-1"' in (root / "predictions.jsonl").read_text(encoding="utf-8")
    assert '"status": "completed"' in (root / "partial-results.jsonl").read_text(encoding="utf-8")
    assert '"instance_id": "demo-1"' in (root / "predictions.partial.jsonl").read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_swebench_runner_limits_instance_concurrency(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    await asyncio.to_thread(subprocess.run, ["git", "init", "-q"], cwd=source, check=True)
    (source / "README.md").write_text("baseline\n", encoding="utf-8")
    await asyncio.to_thread(subprocess.run, ["git", "add", "."], cwd=source, check=True)
    await asyncio.to_thread(
        subprocess.run,
        [
            "git",
            "-c",
            "user.name=CapyCode Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=source,
        check=True,
    )
    tasks = [
        SWEbenchTask(
            instance_id=f"demo-{number}",
            problem_statement="fix it",
            workspace=source,
        )
        for number in range(3)
    ]
    active = 0
    maximum_active = 0

    async def executor(
        task: str, workspace: Path, model: str | None, max_steps: int
    ) -> SessionState:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return SessionState(workspace=str(workspace), task=task, status="completed", step=1)

    report, root = await SWEbenchRunner(output_root=tmp_path / "out").run(
        executor,
        tasks=tasks,
        model_id="demo-model",
        max_concurrency=2,
    )

    assert report.completed_tasks == 3
    assert maximum_active == 2
    predictions = [
        json.loads(line)
        for line in (root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item["instance_id"] for item in predictions] == ["demo-0", "demo-1", "demo-2"]


@pytest.mark.asyncio
async def test_swebench_runner_separates_model_errors_from_infrastructure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    await asyncio.to_thread(subprocess.run, ["git", "init", "-q"], cwd=source, check=True)
    (source / "README.md").write_text("baseline\n", encoding="utf-8")
    await asyncio.to_thread(subprocess.run, ["git", "add", "."], cwd=source, check=True)
    await asyncio.to_thread(
        subprocess.run,
        [
            "git",
            "-c",
            "user.name=CapyCode Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=source,
        check=True,
    )
    task = SWEbenchTask(instance_id="demo-1", problem_statement="fix it", workspace=source)

    async def executor(
        task_text: str, workspace: Path, model: str | None, max_steps: int
    ) -> SessionState:
        raise LLMError(
            LLMErrorKind.BAD_REQUEST,
            "model endpoint returned HTTP 400: unsupported tools",
            retryable=False,
            status_code=400,
        )

    report, _ = await SWEbenchRunner(output_root=tmp_path / "out").run(
        executor, tasks=[task], model_id="demo-model"
    )
    assert report.model_errors == 1
    assert report.infrastructure_errors == 0
    assert report.failed_tasks == 0
    assert report.results[0].status == "model_error"
