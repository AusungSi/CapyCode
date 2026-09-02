from __future__ import annotations

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from capycode.core import AgentRuntime, SessionState
from capycode.llm import LLMResponse, ScriptedLLM, ToolCall
from capycode.profiling import GateRunResult, GateTask, GateTaskCatalog, P0GateRunner
from capycode.tools import build_p0_runtime_tools
from capycode.trace import RunTrackingConfig


def has_hidden_tests(workspace: Path) -> bool:
    return any(workspace.glob("test_hidden*.py"))


def prepare_solved_workspace(task: GateTask, destination: Path) -> None:
    shutil.copytree(task.workspace_source, destination)
    for source in task.oracle_source.rglob("*"):
        if source.is_file():
            target = destination / source.relative_to(task.oracle_source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    task_id = task.manifest.task_id
    if task_id == "p0-01":
        path = destination / "pricing.py"
        path.write_text(path.read_text(encoding="utf-8").replace(" > ", " >= "), encoding="utf-8")
    elif task_id == "p0-02":
        path = destination / "records.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "active=bool(active)", 'active=active.lower() == "true"'
            ),
            encoding="utf-8",
        )
    elif task_id == "p0-03":
        path = destination / "inventory" / "service.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "if product.stock > quantity:", "if product.stock >= quantity:"
            ),
            encoding="utf-8",
        )
    elif task_id == "p0-04":
        path = destination / "request_cache.py"
        content = path.read_text(encoding="utf-8").replace(
            "    _values: dict[str, str] = {}\n",
            "    def __init__(self) -> None:\n        self._values: dict[str, str] = {}\n",
        )
        path.write_text(content, encoding="utf-8")
    elif task_id == "p0-05":
        (destination / "config_loader.py").write_text(
            "from pathlib import Path\n\n\n"
            "def load_port(path: str | Path) -> int:\n"
            "    lines = Path(path).read_text(encoding='utf-8').splitlines()\n"
            "    values = [line.strip() for line in lines "
            "if line.strip() and not line.lstrip().startswith('#')]\n"
            "    if not values:\n"
            "        raise ValueError('missing port')\n"
            "    port = int(values[0])\n"
            "    if not 1 <= port <= 65535:\n"
            "        raise ValueError('port out of range')\n"
            "    return port\n",
            encoding="utf-8",
        )


def make_result(task: GateTask, repeat: int, status: str) -> GateRunResult:
    return GateRunResult(
        task_id=task.manifest.task_id,
        task_title=task.manifest.title,
        capability=task.manifest.capability,
        repeat_index=repeat,
        model_id="fake-model",
        task_fingerprint=task.fingerprint,
        status=status,
        workspace="workspace",
    )


def copy_and_mutate_catalog(destination: Path) -> None:
    shutil.copytree(GateTaskCatalog().root, destination)
    path = destination / "p0-01" / "workspace" / "pricing.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def test_snapshot_does_not_ignore_workspace_because_parent_is_capy(tmp_path: Path) -> None:
    workspace = tmp_path / ".capy" / "benchmarks" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "pricing.py").write_text("value = 1\n", encoding="utf-8")

    snapshot = P0GateRunner._snapshot(workspace)

    assert sorted(snapshot) == ["pricing.py"]


@pytest.mark.asyncio
async def test_bundled_p0_fixtures_are_frozen_and_start_failing(tmp_path: Path) -> None:
    catalog = GateTaskCatalog()
    tasks = catalog.list()

    assert [task.manifest.task_id for task in tasks] == [
        "p0-01",
        "p0-02",
        "p0-03",
        "p0-04",
        "p0-05",
    ]
    assert all(len(task.fingerprint) == 64 for task in tasks)
    assert await P0GateRunner(catalog=catalog).validate_fixtures() == [
        "p0-01",
        "p0-02",
        "p0-03",
        "p0-04",
        "p0-05",
    ]


@pytest.mark.asyncio
async def test_hidden_oracles_accept_the_intended_fixes(tmp_path: Path) -> None:
    runner = P0GateRunner()
    for task in GateTaskCatalog().list():
        workspace = tmp_path / task.manifest.task_id
        prepare_solved_workspace(task, workspace)
        result = await runner.run_unittests(workspace)
        assert result.passed, f"{task.manifest.task_id}: {result.stderr}"


@pytest.mark.asyncio
async def test_git_fixture_stages_explicit_files_despite_global_ignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = GateTaskCatalog().select(["p0-01"])[0]
    workspace = tmp_path / "workspace"
    shutil.copytree(task.workspace_source, workspace)
    ignore = tmp_path / "global-ignore"
    ignore.write_text("*.py\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(ignore))
    runner = P0GateRunner()
    tracked = sorted(runner._snapshot(workspace))

    result = await runner._initialize_git(workspace, tracked)
    committed = await runner._run_process(
        ["git", "show", "--pretty=", "--name-only", "HEAD"], workspace
    )

    assert result.passed, result.stderr
    assert sorted(committed.stdout.splitlines()) == tracked


@pytest.mark.asyncio
async def test_gate_runner_executes_agent_and_hidden_evaluation(tmp_path: Path) -> None:
    python = Path(sys.executable).name
    unittest_argv = [
        python,
        "-m",
        "unittest",
        "discover",
        "-s",
        ".",
        "-p",
        "test_*.py",
        "-q",
    ]

    async def executor(
        task: str,
        workspace: Path,
        model_id: str | None,
        max_steps: int,
    ) -> SessionState:
        assert not has_hidden_tests(workspace)
        client = ScriptedLLM(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="tests-before",
                            name="run_tests",
                            arguments={"argv": unittest_argv},
                        )
                    ]
                ),
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="read-source",
                            name="read_file",
                            arguments={"path": "pricing.py"},
                        )
                    ]
                ),
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="fix-source",
                            name="replace_text",
                            arguments={
                                "path": "pricing.py",
                                "old_text": "order_total >",
                                "new_text": "order_total >=",
                                "replace_all": True,
                            },
                        )
                    ]
                ),
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="tests-after",
                            name="run_tests",
                            arguments={"argv": unittest_argv},
                        ),
                        ToolCall(id="diff", name="git_diff", arguments={}),
                    ]
                ),
                LLMResponse(content="Fixed both inclusive shipping boundaries."),
            ]
        )
        tools = build_p0_runtime_tools()
        try:
            runtime = AgentRuntime(client, tools, max_steps=max_steps)
            return await runtime.run(
                task,
                workspace,
                model_id or "fake-model",
                tracking=RunTrackingConfig(
                    provider="fake",
                    model_id=model_id or "fake-model",
                    input_per_million=0,
                    output_per_million=0,
                    currency="USD",
                    pricing_snapshot_date="2026-08-29",
                ),
            )
        finally:
            await tools.aclose()

    runner = P0GateRunner(output_root=tmp_path / "reports")
    report, report_root = await runner.run(
        executor,
        model_id="fake-model",
        repeats=1,
        task_ids=["p0-01"],
    )

    result = report.results[0]
    assert result.status == "passed", result.reasons
    assert result.initial_tests_failed
    assert result.public_tests_passed
    assert result.hidden_tests_passed
    assert result.diff_scope_passed
    assert result.trace_integrity_passed
    assert result.modified_files == ["pricing.py"]
    assert not report.gate_eligible
    assert not report.gate_passed
    assert report.task_coverage == 1
    assert (report_root / "report.json").is_file()
    assert "p0-01" in (report_root / "report.md").read_text(encoding="utf-8")


def test_catalog_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="unknown P0 benchmark task"):
        GateTaskCatalog().select(["p0-99"])


def test_catalog_rejects_fixture_changed_without_lock_update(tmp_path: Path) -> None:
    copied_catalog = tmp_path / "catalog"
    copy_and_mutate_catalog(copied_catalog)

    with pytest.raises(ValueError, match="fingerprint changed"):
        GateTaskCatalog(copied_catalog).list()


def test_full_gate_requires_five_task_coverage_and_eighty_percent() -> None:
    tasks = GateTaskCatalog().list()
    results = [
        make_result(
            task, repeat, "failed" if task.manifest.task_id == "p0-05" and repeat == 1 else "passed"
        )
        for repeat in (1, 2)
        for task in tasks
    ]
    now = datetime.now(UTC)

    report = P0GateRunner.build_report(
        campaign_id="campaign",
        model_id="fake-model",
        started_at=now,
        finished_at=now,
        repeats=2,
        tasks=tasks,
        results=results,
    )
    partial_report = P0GateRunner.build_report(
        campaign_id="partial",
        model_id="fake-model",
        started_at=now,
        finished_at=now,
        repeats=2,
        tasks=tasks[:4],
        results=[result for result in results if result.task_id != "p0-05"],
    )

    assert report.gate_passed
    assert report.gate_eligible
    assert report.pass_rate == 0.9
    assert report.pass_at_1 == 0.8
    assert report.task_coverage == 1
    assert not partial_report.gate_passed
    assert not partial_report.gate_eligible
