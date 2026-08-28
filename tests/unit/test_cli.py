from __future__ import annotations

from pathlib import Path

import pytest

from capycode.app import cli
from capycode.app.cli import main, run_doctor, show_welcome


def test_no_arguments_starts_branded_entrypoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = show_welcome(tmp_path)

    assert code == 0
    output = capsys.readouterr().out
    assert "CapyCode 0.1.0" in output
    assert f"workspace: {tmp_path.resolve()}" in output
    assert "stage: P0-1 interactive runtime" in output


def test_main_without_arguments_exits_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched = False

    def fake_launch_tui() -> None:
        nonlocal launched
        launched = True

    monkeypatch.setattr(cli, "launch_tui", fake_launch_tui)

    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 0
    assert launched is True


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_doctor_accepts_valid_examples_without_secrets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run_doctor(
        models_path=Path("config/models.example.yaml"),
        profiles_path=Path("config/profiles.example.yaml"),
        strict_secrets=False,
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "configuration: valid" in output
    assert "warning: missing local environment variables" in output


def test_doctor_strict_mode_rejects_missing_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "SMALL_BASE_URL",
        "SMALL_API_KEY",
        "MEDIUM_BASE_URL",
        "MEDIUM_API_KEY",
        "STRONG_BASE_URL",
        "STRONG_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    code = run_doctor(
        models_path=Path("config/models.example.yaml"),
        profiles_path=Path("config/profiles.example.yaml"),
        strict_secrets=True,
    )

    assert code == 1
