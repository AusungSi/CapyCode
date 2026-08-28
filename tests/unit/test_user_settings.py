from __future__ import annotations

import json
from pathlib import Path

from capycode.config import UserSettingsStore, resolve_model
from capycode.config.loader import load_models


def test_user_settings_round_trip_and_resolve(tmp_path: Path) -> None:
    store = UserSettingsStore(tmp_path / "settings.json")

    settings = store.configure_model(
        "small",
        model="demo-model",
        base_url="https://example.test/v1",
        api_key="local-secret",
        available_models=["demo-model", "other-model"],
    )
    registry = load_models(Path("config/models.example.yaml"))
    resolved = resolve_model("small", registry.models["small"], settings)

    assert store.load() == settings
    assert resolved.model == "demo-model"
    assert resolved.base_url == "https://example.test/v1"
    assert resolved.api_key == "local-secret"
    assert settings.default_model == "small"
    assert settings.models["small"].available_models == ["demo-model", "other-model"]

    selected = store.select_model("small", "other-model")
    assert selected.models["small"].model == "other-model"


def test_old_settings_add_selected_model_to_available_models(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "default_model": "small",
                "models": {
                    "small": {
                        "model": "legacy-model",
                        "base_url": "https://example.test/v1",
                        "api_key": "local-secret",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    settings = UserSettingsStore(path).load()

    assert settings.models["small"].available_models == ["legacy-model"]


def test_user_settings_file_is_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "settings.json"
    store = UserSettingsStore(path)

    store.configure_model(
        "medium",
        model="demo-model",
        base_url="https://example.test/v1",
        api_key="local-secret",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["default_model"] == "medium"
    assert payload["models"]["medium"]["api_key"] == "local-secret"
