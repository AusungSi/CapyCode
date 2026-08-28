from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from capycode.config import UserSettingsStore, resolve_model


def test_user_settings_use_real_model_ids_and_per_model_pricing(tmp_path: Path) -> None:
    store = UserSettingsStore(tmp_path / "settings.json")

    settings = store.configure_endpoint(
        model="demo-model",
        base_url="https://example.test/v1",
        api_key="local-secret",
        available_models=["demo-model", "other-model"],
    )
    store.configure_pricing(
        "demo-model",
        input_per_million=2,
        output_per_million=8,
        currency="cny",
        snapshot_date=date(2026, 8, 28),
        context_window=200_000,
    )
    resolved = resolve_model("demo-model", store.load())

    assert settings.default_model == "demo-model"
    assert settings.endpoint is not None
    assert settings.endpoint.available_models == ["demo-model", "other-model"]
    assert resolved.model == "demo-model"
    assert resolved.base_url == "https://example.test/v1"
    assert resolved.api_key == "local-secret"
    assert resolved.input_per_million == 2
    assert resolved.output_per_million == 8
    assert resolved.currency == "CNY"
    assert resolved.context_window == 200_000

    selected = store.select_model("other-model")
    assert selected.default_model == "other-model"
    assert selected.models["other-model"].pricing.input_per_million == 0

    store.configure_pricing(
        "other-model",
        input_per_million=4,
        output_per_million=16,
        currency="USD",
        snapshot_date=date(2026, 8, 29),
        context_window=128_000,
    )
    final = store.load()
    assert final.models["demo-model"].pricing.input_per_million == 2
    assert final.models["other-model"].pricing.input_per_million == 4


def test_legacy_alias_settings_are_migrated_to_real_models(tmp_path: Path) -> None:
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
                        "available_models": ["legacy-model", "other-model"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    settings = UserSettingsStore(path).load()
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert settings.schema_version == 2
    assert settings.default_model == "legacy-model"
    assert settings.endpoint is not None
    assert settings.endpoint.available_models == ["legacy-model", "other-model"]
    assert set(settings.models) == {"legacy-model", "other-model"}
    assert payload["default_model"] == "legacy-model"
    assert "small" not in payload["models"]
    assert payload["endpoint"]["api_key"] == "local-secret"


def test_user_settings_file_is_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "settings.json"
    store = UserSettingsStore(path)

    store.configure_endpoint(
        model="demo-model",
        base_url="https://example.test/v1",
        api_key="local-secret",
        available_models=["demo-model"],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["default_model"] == "demo-model"
    assert payload["endpoint"]["api_key"] == "local-secret"
