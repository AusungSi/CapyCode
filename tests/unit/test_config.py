from __future__ import annotations

from pathlib import Path

import pytest

from capycode.config import load_configuration, load_models


def test_example_configuration_loads() -> None:
    bundle = load_configuration(
        Path("config/models.example.yaml"),
        Path("config/profiles.example.yaml"),
    )

    assert set(bundle.models.models) == {"small", "medium", "strong"}
    assert bundle.profiles.profiles["retrieval_fast"].model == "small"


def test_inline_credentials_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        """
models:
  unsafe:
    provider: openai-compatible
    model: demo
    base_url_env: DEMO_BASE_URL
    api_key_env: DEMO_API_KEY
    api_key: ${INLINE_SECRET_SHOULD_BE_REJECTED}
    context_window: 1000
    pricing:
      input_per_million: 1
      output_per_million: 1
      snapshot_date: 2026-08-28
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="extra_forbidden"):
        load_models(path)


def test_profile_must_reference_registered_model(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        """
profiles:
  invalid:
    capability: repository_retrieval
    model: unavailable
    instruction: prompts/retrieval.md
    tools: [read_file]
    context_policy: retrieval
    budget:
      max_output_tokens: 100
      max_steps: 1
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown models: unavailable"):
        load_configuration("config/models.example.yaml", profiles)
