from __future__ import annotations

import importlib

import capycode


def test_all_top_level_modules_import() -> None:
    modules = (
        "app",
        "capability",
        "config",
        "core",
        "llm",
        "profiling",
        "tools",
        "trace",
        "workspace",
    )

    for module in modules:
        importlib.import_module(f"capycode.{module}")

    assert capycode.__version__ == "0.1.0"
