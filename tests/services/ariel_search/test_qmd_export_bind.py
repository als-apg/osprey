"""QmdExportModule.bind threads the entry expander into every mirror write.

The pipeline path (``create_enhancers_from_config`` -> ``enhance``) and the
best-effort path (``mirror_entry_best_effort`` -> ``mirror_now``) must render
identical bytes for the same row and config, so a file written by one is never
rewritten by the other.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.enhancement.qmd_export.exporter import (
    TOUCH_MARKER_NAME,
    QmdExportModule,
    mirror_entry_best_effort,
)
from osprey.services.ariel_search.enhancement.qmd_export.writer import mirror_path, render_entry

_RAW = "RF-BPM, check."
_EXPANDED = "RF-BPM radio frequency beam position monitor, check."


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "entry_id": "777",
        "source_system": "Example eLog",
        "timestamp": datetime(2024, 5, 17, 13, 45, 9, tzinfo=UTC),
        "author": "jdoe",
        "raw_text": _RAW,
        "attachments": [],
        "metadata": {"category": "operations"},
        "summary": "BPM drift.",
        "keywords": ["bpm"],
    }
    row.update(overrides)
    return row


def _config(mirror, vocabulary_file=None, *, expand: bool = True) -> ARIELConfig:
    data: dict[str, Any] = {
        "database": {"uri": "postgresql://localhost:5432/ariel"},
        "search_modules": {"keyword": {"enabled": True}},
        "enhancement_modules": {"qmd_export": {"enabled": True, "mirror_path": str(mirror)}},
    }
    if vocabulary_file is not None:
        data["vocabulary"] = {
            "enabled": True,
            "path": str(vocabulary_file),
            "expand_entries": expand,
        }
    return ARIELConfig.from_dict(data)


def _pipeline_module(config: ARIELConfig) -> QmdExportModule:
    module = QmdExportModule()
    module.configure(config.get_enhancement_module_config("qmd_export") or {})
    module.bind(config)
    return module


def _read(root, row) -> bytes:
    return mirror_path(root, row).read_bytes()


class TestBindKeyOff:
    @pytest.mark.asyncio
    async def test_key_off_mirror_bytes_are_unchanged(self, tmp_path, unit_vocabulary_file):
        row = _row()
        config = _config(tmp_path / "m", unit_vocabulary_file, expand=False)
        module = _pipeline_module(config)
        await module.enhance(dict(row), None)  # type: ignore[arg-type]
        assert _read(tmp_path / "m", row) == render_entry(row).encode("utf-8")
        assert _EXPANDED.encode() not in _read(tmp_path / "m", row)

    @pytest.mark.asyncio
    async def test_no_vocabulary_mirror_bytes_are_unchanged(self, tmp_path):
        row = _row()
        module = _pipeline_module(_config(tmp_path / "m"))
        await module.enhance(dict(row), None)  # type: ignore[arg-type]
        assert _read(tmp_path / "m", row) == render_entry(row).encode("utf-8")

    def test_unbound_module_renders_unexpanded(self, tmp_path):
        row = _row()
        module = QmdExportModule()
        module.configure({"mirror_path": str(tmp_path / "m")})
        assert module.mirror_now(row) is True
        assert _read(tmp_path / "m", row) == render_entry(row).encode("utf-8")

    def test_unusable_vocabulary_binds_without_raising(self, tmp_path):
        row = _row()
        config = _config(tmp_path / "m", tmp_path / "missing.yml")
        module = _pipeline_module(config)
        assert module.mirror_now(row) is True
        assert _read(tmp_path / "m", row) == render_entry(row).encode("utf-8")


class TestBindExpands:
    @pytest.mark.asyncio
    async def test_enhance_writes_expanded_body(self, tmp_path, unit_vocabulary_file):
        row = _row()
        config = _config(tmp_path / "m", unit_vocabulary_file)
        module = _pipeline_module(config)
        await module.enhance(dict(row), None)  # type: ignore[arg-type]
        text = _read(tmp_path / "m", row).decode("utf-8")
        expander = config.entry_expander_or_none()
        assert expander is not None
        assert text == render_entry(row, expander=expander)
        assert _EXPANDED in text
        assert (tmp_path / "m" / TOUCH_MARKER_NAME).exists()

    def test_mirror_now_writes_expanded_body(self, tmp_path, unit_vocabulary_file):
        row = _row()
        module = _pipeline_module(_config(tmp_path / "m", unit_vocabulary_file))
        assert module.mirror_now(row) is True
        assert _EXPANDED in _read(tmp_path / "m", row).decode("utf-8")
        # Same bytes again: no rewrite.
        assert module.mirror_now(row) is False

    def test_best_effort_writes_expanded_body(self, tmp_path, unit_vocabulary_file):
        row = _row()
        assert mirror_entry_best_effort(_config(tmp_path / "m", unit_vocabulary_file), row)
        assert _EXPANDED in _read(tmp_path / "m", row).decode("utf-8")


class TestPathsAgree:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("expand", [True, False])
    async def test_pipeline_and_best_effort_write_identical_bytes(
        self, tmp_path, unit_vocabulary_file, expand
    ):
        from osprey.services.ariel_search.enhancement.factory import create_enhancers_from_config

        row = _row()
        pipeline_root, inline_root = tmp_path / "pipeline", tmp_path / "inline"

        pipeline_config = _config(pipeline_root, unit_vocabulary_file, expand=expand)
        enhancers = create_enhancers_from_config(pipeline_config, names=["qmd_export"])
        assert len(enhancers) == 1
        await enhancers[0].enhance(dict(row), None)  # type: ignore[arg-type]

        inline_config = _config(inline_root, unit_vocabulary_file, expand=expand)
        assert mirror_entry_best_effort(inline_config, dict(row)) is True

        pipeline_bytes = _read(pipeline_root, row)
        assert pipeline_bytes == _read(inline_root, row)
        assert (_EXPANDED.encode() in pipeline_bytes) is expand

        # The best-effort path finds the pipeline's file already current.
        assert mirror_entry_best_effort(pipeline_config, dict(row)) is False
