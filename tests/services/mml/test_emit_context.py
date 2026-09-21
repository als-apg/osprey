"""Tests for the provenance context shared by every MML emitter.

Every artifact ``emit`` writes carries the same three provenance facts: the
exporter version, and the sha256 of ``ao.json`` and ``mapping.yaml``. The cases
pin where each fact comes from, the exact shape of the channel-DB provenance
STRING, the TTL header lines and OKF front-matter keys, and that the
``knowledge`` extra pre-flight refuses before anything is written.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path
from unittest import mock

import click
import pytest

from osprey.services.mml.canonical import sha256_of
from osprey.services.mml.emit import EmitContext, build_context, require_knowledge_extra
from osprey.services.mml.systems import EXPORTS_KEY, IMPORT_ORDER_KEY


def _files(tmp_path: Path) -> tuple[Path, Path]:
    ao_path = tmp_path / "ao.json"
    mapping_path = tmp_path / "mapping.yaml"
    ao_path.write_bytes(b'{"SR": {}}\n')
    mapping_path.write_bytes(b"facility:\n  token: quokka\n")
    return ao_path, mapping_path


def _ao(exports: dict, order: list[str]) -> dict:
    ao: dict = {EXPORTS_KEY: exports, IMPORT_ORDER_KEY: order}
    for system in order:
        ao[system] = {}
    return ao


class TestDigests:
    def test_both_digests_are_the_file_sha256(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)

        ctx = build_context(ao_path, mapping_path, _ao({}, ["SR"]))

        assert ctx.ao_sha256 == hashlib.sha256(ao_path.read_bytes()).hexdigest()
        assert ctx.mapping_sha256 == hashlib.sha256(mapping_path.read_bytes()).hexdigest()
        assert ctx.ao_sha256 == sha256_of(ao_path)
        assert ctx.ao_sha256 != ctx.mapping_sha256

    def test_accepts_string_paths(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)

        ctx = build_context(str(ao_path), str(mapping_path), _ao({}, ["SR"]))

        assert ctx.ao_sha256 == sha256_of(ao_path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        ao_path, _ = _files(tmp_path)

        with pytest.raises(OSError):
            build_context(ao_path, tmp_path / "absent.yaml", _ao({}, ["SR"]))


class TestExporterVersion:
    def test_reads_the_exporter_of_the_first_imported_system(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)
        exports = {
            "BR": {"exporter": "mml_export 0.9.0"},
            "SR": {"exporter": "mml_export 1.0.0"},
        }

        ctx = build_context(ao_path, mapping_path, _ao(exports, ["SR", "BR"]))

        assert ctx.exporter_version == "mml_export 1.0.0"

    def test_skips_leading_systems_without_an_export_block(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)
        exports = {"BR": {"exporter": "mml_export 1.0.0"}}

        ctx = build_context(ao_path, mapping_path, _ao(exports, ["LTB", "BR"]))

        assert ctx.exporter_version == "mml_export 1.0.0"

    def test_accepts_the_exporter_version_key(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)
        exports = {"SR": {"exporter_version": "2.1"}}

        ctx = build_context(ao_path, mapping_path, _ao(exports, ["SR"]))

        assert ctx.exporter_version == "2.1"

    @pytest.mark.parametrize(
        "ao",
        [
            {"SR": {}},
            {EXPORTS_KEY: {}, IMPORT_ORDER_KEY: ["SR"], "SR": {}},
            {EXPORTS_KEY: {"SR": {"matlab": "R2024b"}}, IMPORT_ORDER_KEY: ["SR"], "SR": {}},
            {EXPORTS_KEY: {"SR": {"exporter": ""}}, IMPORT_ORDER_KEY: ["SR"], "SR": {}},
            {EXPORTS_KEY: {"SR": "not a block"}, IMPORT_ORDER_KEY: ["SR"], "SR": {}},
            {EXPORTS_KEY: "not a map", IMPORT_ORDER_KEY: ["SR"], "SR": {}},
        ],
        ids=["no-exports", "empty-exports", "no-exporter-key", "blank", "bad-block", "bad-map"],
    )
    def test_falls_back_to_none(self, tmp_path: Path, ao: dict) -> None:
        ao_path, mapping_path = _files(tmp_path)

        ctx = build_context(ao_path, mapping_path, ao)

        assert ctx.exporter_version == "none"

    def test_without_import_order_uses_export_order(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)
        ao = {EXPORTS_KEY: {"BR": {"exporter": "a"}, "SR": {"exporter": "b"}}}

        ctx = build_context(ao_path, mapping_path, ao)

        assert ctx.exporter_version == "a"


class TestRenderedProvenance:
    def test_provenance_string_shape(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)
        ao = _ao({"SR": {"exporter": "mml_export 1.0.0"}}, ["SR"])

        ctx = build_context(ao_path, mapping_path, ao)

        assert isinstance(ctx.provenance_string, str)
        assert ctx.provenance_string == (
            f"exporter=mml_export 1.0.0 ao_sha256={sha256_of(ao_path)} "
            f"mapping_sha256={sha256_of(mapping_path)}"
        )

    def test_header_lines_are_three_bare_single_line_entries(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)

        ctx = build_context(ao_path, mapping_path, _ao({}, ["SR"]))

        assert isinstance(ctx.header_lines, tuple)
        assert ctx.header_lines == (
            "exporter=none",
            f"ao_sha256={sha256_of(ao_path)}",
            f"mapping_sha256={sha256_of(mapping_path)}",
        )
        for line in ctx.header_lines:
            assert not line.startswith("#")
            assert "\n" not in line and "\r" not in line

    def test_front_matter_carries_the_three_keys(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)
        ao = _ao({"SR": {"exporter": "mml_export 1.0.0"}}, ["SR"])

        ctx = build_context(ao_path, mapping_path, ao)

        assert ctx.front_matter == {
            "exporter": "mml_export 1.0.0",
            "ao_sha256": sha256_of(ao_path),
            "mapping_sha256": sha256_of(mapping_path),
        }
        assert list(ctx.front_matter) == ["exporter", "ao_sha256", "mapping_sha256"]

    def test_front_matter_is_a_fresh_copy(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)
        ctx = build_context(ao_path, mapping_path, _ao({}, ["SR"]))

        ctx.front_matter["exporter"] = "tampered"

        assert ctx.front_matter["exporter"] == "none"

    def test_context_is_frozen_and_deterministic(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)
        ao = _ao({"SR": {"exporter": "x"}}, ["SR"])

        first = build_context(ao_path, mapping_path, ao)
        second = build_context(ao_path, mapping_path, ao)

        assert isinstance(first, EmitContext)
        assert first == second
        with pytest.raises(dataclasses.FrozenInstanceError):
            first.exporter_version = "y"  # type: ignore[misc]

    def test_does_not_mutate_ao(self, tmp_path: Path) -> None:
        ao_path, mapping_path = _files(tmp_path)
        ao = _ao({"SR": {"exporter": "x"}}, ["SR"])
        snapshot = repr(ao)

        build_context(ao_path, mapping_path, ao)

        assert repr(ao) == snapshot


class TestRequireKnowledgeExtra:
    def test_passes_when_linkml_runtime_is_importable(self) -> None:
        require_knowledge_extra()

    def test_absent_linkml_runtime_names_the_extra(self) -> None:
        with mock.patch.dict(sys.modules, {"linkml_runtime": None}):
            with pytest.raises(click.ClickException) as info:
                require_knowledge_extra()

        message = info.value.format_message()
        assert "knowledge" in message
        assert "Install it with: pip install 'osprey-framework[knowledge]'" in message

    def test_pre_flight_fires_before_any_write(self, tmp_path: Path) -> None:
        target = tmp_path / "data" / "channel_databases" / "middle_layer.json"

        def emit() -> None:
            require_knowledge_extra()
            target.parent.mkdir(parents=True)
            target.write_text("{}\n", encoding="utf-8")

        with mock.patch.dict(sys.modules, {"linkml_runtime": None}):
            with pytest.raises(click.ClickException):
                emit()

        assert not target.exists()
        assert not (tmp_path / "data").exists()
