"""Tests for the LinkML ontology schema ``osprey mml emit`` writes.

The emitted schema must compile through the same ``ontology_compiler`` the
``knowledge compile-ontology`` command uses. The cases pin the family map, the
packaged tree closed under the ancestors of every used class, the aliases a
packaged class gains from the families typed as it, one class per distinct new
family class with the union of its sharing families' aliases, mapping branches
as classes, zero-channel families left out, the provenance in the schema
description, and a deterministic, byte-stable compiled JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from osprey.services.mml.emit.context import EmitContext
from osprey.services.mml.emit.ontology import build_ontology_yaml, compile_to_json
from osprey.services.mml.mapping.branches import ROOT_CLASS, packaged_classes
from osprey.services.mml.mapping.schema import (
    Branch,
    Facility,
    Family,
    Mapping,
    System,
)

pytest.importorskip("linkml_runtime")


def _ctx() -> EmitContext:
    lines = ("exporter=mml-1.2", "ao_sha256=" + "a" * 64, "mapping_sha256=" + "b" * 64)
    return EmitContext(
        ao_sha256="a" * 64,
        mapping_sha256="b" * 64,
        exporter_version="mml-1.2",
        provenance_string=" ".join(lines),
        header_lines=lines,
    )


def _family(
    raw: str,
    *,
    class_: str | None,
    branch: str | None = None,
    aliases: tuple[str, ...] = (),
    description: str | None = None,
    channels: int = 4,
    rename: str | None = None,
) -> Family:
    return Family(
        raw=raw,
        rename=rename,
        branch=branch,
        class_=class_,
        aliases=aliases,
        description=description,
        provenance="authored",
        channels=channels,
        fields={},
    )


def _mapping(families: list[Family], branches: list[Branch] | None = None) -> Mapping:
    return Mapping(
        facility=Facility(
            token="quokka", title="Quokka ring", description="A ring.", provenance="authored"
        ),
        systems={"SR": System(raw="SR", name="SR", description=None, provenance="authored")},
        section_order=("SR",),
        branches={b.name: b for b in branches or []},
        families={f.raw: f for f in families},
    )


def _default_mapping() -> Mapping:
    return _mapping(
        [
            _family("HCM", class_="HCorrector", aliases=("hcm", "h steerer")),
            _family("BPMx", class_="BeamPositionMonitor", rename="BPM"),
            _family(
                "SQF",
                class_="SkewQuad",
                branch="Magnet",
                aliases=("skew", "sqf"),
                description="Skew quadrupole F.",
            ),
            _family(
                "SQD",
                class_="SkewQuad",
                branch="Magnet",
                aliases=("sqd", "skew"),
                description="Skew quadrupole D.",
            ),
            _family("KICK", class_="Kicker", branch="Pulsed", aliases=("kicker",)),
            _family("DEAD", class_=None, channels=0),
        ],
        branches=[Branch(name="Pulsed", parent="Magnet", description="Pulsed magnets.")],
    )


def _compile(tmp_path: Path, schema: dict):
    yaml_path = tmp_path / "quokka.yaml"
    yaml_path.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
    json_path = tmp_path / "facility_ontology.json"
    return compile_to_json(yaml_path, json_path), yaml_path, json_path


class TestSchemaShape:
    def test_header_and_provenance(self) -> None:
        schema = build_ontology_yaml(_default_mapping(), _ctx())
        assert (
            schema["prefixes"]["narad_sem"] == "https://narad.example.org/schema/shared_semantics/"
        )
        assert schema["default_prefix"] == "narad_sem"
        assert schema["imports"] == ["linkml:types"]
        assert _ctx().provenance_string in schema["description"]

    def test_root_has_no_parent(self) -> None:
        classes = build_ontology_yaml(_default_mapping(), _ctx())["classes"]
        root = classes[ROOT_CLASS]
        assert "is_a" not in root
        assert root["class_uri"] == f"narad_sem:{ROOT_CLASS}"

    def test_every_class_uses_only_allowed_fields(self) -> None:
        classes = build_ontology_yaml(_default_mapping(), _ctx())["classes"]
        allowed = {"is_a", "aliases", "class_uri", "description", "comments"}
        for name, body in classes.items():
            assert set(body) <= allowed, name
            assert body["class_uri"] == f"narad_sem:{name}"

    def test_packaged_closure_only(self) -> None:
        classes = build_ontology_yaml(_default_mapping(), _ctx())["classes"]
        packaged = set(classes) & (set(packaged_classes()) | {ROOT_CLASS})
        assert packaged == {
            ROOT_CLASS,
            "Magnet",
            "Corrector",
            "HCorrector",
            "Instrumentation",
            "BeamPositionMonitor",
        }
        assert "Vacuum" not in classes
        assert "Dipole" not in classes

    def test_zero_channel_family_excluded(self) -> None:
        schema = build_ontology_yaml(_default_mapping(), _ctx())
        values = schema["enums"]["DeviceFamily"]["permissible_values"]
        assert "DEAD" not in values
        assert set(values) == {"HCM", "BPM", "SQF", "SQD", "KICK"}

    def test_enum_meaning(self) -> None:
        values = build_ontology_yaml(_default_mapping(), _ctx())["enums"]["DeviceFamily"][
            "permissible_values"
        ]
        assert values["BPM"] == {"meaning": "narad_sem:BeamPositionMonitor"}
        assert values["SQF"] == {"meaning": "narad_sem:SkewQuad"}

    def test_packaged_class_gains_family_aliases(self) -> None:
        classes = build_ontology_yaml(_default_mapping(), _ctx())["classes"]
        aliases = classes["HCorrector"]["aliases"]
        assert set(aliases) == set(packaged_classes()["HCorrector"].alt_labels) | {
            "hcm",
            "h steerer",
        }
        assert aliases == sorted(aliases)

    def test_shared_new_class(self) -> None:
        body = build_ontology_yaml(_default_mapping(), _ctx())["classes"]["SkewQuad"]
        assert body["is_a"] == "Magnet"
        assert body["aliases"] == ["skew", "sqd", "sqf"]
        # First sharing family by raw token: SQD < SQF.
        assert body["description"] == "Skew quadrupole D."

    def test_branch_class(self) -> None:
        classes = build_ontology_yaml(_default_mapping(), _ctx())["classes"]
        assert classes["Pulsed"] == {
            "class_uri": "narad_sem:Pulsed",
            "is_a": "Magnet",
            "description": "Pulsed magnets.",
        }
        assert classes["Kicker"]["is_a"] == "Pulsed"

    def test_new_class_without_branch_is_refused(self) -> None:
        mapping = _mapping([_family("X", class_="Novel", branch=None)])
        with pytest.raises(ValueError, match="X"):
            build_ontology_yaml(mapping, _ctx())

    def test_deterministic(self) -> None:
        first = build_ontology_yaml(_default_mapping(), _ctx())
        second = build_ontology_yaml(_default_mapping(), _ctx())
        assert yaml.safe_dump(first, sort_keys=False) == yaml.safe_dump(second, sort_keys=False)


class TestCompile:
    def test_family_to_class(self, tmp_path: Path) -> None:
        table, _, _ = _compile(tmp_path, build_ontology_yaml(_default_mapping(), _ctx()))
        assert dict(table.family_to_class) == {
            "HCM": "HCorrector",
            "BPM": "BeamPositionMonitor",
            "SQF": "SkewQuad",
            "SQD": "SkewQuad",
            "KICK": "Kicker",
        }

    def test_alt_labels_for_shared_new_class(self, tmp_path: Path) -> None:
        table, _, _ = _compile(tmp_path, build_ontology_yaml(_default_mapping(), _ctx()))
        assert table.classes["SkewQuad"].alt_labels == ("skew", "sqd", "sqf")

    def test_hcorrector_ancestors(self, tmp_path: Path) -> None:
        table, _, _ = _compile(tmp_path, build_ontology_yaml(_default_mapping(), _ctx()))
        cls = table.class_for("HCM").name
        assert cls == "HCorrector"
        assert table.ancestors(cls) == ("Corrector", "Magnet", "AcceleratorDevice")

    def test_branch_chain_ancestors(self, tmp_path: Path) -> None:
        table, _, _ = _compile(tmp_path, build_ontology_yaml(_default_mapping(), _ctx()))
        assert table.ancestors("Kicker") == ("Pulsed", "Magnet", "AcceleratorDevice")

    def test_json_written_and_stable(self, tmp_path: Path) -> None:
        _, yaml_path, json_path = _compile(
            tmp_path, build_ontology_yaml(_default_mapping(), _ctx())
        )
        first = json_path.read_bytes()
        decoded = json.loads(first)
        assert decoded["family_to_class"]["SQD"] == "SkewQuad"
        compile_to_json(yaml_path, json_path)
        assert json_path.read_bytes() == first

    def test_module_import_does_not_pull_linkml(self) -> None:
        import subprocess
        import sys

        code = (
            "import sys; import osprey.services.mml.emit.ontology; "
            "print('linkml_runtime' in sys.modules, 'osprey.cli.knowledge_cmd' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.split()
        assert out == ["False", "False"]
