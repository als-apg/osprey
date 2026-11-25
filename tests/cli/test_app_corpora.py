"""A container's ``corpora:`` checked against the qmd corpora the deployment declares."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.apps.errors import AppsRefusedError
from osprey.apps.loader import corpus_violations
from osprey.apps.manifest import AppManifest, parse_manifest
from osprey.cli.build_profile_load import load_profile_document

QMD = {
    "corpora": [
        {"name": "procedures", "source": "./data/procedures"},
        {"name": "papers", "index": "prebuilt", "index_dir": "/srv/papers-index"},
    ]
}


def _manifest(*corpora: str) -> AppManifest:
    container: dict[str, Any] = {"runtime": "osprey", "corpora": list(corpora)}
    return parse_manifest({"name": "knob", "containers": {"knob": container}}, app="knob")


def test_a_declared_corpus_with_a_source_passes() -> None:
    assert corpus_violations([_manifest("procedures")], QMD) == []
    assert corpus_violations([_manifest()], {}) == []


def test_a_corpus_the_deployment_does_not_declare_is_refused() -> None:
    (violation,) = corpus_violations([_manifest("procedures", "missing")], QMD)

    assert violation.rule == "corpus_unknown"
    assert violation.app == "knob"
    assert violation.field == "containers.knob.corpora.1"
    assert "'missing' is not a corpus services.qmd.corpora declares" in violation.message
    assert "the profile declares: papers, procedures" in violation.message


def test_a_deployment_with_no_corpora_refuses_every_name() -> None:
    (violation,) = corpus_violations([_manifest("procedures")], {})

    assert violation.rule == "corpus_unknown"
    assert "the profile declares: none" in violation.message


def test_a_corpus_declared_without_a_source_is_refused() -> None:
    (violation,) = corpus_violations([_manifest("papers")], QMD)

    assert violation.rule == "corpus_without_source"
    assert violation.field == "containers.knob.corpora.0"
    assert "declares 'papers' with no `source`" in violation.message


def _repo(tmp_path: Path, corpora: list[str], qmd: dict[str, Any]) -> Path:
    manifest = {"name": "knob", "containers": {"knob": {"runtime": "osprey", "corpora": corpora}}}
    profile = {
        "name": "site",
        "data": "data",
        "apps": {"knob": {}},
        "config": {"services": {"qmd": qmd}},
    }
    (tmp_path / "apps" / "knob").mkdir(parents=True)
    (tmp_path / "apps" / "knob" / "app.yml").write_text(yaml.safe_dump(manifest))
    (tmp_path / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False))
    return tmp_path


@pytest.mark.usefixtures("facility_data_tree")
def test_a_profile_that_declares_the_corpus_loads(tmp_path: Path) -> None:
    repo = _repo(tmp_path, ["procedures"], QMD)

    load_profile_document(repo / "profile.yml")


@pytest.mark.usefixtures("facility_data_tree")
@pytest.mark.parametrize(
    ("name", "rule"), [("missing", "corpus_unknown"), ("papers", "corpus_without_source")]
)
def test_validate_refuses_a_corpus_the_deployment_cannot_serve(
    tmp_path: Path, runner, name: str, rule: str
) -> None:
    from osprey.cli.validate_cmd import validate

    repo = _repo(tmp_path, [name], QMD)

    result = runner.invoke(validate, [str(repo / "profile.yml")])
    with pytest.raises(AppsRefusedError) as excinfo:
        load_profile_document(repo / "profile.yml")

    assert [v.rule for v in excinfo.value.violations] == [rule]
    assert result.exit_code == 2, result.output
    assert "apps/knob/app.yml: containers.knob.corpora.0: " in result.output
    assert f"[{rule}]" in result.output
