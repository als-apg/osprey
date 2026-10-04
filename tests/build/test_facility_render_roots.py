"""``osprey build`` writes the facility file at the root of every render.

The build makes the facility file once, from the profile's ``data/facility/``,
before the project venv and before any render; each render then receives it
through ``render_facility_outputs``: the deployment's own (``build/``), each
persona's (``build/<repo>-<persona>/``) and each container image's
(``build/.image/<name>/build/``). Every copy is the same bytes, equal sources
give equal bytes in a second build, and a facility stop ends the build before
the venv is created.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.facility import TEXTURE
from osprey.facility.render import FACILITY_FILE, facility_digest
from osprey.utils.workspace import BUILD_DIR_NAME, IMAGE_DIR_NAME
from tests._builds import BuiltProject, init_project

pytestmark = [pytest.mark.slow]

#: One repo name for both builds: an identity without a ``name`` takes it.
REPO = "demo"


def _build(repo: Path, *extra: str) -> Any:
    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-lifecycle", *extra])


def _render_roots(repo: Path) -> list[Path]:
    """The deployment's render, each persona's and each container image's."""
    build_dir = repo / BUILD_DIR_NAME
    personas = sorted(build_dir.glob(f"{REPO}-*/config.yml"))
    images = sorted((build_dir / IMAGE_DIR_NAME).glob(f"*/{BUILD_DIR_NAME}/config.yml"))
    return [build_dir, *(path.parent for path in personas), *(path.parent for path in images)]


def test_every_render_root_carries_a_byte_equal_facility_file(
    built_control_assistant: BuiltProject,
) -> None:
    roots = _render_roots(built_control_assistant.repo)
    build_dir = built_control_assistant.repo / BUILD_DIR_NAME
    assert any(root.parent == build_dir for root in roots)
    assert any(IMAGE_DIR_NAME in root.parts for root in roots)
    assert (build_dir / IMAGE_DIR_NAME / REPO / BUILD_DIR_NAME) in roots

    main = (build_dir / FACILITY_FILE).read_bytes()
    assert {root: (root / FACILITY_FILE).read_bytes() == main for root in roots} == dict.fromkeys(
        roots, True
    )
    assert sorted(build_dir.rglob(FACILITY_FILE)) == sorted(root / FACILITY_FILE for root in roots)
    assert not list(build_dir.rglob(f"data/{FACILITY_FILE}"))


def test_the_facility_file_is_written_only_through_render_facility_outputs(
    built_control_assistant: BuiltProject,
) -> None:
    outputs = built_control_assistant.outputs
    assert len(outputs) == len(_render_roots(built_control_assistant.repo))
    # The Bluesky view rides only on a render that runs a Bluesky lane.
    for render in outputs:
        assert [path for path in render.files if path != "data/bluesky_devices.yml"] == [
            "data/channel_limits.json",
            "data/facility_facts.json",
            "data/facility_facts.md",
            "data/graph/facility.ttl",
            "data/simulator/addresses.json",
            "data/simulator/decks/SR.json",
            "data/simulator/scenarios.json",
            "data/simulator/seeds.json",
            "data/simulator/served_models.json",
            "data/simulator/variables.json",
            FACILITY_FILE,
        ]


def test_the_facility_file_is_the_demo_facility_without_build_facts(
    built_control_assistant: BuiltProject,
) -> None:
    raw = built_control_assistant.facility_raw
    document = json.loads(raw)

    assert document["schema"] == "osprey.facility.facility/1"
    assert document["identity"]["code"] == "ca"
    assert document["models"][-1] == {"name": TEXTURE, "engine": TEXTURE}
    assert raw.endswith(b"}\n")
    assert str(built_control_assistant.repo).encode() not in raw
    assert str(built_control_assistant.repo.resolve()).encode() not in raw
    assert {"version", "timestamp", "built_at", "generated_at"}.isdisjoint(document)


def test_equal_sources_hash_equal_across_two_builds(
    built_control_assistant: BuiltProject, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A second build of the same ``data/facility/`` elsewhere gives the same bytes."""
    other = init_project(tmp_path_factory.mktemp("second"), "hello-world", REPO)
    shutil.rmtree(other / "data" / "facility")
    shutil.copytree(built_control_assistant.repo / "data" / "facility", other / "data" / "facility")

    result = _build(other, "--skip-deps")
    assert result.exit_code == 0, result.output

    assert facility_digest(other / "data" / "facility") == facility_digest(
        built_control_assistant.repo / "data" / "facility"
    )
    assert (other / BUILD_DIR_NAME / FACILITY_FILE).read_bytes() == (
        built_control_assistant.repo / BUILD_DIR_NAME / FACILITY_FILE
    ).read_bytes()


def test_a_facility_stop_ends_the_build_before_the_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one build here without ``--skip-deps``: the venv step is a spy that must not run."""
    from osprey.cli import build_cmd

    def venv_spy(*args: Any, **kwargs: Any) -> list[str]:
        raise AssertionError("the project venv was created before the facility stop")

    monkeypatch.setattr(build_cmd, "_create_project_venv", venv_spy)
    repo = init_project(tmp_path, "hello-world", REPO)
    (repo / "data" / "facility" / "identity.yaml").write_text("code: 1bad\n", encoding="utf-8")

    result = _build(repo)

    assert result.exit_code == 1, result.output
    assert result.stderr == (
        "facility: source-invalid: path identity.yaml.code — code '1bad' does not match "
        "[A-Za-z_][A-Za-z0-9_]*; fix: rename the code\n"
    )
    assert not (repo / BUILD_DIR_NAME / "config.yml").exists()


def test_equal_trees_hash_equal_and_a_changed_byte_does_not(tmp_path: Path) -> None:
    first = tmp_path / "a" / "facility"
    second = tmp_path / "b" / "facility"
    for root in (first, second):
        (root / "records").mkdir(parents=True)
        (root / "identity.yaml").write_text("code: lab\n", encoding="utf-8")
        (root / "records" / "sr.yaml").write_text("channels: []\n", encoding="utf-8")

    assert facility_digest(first) == facility_digest(second)
    (second / "records" / "sr.yaml").write_text("channels: [] \n", encoding="utf-8")
    assert facility_digest(first) != facility_digest(second)
    (tmp_path / "empty").mkdir()
    assert facility_digest(tmp_path / "absent") == facility_digest(tmp_path / "empty")
