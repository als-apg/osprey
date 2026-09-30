"""The simulator view: what every render carries under ``data/simulator/``.

``render_facility_outputs`` writes ``served_models.json`` (the models the
render's ``simulation.models`` serves, texture last), ``addresses.json`` (every
channel address of the facility file and one status address per served physics
model) and ``decks/<model>.json`` (a byte copy of each deck-bearing model's
deck, served or not) into each render's ``data/simulator/``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.facility import TEXTURE
from osprey.facility.render import FACILITY_FILE, render_facility_outputs
from osprey.utils.workspace import BUILD_DIR_NAME

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

# xdist_group("built_control_assistant"): every module reading the session's one
# control-assistant build shares a worker, so the build runs once per run.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("built_control_assistant")]

SERVED = "data/simulator/served_models.json"
ADDRESSES = "data/simulator/addresses.json"
DECK = "data/simulator/decks/SR.json"


def _render(
    tmp_path: Path, built: BuiltProject, config: dict[str, Any], name: str = "render"
) -> Path:
    render_dir = tmp_path / name
    render_dir.mkdir()
    render_facility_outputs(render_dir, built.facility, config, built.facility_dir)
    return render_dir


def test_every_render_writes_the_three_files(built_control_assistant: BuiltProject) -> None:
    assert built_control_assistant.outputs
    for outputs in built_control_assistant.outputs:
        assert sorted(outputs.files) == sorted([FACILITY_FILE, ADDRESSES, DECK, SERVED])


def test_addresses_are_the_facility_file_channels_and_the_served_status(
    built_control_assistant: BuiltProject,
) -> None:
    addresses = json.loads((built_control_assistant.build_dir / ADDRESSES).read_bytes())
    facility = built_control_assistant.facility

    assert addresses["schema"] == "osprey.facility.addresses/1"
    assert addresses["channels"] == sorted(channel["id"] for channel in facility["channels"])
    assert addresses["status"] == ["ca:SIM:SR:STATUS"]


def test_the_demo_serves_sr_then_texture(built_control_assistant: BuiltProject) -> None:
    served = json.loads((built_control_assistant.build_dir / SERVED).read_bytes())

    assert served == {"schema": "osprey.facility.served_models/1", "models": ["SR", TEXTURE]}


def test_the_deck_is_a_byte_copy_that_pyat_loads(built_control_assistant: BuiltProject) -> None:
    at = pytest.importorskip("at")
    copy = built_control_assistant.build_dir / DECK
    source = built_control_assistant.facility_dir / "decks" / "SR.json"

    assert hashlib.sha256(copy.read_bytes()).hexdigest() == (
        hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert len(at.load_lattice(str(copy))) == 802


@pytest.mark.parametrize("models", [[], [TEXTURE]])
def test_no_physics_serves_texture_alone_and_keeps_the_deck(
    tmp_path: Path, built_control_assistant: BuiltProject, models: list[str]
) -> None:
    render_dir = _render(tmp_path, built_control_assistant, {"simulation": {"models": models}})

    served = json.loads((render_dir / SERVED).read_bytes())
    addresses = json.loads((render_dir / ADDRESSES).read_bytes())
    assert served["models"] == [TEXTURE]
    assert addresses["status"] == []
    assert (render_dir / DECK).read_bytes() == (
        built_control_assistant.build_dir / DECK
    ).read_bytes()


def test_render_writes_are_sorted_and_deterministic(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    first = _render(tmp_path, built_control_assistant, {}, "first")
    second = _render(tmp_path, built_control_assistant, {}, "second")

    for relative in (SERVED, ADDRESSES, DECK):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
        assert (first / relative).read_bytes() == (
            built_control_assistant.build_dir / relative
        ).read_bytes()


def test_two_mock_personas_differing_only_in_the_key_render_different_lists(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    from osprey.cli.build_profile_archiver import _expand_dotted
    from osprey.cli.build_profile_load import load_profile_document

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "profile.yml").write_text(
        "name: demo\ndata: data\nconfig:\n  control_system.type: mock\n  simulation.models: null\n"
    )
    (repo / "personas").mkdir()
    for name, models in (("physics", "[SR]"), ("plain", "[texture]")):
        (repo / "personas" / f"{name}.yml").write_text(
            f"name: {name}\nconfig:\n  simulation.models: {models}\n"
        )

    rendered = {}
    for name in ("physics", "plain"):
        document = load_profile_document(repo / "personas" / f"{name}.yml")
        config = _expand_dotted(document.profile.config)
        assert config["control_system"]["type"] == "mock"
        render_dir = _render(tmp_path, built_control_assistant, config, name)
        rendered[name] = (render_dir / SERVED).read_bytes()

    assert json.loads(rendered["physics"])["models"] == ["SR", TEXTURE]
    assert json.loads(rendered["plain"])["models"] == [TEXTURE]
    assert rendered["physics"] != rendered["plain"]


def test_an_omitted_view_is_named_on_stderr_alone(
    tmp_path: Path,
    built_control_assistant: BuiltProject,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osprey.facility import views

    stub = views.View(
        name="stub",
        path="stub",
        written_when=lambda _inputs: False,
        reason="stub.enabled",
        write=lambda _root, _inputs: pytest.fail("an omitted view is never written"),
    )
    monkeypatch.setattr(views, "VIEWS", (*views.VIEWS, stub))

    render_dir = _render(tmp_path, built_control_assistant, {})

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "view stub not written: stub.enabled" in captured.err
    assert not (render_dir / "data" / "stub").exists()
    assert (render_dir / SERVED).is_file()


def test_validate_names_an_unknown_served_model(
    tmp_path: Path, built_control_assistant: BuiltProject
) -> None:
    from osprey.cli.main import cli

    source = built_control_assistant.repo
    repo = tmp_path / source.name
    shutil.copytree(
        source,
        repo,
        symlinks=True,
        ignore=lambda folder, _names: [BUILD_DIR_NAME] if Path(folder) == source else [],
    )
    profile = repo / "profile.yml"
    document = yaml.safe_load(profile.read_text(encoding="utf-8"))
    document.setdefault("config", {})["simulation.models"] = ["NOPE"]
    profile.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(cli, ["facility", "validate", "--repo", str(repo)])

    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr.startswith("facility: profile-invalid: path simulation.models — ")
    assert "`NOPE`" in result.stderr
