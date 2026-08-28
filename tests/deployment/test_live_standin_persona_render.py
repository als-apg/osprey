"""What a persona login is told about the machine it is pointed at.

A multi-user deployment hands each operator a persona project, not the
deployment's own render, and a persona's ``services:`` block is empty except
for the keys its reach contract projects into it. That is the setting in which
the stand-in's label has to survive: the roster line an operator reads is
derived from the config THEIR container holds, so if the evidence that a
stand-in exists were withheld from persona renders, the same machine would be
called ``LIVE MACHINE (stand-in)`` in a single-user session and ``LIVE MACHINE``
through a login — and one of those two operators would be wrong about where
they are.

So ``services.live_standin.port`` is projected ungated, alone among the service
ports, and this module builds the exemplar deployment for real to prove that
the projection survives an actual persona render and that the label derived
from it is the stand-in's. The unit behind the label
(``tests/mcp_server/test_control_target_roster.py``) stages its config by hand;
what only a build can show is that the config a persona container will really
hold contains what that unit needs.

Two things are asserted together on purpose. The persona's ``services`` block
must stay projected — no ``path``, no deployed-service list, nothing that would
let a persona try to run the deployment's containers — *and* the label must
still come out right from that reduced material. Either one alone would be
satisfiable by breaking the other.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build as build_command
from osprey.mcp_server.control_system.connector_host_manager import target_display_metadata
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

#: Every test here renders a deployment and its personas for real.
pytestmark = pytest.mark.slow

CI_FLAGS = ["--skip-deps", "--skip-lifecycle"]

STANDIN_PORT = 5074

#: The exemplar's two web-terminal personas. Both are checked because the label
#: describes the MACHINE, not the tier: a readonly and a read-write login are
#: pointed at the same endpoint and must be told the same thing about it.
PERSONAS = ("readonly", "readwrite")

STANDIN_LABEL = "LIVE MACHINE (stand-in)"
LIVE_LABEL = "LIVE MACHINE"


def _build_exemplar(dest: Path, *, standin: int | None) -> Path:
    """A seeded exemplar repo, stand-in key set or removed, built for real."""
    from ruamel.yaml import YAML

    repo = build_exemplar_repo(dest / EXEMPLAR_DIRNAME, seed_env=True)
    ruamel = YAML(typ="rt")
    profile_path = repo / "profile.yml"
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = ruamel.load(handle)
    block = profile["virtual_accelerator"]
    if standin is None:
        block.pop("live_standin", None)
    else:
        block["live_standin"] = standin
    with profile_path.open("w", encoding="utf-8") as handle:
        ruamel.dump(profile, handle)

    previous = Path.cwd()
    os.chdir(repo)
    try:
        result = CliRunner().invoke(build_command, CI_FLAGS)
    finally:
        os.chdir(previous)
    assert result.exit_code == 0, result.output
    return repo / "build"


@pytest.fixture(scope="module")
def standin_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_exemplar(tmp_path_factory.mktemp("persona-standin"), standin=STANDIN_PORT)


@pytest.fixture(scope="module")
def plain_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_exemplar(tmp_path_factory.mktemp("persona-plain"), standin=None)


def _persona_config(build: Path, persona: str) -> dict[str, Any]:
    path = build / f"{EXEMPLAR_DIRNAME}-{persona}" / "config.yml"
    assert path.is_file(), f"no render for persona {persona!r} at {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("persona", PERSONAS)
def test_live_standin_persona_render_projects_the_port_and_nothing_else(
    standin_build, persona: str
) -> None:
    """The stand-in reaches a persona as one key: where to dial it.

    ``path`` is the deployment's own business — a persona runs no containers
    and resolves no compose file — so its absence is what makes this a
    projection rather than a copy of the deployment's service block.
    """
    config = _persona_config(standin_build, persona)

    assert config["services"]["live_standin"] == {"port": STANDIN_PORT}
    assert not config.get("deployed_services")
    for name, block in config["services"].items():
        assert "path" not in block, f"{name}: {block}"


@pytest.mark.parametrize("persona", PERSONAS)
def test_live_standin_persona_render_labels_the_live_target_a_stand_in(
    standin_build, persona: str
) -> None:
    """The roster says stand-in, from the persona's own reduced config.

    ``real_machine`` stays true, deliberately: the stand-in IS this
    deployment's live target, so every strict limit, approval prompt and banner
    the real machine gets, it gets. Only the name on the label moves — which is
    why a reader that keys off ``real_machine`` will call it real and must take
    its wording from the label instead.
    """
    metadata = target_display_metadata(_persona_config(standin_build, persona))["live"]

    assert metadata["label"] == STANDIN_LABEL
    assert metadata["real_machine"] is True
    assert metadata["endpoint"] == f"localhost:{STANDIN_PORT}"


def test_live_standin_persona_render_tells_every_persona_the_same_thing(
    standin_build,
) -> None:
    """One machine, one label, whatever the login is allowed to do to it."""
    labels = {
        persona: target_display_metadata(_persona_config(standin_build, persona))["live"]["label"]
        for persona in PERSONAS
    }

    assert set(labels.values()) == {STANDIN_LABEL}, labels


@pytest.mark.parametrize("persona", PERSONAS)
def test_live_standin_persona_render_without_a_stand_in_says_live_machine(
    plain_build, persona: str
) -> None:
    """No stand-in built, no stand-in claimed — the honest default.

    The off-state is the half that keeps the label meaningful: a predicate that
    said "stand-in" whenever a persona's endpoint happened to be loopback would
    call an SSH-tunnelled real gateway a rehearsal, which is the one mistake
    this label exists to prevent.
    """
    config = _persona_config(plain_build, persona)
    assert "live_standin" not in config["services"]

    metadata = target_display_metadata(config)["live"]
    assert metadata["label"] == LIVE_LABEL
    assert metadata["real_machine"] is True
