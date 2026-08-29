"""What a persona login is told about the machines it can be pointed at.

A multi-user deployment hands each operator a persona project, not the
deployment's own render, and a persona's ``services:`` block is empty except
for the keys its reach contract projects into it. That is the setting the
three-target model has to survive: the roster lines an operator reads are
derived from the config THEIR container holds, so a fact withheld from persona
renders is a target described one way in a single-user session and another way
through a login — and one of those two operators would be wrong about where
they are.

Three projections are asserted here, all on a real build of the exemplar:

* ``services.live_standin.port`` — ungated, alone among the service ports,
  because it is the whole evidence the ``standin`` slot's label is derived
  from. The stand-in is its own target now: the parenthesis belongs to the
  ``standin`` slot, and ``live`` stays plain ``LIVE MACHINE`` — it names the
  machine authored under ``epics:``, which a deployment running a stand-in
  does not rewrite.
* ``services.virtual_accelerator.port`` — the port a persona session that
  switches to ``va`` would dial, gated on the ``va`` target resolving rather
  than on the deployment baseline (SC-9). A stand-in-baselined deployment
  still offers the simulator.
* ``services.archiver_recorder.path`` — not an endpoint at all: the host's
  fact THAT it records its own store, which is what
  ``archive_belongs_to_standin`` reads to decide whose history the archive
  holds, and on that answer the ``live`` target is refused. A persona queries
  that same store, so the gate has to hold in a persona session too.

The unit behind the labels (``tests/mcp_server/test_control_target_roster.py``)
stages its config by hand; what only a build can show is that the config a
persona container will really hold contains what those units need — and that
the render the build hands an operator passes the build's own reach validation.

The persona's ``services`` block must stay a projection throughout — no
deployed-service list, no service directory it could try to run — which is why
the projected keys and the labels derived from them are asserted together:
either half alone would be satisfiable by breaking the other.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build as build_command
from osprey.deployment.reach import reach_errors
from osprey.mcp_server.control_system.connector_host_manager import target_display_metadata
from osprey_connectors.standin import archive_belongs_to_standin
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

#: Every test here renders a deployment and its personas for real.
pytestmark = pytest.mark.slow

CI_FLAGS = ["--skip-deps", "--skip-lifecycle"]

STANDIN_PORT = 5074

#: The exemplar's two web-terminal personas. Both are checked because the
#: labels describe MACHINES, not the tier: a readonly and a read-write login
#: are pointed at the same deployment and must be told the same thing about it.
PERSONAS = ("readonly", "readwrite")

STANDIN_LABEL = "LIVE MACHINE (stand-in)"
LIVE_LABEL = "LIVE MACHINE"


def _build_exemplar(dest: Path, *, standin: int | None) -> Path:
    """A seeded exemplar repo, stand-in key set or removed, built for real.

    Removing the key takes the baseline with it: the exemplar is baselined
    ``control_system.type: live_standin``, and the build refuses that type on
    a deployment that stands no stand-in up — the baseline would name a
    machine nothing serves. So the no-stand-in variant is the same repo
    pointed back at the facility's own machine, which is exactly the edit the
    profile's own comment describes.

    Returns:
        The repo root. ``build/`` is under it, and the repo root is also what
        the build passes to :func:`reach_errors` as ``repo_root``.
    """
    from ruamel.yaml import YAML

    repo = build_exemplar_repo(dest / EXEMPLAR_DIRNAME, seed_env=True)
    ruamel = YAML(typ="rt")
    profile_path = repo / "profile.yml"
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = ruamel.load(handle)
    block = profile["virtual_accelerator"]
    if standin is None:
        block.pop("live_standin", None)
        profile["config"]["control_system.type"] = "epics"
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
    return repo


@pytest.fixture(scope="module")
def standin_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_exemplar(tmp_path_factory.mktemp("persona-standin"), standin=STANDIN_PORT)


@pytest.fixture(scope="module")
def plain_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_exemplar(tmp_path_factory.mktemp("persona-plain"), standin=None)


def _persona_config(repo: Path, persona: str) -> dict[str, Any]:
    path = repo / "build" / f"{EXEMPLAR_DIRNAME}-{persona}" / "config.yml"
    assert path.is_file(), f"no render for persona {persona!r} at {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _host_config(repo: Path) -> dict[str, Any]:
    """The deploying render — the one the projections are copied FROM."""
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("persona", PERSONAS)
def test_live_standin_persona_render_projects_the_port_and_nothing_else(
    standin_repo, persona: str
) -> None:
    """The stand-in reaches a persona as one key: where to dial it.

    ``path`` is the deployment's own business — a persona runs no containers
    and resolves no compose file — so its absence from the stand-in's block is
    what makes this a projection rather than a copy of the deployment's
    service block. The recorder's ``path`` is the one exception in the render
    and is not an endpoint at all; it has its own test below.
    """
    config = _persona_config(standin_repo, persona)

    assert config["services"]["live_standin"] == {"port": STANDIN_PORT}
    assert not config.get("deployed_services")
    for name, block in config["services"].items():
        if name == "archiver_recorder":
            continue
        assert "path" not in block, f"{name}: {block}"


@pytest.mark.parametrize("persona", PERSONAS)
def test_the_standin_slot_is_labelled_a_stand_in(standin_repo, persona: str) -> None:
    """The roster says stand-in on the ``standin`` slot, from the persona's
    own reduced config.

    ``real_machine`` stays true, deliberately: the stand-in is a real
    machine's posture — every strict limit, approval prompt and banner
    hardware gets, it gets. Only the name on the label moves, which is why a
    reader that keys off ``real_machine`` will call it real and must take its
    wording from the label instead.
    """
    metadata = target_display_metadata(_persona_config(standin_repo, persona))["standin"]

    assert metadata["label"] == STANDIN_LABEL
    assert metadata["real_machine"] is True
    assert metadata["endpoint"] == f"localhost:{STANDIN_PORT}"


@pytest.mark.parametrize("persona", PERSONAS)
def test_the_live_slot_is_never_renamed_a_stand_in(standin_repo, persona: str) -> None:
    """``live`` is the machine the facility authored under ``epics:``, on a
    deployment running a stand-in exactly as on one that is not.

    The stand-in's presence must not move the parenthesis onto this slot: the
    direction this stack must never fail in is telling an operator the machine
    in front of them is only a rehearsal. The endpoint pins that the two slots
    really are two different machines here — the authored gateway, not the
    loopback port the stand-in serves on.
    """
    metadata = target_display_metadata(_persona_config(standin_repo, persona))["live"]

    assert metadata["label"] == LIVE_LABEL
    assert metadata["real_machine"] is True
    host, _, port = metadata["endpoint"].rpartition(":")
    assert host not in ("localhost", "127.0.0.1"), metadata["endpoint"]
    assert port != str(STANDIN_PORT), metadata["endpoint"]


def test_live_standin_persona_render_tells_every_persona_the_same_thing(standin_repo) -> None:
    """Two machines, two labels, the same pair whatever the login may do.

    The tier decides which gateway role a session selects, never which machine
    it is standing on — so a readonly and a read-write operator reading their
    rosters side by side see the same two names.
    """
    labels = {
        persona: {
            target: meta["label"]
            for target, meta in target_display_metadata(
                _persona_config(standin_repo, persona)
            ).items()
        }
        for persona in PERSONAS
    }

    assert {frozenset(row.items()) for row in labels.values()} == {
        frozenset(labels[PERSONAS[0]].items())
    }, labels
    assert labels[PERSONAS[0]]["standin"] == STANDIN_LABEL
    assert labels[PERSONAS[0]]["live"] == LIVE_LABEL


@pytest.mark.parametrize("persona", PERSONAS)
def test_the_persona_is_told_the_hosts_virtual_accelerator_port(standin_repo, persona: str) -> None:
    """SC-9: the simulator's port reaches a persona of a stand-in-baselined
    deployment, as the host's own number.

    The projection is gated on the ``va`` target resolving, not on
    ``control_system.type`` — a deployment baselined on the stand-in still
    offers ``va`` to any session that switches to it, and a gate keyed on the
    baseline would withhold the port from exactly these renders. The
    connector's port filler always answers (its compiled-in default), so a
    missing projection would not fail loudly: it would quietly point the
    session at whatever port that default names.
    """
    host_port = _host_config(standin_repo)["services"]["virtual_accelerator"]["port"]
    config = _persona_config(standin_repo, persona)

    assert config["services"]["virtual_accelerator"]["port"] == host_port
    assert target_display_metadata(config)["va"]["endpoint"] == f"localhost:{host_port}"


@pytest.mark.parametrize("persona", PERSONAS)
def test_the_persona_render_is_reachable_as_the_build_reads_it(standin_repo, persona: str) -> None:
    """SC-9's other half: the render an operator is handed passes the build's
    own reach validation.

    Read with the same entry point and the same ``repo_root`` the build passes,
    so a consumer switched on in a persona with nothing to dial is caught here
    rather than at that operator's first tool call.
    """
    assert reach_errors(_persona_config(standin_repo, persona), repo_root=standin_repo) == []


@pytest.mark.parametrize("persona", PERSONAS)
def test_the_recorder_fact_reaches_a_persona_of_a_recording_host(
    standin_repo, persona: str
) -> None:
    """A persona is told THAT its host records its own store, not where.

    ``deployed_services`` is the deploying render's spelling of that fact and
    is empty in every persona render, so the recorder's block is projected
    ungated beside the stand-in's port. Its only reader is
    ``archive_belongs_to_standin``, and on that answer the ``live`` target is
    refused: a real machine's readings spliced onto a stand-in's synthesized
    past is the one thing an archive must never contain. The path is the
    host's, and projecting it grows no surface — a service is rendered and
    started off ``deployed_services``, never off a ``services:`` block.
    """
    host_path = _host_config(standin_repo)["services"]["archiver_recorder"]["path"]
    config = _persona_config(standin_repo, persona)

    assert config["services"]["archiver_recorder"] == {"path": host_path}
    assert not config.get("deployed_services")
    assert archive_belongs_to_standin(config) is True


@pytest.mark.parametrize("persona", PERSONAS)
def test_a_persona_of_a_deployment_with_no_stand_in_is_told_of_none(
    plain_repo, persona: str
) -> None:
    """No stand-in built, no stand-in claimed — the honest default.

    The off-state is the half that keeps the label meaningful: a predicate
    that said "stand-in" whenever an endpoint happened to be loopback would
    call an SSH-tunnelled real gateway a rehearsal, which is the one mistake
    this label exists to prevent. With no port projected there is nothing for
    the ``standin`` slot to dial, and neither slot carries the parenthesis.
    """
    config = _persona_config(plain_repo, persona)
    assert "live_standin" not in config["services"]
    assert not archive_belongs_to_standin(config)

    metadata = target_display_metadata(config)
    assert metadata["live"]["label"] == LIVE_LABEL
    assert metadata["live"]["real_machine"] is True
    assert metadata["standin"]["label"] != STANDIN_LABEL
    assert metadata["standin"]["endpoint"] == ""
