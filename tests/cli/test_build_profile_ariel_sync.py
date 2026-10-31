"""End-to-end ``osprey build`` coverage for the bundled ``ariel_sync`` service.

Every other test of this feature holds one piece of it still: the template
suite renders the packaged ``.j2`` against hand-built contexts, and
``test_ariel_ingestion_advisory.py`` calls the advisory function on a config
dict it wrote itself. Neither can say whether a profile an operator actually
writes reaches either one — the service is selected by a ``services:`` entry in
the profile, resolved by the injector, registered in ``deployed_services``, and
only then rendered into ``build/services/ariel_sync/docker-compose.yml`` by a
later pass. This module runs the real build through ``CliRunner`` and asserts on
what it leaves on disk and prints.

Three cases, one build each:

1. The service declared — the compose file lands, carries ``--watch``, and the
   service is registered. The advisory stays silent, because the deployment now
   ingests the source it names.
2. The same remote source with no service — exactly one advisory, naming the
   source and the entry to add. Advisory only: the build still succeeds.
3. The control-assistant preset as shipped — its logbook is seeded from the
   build, which needs no mirroring service, so nothing is said.

Builds run ``--skip-deps --skip-lifecycle``: the virtualenv install and the
profile's shell phases render nothing this module reads and cost minutes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from click.testing import CliRunner

from osprey.cli.init_cmd import init
from tests.cli._lifecycle_build import invoke_build, rendered_config

#: A logbook on another host — the case that needs something polling it.
REMOTE_SOURCE = "https://logbook.example.org/api/entries"

#: The bundled preset that deploys ARIEL; every case starts from its init.
_PRESET = "control-assistant"

#: The profile's selection of the framework's own bundled service template —
#: the spelling the build advisory tells an operator to add.
_ARIEL_SYNC_SERVICE: dict[str, Any] = {"template": "osprey.ariel_sync"}


def _build(
    runner: CliRunner,
    tmp_path: Path,
    *,
    source_url: str | None = None,
    sync_service: bool = False,
) -> tuple[str, Path]:
    """Run a real ``osprey build`` on a freshly initialised ARIEL deployment repo.

    Args:
        runner: The Click runner to invoke the commands through.
        tmp_path: The test's temporary directory; the repo is made under it.
        source_url: A remote logbook to point ``ariel.ingestion.source_url`` at,
            or ``None`` to build the preset as shipped.
        sync_service: Whether the profile selects the ``ariel_sync`` service.

    Returns:
        The build's captured output and the repo root it built.
    """
    repo = tmp_path / "ariel-sync-fixture"
    result = runner.invoke(init, [str(repo), "--preset", _PRESET, "--no-git"])
    assert result.exit_code == 0, result.output

    profile_path = repo / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if source_url is not None:
        profile.setdefault("config", {})["ariel.ingestion.source_url"] = source_url
    if sync_service:
        profile.setdefault("services", {})["ariel_sync"] = dict(_ARIEL_SYNC_SERVICE)
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    result = invoke_build(repo)

    assert result.exit_code == 0, result.output
    return result.output, repo


def _one_line(output: str) -> str:
    """Collapse *output* to single-spaced text, so wrapping cannot hide a match.

    The advisory is printed through Rich, which wraps to the console width. A
    substring assertion against the raw capture would then depend on how wide
    the terminal running the suite happens to be.
    """
    return re.sub(r"\s+", " ", output)


def test_declared_service_renders_its_compose_and_registers(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A ``services: ariel_sync:`` entry becomes a rendered, registered mirror.

    The compose file is what ``osprey up`` runs, and ``deployed_services`` is
    what every later pass reads the stack's membership from — a service present
    in one and missing from the other is deployed by nobody or dialed by
    nothing.
    """
    output, repo = _build(runner, tmp_path, source_url=REMOTE_SOURCE, sync_service=True)

    compose_path = repo / "build" / "services" / "ariel_sync" / "docker-compose.yml"
    assert compose_path.is_file(), "the build rendered no compose file for the mirror"
    document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    assert "ariel-sync" in document["services"]
    assert "--watch" in document["services"]["ariel-sync"]["command"], (
        "the mirror must run as a daemon, not a one-shot sync"
    )
    assert "ariel_sync" in rendered_config(repo)["deployed_services"]
    # The deployment now ingests the source it names, so there is nothing to say.
    assert "osprey.ariel_sync" not in output


def test_remote_source_without_the_service_is_advised_once(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The same profile minus the service names the gap, exactly once.

    One line, not one per render: the build renders the deployment plus a
    project image copy, and the advisory belongs to the single render that has
    ``deployed_services`` in hand.
    """
    output, repo = _build(runner, tmp_path, source_url=REMOTE_SOURCE)

    assert output.count("osprey.ariel_sync") == 1
    advisory = _one_line(output)
    assert REMOTE_SOURCE in advisory
    assert "template: osprey.ariel_sync" in advisory
    # Advisory, not a refusal: the deployment is still built and registered.
    assert "ariel_sync" not in rendered_config(repo)["deployed_services"]


def test_shipped_ariel_profile_is_silent(runner: CliRunner, tmp_path: Path) -> None:
    """The bundled ARIEL deployment seeds its demo logbook from its build, so it needs no mirror.

    This is the case that keeps the advisory from firing on every ARIEL
    deployment the framework ships.
    """
    output, repo = _build(runner, tmp_path)

    assert "osprey.ariel_sync" not in output
    ariel = rendered_config(repo)["ariel"]
    assert "ingestion" not in ariel, "the shipped deployment ingests nothing over the network"
