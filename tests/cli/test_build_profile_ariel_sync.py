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
3. The standalone ARIEL app template as shipped — its seed source is a path on
   disk, which needs no mirroring service, so nothing is said.

Builds run ``--skip-deps --skip-lifecycle``: the virtualenv install and the
profile's shell phases render nothing this module reads and cost minutes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build

#: A logbook on another host — the case that needs something polling it.
REMOTE_SOURCE = "https://logbook.example.org/api/entries"

#: A deploying ARIEL profile pointed at a remote logbook. The ``services:``
#: block is substituted per case, so the two remote-source builds below differ
#: in exactly the one thing under test.
_PROFILE = """\
extends: ariel-standalone
name: Ariel Sync Fixture
{services}config:
  ariel.ingestion.source_url: {source_url}
"""

#: The profile's selection of the framework's own bundled service template —
#: the spelling the build advisory tells an operator to add.
_ARIEL_SYNC_SERVICE = """\
services:
  ariel_sync:
    template: osprey.ariel_sync
"""

#: The shipped standalone ARIEL preset with nothing laid over it. Its app
#: template seeds ``ariel.ingestion.source_url`` with a path under ``data/``.
_STANDALONE_PROFILE = """\
extends: ariel-standalone
name: Ariel Standalone Fixture
"""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _build(runner: CliRunner, tmp_path: Path, profile: str) -> tuple[str, Path]:
    """Run a real ``osprey build`` on *profile* in a fresh deployment repo.

    Args:
        runner: The Click runner to invoke the command through.
        tmp_path: The test's temporary directory; the repo is made under it.
        profile: Contents of the repo's ``profile.yml``.

    Returns:
        The build's captured output and the repo root it built.
    """
    repo = tmp_path / "ariel-sync-fixture"
    repo.mkdir()
    (repo / "profile.yml").write_text(profile, encoding="utf-8")

    result = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])

    assert result.exit_code == 0, result.output
    return result.output, repo


def _rendered_config(repo: Path) -> dict[str, Any]:
    """The parsed ``build/config.yml`` a build left in *repo*."""
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


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
    output, repo = _build(
        runner,
        tmp_path,
        _PROFILE.format(services=_ARIEL_SYNC_SERVICE, source_url=REMOTE_SOURCE),
    )

    compose_path = repo / "build" / "services" / "ariel_sync" / "docker-compose.yml"
    assert compose_path.is_file(), "the build rendered no compose file for the mirror"
    document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    assert "ariel-sync" in document["services"]
    assert "--watch" in document["services"]["ariel-sync"]["command"], (
        "the mirror must run as a daemon, not a one-shot sync"
    )
    assert "ariel_sync" in _rendered_config(repo)["deployed_services"]
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
    output, repo = _build(runner, tmp_path, _PROFILE.format(services="", source_url=REMOTE_SOURCE))

    assert output.count("osprey.ariel_sync") == 1
    advisory = _one_line(output)
    assert REMOTE_SOURCE in advisory
    assert "template: osprey.ariel_sync" in advisory
    # Advisory, not a refusal: the deployment is still built and registered.
    assert "ariel_sync" not in _rendered_config(repo)["deployed_services"]


def test_shipped_standalone_ariel_profile_is_silent(runner: CliRunner, tmp_path: Path) -> None:
    """The bundled ARIEL app ingests a seed file off disk, so it needs no mirror.

    This is the case that keeps the advisory from firing on every ARIEL
    deployment the framework ships.
    """
    output, repo = _build(runner, tmp_path, _STANDALONE_PROFILE)

    assert "osprey.ariel_sync" not in output
    source_url = _rendered_config(repo)["ariel"]["ingestion"]["source_url"]
    assert not source_url.startswith(("http://", "https://")), (
        "the shipped seed source must stay a path on disk for this case to mean anything"
    )
