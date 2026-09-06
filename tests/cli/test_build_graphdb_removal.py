"""Running no graph store is spelled by deleting the keys (issue #714).

The graph store used to be *injected*: the control-assistant app template put a
``services.graphdb`` block, two published host ports and a ``graphdb`` entry in
``deployed_services`` into every render, whether or not the deployment wanted
one. Opting out therefore meant subtracting after the fact — the bare
``services.graphdb:`` (YAML ``null``) whole-block override, finished off by a
build-time step that deleted the null key, withdrew the ``deployed_services``
entry and removed the copied service-template directory.

Nothing injects a store any more. ``profile.yml`` carries the whole declarative
input, so a deployment that runs no graph store simply does not spell one: no
``services.graphdb.*`` keys, no ``graphdb`` in ``deployed_services``. There is
nothing left to subtract, and the subtraction step is gone with it.

These tests build real projects from the ``control-assistant`` preset — the one
that does ship the store — and pin the three halves of the new contract:

* **Deleting the keys removes the store.** The render carries no
  ``services.graphdb`` block, no ``graphdb`` in ``deployed_services``, no
  compose fragment, and no ``graph`` MCP server. A sibling baseline build off
  the untouched profile shows all four are really there to remove, so the
  removal assertions cannot pass vacuously.
* **The mode still needs a store.** ``channel_finder_mode: graph`` on a profile
  with no block is refused by name at validation, before anything renders. That
  refusal is the ``_deploys_slot`` predicate's, asserted here rather than
  re-implemented.
* **A whole-block override is refused.** Neither ``services.graphdb:`` (bare
  null) nor ``services.graphdb: {}`` removes anything; both replace the block
  with something no resolver can read. Each is refused at validation with a
  message that names the working spelling — delete the keys.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.build_cmd import build
from osprey.cli.build_profile import load_profile
from osprey.cli.init_cmd import init
from osprey.errors import BuildProfileError

#: The preset that ships the graph store. Every project here starts from it.
PRESET = "control-assistant"

#: Non-graph paradigm, so the mode never supplies the store the tests remove.
#: The graph mode gets its own project, in :class:`TestGraphModeNeedsAStore`.
MODE = "hierarchical"

BUILD_FLAGS = ["--skip-deps", "--skip-lifecycle"]

#: Dotted-key prefix an operator deletes to run no store, and the
#: ``deployed_services`` member that goes with it.
GRAPHDB_KEY_PREFIX = "  services.graphdb."
GRAPHDB_DEPLOYED_ENTRY = "    - graphdb"


def _init_project(root: Path, name: str, mode: str) -> Path:
    """Materialize an explicit profile from the preset, as an operator would.

    Args:
        root: Directory to create the project directory in.
        name: Project directory name.
        mode: ``channel_finder_mode`` to init with.

    Returns:
        Path of the created project directory.
    """
    project = root / name
    result = CliRunner().invoke(
        init,
        [str(project), "--preset", PRESET, "--no-git", "--set", f"channel_finder_mode={mode}"],
    )
    assert result.exit_code == 0, result.output
    assert (project / "profile.yml").exists()
    return project


def _delete_graphdb_keys(profile_path: Path) -> None:
    """Delete every ``services.graphdb.*`` key and the ``deployed_services`` entry.

    A line-level edit of the emitted ``profile.yml``, which is what deleting the
    keys in an editor amounts to. Asserts that it removed something: a strip
    that silently matched nothing would leave the removal tests passing
    vacuously against a profile that still spells the store.

    Args:
        profile_path: The project's ``profile.yml``.
    """
    lines = profile_path.read_text().splitlines(keepends=True)
    kept = [
        line
        for line in lines
        if not line.startswith(GRAPHDB_KEY_PREFIX) and line.rstrip("\n") != GRAPHDB_DEPLOYED_ENTRY
    ]
    removed = len(lines) - len(kept)
    assert removed > 1, (
        f"expected the store's keys and its deployed entry, stripped {removed} line(s)"
    )
    profile_path.write_text("".join(kept))


def _build(project: Path, monkeypatch) -> Result:
    """Run ``osprey build`` in *project* and return the click result."""
    monkeypatch.chdir(project)
    return CliRunner().invoke(build, BUILD_FLAGS)


def _rendered_config(project: Path) -> dict:
    return yaml.safe_load((project / "build" / "config.yml").read_text())


def _mcp_servers(project: Path) -> dict:
    return json.loads((project / "build" / ".mcp.json").read_text()).get("mcpServers", {})


# ─────────────────────────────────────────────────────────────────────────────
# The two renders the removal is read off
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def baseline_render(tmp_path_factory, monkeypatch_module) -> Path:
    """The preset built untouched — the store this feature lets a profile omit."""
    project = _init_project(tmp_path_factory.mktemp("baseline"), "baseline", MODE)
    result = _build(project, monkeypatch_module)
    assert result.exit_code == 0, result.output
    return project


@pytest.fixture(scope="module")
def storeless_render(tmp_path_factory, monkeypatch_module) -> tuple[Path, Result]:
    """The same preset with the store's keys deleted from ``profile.yml``."""
    project = _init_project(tmp_path_factory.mktemp("storeless"), "storeless", MODE)
    _delete_graphdb_keys(project / "profile.yml")
    return project, _build(project, monkeypatch_module)


@pytest.fixture(scope="module")
def monkeypatch_module():
    """A module-scoped ``monkeypatch``, for the module-scoped builds above."""
    patcher = pytest.MonkeyPatch()
    yield patcher
    patcher.undo()


class TestBaselineShipsTheStore:
    """Guard against the removal tests passing vacuously: the untouched preset
    must really render everything the removal then takes away."""

    def test_render_carries_every_trace_of_the_store(self, baseline_render: Path):
        config = _rendered_config(baseline_render)
        assert "graphdb" in config["services"]
        assert "graphdb" in [str(s) for s in config["deployed_services"]]
        assert (baseline_render / "build" / "services" / "graphdb").is_dir()
        assert "graph" in _mcp_servers(baseline_render)


class TestDeletingTheKeysRemovesTheStore:
    """A root profile that spells no store renders none, anywhere."""

    def test_build_succeeds(self, storeless_render: tuple[Path, Result]):
        _, result = storeless_render
        assert result.exit_code == 0, result.output

    def test_rendered_config_has_no_graphdb_block(self, storeless_render: tuple[Path, Result]):
        project, result = storeless_render
        assert result.exit_code == 0, result.output
        assert "graphdb" not in (_rendered_config(project).get("services") or {})

    def test_graphdb_is_not_in_deployed_services(self, storeless_render: tuple[Path, Result]):
        project, result = storeless_render
        assert result.exit_code == 0, result.output
        deployed = _rendered_config(project).get("deployed_services") or []
        assert "graphdb" not in [str(s) for s in deployed]

    def test_no_compose_fragment_is_bundled(self, storeless_render: tuple[Path, Result]):
        project, result = storeless_render
        assert result.exit_code == 0, result.output
        assert not (project / "build" / "services" / "graphdb").exists()

    def test_no_graph_mcp_server_is_rendered(self, storeless_render: tuple[Path, Result]):
        project, result = storeless_render
        assert result.exit_code == 0, result.output
        assert "graph" not in _mcp_servers(project)

    def test_the_rest_of_the_render_is_untouched(
        self, storeless_render: tuple[Path, Result], baseline_render: Path
    ):
        """Deleting the store's keys removes the store and nothing else."""
        project, result = storeless_render
        assert result.exit_code == 0, result.output
        storeless = _rendered_config(project)
        baseline = _rendered_config(baseline_render)
        assert set(baseline["services"]) - set(storeless["services"]) == {"graphdb"}
        assert [s for s in baseline["deployed_services"] if s != "graphdb"] == list(
            storeless["deployed_services"]
        )
        assert set(_mcp_servers(baseline_render)) - set(_mcp_servers(project)) == {"graph"}


class TestGraphModeNeedsAStore:
    """``channel_finder_mode: graph`` with no block is refused by name.

    The refusal is the deploys-slot predicate's, in ``BuildProfile.validate``;
    asserted here on the profile the operator actually wrote rather than
    re-derived, so the omission spelling and the mode's prerequisite are pinned
    against each other.
    """

    def test_profile_validation_names_the_mode_and_the_block(self, tmp_path: Path):
        project = _init_project(tmp_path, "graphmode", "graph")
        _delete_graphdb_keys(project / "profile.yml")
        with pytest.raises(BuildProfileError) as excinfo:
            load_profile(project / "profile.yml")
        message = str(excinfo.value)
        assert "channel_finder_mode: graph" in message
        assert "services.graphdb" in message

    def test_build_refuses_before_rendering(self, tmp_path: Path, monkeypatch, caplog):
        project = _init_project(tmp_path, "graphmode", "graph")
        _delete_graphdb_keys(project / "profile.yml")
        with caplog.at_level(logging.ERROR):
            result = _build(project, monkeypatch)
        assert result.exit_code != 0
        reported = result.output + caplog.text + str(result.exception or "")
        assert "channel_finder_mode: graph" in reported
        assert "Unexpected error" not in reported
        assert not (project / "build").exists()


class TestWholeBlockOverrideIsRefused:
    """Neither ``services.graphdb:`` nor ``services.graphdb: {}`` is removal.

    Both replace the block with something no resolver downstream can read — the
    null one used to be rescued by a build-time subtraction step that no longer
    exists. Each is refused at validation, and the message hands back the
    spelling that does work.
    """

    @staticmethod
    def _profile_with_whole_block_override(tmp_path: Path, override: str) -> Path:
        project = _init_project(tmp_path, "override", MODE)
        profile = project / "profile.yml"
        _delete_graphdb_keys(profile)
        text = profile.read_text().replace(
            "  deployed_services:", f"  services.graphdb:{override}\n  deployed_services:", 1
        )
        assert "services.graphdb:" in text
        profile.write_text(text)
        return profile

    @pytest.mark.parametrize("override", ["", " {}"], ids=["bare-null", "empty-mapping"])
    def test_refusal_says_to_delete_the_keys(self, tmp_path: Path, override: str):
        profile = self._profile_with_whole_block_override(tmp_path, override)
        with pytest.raises(BuildProfileError) as excinfo:
            load_profile(profile)
        message = str(excinfo.value)
        assert "services.graphdb" in message
        # The working spelling, in both its halves.
        assert "delete" in message.lower()
        assert "deployed_services" in message

    @pytest.mark.parametrize("override", ["", " {}"], ids=["bare-null", "empty-mapping"])
    def test_build_refuses_before_rendering(
        self, tmp_path: Path, monkeypatch, caplog, override: str
    ):
        profile = self._profile_with_whole_block_override(tmp_path, override)
        with caplog.at_level(logging.ERROR):
            result = _build(profile.parent, monkeypatch)
        assert result.exit_code != 0
        reported = result.output + caplog.text + str(result.exception or "")
        assert "services.graphdb" in reported
        # A named refusal, not the resolver crash the null spelling dies with
        # once nothing subtracts it.
        assert "Unexpected error" not in reported
        assert "NoneType" not in reported
        assert not (profile.parent / "build").exists()
