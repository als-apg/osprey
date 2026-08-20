"""A deploy that runs the graph store also makes it answer questions.

Before this, ``osprey up`` would start the Neo4j container and stop there: the
store came up bootstrapped by nobody and empty, so every Cypher query answered
zero rows and the operator's only signal was "the data looks wrong". Seeding
lived behind a verb an operator had to know to run, which a turn-key preset
cannot assume.

Staging the graph store is therefore the knowledge-graph counterpart of
``_stage_ariel_store``: start the store alone, wait for it, bootstrap it, and —
on a first bring-up only — import the configured TTL. Every failure here warns
and names ``osprey knowledge seed-graph``; none of them aborts the deploy, because
a control room whose channels, plans and archive are up should not be denied them
over a search index.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from osprey.deployment import container_lifecycle
from osprey.services.facility_knowledge.seeder import graph_seeder

TTL_TEXT = "@prefix ex: <http://example.org/> .\nex:beam a ex:Device .\n"

#: What a decoy TTL under the repo ROOT (rather than under the render, beside
#: config.yml) would contain — see the anchoring test.
DECOY_TTL_TEXT = "@prefix ex: <http://example.org/> .\nex:wrong a ex:Device .\n"

GRAPHDB_CONFIG: dict = {
    "project_name": "demo",
    "deployed_services": ["graphdb"],
    "services": {
        "graphdb": {
            "port_host": 7687,
            "http_port_host": 7474,
            "ttl_path": "./services/graphdb/als_gtb.ttl",
        }
    },
}


class _FakeSession:
    """Stands in for a neo4j session; the stubbed seeder never reads it."""


@pytest.fixture
def graphdb_stubs(monkeypatch, tmp_path):
    """Reduce ``_stage_graphdb_store`` to its compose call and its seeder calls.

    The project is laid out the way a built one is: the render (and therefore
    ``config.yml`` and the shipped ``services/graphdb/als_gtb.ttl``) one zone
    down in ``build/``, the single ``.env`` at the repo root.
    """
    state: dict[str, Any] = {
        "events": [],
        "cmds": [],
        "sessions": [],
        "imported": [],
        "markers": [],
        "notes": [],
        "resources": 0,
        "bootstrap": graph_seeder.BootstrapResult(status=graph_seeder.BootstrapStatus.INITIALIZED),
        "import_result": graph_seeder.ImportResult(
            termination_status=graph_seeder.TERMINATION_OK, triples_loaded=8114, triples_parsed=8114
        ),
    }

    (tmp_path / ".env").write_text("GRAPHDB_PASSWORD=s3cret\n", encoding="utf-8")
    render = tmp_path / "build"
    (render / "services" / "graphdb").mkdir(parents=True)
    (render / "config.yml").write_text("project_name: demo\n", encoding="utf-8")
    (render / "services" / "graphdb" / "als_gtb.ttl").write_text(TTL_TEXT, encoding="utf-8")

    monkeypatch.setattr(container_lifecycle, "runtime_env", lambda config, env: dict(env))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle, "compose_base_cmd", lambda *a, **k: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "_env_file_args", lambda root=None, provider=None: [])

    def _run(cmd, **kwargs):
        state["events"].append("up")
        state["cmds"].append(list(cmd))

    monkeypatch.setattr(container_lifecycle, "run_captured", _run)
    monkeypatch.setattr(
        container_lifecycle,
        "_wait_for_graphdb_store",
        lambda connection, deadline: state["events"].append("wait"),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_report_fact",
        lambda message, **kwargs: state["notes"].append(message),
    )

    @contextmanager
    def _open_session(uri, username, password, **kwargs):
        state["sessions"].append((uri, username, password))
        yield _FakeSession()

    def _bootstrap(session):
        state["events"].append("bootstrap")
        return state["bootstrap"]

    def _resource_count(session):
        state["events"].append("count")
        return state["resources"]

    def _import_ttl(session, text):
        state["events"].append("import")
        state["imported"].append(text)
        return state["import_result"]

    def _write_marker(session, sha256):
        state["events"].append("marker")
        state["markers"].append(sha256)

    monkeypatch.setattr(graph_seeder, "open_session", _open_session)
    monkeypatch.setattr(graph_seeder, "bootstrap", _bootstrap)
    monkeypatch.setattr(graph_seeder, "resource_count", _resource_count)
    monkeypatch.setattr(graph_seeder, "import_ttl", _import_ttl)
    monkeypatch.setattr(graph_seeder, "write_marker", _write_marker)
    return state


def _stage(config: dict, project_dir: Path) -> None:
    container_lifecycle._stage_graphdb_store(config, ["compose.yml"], {}, project_dir)


def test_the_store_is_started_on_its_own(graphdb_stubs, tmp_path):
    """Started ahead of the stack, like ARIEL's: seeding has to finish before
    anything that queries the graph comes up."""
    # Act
    _stage(GRAPHDB_CONFIG, tmp_path)

    # Assert
    assert graphdb_stubs["cmds"] == [["docker", "compose", "up", "-d", "graphdb"]]


def test_the_staging_runs_its_steps_in_order(graphdb_stubs, tmp_path):
    """Each step depends on the one before: nothing may be imported into a store
    that is not up, not bootstrapped, or not known to be empty -- and the marker
    is written last, because it is what claims the import succeeded."""
    # Act
    _stage(GRAPHDB_CONFIG, tmp_path)

    # Assert
    assert graphdb_stubs["events"] == ["up", "wait", "bootstrap", "count", "import", "marker"]


def test_an_empty_graph_with_a_ttl_path_is_seeded(graphdb_stubs, tmp_path):
    """The defect this staging exists for: a healthy store that answers nothing."""
    # Act
    _stage(GRAPHDB_CONFIG, tmp_path)

    # Assert
    assert graphdb_stubs["imported"] == [TTL_TEXT]
    assert graphdb_stubs["markers"] == [hashlib.sha256(TTL_TEXT.encode("utf-8")).hexdigest()]


def test_a_populated_graph_is_left_alone_and_says_nothing(graphdb_stubs, tmp_path, caplog):
    """The second `osprey up` on the same project. A graph that already carries
    resources is not re-imported, and not complained about either -- this is the
    normal path, not a problem."""
    # Arrange
    graphdb_stubs["resources"] = 8114

    # Act
    with caplog.at_level("WARNING"):
        _stage(GRAPHDB_CONFIG, tmp_path)

    # Assert
    assert graphdb_stubs["imported"] == []
    assert graphdb_stubs["markers"] == []
    assert caplog.text == ""


def test_without_a_ttl_path_the_store_is_bootstrapped_and_noted_not_warned(
    graphdb_stubs, tmp_path, caplog
):
    """A project that means to load its own corpus later is correctly configured,
    not broken: it gets a bootstrapped store and a note, never a warning."""
    # Arrange
    config = {
        **GRAPHDB_CONFIG,
        "services": {"graphdb": {"port_host": 7687}},
    }

    # Act
    with caplog.at_level("WARNING"):
        _stage(config, tmp_path)

    # Assert
    assert graphdb_stubs["events"] == ["up", "wait", "bootstrap", "count"]
    assert any("ttl_path" in note for note in graphdb_stubs["notes"])
    assert caplog.text == ""


def test_a_drifted_graph_config_warns_and_refuses_to_seed(graphdb_stubs, tmp_path, caplog):
    """n10s cannot re-initialize a configured store, so a graph shaped by other
    settings would be imported into under assumptions that do not hold. Say so,
    name the one command that can fix it, and import nothing."""
    # Arrange
    graphdb_stubs["bootstrap"] = graph_seeder.BootstrapResult(
        status=graph_seeder.BootstrapStatus.DIFFERS, differing_keys=("handleMultival",)
    )

    # Act
    with caplog.at_level("WARNING"):
        _stage(GRAPHDB_CONFIG, tmp_path)

    # Assert
    assert graph_seeder.CONFIG_DRIFT_MESSAGE.split(" — ")[0] in caplog.text
    assert "handleMultival" in caplog.text
    assert "osprey knowledge seed-graph --force" in caplog.text
    assert graphdb_stubs["imported"] == []
    assert graphdb_stubs["markers"] == []


def test_a_failed_import_is_never_marked_as_a_good_seed(graphdb_stubs, tmp_path, caplog):
    """n10s commits in batches, so a failed import leaves triples behind. Writing
    the marker anyway would label that half-graph as seeded and every later run
    would skip it forever."""
    # Arrange
    graphdb_stubs["import_result"] = graph_seeder.ImportResult(
        termination_status="KO", triples_loaded=120, triples_parsed=8114, extra_info="bad IRI"
    )

    # Act
    with caplog.at_level("WARNING"):
        _stage(GRAPHDB_CONFIG, tmp_path)

    # Assert
    assert graphdb_stubs["markers"] == []
    assert "osprey knowledge seed-graph" in caplog.text
    assert "bad IRI" in caplog.text


def test_an_unreachable_store_warns_and_leaves_the_deploy_standing(
    graphdb_stubs, tmp_path, monkeypatch, caplog
):
    """A graph store that will not answer must be said out loud, naming the
    command that finishes the job -- but the rest of the deploy is not aborted
    over it."""

    # Arrange
    def _boom(connection, deadline):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(container_lifecycle, "_wait_for_graphdb_store", _boom)

    # Act
    with caplog.at_level("WARNING"):
        _stage(GRAPHDB_CONFIG, tmp_path)

    # Assert
    assert "osprey knowledge seed-graph" in caplog.text
    assert "connection refused" in caplog.text
    assert graphdb_stubs["events"] == ["up"]


def test_a_missing_ttl_file_warns_and_leaves_the_deploy_standing(graphdb_stubs, tmp_path, caplog):
    """A configured corpus that is not on disk is a real problem worth naming --
    and still not a reason to refuse the operator a control room."""
    # Arrange
    (tmp_path / "build" / "services" / "graphdb" / "als_gtb.ttl").unlink()

    # Act
    with caplog.at_level("WARNING"):
        _stage(GRAPHDB_CONFIG, tmp_path)

    # Assert
    assert "osprey knowledge seed-graph" in caplog.text
    assert graphdb_stubs["markers"] == []


def test_the_connection_is_credentialed_from_the_projects_own_dotenv(
    graphdb_stubs, tmp_path, monkeypatch
):
    """Read from the project's own `.env` by name -- never from whatever the
    ambient environment happens to export for another deployment's store."""
    # Arrange
    monkeypatch.setenv("GRAPHDB_PASSWORD", "someone-elses-store")

    # Act
    _stage(GRAPHDB_CONFIG, tmp_path)

    # Assert
    assert graphdb_stubs["sessions"] == [("bolt://localhost:7687", "neo4j", "s3cret")]


def test_a_relative_ttl_path_anchors_on_the_config_directory(graphdb_stubs, tmp_path):
    """The bundle_path rule, shared with `osprey knowledge seed-graph`: relative
    values resolve against the directory holding config.yml. Anchoring on the repo
    root instead would read a different file, and anchoring on the process cwd
    would read whatever the operator happened to be standing in."""
    # Arrange
    decoy = tmp_path / "services" / "graphdb"
    decoy.mkdir(parents=True)
    (decoy / "als_gtb.ttl").write_text(DECOY_TTL_TEXT, encoding="utf-8")

    # Act
    _stage(GRAPHDB_CONFIG, tmp_path)

    # Assert
    assert graphdb_stubs["imported"] == [TTL_TEXT]


def test_an_absolute_ttl_path_is_used_as_written(graphdb_stubs, tmp_path):
    """The other half of the same rule: an operator naming a corpus outside the
    deployment gets that corpus."""
    # Arrange
    elsewhere = tmp_path / "corpus.ttl"
    elsewhere.write_text(DECOY_TTL_TEXT, encoding="utf-8")
    config = {
        **GRAPHDB_CONFIG,
        "services": {"graphdb": {"port_host": 7687, "ttl_path": str(elsewhere)}},
    }

    # Act
    _stage(config, tmp_path)

    # Assert
    assert graphdb_stubs["imported"] == [DECOY_TTL_TEXT]


def test_a_store_this_project_does_not_run_is_left_alone(graphdb_stubs, tmp_path):
    """A project pointed at a graph store someone else runs must never have that
    store bootstrapped or written to by a local deploy."""
    # Arrange
    config = {**GRAPHDB_CONFIG, "deployed_services": []}

    # Act
    _stage(config, tmp_path)

    # Assert
    assert graphdb_stubs["cmds"] == []
    assert graphdb_stubs["events"] == []


def test_the_staging_invocation_is_shaped_by_the_provider_it_is_handed(
    graphdb_stubs, tmp_path, monkeypatch
):
    """The provider must reach all three halves of the invocation contract: the
    argv builder, the env-file arguments, and the process environment. Left
    unthreaded, the store's `up` runs docker-shaped in a podman-shaped deploy."""
    from osprey.deployment.runtime_helper import ComposeProvider

    seen: dict = {}

    def _record_base(runtime_cmd, files, root, env_args, provider=None):
        seen["base_provider"] = provider
        return ["docker", "compose"]

    def _record_env_files(root=None, provider=None):
        seen["env_file_provider"] = provider
        return []

    def _record_run(cmd, *, env=None, **kwargs):
        seen["run_env"] = dict(env or {})

    monkeypatch.setattr(container_lifecycle, "compose_base_cmd", _record_base)
    monkeypatch.setattr(container_lifecycle, "_env_file_args", _record_env_files)
    monkeypatch.setattr(container_lifecycle, "run_captured", _record_run)

    # Act
    container_lifecycle._stage_graphdb_store(
        GRAPHDB_CONFIG, ["compose.yml"], {}, tmp_path, provider=ComposeProvider.PODMAN_COMPOSE
    )

    # Assert
    assert seen["base_provider"] is ComposeProvider.PODMAN_COMPOSE
    assert seen["env_file_provider"] is ComposeProvider.PODMAN_COMPOSE
    assert seen["run_env"]["COMPOSE_PROJECT_DIR"] == str(tmp_path)


def test_a_deploy_listing_graphdb_without_a_block_is_refused_before_anything_starts(
    tmp_path, monkeypatch
):
    """The preflight is wired into the start sequence, not merely available: a
    defaulted store would come up on ports nothing was told about and with no
    corpus, and the first signal would be an empty graph."""
    from osprey.deployment.errors import DeploymentPreconditionError

    # Arrange
    started: list = []
    monkeypatch.setattr(container_lifecycle, "_check_shared_disk_preflight", lambda config: None)
    monkeypatch.setattr(container_lifecycle, "preflight_qmd_models_dir", lambda config: None)
    monkeypatch.setattr(
        container_lifecycle,
        "verify_runtime_is_running",
        lambda config: started.append("runtime") or (True, ""),
    )

    # Act
    with pytest.raises(DeploymentPreconditionError) as excinfo:
        container_lifecycle._start_stack(
            {"project_name": "demo", "deployed_services": ["graphdb"]},
            ["compose.yml"],
            tmp_path,
        )

    # Assert
    assert "services.graphdb" in str(excinfo.value)
    assert started == []
