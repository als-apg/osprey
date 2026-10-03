"""A deploy keeps the graph store in step with the facility file, by stamp.

The build renders the knowledge-graph view ``data/graph/facility.ttl``, whose
first line carries the facility file's hash. The deploy's graph staging step
compares ``ttl_sha256`` of that text with the digest on the store's
``(:_OspreySeed)`` marker: equal leaves the store as it is; anything else wipes
it and imports the view. No manual seeding step is involved.

Three claims, against a real Neo4j + neosemantics store:

* **An unchanged redeploy changes nothing.** The second run of the staging step
  on the same build keeps the marker and the ``(:Resource)`` count, and never
  calls ``import_ttl``.
* **One changed description replaces the graph.** Editing one channel's
  description in the repo's facility records and rebuilding gives a new view, a
  new marker, and the new description in the store.
* **Health reports one digest.** The ``graphdb_seed`` row's value is the
  12-character prefix of the marker's digest and nothing else.
* **The search index follows by the same digest.** After every deploy the
  render's ``graph.duckdb`` carries the marker's digest as its
  ``corpus_sha256``, and the ``channel_finder_search_index`` health row is OK
  with that digest's prefix and no different-corpora warning.

The project is a real ``osprey init`` + ``osprey build`` of the
control-assistant preset. Its rendered ``config.yml`` is pointed at the graph
view and at the throwaway store's published port after every build, and the
staging step is the deploy's own ``_bootstrap_and_seed_graphdb``, run exactly
as ``osprey up`` runs it once the store answers.

Gating: needs Docker for the store; there is no LLM anywhere in this file. The
container takes a random published port and a generated name, so it cannot
collide with a deployed ``graphdb`` service.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment import container_lifecycle
from osprey.services.facility_knowledge.seeder import graph_seeder
from tests._graphdb_container import GRAPHDB_TEST_PASSWORD, graphdb_store_published_port

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    # dockerbuild: a real Neo4j store under testcontainers -- runs in the
    # dedicated graph-reseed-e2e CI job, never the shared e2e-tests lane (the
    # marker->--ignore pairing is enforced by
    # tests/deployment/test_ci_workflow_wiring.py).
    pytest.mark.dockerbuild,
]

#: The rendered graph view, relative to the render's ``config.yml``.
GRAPH_VIEW_TTL_PATH = "./data/graph/facility.ttl"

#: The search index the build writes, relative to the render's ``config.yml``.
SEARCH_INDEX_PATH = Path("data") / "channel_databases" / "graph.duckdb"

#: The repo-side facility records holding the channel descriptions.
CHANNEL_RECORDS = Path("data") / "facility" / "records" / "channels.yaml"

#: The description the test gives its one changed channel.
CHANGED_DESCRIPTION = "Reseed stamp probe: a description the shipped records never carry"


def _osprey(*args: str, cwd: Path) -> None:
    """Run one ``osprey`` command as a subprocess; fail with its output."""
    result = subprocess.run(
        [sys.executable, "-m", "osprey.cli.main", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=900,
    )
    assert result.returncode == 0, (
        f"osprey {' '.join(args)} failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}"
    )


def _build_and_point_at_store(repo: Path, port: int) -> dict[str, Any]:
    """Build *repo*, point its render at the graph view and the store; return the config."""
    _osprey("build", "--repo", str(repo), cwd=repo.parent)
    config_path = repo / "build" / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    graphdb = config["services"]["graphdb"]
    graphdb["ttl_path"] = GRAPH_VIEW_TTL_PATH
    graphdb["port_host"] = port
    config_path.write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False), encoding="utf-8"
    )
    return config


def _graph_view_text(repo: Path) -> str:
    return (repo / "build" / "data" / "graph" / "facility.ttl").read_text(encoding="utf-8")


def _deploy_graph(config: dict[str, Any], repo: Path) -> None:
    """Run the deploy's graph staging step against the store the config names."""
    connection = container_lifecycle._graphdb_connection(config, repo)
    container_lifecycle._bootstrap_and_seed_graphdb(config, repo, connection)


class _StoreReader:
    """Reads the store's marker, count and one channel's description."""

    def __init__(self, config: dict[str, Any], repo: Path) -> None:
        self._connection = container_lifecycle._graphdb_connection(config, repo)

    def _session(self):
        return graph_seeder.open_session(
            self._connection.uri,
            self._connection.username,
            self._connection.password,
            database=self._connection.database,
        )

    def marker(self) -> str | None:
        with self._session() as session:
            return graph_seeder.read_marker(session)

    def resource_count(self) -> int:
        with self._session() as session:
            return graph_seeder.resource_count(session)

    def description(self, address: str) -> str | None:
        with self._session() as session:
            record = session.run(
                "MATCH (b:Resource {fullPv: $address}) RETURN b.description AS d LIMIT 1",
                address=address,
            ).single()
            return None if record is None else record["d"]


def _change_one_description(repo: Path) -> str:
    """Give the first described channel record a new description; return its id."""
    path = repo / CHANNEL_RECORDS
    records = yaml.safe_load(path.read_text(encoding="utf-8"))
    record = next(r for r in records if r.get("description"))
    record["description"] = CHANGED_DESCRIPTION
    path.write_text(yaml.safe_dump(records, sort_keys=False), encoding="utf-8")
    return str(record["id"])


def _index_digest(repo: Path) -> str:
    """The ``corpus_sha256`` the render's search index carries."""
    import duckdb

    con = duckdb.connect(str(repo / "build" / SEARCH_INDEX_PATH), read_only=True)
    try:
        (digest,) = con.execute("SELECT corpus_sha256 FROM meta").fetchone()
    finally:
        con.close()
    return str(digest)


def _search_index_row(config: dict[str, Any], repo: Path):
    from osprey.health.core.channel_finder import channel_finder

    results = asyncio.run(channel_finder(config, cwd=repo / "build")())
    return next(row for row in results if row.name == "channel_finder_search_index")


def _assert_the_index_follows(config: dict[str, Any], repo: Path, marker: str | None) -> None:
    """The render's index carries the store's digest, and health agrees."""
    from osprey.health.models import Status

    assert marker is not None
    assert _index_digest(repo) == marker
    row = _search_index_row(config, repo)
    assert row.status is Status.OK, (row.message, row.value, row.details)
    assert marker[:12] in (row.value or "")
    assert "different corpora" not in row.message


def _seed_row(config: dict[str, Any]):
    from osprey.health.core.graphdb import graphdb

    results = asyncio.run(graphdb(config)())
    return next(row for row in results if row.name == "graphdb_seed")


@pytest.fixture(scope="module")
def store_port(graphdb_plugin_dir: Path) -> Iterator[int]:
    with graphdb_store_published_port(graphdb_plugin_dir, label="graph reseed store") as port:
        yield port


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A control-assistant deployment repo with the store's password in ``.env``."""
    tmp = tmp_path_factory.mktemp("graph-reseed")
    repo = tmp / "reseed"
    try:
        _osprey("init", str(repo), "--preset", "control-assistant", "--no-git", cwd=tmp)
        env_path = repo / ".env"
        existing = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        env_path.write_text(
            f"{existing}GRAPHDB_PASSWORD={GRAPHDB_TEST_PASSWORD}\n", encoding="utf-8"
        )
        yield repo
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_store_follows_the_facility_file_by_stamp(
    repo: Path, store_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GRAPHDB_PASSWORD", GRAPHDB_TEST_PASSWORD)
    imports: list[int] = []
    real_import = graph_seeder.import_ttl

    def _counted_import(session, text):
        imports.append(len(text))
        return real_import(session, text)

    monkeypatch.setattr(graph_seeder, "import_ttl", _counted_import)

    # First deploy: an empty store takes the rendered view.
    config = _build_and_point_at_store(repo, store_port)
    first_text = _graph_view_text(repo)
    _deploy_graph(config, repo)
    store = _StoreReader(config, repo)
    first_marker = store.marker()
    first_count = store.resource_count()
    assert first_marker == graph_seeder.ttl_sha256(first_text)
    assert first_count > 0
    assert len(imports) == 1
    _assert_the_index_follows(config, repo, first_marker)

    # Unchanged redeploy: same marker, same count, no import.
    config = _build_and_point_at_store(repo, store_port)
    assert _graph_view_text(repo) == first_text, "an unchanged repo rendered a different view"
    _deploy_graph(config, repo)
    assert store.marker() == first_marker
    assert store.resource_count() == first_count
    assert len(imports) == 1, "an unchanged redeploy imported the corpus again"
    _assert_the_index_follows(config, repo, store.marker())

    # One changed description: new view, new marker, new description in the store.
    address = _change_one_description(repo)
    config = _build_and_point_at_store(repo, store_port)
    second_text = _graph_view_text(repo)
    assert second_text != first_text
    assert CHANGED_DESCRIPTION in second_text
    _deploy_graph(config, repo)
    second_marker = store.marker()
    assert second_marker == graph_seeder.ttl_sha256(second_text)
    assert second_marker != first_marker
    assert store.description(address) == CHANGED_DESCRIPTION
    assert store.resource_count() == first_count
    assert len(imports) == 2
    _assert_the_index_follows(config, repo, second_marker)

    # Health reports the one digest the marker carries.
    row = _seed_row(config)
    assert row.value == second_marker[:12]
