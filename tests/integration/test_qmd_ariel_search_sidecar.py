"""End-to-end ARIEL search against a real qmd sidecar container.

What this proves that no faked client can: the *whole* round trip. ARIEL writes
markdown files whose names percent-encode a database key, a real qmd daemon
indexes the tree it can see through a read-only bind mount, and the search module
has to turn a path expressed in the container's namespace back into a Postgres
primary key and hydrate the row. Every layer in that chain has its own idea of
what a legal filename is, and the identifiers below are chosen to disagree with
each of them.

**This lane found a real defect with two faces.** qmd does not report the
filenames it indexed: it slugifies them, turning both ``%`` and ``_`` into ``-``,
collapsing runs and dropping a leading one.

*Face one — dropped hits*, now **fixed**. ``a%252%46b.md`` on disk came back as
``a-252-46b.md``, which inverted to a key no row had, so the hit was dropped in
silence; any uppercase letter was enough to make an entry unfindable, covering
identifiers as ordinary as ``ALS-2003-0001``. ``search/qmd.py`` no longer
inverts the reported path — it reads the identifier from the document's own
``# Entry <id>`` heading, which qmd reports unchanged — so every identifier in
this file now survives the round trip and none of these cases is xfailed.
:class:`TestQmdManglesReportedPaths` still pins the *daemon's* renaming, which
has not changed and is precisely why the title channel is load-bearing.

*Face two — a lost document*, pinned in :class:`TestUnderscoreCollision`, and
**still open**. ``_`` is in the writer's literal set, so
``beam_current_setpoint`` is written verbatim and looks completely safe — yet
qmd reports it as ``beam-current-setpoint.md``, which is the real filename of a
**different** entry. The two documents collapse onto one reported path, so only
one is ever indexed. Title-based hydration fixed *which row the survivor maps
to*; it cannot conjure back the document the index never took. That remainder is
a mirror-writer problem — the encoding is not injective under slugification —
and fixing it invalidates existing mirrors. Measured collision rate over the
real 134,996-entry ALS corpus: **0.0000%**, because every ALS ``entry_id`` is a
4-6 digit decimal string.

The other load-bearing property is that **qmd ranks and Postgres answers**. A hit
contributes its position and its snippet and nothing else; every field the caller
sees is re-read from ``enhanced_entries`` by primary key. That is what
:class:`TestPostgresIsAuthoritative` measures, by deliberately drifting a mirror
file away from its row and checking the lie does not reach the caller.

Reranking is off throughout — measured at 0.85 s versus 4.19 s per query on a
two-document corpus — and passed explicitly so these assertions document the mode
they ran in. Nothing here is about rank quality.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from osprey.services.ariel_search.enhancement.qmd_export.writer import encode_entry_id
from osprey.services.ariel_search.search.qmd import ARIEL_COLLECTION
from tests._container_support import is_docker_available
from tests.integration._qmd_ariel_support import (
    DEFAULT_QMD_SIDECAR_IMAGE,
    QMD_IMAGE_ENV,
    VOCABULARY_AUTHOR,
    VOCABULARY_DISTRACTOR_ROWS,
    VOCABULARY_ENTRY_ID,
    VOCABULARY_EXPECTED_PAIRS,
    VOCABULARY_PROSE,
    VOCABULARY_QUERY,
    VOCABULARY_SOURCE_SYSTEM,
    VOCABULARY_TIMESTAMP,
    build_sidecar_container,
    open_migrated_repository,
    qmd_ariel_config_dict,
    qmd_ariel_vocabulary_config,
    qmd_sidecar_image,
    start_qmd_sidecar,
    wait_for_indexed,
    write_vocabulary_file,
)

# xdist_group("docker") pins every container-starting module onto one worker: a
# run must have a single testcontainers session and a single ryuk reaper, because
# concurrent reaper starts race the Docker daemon's port mapper.
pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.xdist_group("docker"),
]

#: Token shared by the encoding-safe entries, so one query retrieves that whole
#: group. Nonsense on purpose: a real word would also match the prose the
#: renderer writes around the body.
SAFE_TOKEN = "zorblatt"

#: Search needle shared by the encoding-hostile entries. A separate group so a hostile
#: identifier cannot perturb the post-filter expectations, which are about
#: filtering rather than about encoding.
HOSTILE_NEEDLE = "quixotry"

#: Entries whose ``entry_id`` needs no encoding, spread across authors, source
#: systems and years so every post-filter has rows on both sides of it.
SAFE_ROWS: list[tuple[str, str, str, datetime]] = [
    ("olog-2003-0001", "Ada Lovelace", "ALS OLOG", datetime(2003, 1, 15, 12, 0, tzinfo=UTC)),
    ("olog-2003-0002", "Grace Hopper", "ALS OLOG", datetime(2003, 2, 15, 12, 0, tzinfo=UTC)),
    ("web-2003-0003", "Ada Lovelace", "ARIEL Web", datetime(2003, 3, 15, 12, 0, tzinfo=UTC)),
    ("olog-2004-0004", "Kathleen Booth", "ALS OLOG", datetime(2004, 1, 15, 12, 0, tzinfo=UTC)),
    ("web-2004-0005", "Grace Hopper", "ARIEL Web", datetime(2004, 2, 15, 12, 0, tzinfo=UTC)),
    ("olog-2004-0006", "Ada Lovelace", "ALS OLOG", datetime(2004, 3, 15, 12, 0, tzinfo=UTC)),
    ("olog-2005-0007", "Grace Hopper", "ALS OLOG", datetime(2005, 1, 15, 12, 0, tzinfo=UTC)),
]

#: Identifiers chosen to break a different assumption each:
#:
#: * ``path_sep`` — separators and a space inside the key, so the encoding has to
#:   flatten a path-shaped id into one filename component.
#: * ``encoded_collision`` — an id that *already looks percent-encoded*. Its
#:   filename is ``a%252%46b.md``; one decode yields ``a%2Fb`` and two yield
#:   ``a/b``, a different entry. This catches a hydration path that decodes once
#:   too often.
#: * ``collection_prefix`` — an id beginning with the collection name. qmd
#:   reports paths prefixed with the collection and the client strips that
#:   prefix, so an id carrying it must survive the stripping.
#: * ``dot_dot`` — traversal syntax in the key.
#: * ``unicode`` / ``uppercase`` — multi-byte and case-folding hazards on a
#:   case-insensitive filesystem.
#: * ``leading_dot`` — would become a dotfile if ``.`` were kept literal.
#: * ``clean`` — the control. Needs no encoding at all, so it isolates the
#:   encoding from the rest of the pipeline: if this one failed too, the fault
#:   would be in the harness rather than in the round trip.
HOSTILE_IDS: dict[str, str] = {
    "path_sep": "2003/01/incident 42",
    "encoded_collision": "a%2Fb",
    "collection_prefix": f"{ARIEL_COLLECTION}/2003/0007",
    "dot_dot": "../escape-attempt",
    "unicode": "Ünïcode-Ω-entry",
    "uppercase": "ALS-CAPS-0002",
    "leading_dot": ".hidden-entry",
    "clean": "pathological-content",
}

#: An id with a Postgres row and a mirror file, used only by the authority test.
#: It carries its own token so drifting its file cannot disturb any other group,
#: and an author of its own so it cannot land inside a post-filter expectation.
AUTHORITY_ID = "authority-probe"
AUTHORITY_TOKEN = "vermiculate"
AUTHORITY_AUTHOR = "Sidecar Probe"

#: A mirror file with NO Postgres row. The index is a copy of the database and is
#: allowed to lag it, so a hit on this must be dropped rather than raised on.
ORPHAN_ID = "orphan-no-row"
ORPHAN_NEEDLE = "orphanium"

#: Control characters that must not reach the markdown file. NUL is deliberately
#: absent: Postgres rejects it in a text column, so it cannot be stored to begin
#: with, and the writer's handling of it belongs to unit coverage.
CONTROL_CHARS = "\x01\x07\x1b\x7f"

#: The two entries of the collision pair. Their ids differ only by ``_`` versus
#: ``-``, and both are legal, unremarkable ARIEL identifiers.
COLLISION_UNDERSCORE = "beam_current_setpoint"
COLLISION_HYPHEN = "beam-current-setpoint"
COLLISION_TOKEN = "kryptonite"

#: The one face of the defect that remains, and the only standing record of it.
#: Distinct from the drop because no hydration strategy can reach it: the losing
#: document is never indexed, so there is no hit to hydrate.
COLLISION_XFAIL = (
    "MIRROR WRITER defect, not a hydration one: the writer's encoding is not "
    "injective under qmd's slugification. 'beam_current_setpoint' and "
    "'beam-current-setpoint' are written to distinct files that qmd reports "
    "under one name, so only ONE is indexed and the other is unreachable by any "
    "query. Title-based hydration fixed which row the survivor maps to — that "
    "part is no longer wrong — but it cannot conjure a document the index never "
    "took. Fixing this means changing encode_entry_id to emit names closed under "
    "slugification (escape with ~ instead of %, drop _ and - from the literal "
    "set), which invalidates every existing mirror and needs a "
    "'qmd-resync --rebuild'. Measured collision rate over the real 134,996-entry "
    "ALS corpus: 0.0000% — every ALS entry_id is a 4-6 digit decimal string, so "
    "no real pair can collide. Pinned in TestUnderscoreCollision."
)


def slugify(filename: str) -> str:
    """Reproduce qmd's transformation of a filename into the name it reports.

    Measured, not documented: ``%`` and ``_`` both become ``-``, runs of ``-``
    collapse to one, and a leading ``-`` is dropped. A description of observed
    behaviour used to make the defect assertions exact, not a contract qmd
    publishes.
    """
    return re.sub(r"-+", "-", re.sub(r"[%_]", "-", filename)).lstrip("-")


def is_mangled(entry_id: str) -> bool:
    """Whether qmd will report this id's filename under a different name.

    Deliberately **not** ``encode_entry_id(entry_id) != entry_id``. That was the
    first cut and it is wrong: ``_`` is in the writer's literal set, so an id
    like ``beam_current_setpoint`` needs no escaping and looks safe, yet qmd
    still renames it. The discriminator has to be the reported name, not the
    encoding.
    """
    encoded = encode_entry_id(entry_id)
    return slugify(encoded) != encoded


def hostile_param(label: str):
    """Parametrise a hostile id.

    Every case runs unmarked now. These were xfailed while hydration inverted
    the reported path, which qmd slugifies; hydration reads the identifier out
    of the document's own title instead, so a renamed path no longer costs the
    hit. :func:`is_mangled` stays live because
    :class:`TestQmdManglesReportedPaths` still asserts qmd does the renaming —
    the daemon's behaviour is unchanged, only ARIEL's dependence on it.
    """
    return pytest.param(label)


def _entry(
    entry_id: str,
    *,
    author: str,
    source_system: str,
    timestamp: datetime,
    raw_text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``enhanced_entries`` row."""
    return {
        "entry_id": entry_id,
        "source_system": source_system,
        "timestamp": timestamp,
        "author": author,
        "raw_text": raw_text,
        "attachments": [],
        "metadata": metadata or {},
        "enhancement_status": {},
    }


def _seed_rows() -> list[dict[str, Any]]:
    """The fixture corpus: three token groups plus the authority probe."""
    rows = [
        _entry(
            entry_id,
            author=author,
            source_system=source,
            timestamp=stamp,
            raw_text=f"Storage ring {SAFE_TOKEN} note {entry_id} logged by {author}.",
        )
        for entry_id, author, source, stamp in SAFE_ROWS
    ]

    hostile_text = {
        "path_sep": f"Quadrupole {HOSTILE_NEEDLE} trim supply replaced.",
        "encoded_collision": f"Sextupole {HOSTILE_NEEDLE} shunt current drifted overnight.",
        "collection_prefix": f"Septum {HOSTILE_NEEDLE} pressure rise during injection.",
        "dot_dot": f"Bumper {HOSTILE_NEEDLE} magnet timing adjusted.",
        "unicode": f"Undulator {HOSTILE_NEEDLE} gap encoder recalibrated — Ω phase check.",
        "uppercase": f"Kicker {HOSTILE_NEEDLE} pulser fault cleared by the crew.",
        "leading_dot": f"Chicane {HOSTILE_NEEDLE} corrector readback stuck at zero.",
    }
    for label, text in hostile_text.items():
        rows.append(
            _entry(
                HOSTILE_IDS[label],
                author="Ada Lovelace",
                source_system="ALS OLOG",
                timestamp=datetime(2006, 1, 15, 12, 0, tzinfo=UTC),
                raw_text=text,
            )
        )

    # The control: an encoding-clean id carrying pathological *content* — C0/C1
    # control bytes, CRLF, a frontmatter fence and a heading inside the body, and
    # enough text to run past the renderer's size cap.
    rows.append(
        _entry(
            HOSTILE_IDS["clean"],
            author="Kathleen Booth",
            source_system="ALS OLOG",
            timestamp=datetime(2006, 2, 15, 12, 0, tzinfo=UTC),
            raw_text=(
                f"Gnarly {HOSTILE_NEEDLE} entry{CONTROL_CHARS} with embedded controls.\r\n"
                "---\n"
                "# Not a real heading\n"
                "---\n" + ("filler " * 40000)
            ),
            metadata={"shift": "owl", "tags": ["vacuum", "rf"]},
        )
    )

    # The collision pair. Both ids are ordinary — neither needs any escaping —
    # but qmd reports both files under one slugified name.
    for entry_id in (COLLISION_UNDERSCORE, COLLISION_HYPHEN):
        rows.append(
            _entry(
                entry_id,
                author=AUTHORITY_AUTHOR,
                source_system="ALS OLOG",
                timestamp=datetime(2008, 1, 15, 12, 0, tzinfo=UTC),
                raw_text=f"The {COLLISION_TOKEN} reading for {entry_id} was nominal.",
            )
        )

    rows.append(
        _entry(
            AUTHORITY_ID,
            author=AUTHORITY_AUTHOR,
            source_system="ALS OLOG",
            timestamp=datetime(2007, 1, 15, 12, 0, tzinfo=UTC),
            raw_text=f"The {AUTHORITY_TOKEN} interlock was reset by the floor operator.",
        )
    )

    # The vocabulary group: one prose entry written entirely in canonical
    # phrases, plus the near misses that give an unexpanded query something to
    # prefer over it. Its own group on purpose — see VOCABULARY_ENTRY_ID for why
    # it must stay out of SAFE_ROWS, which wait_for_indexed counts and three
    # post-filter tests compare against exactly.
    #
    # These four rows grew the indexed corpus from 19 documents to 23 (the rows
    # here plus the ORPHAN_ID mirror file, which has no row). No assertion in the
    # module depends on the total: every group is isolated by its own token,
    # author and year, so the exact-set comparisons and wait_for_indexed's
    # `len(hits) >= len(SAFE_ROWS)` are unaffected. What the total does consume is
    # headroom — `_safe_ids(results) == SAFE_IDS` needs all seven safe rows inside
    # the sidecar's configured candidate_limit of 40, and the corpus is now 23 of
    # those 40. A further group seeded here should raise that limit alongside it.
    for entry_id, text in [
        (VOCABULARY_ENTRY_ID, VOCABULARY_PROSE),
        *VOCABULARY_DISTRACTOR_ROWS,
    ]:
        rows.append(
            _entry(
                entry_id,
                author=VOCABULARY_AUTHOR,
                source_system=VOCABULARY_SOURCE_SYSTEM,
                timestamp=VOCABULARY_TIMESTAMP,
                raw_text=text,
            )
        )
    return rows


@dataclass
class _World:
    """Everything the sidecar tests share."""

    config_dict: dict[str, Any]
    mirror_root: Path
    port: int
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)


async def _prepare(config_dict: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Migrate, seed Postgres, and export the mirror through the real resync."""
    from osprey.services.ariel_search.cli_operations import run_qmd_resync
    from osprey.services.ariel_search.config import ARIELConfig

    config = ARIELConfig.from_dict(config_dict)
    pool, repository = await open_migrated_repository(config)
    try:
        for row in rows:
            await repository.upsert_entry(row)
    finally:
        await pool.close()

    result = await run_qmd_resync(config_dict, rebuild=True)
    assert result is not None, "qmd_export must be enabled for this lane"
    assert result.written == len(rows), (
        f"the rebuild exported {result.written} of {len(rows)} seeded rows "
        f"({result.failed} failed); the sidecar would index an incomplete corpus"
    )


@pytest.fixture(scope="session")
def sidecar_world(request, tmp_path_factory, qmd_ariel_database_factory) -> _World:
    """Seed Postgres, export the mirror, then start the sidecar over it.

    The order is not incidental: the entrypoint refuses to serve an empty index
    and exits, so the corpus has to exist before the container does.
    """
    from osprey.services.ariel_search.enhancement.qmd_export.writer import mirror_path

    mirror_root = Path(tmp_path_factory.mktemp("qmd_ariel_mirror"))
    database_url = qmd_ariel_database_factory("qmd_ariel_search")
    config_dict = qmd_ariel_config_dict(database_url, mirror_root)

    rows = _seed_rows()
    asyncio.run(_prepare(config_dict, rows))

    # A mirrored file with no row behind it. Written by hand precisely because no
    # supported path produces one: the point is that hydration survives an index
    # that has lagged a deletion.
    orphan = mirror_path(mirror_root, {"entry_id": ORPHAN_ID, "timestamp": None})
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text(
        f"# Entry {ORPHAN_ID}\n\nThe {ORPHAN_NEEDLE} record was deleted from Postgres.\n",
        encoding="utf-8",
    )

    _, port = start_qmd_sidecar(request, mirror_root, ARIEL_COLLECTION)

    from osprey.deployment.qmd_service import QMDServiceConfig
    from osprey.services.qmd import QMDClient

    # The daemon only opens its port after the startup pass, so the corpus is
    # already indexed by now; this asserts the whole set arrived rather than
    # assuming it.
    client = QMDClient(QMDServiceConfig(port=port))
    wait_for_indexed(
        client,
        ARIEL_COLLECTION,
        SAFE_TOKEN,
        lambda hits: len(hits) >= len(SAFE_ROWS),
        what=f"all {len(SAFE_ROWS)} encoding-safe documents",
    )

    return _World(
        config_dict=config_dict,
        mirror_root=mirror_root,
        port=port,
        rows={row["entry_id"]: row for row in rows},
    )


@pytest.fixture
def sidecar_client(sidecar_world: _World):
    """A real ``QMDClient`` pointed at the container's published port."""
    from osprey.deployment.qmd_service import QMDServiceConfig
    from osprey.services.qmd import QMDClient

    return QMDClient(QMDServiceConfig(port=sidecar_world.port))


@pytest.fixture
async def sidecar_repository(sidecar_world: _World):
    """A repository on a pool bound to the running test's event loop."""
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database import ARIELRepository, create_connection_pool

    config = ARIELConfig.from_dict(sidecar_world.config_dict)
    pool = await create_connection_pool(config.database)
    try:
        yield ARIELRepository(pool, config)
    finally:
        await pool.close()


async def _search(world, repository, client, query=SAFE_TOKEN, **kwargs):
    """Run the real search module against the real sidecar."""
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.search.qmd import hybrid_search

    return await hybrid_search(
        query,
        repository,
        ARIELConfig.from_dict(world.config_dict),
        client=client,
        **kwargs,
    )


def _ids(results) -> set[str]:
    """The entry ids in a ``hybrid_search`` result list."""
    return {entry["entry_id"] for entry, _score, _snippets in results}


#: Every encoding-safe id, for the assertions that scope themselves to that group.
SAFE_IDS = {row[0] for row in SAFE_ROWS}


def _safe_ids(results) -> set[str]:
    """The encoding-safe ids in a result list.

    Queries here go through qmd's **hybrid** search, so the vector leg pulls in
    documents that do not contain the query token at all — that is the feature,
    not a defect, and it means an exact set comparison against one token group
    would be asserting something qmd never promised. The assertions therefore
    check a subset of the group under test plus a property that must hold of
    *every* row returned, which is both stronger and stable.
    """
    return _ids(results) & SAFE_IDS


# ---------------------------------------------------------------------------
# The sidecar itself
# ---------------------------------------------------------------------------


class TestSidecarImageSelection:
    """The lane must test the image CI built, not a stale local tag.

    No container is *started* here, so these stay fast. Only the one test that
    builds a container object needs a daemon: testcontainers resolves the
    engine in ``DockerContainer.__init__``, which is why it carries the same
    skip every container-touching helper here does. The other three read the
    tag alone and run anywhere.
    """

    async def test_env_override_is_honoured(self, monkeypatch):
        monkeypatch.setenv(QMD_IMAGE_ENV, "ghcr.io/example/qmd:ci-abc123")

        assert qmd_sidecar_image() == "ghcr.io/example/qmd:ci-abc123"

    async def test_falls_back_to_the_local_build_tag(self, monkeypatch):
        """An interactive run needs no environment."""
        monkeypatch.delenv(QMD_IMAGE_ENV, raising=False)

        assert qmd_sidecar_image() == DEFAULT_QMD_SIDECAR_IMAGE

    async def test_the_container_that_gets_built_uses_the_override(self, monkeypatch, tmp_path):
        """The assertion that actually stops the regression.

        Checking :func:`qmd_sidecar_image` alone would not notice a hardcoded tag
        reappearing at the construction site — which is exactly how this broke
        the first time — so this inspects the container that would be started.

        Building that object resolves the Docker engine even though nothing is
        started, so a runner without a daemon skips rather than erroring.
        """
        if not is_docker_available():
            pytest.skip("docker daemon is not reachable")

        monkeypatch.setenv(QMD_IMAGE_ENV, "ghcr.io/example/qmd:ci-abc123")

        container = build_sidecar_container(tmp_path, ARIEL_COLLECTION)

        assert container.image == "ghcr.io/example/qmd:ci-abc123"

    async def test_an_empty_override_falls_back_rather_than_naming_nothing(self, monkeypatch):
        """``OSPREY_QMD_IMAGE=`` is unset in every way that matters."""
        monkeypatch.setenv(QMD_IMAGE_ENV, "")

        assert qmd_sidecar_image() == DEFAULT_QMD_SIDECAR_IMAGE


class TestSidecarIsReal:
    """Guards against the whole lane passing without a container behind it."""

    async def test_client_reports_the_sidecar_available(self, sidecar_client):
        assert sidecar_client.is_configured is True
        assert sidecar_client.is_available() is True

    async def test_collection_name_is_the_shipped_constant(self):
        """The mount, the index and the query must agree on one name."""
        from osprey.deployment.compose_generator import QMD_ARIEL_COLLECTION

        assert ARIEL_COLLECTION == QMD_ARIEL_COLLECTION == "ariel"

    async def test_end_to_end_search_returns_hydrated_rows(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        """The baseline round trip, on identifiers the encoding leaves alone."""
        results = await _search(sidecar_world, sidecar_repository, sidecar_client, max_results=50)

        assert _safe_ids(results) == SAFE_IDS, "an encoding-safe entry did not come back"
        for entry, score, _snippets in results:
            assert entry["raw_text"] == sidecar_world.rows[entry["entry_id"]]["raw_text"]
            assert 0.0 < score <= 1.0

    async def test_snippets_come_from_the_sidecar(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        """A hit contributes its excerpt; an empty list would mean qmd gave none."""
        results = await _search(sidecar_world, sidecar_repository, sidecar_client, max_results=5)

        assert results
        assert any(snippets and SAFE_TOKEN in snippets[0] for _e, _s, snippets in results)


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


class TestQmdManglesReportedPaths:
    """Pins the observed qmd behaviour that breaks hydration for most ids.

    This class passes today. It is not a wish; it is a measurement, kept so the
    defect is an executable fact rather than a paragraph someone has to trust,
    and so that a qmd upgrade that changes the behaviour fails loudly here
    instead of silently changing which entries are findable.
    """

    async def test_reported_path_differs_from_the_file_on_disk(self, sidecar_world, sidecar_client):
        """qmd reports ``a-252-46b.md`` for a file named ``a%252%46b.md``."""
        from osprey.services.ariel_search.enhancement.qmd_export.writer import mirror_path

        row = sidecar_world.rows[HOSTILE_IDS["encoded_collision"]]
        on_disk = mirror_path(sidecar_world.mirror_root, row)
        assert on_disk.name == "a%252%46b.md", on_disk.name
        assert on_disk.exists()

        hits = sidecar_client.query(
            ARIEL_COLLECTION, HOSTILE_NEEDLE, limit=50, rerank=False, candidate_limit=40
        )
        reported = {Path(hit.file).name for hit in hits}

        assert "a-252-46b.md" in reported, reported
        assert on_disk.name not in reported, "qmd started reporting the real filename"

    async def test_every_escape_is_slugified(self, sidecar_world, sidecar_client):
        """The transformation, spelled out: ``%`` and ``_`` -> ``-``, runs collapsed.

        Derived from the reported names rather than assumed — ``-%`` in an id like
        ``Ünïcode-Ω-entry`` becomes a single ``-``, not two, which is why this
        asserts against :func:`slugify` rather than a bare ``replace``.
        """
        hits = sidecar_client.query(
            ARIEL_COLLECTION, HOSTILE_NEEDLE, limit=50, rerank=False, candidate_limit=40
        )
        reported = {Path(hit.file).name for hit in hits}

        for label, entry_id in HOSTILE_IDS.items():
            encoded = f"{encode_entry_id(entry_id)}.md"
            if not is_mangled(entry_id):
                assert encoded in reported, f"{label}: clean names must survive intact"
                continue
            assert slugify(encoded) in reported, (
                f"{label}: expected {slugify(encoded)!r} in {sorted(reported)}"
            )
            assert encoded not in reported, f"{label}: the real filename came back after all"


class TestUnderscoreCollision:
    """A mangled name that is also the real filename of a *different* entry.

    ``_`` is in the writer's literal set, so ``beam_current_setpoint`` is written
    to disk verbatim and looks entirely safe. qmd still reports it as
    ``beam-current-setpoint.md`` — which is the real filename of a different,
    equally ordinary entry.

    Two consequences, one of which has already been closed:

    1. **Still live.** Both documents collapse onto one reported path, so one of
       them is simply unreachable through search — a whole entry that no query
       can reach while its twin exists. This is what
       :meth:`test_both_entries_are_reachable` xfails on, and it is a mirror-writer
       problem: two distinct ids must not produce names qmd cannot tell apart.

    2. **Closed — do not escalate this.** The reported path *would* invert to the
       other entry's key, which is exactly why ``hybrid_search`` no longer inverts
       the path: it reads the identifier from the document's own
       ``# Entry <id>`` heading, which qmd reports unchanged. The survivor
       therefore hydrates to ``beam_current_setpoint``, the correct row, logging
       ``reported path disagrees with the document's own title; trusting the
       title``. Had that channel not been added, this would have been a **wrong**
       answer rather than a missing one — worse than a dropped hit, because
       "Postgres is authoritative" cannot help when the key handed to it is the
       wrong key.

    :meth:`test_the_reported_path_would_have_hydrated_the_wrong_row` keeps
    consequence 2 measured rather than assumed, so the title channel's value stays
    visible and nobody removes it as redundant.
    """

    async def test_the_pair_needs_no_escaping_at_all(self):
        """Both ids are written to disk verbatim — nothing here looks risky."""
        assert encode_entry_id(COLLISION_UNDERSCORE) == COLLISION_UNDERSCORE
        assert encode_entry_id(COLLISION_HYPHEN) == COLLISION_HYPHEN

    async def test_both_files_exist_on_disk_under_distinct_names(self, sidecar_world):
        from osprey.services.ariel_search.enhancement.qmd_export.writer import mirror_path

        under = mirror_path(sidecar_world.mirror_root, sidecar_world.rows[COLLISION_UNDERSCORE])
        hyphen = mirror_path(sidecar_world.mirror_root, sidecar_world.rows[COLLISION_HYPHEN])

        assert under.name == "beam_current_setpoint.md"
        assert hyphen.name == "beam-current-setpoint.md"
        assert under.exists() and hyphen.exists()
        assert under != hyphen

    async def test_qmd_collapses_the_pair_onto_one_reported_path(
        self, sidecar_world, sidecar_client
    ):
        """Two documents go in; one reported name comes out."""
        hits = sidecar_client.query(
            ARIEL_COLLECTION, COLLISION_TOKEN, limit=50, rerank=False, candidate_limit=40
        )
        reported = [Path(hit.file).name for hit in hits]

        assert "beam-current-setpoint.md" in reported
        assert "beam_current_setpoint.md" not in reported, (
            "qmd started reporting the underscore filename; the collision may be fixed"
        )
        assert reported.count("beam-current-setpoint.md") == 1, (
            f"expected the pair to collapse onto one path, got {reported}"
        )

    @pytest.mark.xfail(strict=True, reason=COLLISION_XFAIL)
    async def test_both_entries_are_reachable(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        """Neither entry should be lost just because the other exists."""
        results = await _search(
            sidecar_world,
            sidecar_repository,
            sidecar_client,
            query=COLLISION_TOKEN,
            max_results=50,
        )

        assert {COLLISION_UNDERSCORE, COLLISION_HYPHEN} <= _ids(results)

    async def test_the_reported_path_would_have_hydrated_the_wrong_row(
        self, sidecar_world, sidecar_client
    ):
        """Pins the path channel that hydration deliberately no longer consults.

        This measures ``entry_id_from_path(hit.file)`` directly — a channel
        ``hybrid_search`` stopped using — so it is a counterfactual, not a report of
        current behaviour: *had* hydration kept inverting the reported path, this
        hit would have returned the hyphen entry's row for the underscore entry's
        document. Kept measured rather than asserted in prose so the title
        channel's value stays visible and nobody deletes it as redundant.

        What hydration actually does now is covered by
        :meth:`test_the_survivor_hydrates_to_the_right_row`.
        """
        from osprey.services.ariel_search.enhancement.qmd_export.writer import entry_id_from_path

        hits = sidecar_client.query(
            ARIEL_COLLECTION, COLLISION_TOKEN, limit=50, rerank=False, candidate_limit=40
        )
        hit = next(h for h in hits if Path(h.file).name == "beam-current-setpoint.md")

        # The two channels disagree: this is the whole reason one of them won.
        path_says = entry_id_from_path(hit.file)
        title_says = hit.title.removeprefix("Entry ")

        assert path_says == COLLISION_HYPHEN
        assert title_says == COLLISION_UNDERSCORE
        assert path_says != title_says, (
            "the two channels agree again — qmd may have stopped renaming, so "
            "re-check whether the title channel is still load-bearing"
        )

    async def test_the_survivor_hydrates_to_the_right_row(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        """The positive control: the reachable entry comes back correct.

        Without this, the class would only ever describe what is broken, and a
        reader could not tell whether the surviving hit is right or merely
        untested.
        """
        results = await _search(
            sidecar_world,
            sidecar_repository,
            sidecar_client,
            query=COLLISION_TOKEN,
            max_results=50,
        )
        by_id = {entry["entry_id"]: entry for entry, _s, _n in results}

        assert COLLISION_UNDERSCORE in by_id, "the reachable half of the pair went missing"
        assert (
            by_id[COLLISION_UNDERSCORE]["raw_text"]
            == sidecar_world.rows[COLLISION_UNDERSCORE]["raw_text"]
        )

    async def test_mangled_path_decodes_to_a_key_no_row_has(self, sidecar_world):
        """Why the hit vanishes: the inversion produces a different identifier."""
        from osprey.services.ariel_search.enhancement.qmd_export.writer import entry_id_from_path

        assert entry_id_from_path("2003/02/a-252-46b.md") == "a-252-46b"
        assert entry_id_from_path("2003/02/a-252-46b.md") != HOSTILE_IDS["encoded_collision"]

    async def test_the_original_id_is_still_recoverable_from_the_document(self, sidecar_world):
        """There is a fix available: the renderer already writes the real key.

        ``render_entry`` emits ``# Entry <entry_id>`` as the first line, and qmd
        surfaces that as the hit's ``title``. Recorded here so the repair is not
        re-derived from scratch — this is a source change in
        ``search/qmd.py``, which this lane does not own.
        """
        from osprey.services.ariel_search.enhancement.qmd_export.writer import mirror_path

        row = sidecar_world.rows[HOSTILE_IDS["encoded_collision"]]
        first_line = (
            mirror_path(sidecar_world.mirror_root, row).read_text(encoding="utf-8").splitlines()[0]
        )

        assert first_line == f"# Entry {HOSTILE_IDS['encoded_collision']}"


# ---------------------------------------------------------------------------
# Hostile identifiers
# ---------------------------------------------------------------------------


class TestHostileEntryIds:
    """Every fixture identifier should survive the filename round trip."""

    async def test_every_hostile_id_comes_back(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        results = await _search(
            sidecar_world, sidecar_repository, sidecar_client, query=HOSTILE_NEEDLE, max_results=50
        )

        assert set(HOSTILE_IDS.values()) <= _ids(results), (
            f"missing: {sorted(set(HOSTILE_IDS.values()) - _ids(results))}"
        )

    @pytest.mark.parametrize("label", [hostile_param(label) for label in sorted(HOSTILE_IDS)])
    async def test_hostile_id_hydrates_its_own_row(
        self, label, sidecar_world, sidecar_repository, sidecar_client
    ):
        """The row behind each hit is the one the document's title declares.

        Parametrised rather than folded into the sweep above so a single bad
        identifier names itself instead of collapsing a set comparison. Every
        case now runs unmarked; while hydration inverted the reported path, only
        ``clean`` passed and the rest were xfailed.
        """
        entry_id = HOSTILE_IDS[label]
        results = await _search(
            sidecar_world, sidecar_repository, sidecar_client, query=HOSTILE_NEEDLE, max_results=50
        )
        by_id = {entry["entry_id"]: entry for entry, _s, _n in results}

        assert entry_id in by_id, f"{label}: {entry_id!r} did not come back"
        # raw_text, not author or source_system: seven of the eight hostile rows
        # share one author and one source system, so those fields would hold just
        # as well if the wrong row had been hydrated. raw_text embeds the id and
        # is the only field here that can tell these rows apart.
        assert by_id[entry_id]["raw_text"] == sidecar_world.rows[entry_id]["raw_text"]

    async def test_encode_decode_round_trips_in_isolation(self):
        """The writer's own encoding is invertible; only qmd's report is not.

        Keeping this live separates the two failures. If this ever breaks, the
        defect is ARIEL's; while it holds, the defect is qmd's.
        """
        from osprey.services.ariel_search.enhancement.qmd_export.writer import entry_id_from_path

        for entry_id in HOSTILE_IDS.values():
            encoded = encode_entry_id(entry_id)
            assert entry_id_from_path(f"2003/02/{encoded}.md") == entry_id

        assert encode_entry_id(HOSTILE_IDS["encoded_collision"]) == "a%252%46b"

    async def test_encoded_looking_id_is_not_double_decoded(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        """``a%2Fb`` must come back as itself, never as ``a/b``."""
        results = await _search(
            sidecar_world, sidecar_repository, sidecar_client, query=HOSTILE_NEEDLE, max_results=50
        )
        found = _ids(results)

        assert "a%2Fb" in found
        assert "a/b" not in found

    async def test_collection_prefixed_id_is_not_truncated(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        """An id starting with the collection name survives prefix stripping."""
        results = await _search(
            sidecar_world, sidecar_repository, sidecar_client, query=HOSTILE_NEEDLE, max_results=50
        )
        found = _ids(results)

        assert HOSTILE_IDS["collection_prefix"] in found
        assert "2003/0007" not in found, "the collection prefix was stripped from the id itself"


# ---------------------------------------------------------------------------
# Pathological content
# ---------------------------------------------------------------------------


class TestPathologicalContent:
    """A row whose text fights the markdown format still indexes and hydrates."""

    async def test_pathological_entry_is_searchable(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        results = await _search(
            sidecar_world, sidecar_repository, sidecar_client, query=HOSTILE_NEEDLE, max_results=50
        )

        assert HOSTILE_IDS["clean"] in _ids(results)

    async def test_control_characters_never_reach_the_mirror(self, sidecar_world):
        """The bytes qmd indexes carry no C0/C1 controls except tab, CR and LF."""
        from osprey.services.ariel_search.enhancement.qmd_export.writer import mirror_path

        row = sidecar_world.rows[HOSTILE_IDS["clean"]]
        text = mirror_path(sidecar_world.mirror_root, row).read_text(encoding="utf-8")

        # Guard first: an emptied CONTROL_CHARS would make the loop below vacuous,
        # and reusing the loop variable afterwards would raise NameError instead
        # of failing cleanly — which is exactly the drift this is here to catch.
        assert CONTROL_CHARS and all(char in row["raw_text"] for char in CONTROL_CHARS), (
            "the fixture no longer carries control characters"
        )
        for char in CONTROL_CHARS:
            assert char not in text, f"{char!r} survived into the mirror"

    async def test_oversized_entry_is_capped_on_disk(self, sidecar_world):
        """The renderer's size cap holds for a row far past it."""
        from osprey.services.ariel_search.enhancement.qmd_export.writer import (
            BODY_CAP_BYTES,
            TRUNCATION_MARKER,
            mirror_path,
        )

        row = sidecar_world.rows[HOSTILE_IDS["clean"]]
        assert len(row["raw_text"]) > BODY_CAP_BYTES, "fixture is not actually oversized"

        data = mirror_path(sidecar_world.mirror_root, row).read_bytes()

        assert len(data) <= BODY_CAP_BYTES
        assert data.decode("utf-8").endswith(TRUNCATION_MARKER)

    async def test_hydrated_text_is_the_uncapped_row(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        """Truncation is a property of the index, never of the answer."""
        results = await _search(
            sidecar_world, sidecar_repository, sidecar_client, query=HOSTILE_NEEDLE, max_results=50
        )
        by_id = {entry["entry_id"]: entry for entry, _s, _n in results}
        row = sidecar_world.rows[HOSTILE_IDS["clean"]]

        assert by_id[HOSTILE_IDS["clean"]]["raw_text"] == row["raw_text"]


# ---------------------------------------------------------------------------
# Post-filtering
# ---------------------------------------------------------------------------


class TestPostFiltering:
    """Filters qmd cannot answer run after hydration, against the real rows.

    Each test asserts two things: that every row which came back satisfies the
    filter (nothing leaked past it) and that every encoding-safe row which should
    have satisfied it did come back (nothing was over-filtered).
    """

    async def test_author_filter_is_a_case_insensitive_substring(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        results = await _search(
            sidecar_world, sidecar_repository, sidecar_client, max_results=50, author="LOVELACE"
        )

        assert results
        for entry, _s, _n in results:
            assert "lovelace" in entry["author"].casefold()
        assert _safe_ids(results) == {"olog-2003-0001", "web-2003-0003", "olog-2004-0006"}

    async def test_source_system_filter_is_exact(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        results = await _search(
            sidecar_world,
            sidecar_repository,
            sidecar_client,
            max_results=50,
            source_system="ARIEL Web",
        )

        assert results
        for entry, _s, _n in results:
            assert entry["source_system"] == "ARIEL Web"
        assert _safe_ids(results) == {"web-2003-0003", "web-2004-0005"}

    async def test_source_system_filter_does_not_substring_match(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        """``ARIEL`` is a prefix of ``ARIEL Web`` and must still match nothing."""
        results = await _search(
            sidecar_world, sidecar_repository, sidecar_client, max_results=50, source_system="ARIEL"
        )

        assert results == []

    async def test_date_range_filter(self, sidecar_world, sidecar_repository, sidecar_client):
        results = await _search(
            sidecar_world,
            sidecar_repository,
            sidecar_client,
            max_results=50,
            start_date=datetime(2004, 1, 1, tzinfo=UTC),
            end_date=datetime(2004, 12, 31, tzinfo=UTC),
        )

        assert results
        for entry, _s, _n in results:
            assert entry["timestamp"].year == 2004
        assert _safe_ids(results) == {"olog-2004-0004", "web-2004-0005", "olog-2004-0006"}

    async def test_filters_compose(self, sidecar_world, sidecar_repository, sidecar_client):
        """Two active filters intersect rather than either one winning."""
        results = await _search(
            sidecar_world,
            sidecar_repository,
            sidecar_client,
            max_results=50,
            author="Grace",
            start_date=datetime(2004, 1, 1, tzinfo=UTC),
            end_date=datetime(2004, 12, 31, tzinfo=UTC),
        )

        assert results
        for entry, _s, _n in results:
            assert "grace" in entry["author"].casefold()
            assert entry["timestamp"].year == 2004
        assert _safe_ids(results) == {"web-2004-0005"}

    async def test_max_results_is_honoured_after_filtering(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        """The cap applies to the survivors, not to the candidates fetched."""
        results = await _search(
            sidecar_world, sidecar_repository, sidecar_client, max_results=2, author="Lovelace"
        )

        assert len(results) == 2
        for entry, _s, _n in results:
            assert "lovelace" in entry["author"].casefold()


# ---------------------------------------------------------------------------
# Postgres is authoritative
# ---------------------------------------------------------------------------


class TestPostgresIsAuthoritative:
    """A mirror file that has drifted from its row cannot mislead a caller."""

    async def test_hit_without_a_row_is_dropped(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        """The index may lag a deletion; a hit with no row is dropped silently.

        The assertion is that the orphan is absent, not that the answer is empty:
        the hybrid search also returns semantically nearby documents that do have
        rows, and dropping *those* would be a different bug.
        """
        orphan_file = f"{encode_entry_id(ORPHAN_ID)}.md"
        raw = wait_for_indexed(
            sidecar_client,
            ARIEL_COLLECTION,
            ORPHAN_NEEDLE,
            lambda hits: any(hit.file.endswith(orphan_file) for hit in hits),
            what="the row-less mirror file",
        )
        assert any(hit.file.endswith(orphan_file) for hit in raw), (
            "qmd never offered the orphan, so the drop path was not exercised"
        )

        results = await _search(
            sidecar_world, sidecar_repository, sidecar_client, query=ORPHAN_NEEDLE, max_results=10
        )

        assert ORPHAN_ID not in _ids(results)
        for entry, _s, _n in results:
            assert entry["entry_id"] in sidecar_world.rows, "a hit hydrated to a row that is gone"

    async def test_drifted_file_does_not_leak_into_the_answer(
        self, sidecar_world, sidecar_repository, sidecar_client
    ):
        """Overwrite a mirror file with a lie; the caller still sees the row.

        Runs against ``AUTHORITY_ID``, which carries its own token, so drifting
        its file cannot perturb any other test in this module.
        """
        from osprey.services.ariel_search.enhancement.qmd_export import TOUCH_MARKER_NAME
        from osprey.services.ariel_search.enhancement.qmd_export.writer import mirror_path

        row = sidecar_world.rows[AUTHORITY_ID]
        path = mirror_path(sidecar_world.mirror_root, row)
        lie = "the interlock was NEVER reset and the shutter is open"
        path.write_text(
            f"# Entry {AUTHORITY_ID}\n\nThe {AUTHORITY_TOKEN} record says {lie}.\n",
            encoding="utf-8",
        )
        (sidecar_world.mirror_root / TOUCH_MARKER_NAME).write_text(str(time.time()))

        wait_for_indexed(
            sidecar_client,
            ARIEL_COLLECTION,
            AUTHORITY_TOKEN,
            lambda hits: any("NEVER" in (hit.snippet or "") for hit in hits),
            what="the drifted mirror file",
        )

        results = await _search(
            sidecar_world,
            sidecar_repository,
            sidecar_client,
            query=AUTHORITY_TOKEN,
            max_results=10,
        )

        assert results, "the drifted entry should still be findable"
        entry, _score, _snippets = results[0]
        assert entry["entry_id"] == AUTHORITY_ID
        assert entry["raw_text"] == row["raw_text"]
        assert lie not in entry["raw_text"]


# ---------------------------------------------------------------------------
# Vocabulary expansion
# ---------------------------------------------------------------------------


#: ``candidateLimit`` for the two reranked queries, and the only reason these
#: cases pass one at all. **Measured defect, not a tuning preference:** with the
#: lane's configured 40, the reranker's candidate set covers the whole corpus,
#: which includes the 256 KB ``pathological-content`` document — and a reranked
#: query over that document wedges the qmd daemon permanently. It stops
#: answering ``/mcp`` and ``/health`` alike while its container stays up, so
#: every later test in this module fails too, against a session-scoped sidecar
#: no test can restart. Reproduced outside pytest: an 11-document corpus reranks
#: in ~2 s, the same corpus plus one 256 KB document never returns, and the
#: daemon is dead for every session afterwards. Ten keeps the giant document out
#: of the candidate set for these two queries — measured, it does not enter the
#: top ten for either — without weakening what the reranked ordering proves.
#: That exclusion is a ranking accident rather than a structural guarantee, which
#: is why :func:`_assert_the_giant_document_is_out_of_the_candidate_set` reads
#: the precondition before either reranked query is sent.
RERANK_CANDIDATE_LIMIT = 10


class _RecordingClient:
    """A real ``QMDClient`` that remembers the query text it was handed.

    Everything except :meth:`query` is delegated untouched, so the sidecar sees
    exactly the request it would otherwise have seen — this records what crossed
    the wire, it does not stand in for it.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self.queries: list[str] = []

    def __getattr__(self, name: str) -> Any:
        """Delegate every other attribute — ``is_configured``, ``is_available``."""
        return getattr(self._client, name)

    def query(self, collection: str, query: str, **kwargs: Any) -> Any:
        """Record the query text, then run the real one."""
        self.queries.append(query)
        return self._client.query(collection, query, **kwargs)


@pytest.fixture(scope="session")
def vocabulary_file(tmp_path_factory) -> Path:
    """The ``vocabulary.yml`` the expansion cases load, written once."""
    return write_vocabulary_file(Path(tmp_path_factory.mktemp("qmd_ariel_vocabulary")))


@pytest.fixture
def framework_registry(monkeypatch):
    """Route the service's mode lookup through the real framework registry.

    The service resolves ``hybrid`` through ``get_registry()``, which builds
    itself from a project ``config.yml`` this test process does not have. A
    framework-only :class:`RegistryManager` needs no config and registers the
    same real modules, so the dispatch under test is the shipped one — not a
    stand-in descriptor that could agree with a service that had drifted.
    """
    from osprey.registry.manager import RegistryManager

    registry = RegistryManager()
    registry.initialize(silent=True)
    monkeypatch.setattr("osprey.registry.get_registry", lambda *a, **k: registry)
    return registry


async def _service_search(
    world: _World,
    vocabulary_path: Path,
    client: Any,
    *,
    expand_query: bool,
    rerank: bool | None = None,
    candidate_limit: int | None = None,
    expand_modes: list[str] | None = None,
    max_results: int = 10,
):
    """Run one hybrid search through the real service, vocabulary enabled.

    The service is the layer that owns expansion (it resolves ``expand_query``
    against ``expand_by_default`` and ``expand_modes`` and decides whether the
    module is handed a ``query_expansion``), so these cases go through it rather
    than calling ``hybrid_search`` directly the way the rest of this module does.

    ``client`` and ``rerank`` travel in ``advanced_params``, which the service
    forwards to the module as keyword arguments: that is how a per-call
    ``rerank=True`` reaches the sidecar while the lane's config keeps
    ``search_modules.hybrid.settings.rerank: false`` for every other test.

    Args:
        world: The sidecar world, for its config dict.
        vocabulary_path: The vocabulary file to load.
        client: Client for the sidecar, passed straight through to the module.
        expand_query: The caller's explicit ``expand_query`` control.
        rerank: Per-call rerank override, or None to leave config's value.
        candidate_limit: Per-call ``candidateLimit`` override, or None for
            config's 40. See :data:`RERANK_CANDIDATE_LIMIT`.
        expand_modes: ``ariel.vocabulary.expand_modes``, or None for the default.
        max_results: Result cap.

    Returns:
        The ``ARIELSearchResult``.
    """
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.service import create_ariel_service

    config = ARIELConfig.from_dict(
        qmd_ariel_vocabulary_config(world.config_dict, vocabulary_path, expand_modes=expand_modes)
    )
    advanced: dict[str, Any] = {"expand_query": expand_query, "client": client}
    if rerank is not None:
        advanced["rerank"] = rerank
    if candidate_limit is not None:
        advanced["candidate_limit"] = candidate_limit

    async with await create_ariel_service(config) as service:
        return await service.search(
            VOCABULARY_QUERY,
            mode="hybrid",
            max_results=max_results,
            advanced_params=advanced,
        )


def _result_ids(result) -> list[str]:
    """The entry ids of an ``ARIELSearchResult``, in ranking order."""
    return [entry["entry_id"] for entry in result.entries]


def _assert_the_giant_document_is_out_of_the_candidate_set(client: Any, query: str) -> None:
    """Fail in one line rather than hang, if the wedge precondition has moved.

    :data:`RERANK_CANDIDATE_LIMIT` protects the reranked cases by *ranking
    accident*: the 256 KB ``pathological-content`` document simply does not reach
    the top ten for these queries. Nothing structural keeps it out, so a corpus
    edit, a model bump or a reworded needle could pull it in — and the symptom
    would not be a failed assertion but a wedged daemon and a timed-out step,
    with the module's other 45 tests failing behind it.

    A non-reranked query at the same limit returns exactly the candidate set the
    reranker would be handed (same shape ``wait_for_indexed`` uses), so this
    reads the precondition directly, before anything reranked runs.

    Args:
        client: A ``QMDClient`` pointed at the sidecar.
        query: The exact query text that is about to be sent reranked.
    """
    hits = client.query(
        ARIEL_COLLECTION,
        query,
        limit=RERANK_CANDIDATE_LIMIT,
        rerank=False,
        candidate_limit=RERANK_CANDIDATE_LIMIT,
    )
    reported = [Path(hit.file).name for hit in hits]
    giant = f"{encode_entry_id(HOSTILE_IDS['clean'])}.md"

    assert giant not in reported, (
        f"the 256 KB {HOSTILE_IDS['clean']!r} document entered the reranker's candidate set "
        f"for {query!r} — reranking it WEDGES the qmd daemon permanently (it answers neither "
        f"/mcp nor /health afterwards, failing every later test in this module). Refusing to "
        f"send the reranked query. Candidate set: {reported}. "
        f"See RERANK_CANDIDATE_LIMIT for the measurement."
    )


class TestVocabularyExpansionReachesTheSidecar:
    """Expansion changes what the sidecar ranks, measured against the real one.

    The seeded ``VOCABULARY_ENTRY_ID`` row is written entirely in canonical
    phrases (``troubleshoot``, ``beam position monitor``) and the query entirely
    in facility shorthand (``t/s bpm``), so nothing lexical connects them — and
    ``VOCABULARY_DISTRACTOR_ROWS`` gives the unexpanded query somewhere else to
    go, without which the sidecar's vector leg reaches the entry unaided and the
    comparison measures nothing. Measured on this corpus, twice on fresh
    containers and fresh indexes, byte-identical both times: **rank 2 expanded**
    (one slot of headroom under the top-3 assertion) and **rank 7 unexpanded**
    (four slots below the same boundary). How many distractor rows it takes to
    sit there is itself measured — see ``VOCABULARY_DISTRACTOR_ROWS``.

    Reranking is switched on **per call** here — the one place in this module
    where rank quality is the assertion — while the lane's config keeps
    ``rerank: false`` for every other test, which is what keeps the rest of the
    suite at its measured 0.85 s per query. See :data:`RERANK_CANDIDATE_LIMIT`
    for the one knob that has to travel with it.
    """

    async def test_expansion_lifts_the_canonical_entry_into_the_top_three(
        self, sidecar_world, sidecar_client, vocabulary_file, framework_registry
    ):
        """The differential: same query, same corpus, expansion the only change.

        Both halves run in one test on purpose. Split apart, a change in the
        corpus or the reranker could move the entry for reasons that have
        nothing to do with expansion and each half would still look sound; the
        claim being made is a comparison, so it is asserted as one.
        """
        # The reranked calls below are the ones that can wedge the daemon, so the
        # precondition that keeps them safe is read first — for the expanded query
        # as it actually crosses the wire, which only the module can spell.
        recorder = _RecordingClient(sidecar_client)
        await _service_search(sidecar_world, vocabulary_file, recorder, expand_query=True)
        for wire_query in (recorder.queries[0], VOCABULARY_QUERY):
            _assert_the_giant_document_is_out_of_the_candidate_set(sidecar_client, wire_query)

        expanded = await _service_search(
            sidecar_world,
            vocabulary_file,
            sidecar_client,
            expand_query=True,
            rerank=True,
            candidate_limit=RERANK_CANDIDATE_LIMIT,
        )
        unexpanded = await _service_search(
            sidecar_world,
            vocabulary_file,
            sidecar_client,
            expand_query=False,
            rerank=True,
            candidate_limit=RERANK_CANDIDATE_LIMIT,
        )

        assert VOCABULARY_ENTRY_ID in _result_ids(expanded)[:3], (
            f"the expanded query did not surface {VOCABULARY_ENTRY_ID!r} in the top 3: "
            f"{_result_ids(expanded)}"
        )
        assert VOCABULARY_ENTRY_ID not in _result_ids(unexpanded)[:3], (
            "the unexpanded query already reaches the entry, so this measures nothing: "
            f"{_result_ids(unexpanded)}"
        )

    async def test_expanded_terms_name_both_shorthand_pairs(
        self, sidecar_world, sidecar_client, vocabulary_file, framework_registry
    ):
        """The transparency payload the caller sees alongside the results.

        The entry-count assertion is not decoration: a search that fails is
        still reported with the expansion it carried, so without it this case
        would pass just as happily against a sidecar that never answered.
        """
        result = await _service_search(
            sidecar_world, vocabulary_file, sidecar_client, expand_query=True
        )

        assert result.entries, f"the search returned nothing: {result.reasoning}"
        pairs = {group["original"]: tuple(group["alternatives"]) for group in result.expanded_terms}
        assert pairs == VOCABULARY_EXPECTED_PAIRS

    async def test_the_sidecar_receives_the_flattened_query(
        self, sidecar_world, sidecar_client, vocabulary_file, framework_registry
    ):
        """Expansion has to reach the daemon, not merely the result payload.

        Without this, a module that resolved an expansion and then queried with
        the raw text would still satisfy the transparency assertion above.
        """
        recorder = _RecordingClient(sidecar_client)

        await _service_search(sidecar_world, vocabulary_file, recorder, expand_query=True)

        assert len(recorder.queries) == 1, recorder.queries
        sent = recorder.queries[0]
        assert sent != VOCABULARY_QUERY
        for alternatives in VOCABULARY_EXPECTED_PAIRS.values():
            for alternative in alternatives:
                assert alternative in sent, f"{alternative!r} missing from {sent!r}"


class TestVocabularyExpandModesGateTheSidecar:
    """``expand_modes`` that omits ``hybrid`` leaves this module unexpanded."""

    async def test_keyword_only_expand_modes_sends_the_unexpanded_query(
        self, sidecar_world, sidecar_client, vocabulary_file, framework_registry
    ):
        """A usable vocabulary the caller asked for, scoped away from hybrid.

        Both halves matter: an empty ``expanded_terms`` alone would also be
        produced by a module that expanded the query and forgot to report it,
        and an unexpanded query alone would also be produced by a vocabulary
        that failed to load.

        What this test does **not** rule out on its own is that same failed
        load: ``ARIELConfig.from_dict`` does not raise when a vocabulary file
        cannot be read — it collects the problems into ``vocabulary_errors`` and
        leaves ``loaded_vocabulary`` unset — and an inert vocabulary produces
        exactly these two observations. That the session's vocabulary file loads
        is proved by
        :meth:`TestVocabularyExpansionReachesTheSidecar.test_expanded_terms_name_both_shorthand_pairs`,
        which runs the same file through the same fixture and reads real
        expansions back. This case is only meaningful alongside it.
        """
        recorder = _RecordingClient(sidecar_client)

        result = await _service_search(
            sidecar_world,
            vocabulary_file,
            recorder,
            expand_query=True,
            expand_modes=["keyword"],
        )

        assert result.expanded_terms == ()
        assert recorder.queries == [VOCABULARY_QUERY]
