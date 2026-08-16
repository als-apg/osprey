"""Tests for the qmd-backed ranked backend of :meth:`OKFBundle.search`.

The MCP protocol itself is covered in ``tests/services/qmd/test_client.py``.
What is proven here is everything the bundle adds on top of it: which knobs
reach the daemon, how a qmd result path maps back onto an OKF §2 concept ID,
which hits are refused, and — the part with operational consequences — that a
missing sidecar degrades silently while a broken one degrades loudly, exactly
once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from osprey.deployment.qmd_service import QMDServiceConfig
from osprey.services.facility_knowledge.okf.bundle import (
    OKF_COLLECTION,
    OKFBundle,
    OKFSearchSettings,
)
from osprey.services.qmd import QMDClient, QMDClientError, QMDSearchResult, QMDUnavailableError

# ---------------------------------------------------------------------------
# Fixtures and doubles
# ---------------------------------------------------------------------------


def _write(path: Path, body: str) -> None:
    """Write *body* to *path*, creating parents and stripping indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(body).lstrip(), encoding="utf-8")


@pytest.fixture
def bundle_root(tmp_path: Path) -> Path:
    """A small bundle with two concepts, an index, and a log."""
    _write(
        tmp_path / "index.md",
        """\
        # Facility Bundle

        * [Corrector Magnets](/magnets/corrector_magnets.md) - Steering.
        """,
    )
    _write(tmp_path / "log.md", "# Change log\n\nNothing yet.\n")
    _write(
        tmp_path / "magnets" / "corrector_magnets.md",
        """\
        ---
        title: Corrector Magnets
        type: component
        ---

        Corrector magnets steer the beam.
        """,
    )
    _write(
        tmp_path / "diagnostics" / "beam position monitors.md",
        """\
        ---
        title: Beam Position Monitors
        type: component
        ---

        BPMs measure the beam orbit.
        """,
    )
    return tmp_path


def _hit(file: str, *, score: float = 1.0, snippet: str = "1: a line") -> QMDSearchResult:
    """Build one qmd hit with its collection prefix already stripped."""
    return QMDSearchResult(
        docid="#abc123",
        file=file,
        collection=OKF_COLLECTION,
        title="",
        score=score,
        line=1,
        snippet=snippet,
    )


class _FakeQMDClient(QMDClient):
    """A configured client whose protocol is replaced by a canned answer.

    Subclasses the real client so the bundle is exercised through the type it
    actually holds, while the HTTP conversation — proven elsewhere — is out of
    the way.

    Args:
        hits: Results :meth:`query` returns.
        available: What :meth:`is_available` reports.
        error: Raised by :meth:`query` instead of returning *hits*.
    """

    def __init__(
        self,
        hits: list[QMDSearchResult] | None = None,
        *,
        available: bool = True,
        error: Exception | None = None,
    ) -> None:
        super().__init__(QMDServiceConfig(port=8180))
        self._hits = hits or []
        self.available = available
        self._error = error
        self.queries: list[dict[str, Any]] = []

    def is_available(self) -> bool:
        """Report the canned availability without probing anything."""
        return self.available

    def query(self, *args: Any, **kwargs: Any) -> list[QMDSearchResult]:
        """Record the call and return the canned hits.

        Raises:
            Exception: Whatever *error* was constructed with, if any.
        """
        self.queries.append({"args": args, "kwargs": kwargs})
        if self._error is not None:
            raise self._error
        return self._hits


# ---------------------------------------------------------------------------
# Query knobs
# ---------------------------------------------------------------------------


class TestQueryKnobs:
    """What the bundle asks the daemon for."""

    def test_queries_the_okf_collection(self, bundle_root: Path) -> None:
        client = _FakeQMDClient([])
        OKFBundle(bundle_root, qmd_client=client).search("steering")

        assert client.queries[0]["args"] == (OKF_COLLECTION, "steering")

    def test_rerank_defaults_to_false(self, bundle_root: Path) -> None:
        """Reranking costs ~4x and is near-constant in corpus size.

        No OKF bundle is small enough to outrun it, so the interactive default
        is off — and it is passed explicitly, because qmd's own default is on.
        """
        client = _FakeQMDClient([])
        OKFBundle(bundle_root, qmd_client=client).search("steering")

        assert client.queries[0]["kwargs"]["rerank"] is False

    def test_settings_reach_the_daemon(self, bundle_root: Path) -> None:
        client = _FakeQMDClient([])
        bundle = OKFBundle(
            bundle_root,
            qmd_client=client,
            search_settings=OKFSearchSettings(rerank=True, candidate_limit=12),
        )
        bundle.search("steering", limit=3)

        kwargs = client.queries[0]["kwargs"]
        assert kwargs["rerank"] is True
        assert kwargs["candidate_limit"] == 12
        assert kwargs["limit"] == 3


# ---------------------------------------------------------------------------
# Result mapping
# ---------------------------------------------------------------------------


class TestResultMapping:
    """Turning qmd hits back into concepts of this bundle."""

    def test_maps_paths_to_concept_ids_in_rank_order(self, bundle_root: Path) -> None:
        client = _FakeQMDClient(
            [
                _hit("magnets/corrector_magnets.md", score=1.0),
                _hit("diagnostics/beam position monitors.md", score=0.5),
            ]
        )
        results = OKFBundle(bundle_root, qmd_client=client).search("steering")

        assert [r.concept_id for r in results] == [
            "magnets/corrector_magnets",
            "diagnostics/beam position monitors",
        ]
        assert [r.score for r in results] == [1.0, 0.5]

    def test_carries_qmd_snippet_and_parsed_document(self, bundle_root: Path) -> None:
        client = _FakeQMDClient([_hit("magnets/corrector_magnets.md", snippet="3: steer the beam")])
        results = OKFBundle(bundle_root, qmd_client=client).search("steering")

        assert results[0].snippet == "3: steer the beam"
        assert results[0].document.frontmatter["title"] == "Corrector Magnets"

    def test_percent_encoded_paths_are_decoded(self, bundle_root: Path) -> None:
        """qmd percent-encodes result paths; the concept ID is the decoded one."""
        client = _FakeQMDClient([_hit("diagnostics/beam%20position%20monitors.md")])
        results = OKFBundle(bundle_root, qmd_client=client).search("orbit")

        assert [r.concept_id for r in results] == ["diagnostics/beam position monitors"]

    @pytest.mark.parametrize("reserved", ["index.md", "magnets/index.md", "log.md"])
    def test_reserved_names_are_never_returned(self, bundle_root: Path, reserved: str) -> None:
        """OKF §3.1 reserved files are not concepts, however they were indexed."""
        client = _FakeQMDClient([_hit(reserved), _hit("magnets/corrector_magnets.md")])
        results = OKFBundle(bundle_root, qmd_client=client).search("facility")

        assert [r.concept_id for r in results] == ["magnets/corrector_magnets"]

    def test_paths_escaping_the_bundle_root_are_refused(self, bundle_root: Path) -> None:
        """The containment check that guards read_concept guards this too."""
        client = _FakeQMDClient([_hit("../../etc/passwd.md"), _hit("magnets/corrector_magnets.md")])
        results = OKFBundle(bundle_root, qmd_client=client).search("passwd")

        assert [r.concept_id for r in results] == ["magnets/corrector_magnets"]

    def test_hits_missing_from_disk_are_dropped(self, bundle_root: Path) -> None:
        """The index may lag the filesystem; a stale hit is not an error."""
        client = _FakeQMDClient([_hit("magnets/deleted_yesterday.md")])
        results = OKFBundle(bundle_root, qmd_client=client).search("steering")

        assert results == []

    def test_unparseable_documents_are_dropped(self, bundle_root: Path) -> None:
        _write(bundle_root / "broken.md", "---\n: not: yaml: at all\n---\n\nBody.\n")
        client = _FakeQMDClient([_hit("broken.md"), _hit("magnets/corrector_magnets.md")])
        results = OKFBundle(bundle_root, qmd_client=client).search("broken")

        assert [r.concept_id for r in results] == ["magnets/corrector_magnets"]

    def test_duplicate_hits_are_collapsed(self, bundle_root: Path) -> None:
        client = _FakeQMDClient(
            [_hit("magnets/corrector_magnets.md"), _hit("magnets/corrector_magnets.md", score=0.5)]
        )
        results = OKFBundle(bundle_root, qmd_client=client).search("steering")

        assert [r.concept_id for r in results] == ["magnets/corrector_magnets"]


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


class TestFallback:
    """When the ranked backend cannot answer."""

    def test_unconfigured_falls_back_without_logging(
        self, bundle_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No sidecar is a supported deployment, not a fault to report."""
        bundle = OKFBundle(bundle_root, qmd_client=QMDClient(None))

        with caplog.at_level(logging.DEBUG):
            results = bundle.search("steer")

        assert [r.concept_id for r in results] == ["magnets/corrector_magnets"]
        assert results[0].score is None
        assert caplog.records == []

    def test_configured_but_down_warns_and_falls_back(
        self, bundle_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        bundle = OKFBundle(bundle_root, qmd_client=_FakeQMDClient(available=False))

        with caplog.at_level(logging.WARNING):
            results = bundle.search("steer")

        assert [r.concept_id for r in results] == ["magnets/corrector_magnets"]
        assert results[0].score is None
        assert [r.levelname for r in caplog.records] == ["WARNING"]
        assert "falling back to substring search" in caplog.text

    def test_outage_warns_once_not_once_per_query(
        self, bundle_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An open panel polls search; a stopped sidecar must not flood the log."""
        bundle = OKFBundle(bundle_root, qmd_client=_FakeQMDClient(available=False))

        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                bundle.search("steer")

        assert len(caplog.records) == 1

    def test_recovery_re_arms_the_warning(
        self, bundle_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A second outage is a second event, and is reported as one."""
        client = _FakeQMDClient([_hit("magnets/corrector_magnets.md")], available=False)
        bundle = OKFBundle(bundle_root, qmd_client=client)

        with caplog.at_level(logging.WARNING):
            bundle.search("steer")
            client.available = True
            assert bundle.search("steer")[0].score == 1.0
            client.available = False
            bundle.search("steer")

        assert len(caplog.records) == 2

    @pytest.mark.parametrize(
        "error",
        [QMDClientError("tool failed"), QMDUnavailableError("host went away")],
        ids=["tool-error", "unreachable"],
    )
    def test_a_failed_query_falls_back_rather_than_raising(
        self, bundle_root: Path, caplog: pytest.LogCaptureFixture, error: Exception
    ) -> None:
        """Reached-but-broken looks the same to a caller as stopped."""
        bundle = OKFBundle(bundle_root, qmd_client=_FakeQMDClient(error=error))

        with caplog.at_level(logging.WARNING):
            results = bundle.search("steer")

        assert [r.concept_id for r in results] == ["magnets/corrector_magnets"]
        assert results[0].score is None
        assert len(caplog.records) == 1

    def test_an_empty_ranked_result_is_not_a_fallback(self, bundle_root: Path) -> None:
        """A qmd result set of zero is an answer, and substring must not override it."""
        bundle = OKFBundle(bundle_root, qmd_client=_FakeQMDClient([]))

        assert bundle.search("steer") == []


# ---------------------------------------------------------------------------
# Settings resolution
# ---------------------------------------------------------------------------


class TestSearchSettings:
    """Reading ``facility_knowledge.search`` out of a project config."""

    @pytest.mark.parametrize(
        "config",
        [None, {}, {"facility_knowledge": {}}, {"facility_knowledge": {"search": {}}}],
        ids=["none", "empty", "no-search-block", "empty-search-block"],
    )
    def test_absent_config_yields_the_interactive_defaults(self, config: Any) -> None:
        settings = OKFSearchSettings.from_config(config)

        assert settings.rerank is False
        assert settings.candidate_limit is None
        assert settings.collection == OKF_COLLECTION

    def test_reads_both_knobs(self) -> None:
        settings = OKFSearchSettings.from_config(
            {"facility_knowledge": {"search": {"rerank": True, "candidate_limit": 20}}}
        )

        assert settings.rerank is True
        assert settings.candidate_limit == 20

    @pytest.mark.parametrize("value", ["false", 0, None], ids=["string", "int", "null"])
    def test_a_non_boolean_rerank_is_refused(self, value: Any) -> None:
        """Defaulting past ``rerank: "false"`` would silently keep the slow path."""
        with pytest.raises(ValueError, match="rerank"):
            OKFSearchSettings.from_config({"facility_knowledge": {"search": {"rerank": value}}})

    @pytest.mark.parametrize(
        "value", [0, -1, "40", 1.5, True], ids=["zero", "neg", "str", "float", "bool"]
    )
    def test_a_bad_candidate_limit_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match="candidate_limit"):
            OKFSearchSettings.from_config(
                {"facility_knowledge": {"search": {"candidate_limit": value}}}
            )


# ---------------------------------------------------------------------------
# Recovering an ID the daemon normalised
# ---------------------------------------------------------------------------


class TestNormalisedPathRecovery:
    """qmd does not echo the filename it indexed — it normalises it.

    Measured against qmd 2.5.3 over a real bundle: every ``_`` in a reported
    path comes back as ``-``, so ``magnets/corrector_magnets.md`` is reported as
    ``magnets/corrector-magnets.md``. Taken at face value that names no file on
    disk, the read fails, and the hit is dropped with no error at any layer —
    the search just returns fewer results, which reads exactly like a bundle
    that does not cover the topic.

    Underscore is one of the two conventions real bundles use for concept IDs,
    so for a facility that picked it this is silent, total loss of ranked
    search.
    """

    def test_a_normalised_underscore_path_is_recovered(self, bundle_root: Path) -> None:
        client = _FakeQMDClient([_hit("magnets/corrector-magnets.md")])
        bundle = OKFBundle(bundle_root, qmd_client=client)

        results = bundle.search("steering")

        assert [r.concept_id for r in results] == ["magnets/corrector_magnets"]

    def test_a_normalised_space_path_is_recovered(self, bundle_root: Path) -> None:
        """Spaces fold the same way underscores do, and a concept named with
        them is no less real."""
        client = _FakeQMDClient([_hit("diagnostics/beam-position-monitors.md")])
        bundle = OKFBundle(bundle_root, qmd_client=client)

        results = bundle.search("orbit")

        assert [r.concept_id for r in results] == ["diagnostics/beam position monitors"]

    def test_an_exact_path_needs_no_recovery(self, bundle_root: Path) -> None:
        """The common case — a bundle whose IDs survive normalisation unchanged
        — resolves directly and never scans the bundle."""
        client = _FakeQMDClient([_hit("magnets/corrector_magnets.md")])
        bundle = OKFBundle(bundle_root, qmd_client=client)

        assert [r.concept_id for r in bundle.search("steering")] == ["magnets/corrector_magnets"]

    def test_directory_structure_still_separates_concepts(self, bundle_root: Path) -> None:
        """``alpha/beta_gamma`` and ``alpha_beta/gamma`` must not collide.

        Folding the separator along with the punctuation would make these one
        key, turning two perfectly distinct concepts into an ambiguity that
        drops both.
        """
        for concept_id in ("alpha/beta_gamma", "alpha_beta/gamma"):
            _write(
                bundle_root / f"{concept_id}.md",
                f"""\
                ---
                title: {concept_id}
                type: component
                ---

                Body.
                """,
            )
        client = _FakeQMDClient([_hit("alpha/beta-gamma.md")])
        bundle = OKFBundle(bundle_root, qmd_client=client)

        assert [r.concept_id for r in bundle.search("anything")] == ["alpha/beta_gamma"]

    def test_an_ambiguous_normalisation_drops_the_hit_and_warns(
        self, bundle_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two concepts the daemon would report identically is a coin flip.

        Returning either would present as a correct answer, so the hit is
        dropped — but the collision is a property of the bundle that an operator
        can fix, so it has to be discoverable rather than silent.
        """
        _write(
            bundle_root / "magnets" / "corrector magnets.md",
            """\
            ---
            title: Corrector Magnets (duplicate)
            type: component
            ---

            A second document whose ID normalises the same way as the first.
            """,
        )
        client = _FakeQMDClient([_hit("magnets/corrector-magnets.md")])
        bundle = OKFBundle(bundle_root, qmd_client=client)

        with caplog.at_level(logging.WARNING, logger=OKFBundle.__module__):
            results = bundle.search("steering")

        assert results == []
        collisions = [r for r in caplog.records if "could be" in r.getMessage()]
        assert len(collisions) == 1
        assert "magnets/corrector magnets" in collisions[0].getMessage()
        assert "magnets/corrector_magnets" in collisions[0].getMessage()

    def test_an_exact_match_wins_over_an_otherwise_ambiguous_fold(
        self, bundle_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Ambiguity is only ambiguity when nothing matches exactly.

        A bundle can hold ``corrector-magnets`` and ``corrector_magnets`` side by
        side, and a report naming the first is not a coin flip — it names a file
        that is there. Only a report matching neither has to choose.
        """
        _write(
            bundle_root / "magnets" / "corrector-magnets.md",
            """\
            ---
            title: Corrector Magnets (hyphenated sibling)
            type: component
            ---

            A distinct concept whose ID the daemon reports unchanged.
            """,
        )
        client = _FakeQMDClient([_hit("magnets/corrector-magnets.md")])
        bundle = OKFBundle(bundle_root, qmd_client=client)

        with caplog.at_level(logging.WARNING, logger=OKFBundle.__module__):
            results = bundle.search("steering")

        assert [r.concept_id for r in results] == ["magnets/corrector-magnets"]
        assert [r for r in caplog.records if "could be" in r.getMessage()] == []

    def test_a_path_matching_no_concept_is_dropped_quietly(self, bundle_root: Path) -> None:
        """The index is a copy of the filesystem and is allowed to lag it, so a
        hit for a document that is simply gone is not a warning."""
        client = _FakeQMDClient([_hit("magnets/deleted-since-indexing.md")])
        bundle = OKFBundle(bundle_root, qmd_client=client)

        assert bundle.search("steering") == []

    def test_the_bundle_is_swept_once_per_search_not_once_per_hit(
        self, bundle_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recovery is needed by EVERY hit of an underscore-convention bundle.

        So the sweep has to be hoisted out of the result loop: rebuilt per hit,
        a default ``limit=10`` search performs ten identical filesystem walks on
        the panel's interactive path. Counted rather than timed, because a
        timing assertion here would be a flake generator.
        """
        import osprey.services.facility_knowledge.okf.bundle as bundle_module

        sweeps = 0
        real = bundle_module._normalised_concept_index

        def counting(root: Path) -> dict[str, list[str]]:
            nonlocal sweeps
            sweeps += 1
            return real(root)

        monkeypatch.setattr(bundle_module, "_normalised_concept_index", counting)
        client = _FakeQMDClient([_hit("magnets/corrector-magnets.md") for _ in range(10)])
        bundle = OKFBundle(bundle_root, qmd_client=client)

        results = bundle.search("steering")

        assert [r.concept_id for r in results] == ["magnets/corrector_magnets"]
        assert sweeps == 1, f"expected one sweep for the whole search, got {sweeps}"

    def test_a_bundle_needing_no_recovery_is_never_swept(
        self, bundle_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hoist must not become a tax on bundles that do not need it.

        A hyphen-convention bundle resolves every reported path directly, so the
        sweep is never built — which is why it is lazy rather than unconditional.
        """
        import osprey.services.facility_knowledge.okf.bundle as bundle_module

        _write(
            bundle_root / "magnets" / "corrector-magnets.md",
            """\
            ---
            title: Corrector Magnets (hyphenated)
            type: component
            ---

            Reported by the daemon exactly as it is stored.
            """,
        )

        def refuse(root: Path) -> dict[str, list[str]]:
            raise AssertionError("swept the bundle for a search that needed no recovery")

        monkeypatch.setattr(bundle_module, "_normalised_concept_index", refuse)
        client = _FakeQMDClient([_hit("magnets/corrector-magnets.md")])
        bundle = OKFBundle(bundle_root, qmd_client=client)

        assert [r.concept_id for r in bundle.search("steering")] == ["magnets/corrector-magnets"]

    def test_a_concept_drafted_after_an_earlier_search_is_recoverable(
        self, bundle_root: Path
    ) -> None:
        """The invalidation requirement, stated as behaviour.

        A bundle gains concepts at runtime — that is what ``draft_concept`` plus
        the ``.qmd-touch`` marker exist to do. Recovery therefore re-reads the
        bundle on every call that needs it; a map built once at first search
        would make every later draft permanently unfindable, which is a worse
        failure than the one this recovery exists to fix.
        """
        client = _FakeQMDClient([_hit("magnets/corrector-magnets.md")])
        bundle = OKFBundle(bundle_root, qmd_client=client)
        assert [r.concept_id for r in bundle.search("steering")] == ["magnets/corrector_magnets"]

        _write(
            bundle_root / "operations" / "beam_dump_recovery.md",
            """\
            ---
            title: Beam Dump Recovery
            type: procedure
            ---

            Drafted after the first search ran.
            """,
        )
        client._hits = [_hit("operations/beam-dump-recovery.md")]

        assert [r.concept_id for r in bundle.search("recovery")] == [
            "operations/beam_dump_recovery"
        ]
