"""``osprey.apps.sdk.corpus``: a declared qmd corpus read and searched from application code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.apps.sdk import Corpus, CorpusHit, CorpusUnavailable, corpus
from osprey.apps.sdk.testing import corpus_stand_in
from osprey.services.qmd import client as qmd_client
from tests.services.qmd.test_client import StubTransport

URL = "http://qmd-procedures:10062"


def _result(*files: str) -> dict[str, Any]:
    """A ``query`` result naming *files*, as the daemon reports them."""
    rows = [
        {
            "docid": f"#{position}",
            "file": f"procedures/{file}",
            "title": f"Title {position}",
            "score": 1 / (position + 1),
            "context": None,
            "line": position + 1,
            "snippet": f"{position + 1}: text",
        }
        for position, file in enumerate(files)
    ]
    return {"content": [], "structuredContent": {"results": rows}}


@pytest.fixture
def bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The environment the runtime template hands a container that lists ``procedures``."""
    monkeypatch.setenv("OSPREY_APP_CORPORA", f"procedures={tmp_path}:ops_notes=/nowhere")
    monkeypatch.setenv("OSPREY_QMD_PROCEDURES_URL", URL)
    return tmp_path


def _sidecar(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> StubTransport:
    transport = StubTransport(**kwargs)
    monkeypatch.setattr(qmd_client, "_UrllibTransport", lambda: transport)
    return transport


def test_a_declared_corpus_reads_from_its_bound_directory(bound: Path) -> None:
    (bound / "bakeout.md").write_text("# Bakeout\n")

    procedures = corpus("procedures")

    assert isinstance(procedures, Corpus)
    assert procedures.name == "procedures"
    assert procedures.path == bound
    assert (procedures.path / "bakeout.md").read_text() == "# Bakeout\n"
    assert corpus("ops_notes").path == Path("/nowhere")


def test_search_asks_the_corpus_sidecar_and_returns_its_hits(
    bound: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (bound / "vacuum").mkdir()
    (bound / "vacuum" / "bakeout.md").write_text("# Bakeout\n")
    sidecar = _sidecar(monkeypatch, tool_result=_result("vacuum/bakeout.md"))

    hits = corpus("procedures").search("bakeout temperature", limit=3)

    assert hits == [
        CorpusHit(path="vacuum/bakeout.md", title="Title 0", score=1.0, line=1, snippet="1: text")
    ]
    (call,) = sidecar.tool_calls
    assert call["arguments"]["collections"] == ["procedures"]
    assert call["arguments"]["limit"] == 3
    assert sidecar.posts[0][1] is not None
    assert all(URL in url for url in sidecar.health_calls)


def test_a_hit_names_the_file_on_disk_not_the_sidecars_spelling(
    bound: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (bound / "RF_Systems").mkdir()
    (bound / "RF_Systems" / "Cavity_Tuning v2.md").write_text("# Tuning\n")
    _sidecar(
        monkeypatch,
        tool_result=_result("rf-systems/cavity-tuning-v2.md", "gone/no-such-file.md"),
    )

    hits = corpus("procedures").search("tuning")

    assert [hit.path for hit in hits] == ["RF_Systems/Cavity_Tuning v2.md", "gone/no-such-file.md"]


def test_two_files_one_reported_name_keeps_the_sidecars_spelling(
    bound: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (bound / "a_b.md").write_text("one\n")
    (bound / "a b.md").write_text("two\n")
    _sidecar(monkeypatch, tool_result=_result("a-b.md"))

    assert [hit.path for hit in corpus("procedures").search("x")] == ["a-b.md"]


def test_a_sidecar_that_is_down_is_corpus_unavailable(
    bound: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sidecar(monkeypatch, unreachable=True)

    with pytest.raises(CorpusUnavailable, match="corpus 'procedures' cannot be searched"):
        corpus("procedures").search("bakeout")
    assert corpus("procedures").path == bound


@pytest.mark.usefixtures("bound")
def test_a_corpus_with_no_sidecar_address_is_corpus_unavailable() -> None:
    with pytest.raises(CorpusUnavailable, match="OSPREY_QMD_OPS_NOTES_URL is unset"):
        corpus("ops_notes").search("anything")


@pytest.mark.usefixtures("bound")
def test_a_corpus_the_container_did_not_declare_names_the_manifest_key() -> None:
    with pytest.raises(LookupError, match=r"containers\.<key>\.corpora"):
        corpus("papers")


def test_a_process_handed_no_corpus_names_the_manifest_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OSPREY_APP_CORPORA", raising=False)

    with pytest.raises(LookupError, match="no corpus named 'procedures' is bound here"):
        corpus("procedures")


# ---------------------------------------------------------------------------
# The stand-in an application's own tests use
# ---------------------------------------------------------------------------

HITS = [
    CorpusHit("bakeout.md", "Bakeout", 1.0, 3, "3: bake at 150 C"),
    CorpusHit("venting.md", "Venting", 0.4, 9, "9: vent slowly"),
]


def test_a_stand_in_answers_from_a_directory_and_canned_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OSPREY_APP_CORPORA", raising=False)
    (tmp_path / "bakeout.md").write_text("# Bakeout\n")

    with corpus_stand_in("procedures", tmp_path, HITS) as stood_in:
        procedures = corpus("procedures")
        assert procedures is stood_in
        assert procedures.path == tmp_path
        assert procedures.search("anything") == HITS
        assert procedures.search("anything", limit=1) == HITS[:1]

    with pytest.raises(LookupError):
        corpus("procedures")


def test_an_unavailable_stand_in_raises_as_a_sidecar_that_is_down(tmp_path: Path) -> None:
    with corpus_stand_in("procedures", tmp_path, unavailable=True):
        with pytest.raises(CorpusUnavailable):
            corpus("procedures").search("anything")
        assert corpus("procedures").path == tmp_path


def test_the_fixture_stands_each_corpus_in_for_the_test(corpus_stand_in, tmp_path: Path) -> None:
    """The ``pytest11`` fixture: the same arguments, undone when the test ends."""
    stood_in = corpus_stand_in("procedures", tmp_path, HITS)
    corpus_stand_in("ops_notes", tmp_path, unavailable=True)

    assert corpus("procedures") is stood_in
    assert corpus("procedures").search("anything") == HITS
    with pytest.raises(CorpusUnavailable):
        corpus("ops_notes").search("anything")
