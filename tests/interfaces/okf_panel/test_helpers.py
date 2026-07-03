"""Unit tests for okf_panel.helpers (make_snippet, group_concepts, structure md).

Mirrors the ALS ``mcp_servers/okf_panel/tests/test_helpers.py`` style, adapted
to the core import paths and the facility-neutral structure title.
"""

from osprey.interfaces.okf_panel.helpers import (
    build_structure_markdown,
    group_concepts,
    make_snippet,
)
from osprey.services.facility_knowledge.okf.bundle import ConceptEntry

# ---------------------------------------------------------------------------
# make_snippet
# ---------------------------------------------------------------------------


def test_snippet_found_mid_body():
    body = (
        "The accelerator stores beam in the storage ring.\n"
        "Operators monitor the global orbit feedback system continuously\n"
        "to keep the closed orbit stable during user operations. "
        + "Trailing padding text to push the end well past the match. "
        * 5
    )
    snippet = make_snippet(body, "orbit feedback")

    assert "orbit feedback" in snippet
    assert snippet.startswith("…")
    assert snippet.endswith("…")
    assert "\n" not in snippet
    assert "  " not in snippet


def test_snippet_query_at_start_no_leading_ellipsis():
    body = "Orbit correction adjusts the beam path using corrector magnets " * 5
    snippet = make_snippet(body, "Orbit")

    assert not snippet.startswith("…")
    assert "Orbit" in snippet


def test_snippet_query_near_end_no_trailing_ellipsis():
    body = "Some leading context that is reasonably long. " * 5 + "tail token"
    snippet = make_snippet(body, "tail token")

    assert snippet.endswith("token")
    assert not snippet.endswith("…")
    assert snippet.startswith("…")


def test_snippet_query_absent_returns_empty():
    body = "The beam current decays slowly between top-off injections."
    assert make_snippet(body, "nonexistent phrase") == ""


def test_snippet_empty_query_returns_empty():
    body = "Any non-empty body text goes here for the test."
    assert make_snippet(body, "") == ""


def test_snippet_case_insensitive_match():
    body = "Operators watch the orbit drift throughout the shift."
    snippet = make_snippet(body, "ORBIT")
    assert "orbit" in snippet


def test_snippet_collapses_whitespace_runs():
    body = "alpha\n\n\t  beta   query-term\n\n   gamma   delta"
    snippet = make_snippet(body, "query-term")

    assert "query-term" in snippet
    assert "  " not in snippet
    assert "\n" not in snippet
    assert "\t" not in snippet


# ---------------------------------------------------------------------------
# group_concepts
# ---------------------------------------------------------------------------


def test_group_concepts_groups_by_first_segment():
    entries = [
        ConceptEntry(concept_id="devices/bpm", title="BPM"),
        ConceptEntry(concept_id="devices/rf-system", title="RF System"),
        ConceptEntry(concept_id="references/als-terminology", title="ALS Terminology"),
    ]
    payload = group_concepts(entries)

    ids = [g["id"] for g in payload["groups"]]
    assert ids == ["devices", "references"]

    devices = next(g for g in payload["groups"] if g["id"] == "devices")
    assert {c["id"] for c in devices["concepts"]} == {"devices/bpm", "devices/rf-system"}


def test_group_concepts_label_title_cases_hyphen_and_underscore():
    entries = [
        ConceptEntry(concept_id="rf-systems/klystron", title="Klystron"),
        ConceptEntry(concept_id="beam_dynamics/tune", title="Tune"),
    ]
    payload = group_concepts(entries)

    labels = {g["id"]: g["label"] for g in payload["groups"]}
    assert labels["rf-systems"] == "Rf Systems"
    assert labels["beam_dynamics"] == "Beam Dynamics"


def test_group_concepts_groups_sorted_alpha_and_concepts_sorted_by_title():
    entries = [
        ConceptEntry(concept_id="zebra/widget", title="Widget"),
        ConceptEntry(concept_id="alpha/zulu", title="Zulu"),
        ConceptEntry(concept_id="alpha/mike", title="mike"),
        ConceptEntry(concept_id="alpha/alpha", title="Alpha"),
    ]
    payload = group_concepts(entries)

    assert [g["id"] for g in payload["groups"]] == ["alpha", "zebra"]

    alpha = next(g for g in payload["groups"] if g["id"] == "alpha")
    titles = [c["title"] for c in alpha["concepts"]]
    assert titles == ["Alpha", "mike", "Zulu"]


def test_group_concepts_per_concept_dict_excludes_type():
    entries = [
        ConceptEntry(concept_id="devices/bpm", title="BPM", description="A monitor", type="device"),
    ]
    payload = group_concepts(entries)

    concept = payload["groups"][0]["concepts"][0]
    assert concept == {"id": "devices/bpm", "title": "BPM", "description": "A monitor"}
    assert "type" not in concept


def test_group_concepts_empty_input():
    assert group_concepts([]) == {"groups": []}


# ---------------------------------------------------------------------------
# build_structure_markdown
# ---------------------------------------------------------------------------


def test_build_structure_markdown_shape():
    entries = [
        ConceptEntry(concept_id="devices/bpm", title="BPM", description="A monitor"),
        ConceptEntry(concept_id="devices/rf", title="RF System", description=""),
        ConceptEntry(concept_id="references/terms", title="Terms", description="Glossary"),
    ]
    grouped = group_concepts(entries)
    md = build_structure_markdown(grouped)

    # Facility-neutral title (deviation from ALS original, which hardcoded "ALS").
    assert md.startswith("# Facility Knowledge Base")
    assert "_3 concepts across 2 groups._" in md

    assert "## Devices" in md
    assert "## References" in md

    # Links use the /id.md root-absolute form (cross-link-nav compatible).
    assert "[BPM](/devices/bpm.md)" in md
    assert "[RF System](/devices/rf.md)" in md
    assert "[Terms](/references/terms.md)" in md

    # Description suffix present when non-empty, omitted (no dash) when empty.
    assert "[BPM](/devices/bpm.md) — A monitor" in md
    assert "[RF System](/devices/rf.md)\n" in md
    assert "[RF System](/devices/rf.md) —" not in md


def test_build_structure_markdown_empty_bundle():
    md = build_structure_markdown({"groups": []})
    assert md.startswith("# Facility Knowledge Base")
    assert "_0 concepts across 0 groups._" in md
