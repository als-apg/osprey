"""Tests that a split Turtle corpus states exactly the triples of the whole.

The graph store is seeded in bounded requests: ``split_ttl`` cuts the corpus
between statements and ``import_ttl`` sends the parts one by one.  The seed is
only right if the parts, parsed apart, add up to the document parsed in one go.
rdflib is the referee: every test parses the whole text once, parses each part
as a document of its own, and compares the two graphs.

Each part is parsed the way a store parses a request, with blank node labels
scoped to that part.  rdflib alone would hide the one trap that matters there:
it keeps a label as the node's identity, so the same ``_:b0`` in two parts
would merge back into one node and the comparison would pass for a split that
tears a blank node in two.
"""

from __future__ import annotations

import random

import pytest
from rdflib import BNode, Graph
from rdflib.compare import isomorphic

from osprey.services.graph_store.seeder import IMPORT_CHUNK_CHARS, split_ttl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _whole(text: str) -> Graph:
    """The document parsed in one go."""
    return Graph().parse(data=text, format="turtle")


def _parts(parts: list[str]) -> Graph:
    """The parts parsed apart, each with blank nodes of its own, and united."""
    union = Graph()
    for number, part in enumerate(parts):
        for triple in Graph().parse(data=part, format="turtle"):
            union.add(
                tuple(
                    BNode(f"part{number}-{term}") if isinstance(term, BNode) else term
                    for term in triple
                )
            )
    return union


def _assert_same_triples(text: str, max_chars: int) -> list[str]:
    """Split *text*, check the parts state its triples, and return the parts."""
    parts = list(split_ttl(text, max_chars))
    whole, united = _whole(text), _parts(parts)
    assert len(united) == len(whole)
    assert isomorphic(united, whole)
    return parts


def _generated_corpus(subjects: int, seed: int = 2915) -> str:
    """A corpus in the shape the graph view writes, of *subjects* subject blocks.

    A leading comment line, the prefix table, then one block per subject:
    ``a`` first, one predicate per line closed by ``;``, the objects of a
    repeated predicate one per line after a ``,``, strings with escapes, and
    integers and decimals.
    """
    rng = random.Random(seed)
    awkward = ["plain", "ends in a dot .", 'a "quoted" word', "hash # inside", "line\nbreak"]
    lines = [
        "# facility-sha256: " + "0" * 64,
        "@prefix ex: <https://example.org/> .",
        "@prefix ex_p: <https://example.org/property/> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
    ]
    for index in range(subjects):
        label = rng.choice(awkward).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        names = ",\n        ".join(f'"name-{index}-{n}"' for n in range(rng.randint(1, 4)))
        lines += [
            "",
            f"ex:device-{index} a ex:Device ;",
            f"    ex_p:index {index} ;",
            f"    ex_p:position {rng.uniform(0, 200):.4f} ;",
            f"    ex_p:partOf ex:device-{rng.randrange(subjects)} ;",
            f"    ex_p:page <https://example.org/page/{index}#top> ;",
            f'    rdfs:label "{label}" ;',
            f"    skos:altLabel {names} .",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# A generated corpus
# ---------------------------------------------------------------------------


class TestGeneratedCorpus:
    """A corpus larger than one request is cut into parts equal to the whole."""

    def test_the_parts_state_the_triples_of_the_whole(self):
        text = _generated_corpus(1500)

        parts = _assert_same_triples(text, max_chars=20_000)

        assert len(parts) > 10

    def test_no_part_is_larger_than_the_bound(self):
        text = _generated_corpus(1500)

        parts = list(split_ttl(text, 20_000))

        assert max(len(part) for part in parts) <= 20_000

    def test_every_part_is_a_document_of_its_own(self):
        """Each part opens with the prefix table, so it parses with no other."""
        text = _generated_corpus(300)

        parts = list(split_ttl(text, 5_000))

        for part in parts:
            assert "@prefix ex: <https://example.org/> ." in part
            Graph().parse(data=part, format="turtle")

    def test_the_bodies_of_the_parts_are_the_text_in_order(self):
        """Nothing is dropped, repeated or reordered: only directives are added."""
        text = _generated_corpus(300)
        # The comment line and the prefix table, up to the first subject block.
        head = text[: text.index("\n\n") + 1]

        first, *rest = split_ttl(text, 5_000)

        assert all(part.startswith(head) for part in rest)
        assert first + "".join(part[len(head) :] for part in rest) == text

    def test_a_text_within_the_bound_is_yielded_unchanged(self):
        """The common case, and what keeps a small seed one request as before."""
        text = _generated_corpus(5)

        assert list(split_ttl(text, len(text))) == [text]
        assert list(split_ttl(text)) == [text]

    def test_the_default_bound_cuts_a_large_corpus(self):
        text = _generated_corpus(40_000)
        assert len(text) > 2 * IMPORT_CHUNK_CHARS

        parts = list(split_ttl(text))

        assert len(parts) >= 3
        assert max(len(part) for part in parts) <= IMPORT_CHUNK_CHARS

    def test_the_graph_view_of_a_facility_document_survives_the_cut(self):
        """The real writer's output, not only this module's imitation of it."""
        from osprey.facility.views.graph import graph_text
        from osprey_facility.records import FacilityRecords
        from tests.facility.test_graph_view import _doc

        text = graph_text(FacilityRecords(_doc()))

        parts = _assert_same_triples(text, max_chars=600)

        assert len(parts) > 3


# ---------------------------------------------------------------------------
# The traps
# ---------------------------------------------------------------------------

_HEAD = "@prefix ex: <https://example.org/> .\n"


def _padded(statements: list[str]) -> str:
    """*statements* between runs of filler statements, so every cut lands near one."""
    filler = [f'ex:filler-{n} ex:value "{n}" .' for n in range(40)]
    body: list[str] = []
    for statement in statements:
        body += filler[:20] + [statement] + filler[20:]
    return _HEAD + "\n".join(body) + "\n"


class TestStatementBoundaries:
    """A ``.`` ends a statement only where Turtle says it does."""

    @pytest.mark.parametrize(
        "statement",
        [
            'ex:s ex:p "one . two . three" .',
            'ex:s ex:p "an escaped quote \\" . still the string" .',
            "ex:s ex:p 'single quoted . text' .",
            'ex:s ex:p """a long string .\nwith a line that ends in a dot .\n\nex:not ex:a ex:statement .\n""" .',
            "ex:s ex:p '''long single quoted .\n@prefix ex: <https://other.example/> .\n''' .",
            'ex:s ex:p """a "quoted" word and two "" quotes .\n""" .',
            'ex:s ex:p "a # is no comment in a string" .',
            "ex:s ex:p <https://example.org/a.b/c#frag> .",
            "ex:s ex:p ex:local.name .",
            "ex:s ex:p ex:escaped\\.dot .",
            "ex:s ex:p 1.5, 2.0e3, .5, 3 .",
            "ex:s ex:p 4. # an integer, then the end of the statement",
            "ex:s ex:p ex:o. # no space before the dot",
            'ex:s ex:p "text"@en, "texte"@fr-BE .',
            'ex:s ex:p "a language tag that reads as a directive"@base .',
            'ex:s ex:p "1.5"^^<http://www.w3.org/2001/XMLSchema#decimal> .',
            'ex:s # a comment with a dot . and a quote " in it\n    ex:p ex:o .',
            'ex:s ex:p [ ex:q "inner . dot" ; ex:r [ ex:deep 1.5 ] ] .',
            'ex:s ex:p ( 1 2.5 "three . " ex:four ) .',
            '[ ex:p "anonymous subject" ] ex:q ex:o .',
        ],
    )
    def test_a_statement_is_not_cut_inside(self, statement):
        text = _padded([statement])

        for max_chars in (120, 300, 700):
            _assert_same_triples(text, max_chars)

    def test_a_statement_larger_than_the_bound_travels_whole(self):
        objects = ",\n    ".join(f'"value {n} ."' for n in range(200))
        text = _padded([f"ex:s ex:p {objects} ."])

        parts = _assert_same_triples(text, max_chars=300)

        assert any(len(part) > 300 for part in parts)

    def test_no_part_is_directives_alone(self):
        """A statement that does not fit beside the directives still goes with
        them: a part that states nothing would be a request that imports
        nothing."""
        objects = ",\n    ".join(f'"value {n} ."' for n in range(200))
        text = _padded([f"ex:s ex:p {objects} .", "@prefix late: <https://late.example/> ."])

        parts = _assert_same_triples(text, max_chars=300)

        assert all(len(Graph().parse(data=part, format="turtle")) for part in parts)

    def test_a_text_of_directives_alone_is_yielded_as_it_is(self):
        text = "".join(f"@prefix p{n}: <https://example.org/{n}/> .\n" for n in range(40))

        assert list(split_ttl(text, 200)) == [text]

    def test_text_after_the_last_statement_is_kept(self):
        text = _padded(["ex:s ex:p ex:o ."]) + "\n# a closing comment\n\n"

        parts = list(split_ttl(text, 200))

        assert parts[-1].endswith("# a closing comment\n\n")
        _assert_same_triples(text, 200)


class TestDirectives:
    """A name resolves in its part as it does at its place in the whole."""

    def test_a_prefix_stated_late_reaches_the_parts_after_it(self):
        text = _padded(
            [
                "@prefix late: <https://late.example/> .",
                "late:s late:p late:o .",
            ]
        )

        parts = _assert_same_triples(text, max_chars=200)

        assert "late:" not in parts[0]

    def test_a_redefined_prefix_keeps_both_meanings(self):
        text = _padded(
            [
                "ex:before ex:p ex:o .",
                "@prefix ex: <https://second.example/> .",
                "ex:after ex:p ex:o .",
            ]
        )

        _assert_same_triples(text, max_chars=200)
        whole = _whole(text)
        assert any(str(s) == "https://example.org/before" for s in whole.subjects())
        assert any(str(s) == "https://second.example/after" for s in whole.subjects())

    @pytest.mark.parametrize(
        "lines",
        [
            pytest.param(
                [
                    "@base <https://base.example/one/> .",
                    "<a> ex:p <b> .",
                    "@base <../two/> .",
                    "<c> ex:p <d> .",
                ],
                id="a-base-is-carried-and-a-later-base-replaces-it",
            ),
            pytest.param(
                [
                    "PREFIX sp: <https://sparql.example/>",
                    "sp:s sp:p sp:o .",
                    "base <https://base.example/>",
                    "<relative> sp:p sp:o .",
                    "prefix lower: <https://lower.example/>\nlower:s lower:p lower:o .",
                ],
                id="sparql-style-directives-are-carried",
            ),
            pytest.param(
                [
                    "@prefix base: <https://base.example/> .",
                    "@prefix prefix: <https://prefix.example/> .",
                    "base:s prefix:p ex:BASE .",
                    "ex:prefix ex:base base:o .",
                ],
                id="a-name-that-reads-like-a-directive-is-a-name",
            ),
        ],
    )
    def test_the_parts_hold_the_same_triples_as_the_whole(self, lines):
        _assert_same_triples(_padded(lines), max_chars=200)


class TestBlankNodes:
    """A blank node is one node however the text is cut."""

    def test_a_label_used_far_apart_names_one_node(self):
        text = _padded(
            [
                '_:shared ex:name "the one node" .',
                "ex:first ex:points _:shared .",
                "ex:second ex:points _:shared .",
                '_:other ex:knows _:shared ; ex:name "another" .',
            ]
        )

        parts = _assert_same_triples(text, max_chars=200)

        assert sum("_:shared" in part for part in parts) == 1

    def test_labelled_statements_keep_the_prefix_of_their_place(self):
        text = _padded(
            [
                "_:n ex:p ex:before .",
                "@prefix ex: <https://second.example/> .",
                "_:n ex:p ex:after .",
            ]
        )

        _assert_same_triples(text, max_chars=200)

    def test_an_anonymous_node_stays_in_its_statement(self):
        text = _padded(
            [
                'ex:a ex:has [ ex:name "first" ; ex:nested [ ex:name "inner" ] ] .',
                'ex:b ex:has [ ex:name "second" ] .',
            ]
        )

        parts = _assert_same_triples(text, max_chars=200)

        assert len(parts) > 4

    def test_a_local_name_with_an_underscore_and_colon_is_no_label(self):
        text = _padded(["ex:a_:b ex:p ex:_:c ."])

        parts = _assert_same_triples(text, max_chars=200)

        # No labelled document was held back: the name stays in its place.
        assert "ex:filler-39" in parts[-1]
