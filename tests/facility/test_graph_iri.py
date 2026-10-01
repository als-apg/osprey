"""The graph view's IRIs: one per (code, kind, id), and back to the raw id.

A node's local name is ``<code>_<kind>_esc(<id>)`` under the kind's namespace.
``esc`` writes every byte outside ``[A-Za-z0-9]`` as ``_xHH_``, so two ids never
share a local name and every local name is ``PN_LOCAL``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from osprey.facility import PN_LOCAL
from osprey.facility.views import graph_iri

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

DOCUMENT_KEYS = {"place": "places", "device": "devices", "channel": "channels", "group": "groups"}


def test_kinds_and_namespaces() -> None:
    assert graph_iri.KINDS == ("place", "device", "channel", "group")
    for kind in graph_iri.KINDS:
        assert graph_iri.namespace(kind) == f"https://narad.example.org/{kind}/"
        assert graph_iri.iri("demo", kind, "A") == f"https://narad.example.org/{kind}/demo_{kind}_A"


def test_escape_writes_every_byte_outside_alphanumerics_as_upper_case_hex() -> None:
    assert graph_iri.escape("SRq01") == "SRq01"
    assert graph_iri.escape("A/B") == "A_x2F_B"
    assert graph_iri.escape("a-b_c.d") == "a_x2D_b_x5F_c_x2E_d"
    assert graph_iri.escape("SR:Q1") == "SR_x3A_Q1"
    assert graph_iri.escape("é") == "_xC3__xA9_"
    assert graph_iri.escape("") == ""


@pytest.mark.parametrize(("left", "right"), [("SR/Q_1", "SR_Q/1"), ("A/B", "A_x2F_B")])
def test_ids_differing_only_in_punctuation_mint_distinct_iris(left: str, right: str) -> None:
    for kind in graph_iri.KINDS:
        minted = {graph_iri.iri("demo", kind, left), graph_iri.iri("demo", kind, right)}
        assert len(minted) == 2
        assert {graph_iri.decode(i, "demo") for i in minted} == {left, right}


def test_a_place_and_a_device_of_one_id_are_two_nodes() -> None:
    place = graph_iri.iri("demo", "place", "A/B")
    device = graph_iri.iri("demo", "device", "A/B")

    assert place != device
    assert graph_iri.local_name("demo", "place", "A/B") != graph_iri.local_name(
        "demo", "device", "A/B"
    )
    assert graph_iri.decode(place, "demo") == graph_iri.decode(device, "demo") == "A/B"


@pytest.mark.parametrize("raw", ["device_x", "x"])
def test_a_code_with_an_underscore_round_trips(raw: str) -> None:
    for kind in graph_iri.KINDS:
        minted = graph_iri.iri("als_u", kind, raw)
        assert graph_iri.decode(minted, "als_u") == raw
        assert graph_iri.decode(graph_iri.local_name("als_u", kind, raw), "als_u") == raw
    assert graph_iri.local_name("als_u", "device", "device_x") == "als_u_device_device_x5F_x"
    assert graph_iri.local_name("als_u", "device", "x") == "als_u_device_x"


def test_an_id_with_a_hyphen_round_trips_and_is_pn_local() -> None:
    local = graph_iri.local_name("demo", "device", "SR-Q-01")

    assert local == "demo_device_SR_x2D_Q_x2D_01"
    assert PN_LOCAL.fullmatch(local)
    assert graph_iri.decode(local, "demo") == "SR-Q-01"


@pytest.mark.parametrize("raw", ["", "A/B", "SR:Q1.", "é/ü", "_x2F_", "x2F", "a b", "__", "_x"])
def test_every_local_name_is_pn_local_and_decodes(raw: str) -> None:
    for kind in graph_iri.KINDS:
        local = graph_iri.local_name("demo", kind, raw)
        assert PN_LOCAL.fullmatch(local)
        assert graph_iri.decode(local, "demo") == raw


def test_an_id_ending_in_a_dot_round_trips_through_rdflib() -> None:
    from rdflib import RDF, Graph, Namespace, URIRef

    raw = "SR:Q1.CURRENT."
    minted = graph_iri.iri("demo", "channel", raw)
    sem = Namespace("https://narad.example.org/schema/shared_semantics/")
    graph = Graph()
    graph.bind("channel", graph_iri.namespace("channel"))
    graph.add((URIRef(minted), RDF.type, sem.ChannelBinding))

    parsed = Graph().parse(data=graph.serialize(format="turtle"), format="turtle")

    [subject] = parsed.subjects(RDF.type, sem.ChannelBinding)
    assert str(subject) == minted
    assert graph_iri.decode(str(subject), "demo") == raw


@pytest.mark.parametrize(
    "iri",
    [
        "other_device_A",
        "demo_magnet_A",
        "demo_device",
        "https://narad.example.org/place/demo_device_A",
        "https://narad.example.org/signal/demo_device_A",
        "demo_device_A/B",
        "demo_device_A_x2f_B",
        "demo_device_A_x2F",
        "demo_device_A_B",
        "demo_device__xFF_",
    ],
)
def test_decode_refuses_what_it_did_not_mint(iri: str) -> None:
    with pytest.raises(ValueError, match="demo"):
        graph_iri.decode(iri, "demo")


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="magnet"):
        graph_iri.iri("demo", "magnet", "A")


# xdist_group("built_control_assistant"): every module reading the session's one
# control-assistant build shares a worker, so the build runs once per run.
@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_every_demo_iri_decodes_to_its_raw_id(built_control_assistant: BuiltProject) -> None:
    facility = built_control_assistant.facility
    code = facility["identity"]["code"]

    minted: dict[str, tuple[str, str]] = {}
    for kind in graph_iri.KINDS:
        for record in facility[DOCUMENT_KEYS[kind]]:
            raw = record["id"]
            iri = graph_iri.iri(code, kind, raw)
            assert graph_iri.decode(iri, code) == raw
            assert PN_LOCAL.fullmatch(iri.removeprefix(graph_iri.namespace(kind)))
            assert minted.setdefault(iri, (kind, raw)) == (kind, raw)

    assert len(minted) == sum(len(facility[key]) for key in DOCUMENT_KEYS.values())
    assert {kind for kind, _ in minted.values()} >= {"place", "device", "channel"}
