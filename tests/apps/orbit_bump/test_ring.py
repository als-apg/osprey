"""The ring view: per-plane members in lattice order, ``next_cor`` and the unsolved map."""

import pytest
from orbit_bump import ring
from orbit_bump.ring import PlaneView, RingView, UnsolvedCorrector, build_ring

from osprey.apps.sdk import Member, ModelMembers

# A synthetic two-plane ring, in lattice-index order (s in metres in brackets):
#
#   x: C0@2 B0@4[1.0] C1@5 C2@7 B1@8 B2@9 C3@11[3.0] B3@12[3.0] C4@14 B4@16
#   y: B0@4 C0@6 C1@11[3.0] B1@12[3.0] B2@16
#
# Non-alternating spacing (x: two correctors between B0 and B1, none between B1 and
# B2), a wrap at the ring end (x B4 and y B2 have no downstream corrector), and a
# corrector and a BPM sharing one s where the lattice index decides (index 11 and 12).


def _bpm(element, plane, index, s_m):
    return Member(
        element=element,
        plane=plane,
        index=index,
        s_m=s_m,
        unit="mm",
        position_addr=f"pos:{element}:{plane}",
        golden_addr=f"gold:{element}:{plane}",
    )


def _cor(element, plane, index, s_m, tolerance=0.01):
    return Member(
        element=element,
        plane=plane,
        index=index,
        s_m=s_m,
        unit="A",
        setpoint_addr=f"sp:{element}",
        readback_addr=f"rb:{element}",
        tolerance=tolerance,
    )


BPMS = [
    _bpm("m4", "x", 4, 1.0),
    _bpm("m4", "y", 4, 1.0),
    _bpm("m8", "x", 8, 2.0),
    _bpm("m9", "x", 9, 2.2),
    _bpm("m12", "x", 12, 3.0),
    _bpm("m12", "y", 12, 3.0),
    _bpm("m16", "x", 16, 4.0),
    _bpm("m16", "y", 16, 4.0),
]
CORRECTORS = [
    _cor("h2", "x", 2, 0.5),
    _cor("h5", "x", 5, 1.2),
    _cor("v6", "y", 6, 1.5),
    _cor("h7", "x", 7, 1.8),
    _cor("h11", "x", 11, 3.0, tolerance=None),
    _cor("v11b", "y", 11, 3.0),
    _cor("h14", "x", 14, 3.5),
]


def _members(bpms=BPMS, correctors=CORRECTORS):
    return ModelMembers(model="ring-a", length_m=5.0, bpms=list(bpms), correctors=list(correctors))


@pytest.fixture
def view():
    return build_ring(_members())


def test_view_carries_model_and_length(view):
    assert isinstance(view, RingView)
    assert view.model == "ring-a"
    assert view.length_m == 5.0


def test_planes_split_members_by_plane_in_lattice_order(view):
    x = view.plane("x")
    assert isinstance(x, PlaneView)
    assert [m.element for m in x.bpms] == ["m4", "m8", "m9", "m12", "m16"]
    assert [m.element for m in x.correctors] == ["h2", "h5", "h7", "h11", "h14"]
    y = view.plane("y")
    assert [m.element for m in y.bpms] == ["m4", "m12", "m16"]
    assert [m.element for m in y.correctors] == ["v6", "v11b"]
    assert all(m.plane == "y" for m in (*y.bpms, *y.correctors))


def test_members_keep_their_addresses_units_s_and_tolerance(view):
    x = view.plane("x")
    b = x.bpms[0]
    assert (b.position_addr, b.golden_addr, b.unit, b.s_m) == ("pos:m4:x", "gold:m4:x", "mm", 1.0)
    c = x.correctors[3]
    assert (c.setpoint_addr, c.readback_addr, c.unit, c.tolerance) == (
        "sp:h11",
        "rb:h11",
        "A",
        None,
    )


def test_order_is_by_lattice_index_whatever_the_input_order():
    shuffled = build_ring(_members(bpms=BPMS[::-1], correctors=CORRECTORS[::-1]))
    assert shuffled.plane("x").bpm_idx == [4, 8, 9, 12, 16]
    assert shuffled.plane("x").cor_idx == [2, 5, 7, 11, 14]


def test_lattice_indices_for_split_rows(view):
    x = view.plane("x")
    assert x.bpm_idx == [4, 8, 9, 12, 16]
    assert x.cor_idx == [2, 5, 7, 11, 14]
    assert view.plane("y").bpm_idx == [4, 12, 16]
    assert view.plane("y").cor_idx == [6, 11]
    # Every lattice index of the model lies in [0, n_index).
    assert x.n_index == view.plane("y").n_index == 17


def test_next_cor_is_the_first_corrector_downstream_by_index(view):
    # x: B0@4 -> h5 (1); B1@8 and B2@9 -> h11 (3) (non-alternating);
    # B3@12 -> h14 (4) although h11 shares its s; B4@16 wraps to h2 (0).
    assert view.plane("x").next_cor == [1, 3, 3, 4, 0]
    # y: B0@4 -> v6 (0); B1@12 -> wraps to v6 (0), v11b at the same s is upstream;
    # B2@16 wraps to v6 (0).
    assert view.plane("y").next_cor == [0, 0, 0]


def test_shared_s_index_decides_downstream():
    # A corrector at the BPM's s but a higher index is downstream of it.
    bpms = [_bpm("m3", "x", 3, 1.0)]
    cors = [_cor("h1", "x", 1, 0.2), _cor("h4", "x", 4, 1.0)]
    assert build_ring(_members(bpms, cors)).plane("x").next_cor == [1]


def test_unknown_plane_is_refused(view):
    with pytest.raises(ValueError, match="plane"):
        view.plane("z")


def test_plane_without_members_is_empty():
    view = build_ring(_members(bpms=BPMS[:1], correctors=[]))
    y = view.plane("y")
    assert (y.bpms, y.correctors, y.bpm_idx, y.cor_idx, y.next_cor) == ((), (), [], [], [])
    assert view.plane("x").next_cor == []


def test_corrector_lookup_by_setpoint_address(view):
    assert view.corrector("sp:h7") == UnsolvedCorrector(
        element="h7", plane="x", position=2, setpoint_addr="sp:h7", reason=None
    )
    assert view.corrector("sp:v11b").position == 1
    assert view.corrector("sp:v11b").plane == "y"
    assert view.corrector("nowhere") is None


def test_unsolved_maps_addresses_to_elements_and_ring_positions(view):
    unsolved = {"sp:h11": "no closed orbit", "sp:v6": "unstable", "sp:h2": "singular"}
    entries = view.unsolved(unsolved)
    # Ring order: plane x then y, each by ring position.
    assert entries == [
        UnsolvedCorrector("h2", "x", 0, "sp:h2", "singular"),
        UnsolvedCorrector("h11", "x", 3, "sp:h11", "no closed orbit"),
        UnsolvedCorrector("v6", "y", 0, "sp:v6", "unstable"),
    ]
    assert view.unsolved_positions(unsolved, "x") == {0, 3}
    assert view.unsolved_positions(unsolved, "y") == {0}
    assert view.unsolved_elements(unsolved) == ["h2", "h11", "v6"]


def test_unsolved_address_outside_the_model_is_left_out(view):
    assert view.unsolved({"sp:gone": "why"}) == []
    assert view.unsolved_positions({"sp:gone": "why"}, "x") == set()
    assert view.unsolved_positions({}, "y") == set()


def test_next_cor_feeds_select_correctors(view):
    from orbit_bump.solver import select_correctors

    x = view.plane("x")
    unsolved = view.unsolved_positions({"sp:h14": "unstable"}, "x")
    picked = select_correctors([4], x.next_cor, len(x.correctors), "auto", None, unsolved)
    # B4 wraps: upstream h14 (4, unsolved), then h2 and h5 downstream.
    assert picked.selected == [0, 1]
    assert picked.skipped_unsolved == [4]


def test_ring_view_reads_the_sdk(monkeypatch):
    seen = {}

    def fake_members(model, render_root=None):
        seen["args"] = (model, render_root)
        return _members()

    monkeypatch.setattr(ring, "model_members", fake_members)
    view = ring.ring_view("ring-a", render_root="/r")
    assert seen["args"] == ("ring-a", "/r")
    assert view.plane("x").next_cor == [1, 3, 3, 4, 0]


def test_offered_models_are_the_closed_orbit_models(monkeypatch):
    monkeypatch.setattr(ring, "closed_orbit_models", lambda render_root=None: ["a", "b"])
    assert ring.offered_models() == ["a", "b"]


def test_module_names_no_facility_literal():
    import inspect

    source = inspect.getsource(ring)
    for word in ("SR", "BPM", "HCM", "VCM", "MAG", "DIAG"):
        for quote in ('"', "'"):
            assert f"{quote}{word}" not in source
    assert "SR:" not in source
