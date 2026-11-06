"""The ``agentic`` tier marker's collection check (tests/e2e/agentic_tiers.py).

Every agentic e2e test carries its tier and a plain-words description on one
marker, and collection refuses a test whose marker is incomplete. The check is
pinned here on hand-built items, without collecting the e2e suite.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e import agentic_tiers

E2E_ROOT = Path(__file__).resolve().parent / "e2e"

COMPLETE = {
    "tier": 1,
    "title": "Read one channel",
    "asks": "What is the beam current?",
    "passes_if": "answers with the live value; no writes",
}


def _item(path: Path, **markers: pytest.Mark | None) -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        nodeid=f"{path.name}::test_x",
        get_closest_marker=lambda name: markers.get(name),
    )


def _agentic(*args, **kwargs) -> pytest.Mark:
    return pytest.mark.agentic(*args, **kwargs).mark


def test_a_complete_marker_passes():
    assert agentic_tiers.marker_problem(_agentic(**COMPLETE)) is None


@pytest.mark.parametrize("field", agentic_tiers.FIELDS)
def test_a_missing_field_is_named(field):
    kwargs = {k: v for k, v in COMPLETE.items() if k != field}
    assert agentic_tiers.marker_problem(_agentic(**kwargs)) == f"missing {field}"


@pytest.mark.parametrize("tier", [0, 3, "1", True, None, [1]])
def test_an_unknown_tier_is_refused(tier):
    problem = agentic_tiers.marker_problem(_agentic(**{**COMPLETE, "tier": tier}))
    assert problem == f"tier={tier!r} is not one of [1, 2]"


def test_an_empty_description_is_refused():
    problem = agentic_tiers.marker_problem(_agentic(**{**COMPLETE, "asks": "  "}))
    assert problem == "empty asks"


def test_positional_and_unknown_arguments_are_refused():
    assert agentic_tiers.marker_problem(_agentic(1)) == "takes keyword arguments only"
    problem = agentic_tiers.marker_problem(_agentic(**COMPLETE, owner="x"))
    assert problem == "unknown argument owner"


def test_a_test_in_tier2_must_carry_tier_2():
    path = E2E_ROOT / "tier2" / "test_x.py"
    unmarked = _item(path)
    tier1 = _item(path, agentic=_agentic(**COMPLETE))
    tier2 = _item(path, agentic=_agentic(**{**COMPLETE, "tier": 2}))

    assert "without an agentic(tier=2) marker" in agentic_tiers.item_problem(unmarked, E2E_ROOT)
    assert "carries tier=1" in agentic_tiers.item_problem(tier1, E2E_ROOT)
    assert agentic_tiers.item_problem(tier2, E2E_ROOT) is None


def test_a_test_outside_tier2_may_carry_either_tier_or_none():
    path = E2E_ROOT / "test_x.py"
    assert agentic_tiers.item_problem(_item(path), E2E_ROOT) is None
    for tier in agentic_tiers.TIERS:
        item = _item(path, agentic=_agentic(**{**COMPLETE, "tier": tier}))
        assert agentic_tiers.item_problem(item, E2E_ROOT) is None


def test_agentic_and_harness_on_one_test_is_refused():
    item = _item(
        E2E_ROOT / "test_x.py",
        agentic=_agentic(**COMPLETE),
        harness_benchmark=pytest.mark.harness_benchmark.mark,
    )
    assert "both agentic and harness_benchmark" in agentic_tiers.item_problem(item, E2E_ROOT)


def test_the_refusal_names_every_test_and_says_what_is_needed():
    message = agentic_tiers.collection_refusal(
        [("b.py::test_b", "agentic marker missing asks"), ("a.py::test_a", "tier=3 ...")]
    )
    assert message.startswith("2 agentic test(s)")
    assert "@pytest.mark.agentic(tier=1|2, title=..., asks=..., passes_if=...)" in message
    assert message.index("a.py::test_a") < message.index("b.py::test_b")
    assert agentic_tiers.collection_refusal([]) is None


def test_the_manifest_carries_the_tier_and_description():
    item = _item(E2E_ROOT / "test_x.py", agentic=_agentic(**COMPLETE))
    assert agentic_tiers.manifest_entry(item) == {"lane": "agentic", **COMPLETE}
    harness = _item(E2E_ROOT / "test_y.py", harness_benchmark=pytest.mark.harness_benchmark.mark)
    assert agentic_tiers.manifest_entry(harness) == {"lane": "harness"}
    assert agentic_tiers.manifest_entry(_item(E2E_ROOT / "test_z.py")) == {"lane": "unmarked"}


def test_a_not_applicable_skip_reason_carries_the_label_the_board_shows():
    reason = agentic_tiers.not_applicable("no vision", "provider p has no images")
    assert reason == "n/a: no vision (provider p has no images)"
    assert reason.startswith(agentic_tiers.NOT_APPLICABLE)


class TestTheSwitchBuildFollowsTheCell:
    """A build writes each agent's model into its definition, and a subagent runs
    the id written there, so the machine-switch tests must build with the model a
    board cell forces. Outside the board they keep the pair CI uses."""

    def _built_with(self, monkeypatch, tmp_path) -> dict[str, str]:
        from tests.e2e import test_target_switch_agentic as switch

        calls = []
        monkeypatch.setattr(switch, "_run_osprey", lambda verb, args, **_k: calls.append(args))
        monkeypatch.setattr(switch, "_profile_edits", lambda **_k: {})
        with contextlib.suppress(Exception):  # the steps after init need a real render
            switch._init_and_build(tmp_path, bench_port=1, va_port=2)
        init = calls[0]
        pairs = [init[i + 1] for i, a in enumerate(init) if a == "--set"]
        return dict(p.split("=", 1) for p in pairs)

    def test_ci_keeps_its_pinned_pair(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OSPREY_E2E_FORCE_PROVIDER", raising=False)
        monkeypatch.delenv("OSPREY_E2E_FORCE_MODEL", raising=False)
        built = self._built_with(monkeypatch, tmp_path)
        assert (built["provider"], built["model"]) == ("als-apg", "claude-haiku-5-5")

    def test_a_board_cell_forces_both(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OSPREY_E2E_FORCE_PROVIDER", "cborg")
        monkeypatch.setenv("OSPREY_E2E_FORCE_MODEL", "gpt-oss-120b")
        built = self._built_with(monkeypatch, tmp_path)
        assert (built["provider"], built["model"]) == ("cborg", "gpt-oss-120b")
