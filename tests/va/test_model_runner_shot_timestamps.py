"""The served VA posts a turn series a shot filled with the time of that shot.

:meth:`ModelRunner._post_outputs` hands the base class one group per time: the
filled series at their shot's time, every other value at the pass's time. These
tests stand in for the base class's post, so they need the serving package but
no transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip(
    "lume_pva_apg", reason="the served VA runs on lume-pva-apg, which ships for Linux only"
)

from lume_pva_apg.runner import Runner

from osprey.services.virtual_accelerator.serving.runner import ModelRunner

PASS_TS = 1_760_000_100.0
SHOT_TS = 1_760_000_000.0
SERIES = ("SR:BPM:TBT:X", "SR:BPM:TBT:SUM")
OTHER = "SR:MAG:HCM:01:CURRENT:SP"


@dataclass
class _Model:
    shot_times: dict[str, float]


@dataclass
class _Child:
    model: Any


@dataclass
class _Composite:
    children: dict[str, _Child] = field(default_factory=dict)

    @property
    def _children(self) -> dict[str, _Child]:
        return self.children


def _posts(
    monkeypatch: pytest.MonkeyPatch, composite: Any, values: dict[str, Any]
) -> list[tuple[float, dict[str, Any]]]:
    posted: list[tuple[float, dict[str, Any]]] = []
    monkeypatch.setattr(
        Runner, "_post_outputs", lambda _self, out, ts: posted.append((ts, dict(out)))
    )
    runner = object.__new__(ModelRunner)
    runner.model = composite
    ModelRunner._post_outputs(runner, values, PASS_TS)
    return posted


def test_a_filled_series_is_posted_at_its_shot_time_and_the_rest_at_the_pass_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composite = _Composite({"SR": _Child(_Model(dict.fromkeys(SERIES, SHOT_TS)))})
    values = {SERIES[0]: [1.0], SERIES[1]: [1.0], OTHER: 2.0}

    posted = dict(_posts(monkeypatch, composite, values))

    assert posted == {SHOT_TS: {SERIES[0]: [1.0], SERIES[1]: [1.0]}, PASS_TS: {OTHER: 2.0}}


def test_a_pass_with_no_filled_series_posts_once_at_the_pass_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composite = _Composite({"SR": _Child(_Model({}))})
    values = {SERIES[0]: [0.0], OTHER: 2.0}

    assert _posts(monkeypatch, composite, values) == [(PASS_TS, values)]


def test_a_pass_reading_nothing_still_reaches_the_base_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _posts(monkeypatch, _Composite(), {}) == [(PASS_TS, {})]
