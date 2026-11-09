"""The collection hook that skips browser tests before their fixtures run.

``tests/interfaces/conftest.py`` probes each Playwright engine once and marks
every test needing an engine that cannot launch as skipped, so a host without
the binary does not build a deployment or a served hub only to skip. The hook
has to be right in both directions: a test that needs the engine is marked
with the fixture's own reason, and nothing is marked when the engine launches
— a lane with the browsers installed must run every one of these tests. The
hook is called directly here on fake items, with the probe stubbed, so no
Playwright driver starts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.interfaces import conftest

HERE = Path(conftest.__file__).parent
REASON = "chromium binary not available: planted"


class _FakeCallSpec:
    def __init__(self, params: dict[str, object]) -> None:
        self.params = params


class _FakeItem:
    """The attributes ``pytest_collection_modifyitems`` reads and writes."""

    def __init__(
        self,
        path: Path,
        fixturenames: tuple[str, ...] = (),
        params: dict[str, object] | None = None,
    ) -> None:
        self.path = path
        self.fixturenames = list(fixturenames)
        if params is not None:
            self.callspec = _FakeCallSpec(params)
        self.added: list[pytest.MarkDecorator] = []

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.added.append(marker)

    def skip_reasons(self) -> list[str]:
        return [m.kwargs["reason"] for m in self.added if m.name == "skip"]


def _probe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        conftest,
        "_engine_skip_reason",
        lambda engine: f"{engine} binary not available: planted",
    )


def _closure_item() -> _FakeItem:
    return _FakeItem(HERE / "test_x_browser.py", ("tmp_path", "context", "chromium_browser"))


def _param_item() -> _FakeItem:
    return _FakeItem(
        HERE / "channel_finder" / "test_x_browser.py",
        ("finder_url", "engine_browser"),
        {"engine_browser": "chromium_browser"},
    )


def test_a_closure_holding_the_browser_fixture_is_skipped_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _probe_fails(monkeypatch)
    item = _closure_item()

    conftest.pytest_collection_modifyitems([item])

    assert item.skip_reasons() == [REASON]


def test_a_parametrized_fixture_naming_the_browser_fixture_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _probe_fails(monkeypatch)
    item = _param_item()

    conftest.pytest_collection_modifyitems([item])

    assert item.skip_reasons() == [REASON]


def test_items_outside_the_tree_or_needing_no_browser_are_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _probe_fails(monkeypatch)
    outside = _FakeItem(HERE.parent / "docs" / "test_x.py", ("chromium_browser",))
    no_browser = _FakeItem(HERE / "test_x.py", ("tmp_path",), {"name": "webkit"})

    conftest.pytest_collection_modifyitems([outside, no_browser])

    assert outside.added == []
    assert no_browser.added == []


def test_nothing_is_marked_when_the_engine_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    probed: list[str] = []

    def launches(engine: str) -> None:
        probed.append(engine)

    monkeypatch.setattr(conftest, "_engine_skip_reason", launches)
    items = [_closure_item(), _param_item()]

    conftest.pytest_collection_modifyitems(items)

    assert probed  # the probe was consulted, and it said the engine launches
    assert [item.added for item in items] == [[], []]


def test_a_parametrized_fixture_naming_the_browser_fixture_is_skipped__mutation_drops_the_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(conftest, "_requested_fixtures", lambda item: set(item.fixturenames))
    with pytest.raises(AssertionError):
        test_a_parametrized_fixture_naming_the_browser_fixture_is_skipped(monkeypatch)


def test_a_closure_holding_the_browser_fixture_is_skipped__mutation_drops_the_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        conftest,
        "_requested_fixtures",
        lambda item: set(getattr(item, "callspec", _FakeCallSpec({})).params.values()),
    )
    with pytest.raises(AssertionError):
        test_a_closure_holding_the_browser_fixture_is_skipped_with_its_reason(monkeypatch)
