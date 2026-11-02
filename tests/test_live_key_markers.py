"""The skip gates behind the live Jev and OpenRouter tests.

The two markers arm tests that spend real money against a live API, so the
gate has to be right in both directions: a run with no key skips rather than
errors, and a key that reached the process only because litellm loaded an
ancestor checkout's ``.env`` during collection does not count as one the run
was given. The predicates and the collection hook in ``tests/conftest.py`` are
exercised directly here, with synthetic values only.

Marker names are assembled from parts and the marks are built as
``pytest.Mark`` objects: the placement guard in ``test_api_marker_placement.py``
scans every fast-lane source line for the decorator spelling and would flag it
even inside a string.
"""

from __future__ import annotations

import pytest

from tests import conftest

#: (marker suffix, environment variable the marker gates on).
LIVE_KEYS = (
    ("typesafe", "TYPESAFE_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
)

SYNTHETIC_KEY = "synthetic-live-key-for-tests"


def _marker_name(suffix: str) -> str:
    return "requires_" + suffix


class _FakeItem:
    """The two attributes ``pytest_collection_modifyitems`` touches."""

    def __init__(self, *marks: pytest.Mark) -> None:
        self.keywords = {mark.name: mark for mark in marks}
        self.added: list[pytest.MarkDecorator] = []

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.added.append(marker)

    def skip_reasons(self) -> list[str]:
        return [m.kwargs["reason"] for m in self.added if m.name == "skip"]


def _item_for(suffix: str) -> _FakeItem:
    return _FakeItem(pytest.Mark(_marker_name(suffix), (), {}, _ispytest=True))


@pytest.fixture(autouse=True)
def _no_live_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from a run that was given neither key and has no ancestor .env."""
    for _, var in LIVE_KEYS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        conftest,
        "_PRE_COLLECTION_ENV",
        {
            k: v
            for k, v in conftest._PRE_COLLECTION_ENV.items()
            if k not in dict(LIVE_KEYS).values()
        },
    )
    monkeypatch.setattr(conftest, "_ancestor_dotenv_values", dict)


def test_both_markers_are_registered(pytestconfig: pytest.Config) -> None:
    registered = {line.split(":", 1)[0].strip() for line in pytestconfig.getini("markers")}
    for suffix, _ in LIVE_KEYS:
        assert _marker_name(suffix) in registered


def test_both_markers_have_resource_checks() -> None:
    for suffix, _ in LIVE_KEYS:
        assert _marker_name(suffix) in conftest._RESOURCE_CHECKS


@pytest.mark.parametrize(("suffix", "var"), LIVE_KEYS)
def test_unset_key_skips_the_marked_item(suffix: str, var: str) -> None:
    item = _item_for(suffix)

    conftest.pytest_collection_modifyitems([item])

    reasons = item.skip_reasons()
    assert len(reasons) == 1
    assert var in reasons[0]


@pytest.mark.parametrize(("suffix", "var"), LIVE_KEYS)
def test_empty_key_skips_the_marked_item(
    suffix: str, var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(var, "")
    item = _item_for(suffix)

    conftest.pytest_collection_modifyitems([item])

    assert len(item.skip_reasons()) == 1


@pytest.mark.parametrize(("suffix", "var"), LIVE_KEYS)
def test_key_only_from_an_ancestor_dotenv_skips(
    suffix: str, var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Appeared during collection with exactly the ancestor .env's value.
    monkeypatch.setenv(var, SYNTHETIC_KEY)
    monkeypatch.setattr(conftest, "_ancestor_dotenv_values", lambda: {var: SYNTHETIC_KEY})
    item = _item_for(suffix)

    conftest.pytest_collection_modifyitems([item])

    assert len(item.skip_reasons()) == 1


@pytest.mark.parametrize(("suffix", "var"), LIVE_KEYS)
def test_exported_key_runs_the_marked_item(
    suffix: str, var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exported before collection counts even when an ancestor .env agrees.
    monkeypatch.setenv(var, SYNTHETIC_KEY)
    monkeypatch.setitem(conftest._PRE_COLLECTION_ENV, var, SYNTHETIC_KEY)
    monkeypatch.setattr(conftest, "_ancestor_dotenv_values", lambda: {var: SYNTHETIC_KEY})
    item = _item_for(suffix)

    conftest.pytest_collection_modifyitems([item])

    assert item.added == []


@pytest.mark.parametrize(("suffix", "var"), LIVE_KEYS)
def test_key_set_during_collection_by_something_else_counts(
    suffix: str, var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Added during collection, but not with the value an ancestor .env holds.
    monkeypatch.setenv(var, SYNTHETIC_KEY)
    monkeypatch.setattr(conftest, "_ancestor_dotenv_values", lambda: {var: "a-different-value"})
    item = _item_for(suffix)

    conftest.pytest_collection_modifyitems([item])

    assert item.added == []


def test_unmarked_item_is_untouched() -> None:
    item = _FakeItem()

    conftest.pytest_collection_modifyitems([item])

    assert item.added == []


def test_each_marker_gates_only_on_its_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", SYNTHETIC_KEY)
    monkeypatch.setitem(conftest._PRE_COLLECTION_ENV, "OPENROUTER_API_KEY", SYNTHETIC_KEY)
    jev_item = _item_for("typesafe")
    openrouter_item = _item_for("openrouter")

    conftest.pytest_collection_modifyitems([jev_item, openrouter_item])

    assert len(jev_item.skip_reasons()) == 1
    assert openrouter_item.added == []
