"""Tests for the per-user bar-layout store.

Three contracts, one per public entry point:

* **load never fails.** A document that is absent, truncated, written by a
  build with a different schema version, or structurally not a layout all
  degrade to the deployment default. A corrupt preferences blob must cost the
  operator their arrangement, never the terminal.
* **save refuses exactly what the browser's normalizer refuses** — an unknown
  item type, a type placed in a bar that cannot hold it, more items than a bar
  may carry, a malformed entry, and an option value outside its spec. Anything
  else and the server-rendered first paint would disagree with what the client
  hydrates.
* **rev is monotonic.** Every accepted save lands one above whatever is on
  disk, whatever the caller claimed, so a stale editor's revision can be
  recognised rather than believed.

The vocabulary is a parameter, never a module-level fact, so these tests run
against a fixture catalog and the store cannot drift into holding item
knowledge of its own.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from osprey.interfaces.web_terminal import bar_items_store
from osprey.interfaces.web_terminal.bar_items_store import (
    BarLayoutConflict,
    BarLayoutInvalid,
    BarVocabulary,
)

LOGGER_NAME = "osprey.interfaces.web_terminal.bar_items_store"


# ── fixtures ───────────────────────────────────────────────────────────────


#: A fixture catalog shaped exactly like ``static/js/bar-catalog.js``: every
#: type declares the hosts that may hold it and a spec per option it accepts.
#: Four types are enough to exercise every refusal class — a header-only type,
#: an option-free type, an enum/boolean pair, and a bounded number.
CATALOG: dict[str, dict[str, Any]] = {
    "logo": {"hosts": ("header",), "options": {}},
    "activity": {"hosts": ("header", "status"), "options": {}},
    "clock": {
        "hosts": ("header", "status"),
        "options": {
            "zone": {"kind": "enum", "values": ("local", "utc", "both"), "default": "local"},
            "seconds": {"kind": "boolean", "default": False},
        },
    },
    "gap": {
        "hosts": ("header", "status"),
        "options": {"size": {"kind": "number", "min": 4, "max": 400, "default": 12}},
    },
}

VOCABULARY = BarVocabulary(items=CATALOG, version=1, max_items_per_host=20)

#: The deployment default this build would render with nothing saved.
DEFAULT_LAYOUT: dict[str, Any] = {
    "version": 1,
    "rev": 0,
    "header": [{"type": "logo"}, {"type": "activity"}],
    "status": [{"type": "clock", "options": {"zone": "utc", "seconds": True}}],
    "status_visible": True,
}


def load(store_dir: Path) -> dict[str, Any]:
    """Load with the fixture vocabulary and default, which every test shares."""
    return bar_items_store.load_layout(store_dir, vocabulary=VOCABULARY, default=DEFAULT_LAYOUT)


def save(store_dir: Path, layout: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Save with the fixture vocabulary, which every test shares."""
    return bar_items_store.save_layout(store_dir, layout, vocabulary=VOCABULARY, **kwargs)


def document(**overrides: Any) -> dict[str, Any]:
    """A valid layout the caller can spoil one field of."""
    base: dict[str, Any] = {
        "version": 1,
        "rev": 0,
        "header": [{"type": "logo"}],
        "status": [{"type": "activity"}],
        "status_visible": True,
    }
    base.update(overrides)
    return base


# ── load_layout ────────────────────────────────────────────────────────────


def test_load_returns_the_deployment_default_when_nothing_is_stored(tmp_path: Path) -> None:
    result = load(tmp_path)

    assert result["header"] == [{"type": "logo"}, {"type": "activity"}]
    assert result["status_visible"] is True
    # Never saved, so the revision a conditional save compares against is zero.
    assert result["rev"] == 0


def test_load_leaves_no_directory_behind_when_nothing_is_stored(tmp_path: Path) -> None:
    store_dir = tmp_path / "bar_items"

    load(store_dir)

    # Reading is not a reason to create a mount point; only a save is.
    assert not store_dir.exists()


def test_load_returns_a_copy_the_caller_cannot_mutate_the_default_through(
    tmp_path: Path,
) -> None:
    first = load(tmp_path)
    first["header"].append({"type": "clock"})
    first["status_visible"] = False

    assert load(tmp_path)["header"] == [{"type": "logo"}, {"type": "activity"}]
    assert load(tmp_path)["status_visible"] is True


def test_load_reads_back_what_save_wrote(tmp_path: Path) -> None:
    saved = save(
        tmp_path,
        document(
            header=[{"type": "logo"}, {"type": "gap", "options": {"size": 40}}],
            status=[{"type": "clock", "options": {"zone": "utc", "seconds": True}}],
            status_visible=False,
        ),
    )

    assert load(tmp_path) == saved
    assert saved["header"] == [
        {"type": "logo", "options": {}},
        {"type": "gap", "options": {"size": 40}},
    ]
    assert saved["status_visible"] is False


def test_load_falls_back_to_the_default_for_a_truncated_document(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bar_items_store.layout_path(tmp_path).write_text("{ truncated")

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert load(tmp_path)["header"] == DEFAULT_LAYOUT["header"]


def test_load_falls_back_to_the_default_for_an_unreadable_schema_version(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A document a NEWER deployment wrote. This build cannot claim to
    # understand it, so it renders the default rather than guessing.
    bar_items_store.layout_path(tmp_path).write_text(json.dumps(document(version=2, rev=7)))

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = load(tmp_path)

    assert result["header"] == DEFAULT_LAYOUT["header"]
    assert result["rev"] == 0
    assert "version" in caplog.text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("header", "logo"),
        ("header", {"type": "logo"}),
        ("status", None),
        ("status_visible", "yes"),
        ("rev", -1),
        ("rev", 1.5),
        ("rev", "3"),
    ],
)
def test_load_falls_back_to_the_default_for_a_structurally_broken_document(
    tmp_path: Path, field: str, value: Any
) -> None:
    bar_items_store.layout_path(tmp_path).write_text(json.dumps(document(**{field: value})))

    assert load(tmp_path)["header"] == DEFAULT_LAYOUT["header"]


def test_load_falls_back_to_the_default_for_an_entry_that_is_not_an_item(
    tmp_path: Path,
) -> None:
    bar_items_store.layout_path(tmp_path).write_text(
        json.dumps(document(header=[{"type": "logo"}, "clock"]))
    )

    assert load(tmp_path)["header"] == DEFAULT_LAYOUT["header"]


def test_load_keeps_an_item_this_build_no_longer_knows(tmp_path: Path) -> None:
    # Envelope only: the client normalizer drops an unrenderable item and marks
    # the layout read-only. Discarding the whole document here would destroy an
    # arrangement that a rollback, or a re-enabled panel, would make valid again.
    stored = document(header=[{"type": "logo"}, {"type": "retired-item"}])
    bar_items_store.layout_path(tmp_path).write_text(json.dumps(stored))

    assert load(tmp_path)["header"] == [{"type": "logo"}, {"type": "retired-item"}]


def test_load_falls_back_to_an_empty_document_when_the_default_is_itself_broken(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = bar_items_store.load_layout(
            tmp_path, vocabulary=VOCABULARY, default={"version": 1, "header": "nope"}
        )

    # Renders nothing rather than inventing an order the deployment never declared.
    assert result == {
        "version": 1,
        "rev": 0,
        "header": [],
        "status": [],
        "status_visible": True,
    }


def test_load_reports_a_directory_where_the_document_should_be_as_absent(
    tmp_path: Path,
) -> None:
    bar_items_store.layout_path(tmp_path).mkdir(parents=True)

    assert load(tmp_path)["header"] == DEFAULT_LAYOUT["header"]


# ── save_layout: the refusal classes ───────────────────────────────────────


def test_save_refuses_an_unknown_item_type(tmp_path: Path) -> None:
    with pytest.raises(BarLayoutInvalid) as excinfo:
        save(tmp_path, document(header=[{"type": "telemetry"}]))

    assert excinfo.value.reason == "unknown-type"
    assert "telemetry" in str(excinfo.value)


def test_save_refuses_a_type_the_bar_cannot_hold(tmp_path: Path) -> None:
    # `hosts` is a hard capability on the client: a 28 px wordmark has no body
    # that fits a status bar, so a status placement is refused, not re-homed.
    with pytest.raises(BarLayoutInvalid) as excinfo:
        save(tmp_path, document(header=[], status=[{"type": "logo"}]))

    assert excinfo.value.reason == "host-mismatch"
    assert "logo" in str(excinfo.value)


def test_save_refuses_more_items_than_a_bar_may_carry(tmp_path: Path) -> None:
    with pytest.raises(BarLayoutInvalid) as excinfo:
        save(tmp_path, document(header=[{"type": "activity"}] * 21))

    assert excinfo.value.reason == "overflow"


def test_save_accepts_a_bar_filled_to_the_cap(tmp_path: Path) -> None:
    saved = save(tmp_path, document(header=[{"type": "activity"}] * 20))

    assert len(saved["header"]) == 20


def test_save_caps_each_bar_separately(tmp_path: Path) -> None:
    # Per host, not per document: a legal header edit must not fail because of
    # what is in the status bar.
    saved = save(
        tmp_path,
        document(header=[{"type": "activity"}] * 20, status=[{"type": "activity"}] * 20),
    )

    assert (len(saved["header"]), len(saved["status"])) == (20, 20)


@pytest.mark.parametrize(
    "entry",
    ["clock", None, 7, [], {}, {"type": 3}, {"type": "clock", "options": ["zone"]}],
)
def test_save_refuses_an_entry_that_is_not_an_item(tmp_path: Path, entry: Any) -> None:
    with pytest.raises(BarLayoutInvalid) as excinfo:
        save(tmp_path, document(header=[entry]))

    assert excinfo.value.reason == "malformed"


@pytest.mark.parametrize(
    "options",
    [
        {"zone": "mars"},
        {"zone": 3},
        {"zone": None},
        {"seconds": "true"},
        {"seconds": 1},
    ],
)
def test_save_refuses_an_option_value_outside_its_spec(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    with pytest.raises(BarLayoutInvalid) as excinfo:
        save(tmp_path, document(status=[{"type": "clock", "options": options}]))

    assert excinfo.value.reason == "bad-option"


@pytest.mark.parametrize("size", [3, 401, "12", True, None, float("nan")])
def test_save_refuses_a_number_option_outside_its_bounds(tmp_path: Path, size: Any) -> None:
    with pytest.raises(BarLayoutInvalid) as excinfo:
        save(tmp_path, document(header=[{"type": "gap", "options": {"size": size}}]))

    assert excinfo.value.reason == "bad-option"
    assert "size" in str(excinfo.value)


def test_save_accepts_a_number_option_at_its_bounds(tmp_path: Path) -> None:
    saved = save(
        tmp_path,
        document(header=[{"type": "gap", "options": {"size": 4}}], status=[]),
    )

    assert saved["header"] == [{"type": "gap", "options": {"size": 4}}]


@pytest.mark.parametrize(
    ("field", "value"),
    [("version", 2), ("version", None), ("header", "logo"), ("status", None)],
)
def test_save_refuses_a_document_this_build_cannot_read(
    tmp_path: Path, field: str, value: Any
) -> None:
    with pytest.raises(BarLayoutInvalid):
        save(tmp_path, document(**{field: value}))


def test_save_refuses_a_layout_that_is_not_a_document(tmp_path: Path) -> None:
    with pytest.raises(BarLayoutInvalid) as excinfo:
        save(tmp_path, ["logo", "clock"])  # type: ignore[arg-type]

    assert excinfo.value.reason == "malformed"


def test_save_leaves_the_stored_document_untouched_when_it_refuses(tmp_path: Path) -> None:
    save(tmp_path, document(header=[{"type": "logo"}]))

    with pytest.raises(BarLayoutInvalid):
        save(tmp_path, document(header=[{"type": "telemetry"}]))

    stored = load(tmp_path)
    assert stored["header"] == [{"type": "logo", "options": {}}]
    assert stored["rev"] == 1


def test_save_does_not_check_availability(tmp_path: Path) -> None:
    # The client refuses an item whose runtime dependency is missing; the server
    # cannot see that, and refusing here would make a layout unsavable for as
    # long as a bridge happened to be down.
    saved = save(tmp_path, document(status=[{"type": "activity"}]))

    assert saved["status"] == [{"type": "activity", "options": {}}]


# ── save_layout: options are completed, not trusted ────────────────────────


def test_save_fills_in_every_option_the_type_declares(tmp_path: Path) -> None:
    saved = save(tmp_path, document(status=[{"type": "clock"}]))

    # A complete document normalizes clean in the browser; a partial one would
    # arrive `changed` and make a freshly saved layout look edited.
    assert saved["status"] == [{"type": "clock", "options": {"zone": "local", "seconds": False}}]


def test_save_discards_an_option_the_type_does_not_declare(tmp_path: Path) -> None:
    # The client discards it too. A refusal here would leave an older tab
    # permanently unable to save.
    saved = save(
        tmp_path,
        document(status=[{"type": "clock", "options": {"zone": "utc", "blink": True}}]),
    )

    assert saved["status"] == [{"type": "clock", "options": {"zone": "utc", "seconds": False}}]


def test_save_gives_an_option_free_type_an_empty_option_map(tmp_path: Path) -> None:
    saved = save(tmp_path, document(header=[{"type": "logo", "options": {"size": 3}}]))

    assert saved["header"] == [{"type": "logo", "options": {}}]


# ── save_layout: revisions ─────────────────────────────────────────────────


def test_save_starts_the_revision_at_one(tmp_path: Path) -> None:
    assert save(tmp_path, document())["rev"] == 1


def test_save_increments_the_revision_on_every_write(tmp_path: Path) -> None:
    revisions = [save(tmp_path, document())["rev"] for _ in range(4)]

    assert revisions == [1, 2, 3, 4]
    assert load(tmp_path)["rev"] == 4


def test_save_ignores_the_revision_the_caller_claims(tmp_path: Path) -> None:
    save(tmp_path, document())

    # The revision is the store's to assign; a caller that echoed 900 back must
    # not be able to strand every later editor.
    assert save(tmp_path, document(rev=900))["rev"] == 2


def test_save_resumes_the_sequence_after_a_corrupt_document(tmp_path: Path) -> None:
    save(tmp_path, document())
    save(tmp_path, document())
    bar_items_store.layout_path(tmp_path).write_text("{ truncated")

    # Nothing readable is on disk, so there is no revision to build on and the
    # sequence restarts rather than inventing one.
    assert save(tmp_path, document())["rev"] == 1


def test_save_accepts_a_matching_expected_revision(tmp_path: Path) -> None:
    save(tmp_path, document())

    assert save(tmp_path, document(), expected_rev=1)["rev"] == 2


def test_save_accepts_expected_revision_zero_on_a_store_never_written(tmp_path: Path) -> None:
    assert save(tmp_path, document(), expected_rev=0)["rev"] == 1


def test_save_refuses_a_stale_expected_revision(tmp_path: Path) -> None:
    save(tmp_path, document())
    save(tmp_path, document())

    with pytest.raises(BarLayoutConflict) as excinfo:
        save(tmp_path, document(header=[{"type": "activity"}]), expected_rev=1)

    assert (excinfo.value.expected, excinfo.value.actual) == (1, 2)
    # The losing write left nothing behind.
    assert load(tmp_path)["header"] == [{"type": "logo", "options": {}}]
    assert load(tmp_path)["rev"] == 2


def test_save_validates_before_it_compares_revisions(tmp_path: Path) -> None:
    # A malformed document is malformed whatever revision it claims, and the
    # caller deserves the more specific answer.
    with pytest.raises(BarLayoutInvalid):
        save(tmp_path, document(header=[{"type": "telemetry"}]), expected_rev=99)


# ── save_layout: the file it leaves behind ─────────────────────────────────


def test_save_creates_the_store_directory(tmp_path: Path) -> None:
    store_dir = tmp_path / "agent_data" / "bar_items"

    save(store_dir, document())

    assert bar_items_store.layout_path(store_dir).is_file()


def test_save_writes_a_document_an_operator_can_read(tmp_path: Path) -> None:
    save(tmp_path, document())

    text = bar_items_store.layout_path(tmp_path).read_text()
    assert "\n" in text
    assert json.loads(text)["version"] == 1


def test_save_lands_atomically_and_leaves_no_debris(tmp_path: Path) -> None:
    save(tmp_path, document())
    save(tmp_path, document(status_visible=False))

    assert [entry.name for entry in tmp_path.iterdir()] == [bar_items_store.LAYOUT_FILENAME]
    assert load(tmp_path)["status_visible"] is False


def test_save_returns_a_copy_the_caller_cannot_mutate_the_store_through(
    tmp_path: Path,
) -> None:
    saved = save(tmp_path, document())
    saved["header"].clear()

    assert load(tmp_path)["header"] == [{"type": "logo", "options": {}}]


# ── reset_layout ───────────────────────────────────────────────────────────


def test_reset_deletes_the_document(tmp_path: Path) -> None:
    save(tmp_path, document(header=[{"type": "activity"}]))

    assert bar_items_store.reset_layout(tmp_path) is True
    assert not bar_items_store.layout_path(tmp_path).exists()
    # Back to what the deployment configured, not to an empty bar.
    assert load(tmp_path)["header"] == DEFAULT_LAYOUT["header"]


def test_reset_is_quiet_when_there_is_nothing_to_delete(tmp_path: Path) -> None:
    assert bar_items_store.reset_layout(tmp_path) is False
    assert bar_items_store.reset_layout(tmp_path / "never-created") is False


def test_reset_restarts_the_revision_sequence(tmp_path: Path) -> None:
    save(tmp_path, document())
    save(tmp_path, document())
    bar_items_store.reset_layout(tmp_path)

    # Reset is a delete, so the next save is a first save.
    assert save(tmp_path, document())["rev"] == 1


def test_reset_leaves_a_corrupt_document_gone_rather_than_stuck(tmp_path: Path) -> None:
    bar_items_store.layout_path(tmp_path).write_text("{ truncated")

    assert bar_items_store.reset_layout(tmp_path) is True
    assert not bar_items_store.layout_path(tmp_path).exists()


# ── the vocabulary is a parameter ──────────────────────────────────────────


def test_the_store_holds_no_item_knowledge_of_its_own(tmp_path: Path) -> None:
    narrow = BarVocabulary(
        items={"widget": {"hosts": ("status",), "options": {}}},
        version=1,
        max_items_per_host=2,
    )

    saved = bar_items_store.save_layout(
        tmp_path,
        document(header=[], status=[{"type": "widget"}]),
        vocabulary=narrow,
    )
    assert saved["status"] == [{"type": "widget", "options": {}}]

    # Every real item name is unknown under this catalog, and the cap is the
    # one this vocabulary states.
    with pytest.raises(BarLayoutInvalid) as unknown:
        bar_items_store.save_layout(
            tmp_path, document(header=[{"type": "logo"}]), vocabulary=narrow
        )
    assert unknown.value.reason == "unknown-type"

    with pytest.raises(BarLayoutInvalid) as overflow:
        bar_items_store.save_layout(
            tmp_path,
            document(header=[], status=[{"type": "widget"}] * 3),
            vocabulary=narrow,
        )
    assert overflow.value.reason == "overflow"


def test_the_store_writes_the_schema_version_the_vocabulary_states(tmp_path: Path) -> None:
    future = BarVocabulary(items=CATALOG, version=9, max_items_per_host=20)

    saved = bar_items_store.save_layout(tmp_path, document(version=9), vocabulary=future)

    assert saved["version"] == 9
    # And a version-1 document is now the unreadable one.
    assert (
        bar_items_store.load_layout(tmp_path, vocabulary=VOCABULARY, default=DEFAULT_LAYOUT)[
            "header"
        ]
        == DEFAULT_LAYOUT["header"]
    )
