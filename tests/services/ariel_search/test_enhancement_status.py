"""Unit tests for :mod:`osprey.services.ariel_search.enhancement.status`.

The Python decision table (:func:`enhancement_view`) as a parametrized case
table, the SQL form's shape (:func:`enhancement_state_sql`, :func:`has_images_sql`),
and the config-derived pieces (:func:`current_markers`, :func:`enabled_modules`).
The SQL/Python parity on real rows is pinned by the filter SQL task's
integration table; here the SQL is checked for its rows and parameters.
"""

from __future__ import annotations

import pytest

from osprey.imaging.formats import viewable_sql
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import (
    IMAGE_STATUS_KEYS,
    MAX_ENHANCEMENT_ATTEMPTS,
)
from osprey.services.ariel_search.enhancement.status import (
    IMAGE_MODULES,
    EnhancementContext,
    current_markers,
    enabled_modules,
    enhancement_context,
    enhancement_state_sql,
    enhancement_view,
    has_images,
    has_images_sql,
    image_embedding_marker,
)

_DB = {"database": {"uri": "postgresql://localhost/test"}}
_ALL = ["image_caption", "semantic_processor", "text_embedding", "image_embedding", "qmd_export"]
_MARKERS = {"image_caption": "vis-a", "image_embedding": "image_embeddings_x"}


def _state(module, status, *, images=True, markers=_MARKERS):
    return enhancement_view(status, has_images=images, enabled=[module], markers=markers)[module]


# ---------------------------------------------------------------------------
# The decision table
# ---------------------------------------------------------------------------

_CASES = [
    # (id, module, stored status object or _MISSING, has_images, expected)
    ("missing-text", "text_embedding", None, True, "pending"),
    ("missing-image", "image_caption", None, True, "pending"),
    ("missing-image-no-pictures", "image_caption", None, False, "n/a"),
    (
        "image-marker-mismatch-stale",
        "image_caption",
        {"status": "complete", "marker": "vis-old"},
        True,
        "stale",
    ),
    (
        "image-marker-absent-stale",
        "image_embedding",
        {"status": "complete"},
        True,
        "stale",
    ),
    (
        "image-stale-wins-over-failed",
        "image_caption",
        {"status": "failed", "gave_up": True, "attempts": 3, "marker": "vis-old"},
        True,
        "stale",
    ),
    (
        "image-old-marker-no-pictures-na",
        "image_caption",
        {"status": "complete", "marker": "vis-old"},
        False,
        "n/a",
    ),
    (
        "text-never-stale",
        "text_embedding",
        {"status": "complete", "marker": "anything", "md5": "x"},
        True,
        "complete",
    ),
    (
        "failed-gave-up",
        "image_caption",
        {"status": "failed", "gave_up": True, "attempts": 3, "marker": "vis-a"},
        True,
        "failed",
    ),
    (
        "image-failed-not-given-up",
        "image_caption",
        {"status": "failed", "gave_up": False, "attempts": 1, "marker": "vis-a"},
        True,
        "failed",
    ),
    (
        "image-failed-no-pictures-na",
        "image_embedding",
        {"status": "failed", "gave_up": True, "attempts": 3, "marker": "image_embeddings_x"},
        False,
        "n/a",
    ),
    (
        "text-attempts-at-max-failed",
        "semantic_processor",
        {"status": "pending", "attempts": MAX_ENHANCEMENT_ATTEMPTS},
        True,
        "failed",
    ),
    (
        "text-attempts-below-max-pending",
        "semantic_processor",
        {"status": "pending", "attempts": MAX_ENHANCEMENT_ATTEMPTS - 1},
        True,
        "pending",
    ),
    (
        "text-attempts-string",
        "qmd_export",
        {"status": "pending", "attempts": str(MAX_ENHANCEMENT_ATTEMPTS)},
        True,
        "failed",
    ),
    ("text-failed", "qmd_export", {"status": "failed", "attempts": 1}, True, "failed"),
    ("text-complete", "text_embedding", {"status": "complete"}, True, "complete"),
    ("text-complete-no-pictures", "text_embedding", {"status": "complete"}, False, "complete"),
    (
        "image-complete-current-marker",
        "image_caption",
        {"status": "complete", "marker": "vis-a"},
        True,
        "complete",
    ),
    ("unknown-status-pending", "text_embedding", {"status": "running"}, True, "pending"),
    ("non-object-value-text-pending", "text_embedding", "complete", True, "pending"),
]


@pytest.mark.parametrize(
    ("module", "stored", "images", "expected"),
    [c[1:] for c in _CASES],
    ids=[c[0] for c in _CASES],
)
def test_decision_table(module, stored, images, expected):
    status = {} if stored is None else {module: stored}
    assert _state(module, status, images=images) == expected


def test_image_module_without_current_marker_is_never_stale():
    status = {"image_embedding": {"status": "complete", "marker": "old_table"}}
    markers = {"image_caption": "vis-a", "image_embedding": None}
    assert _state("image_embedding", status, markers=markers) == "complete"


def test_disabled_module_is_absent_and_order_follows_enabled():
    status = {"text_embedding": {"status": "complete"}, "qmd_export": {"status": "complete"}}
    view = enhancement_view(
        status, has_images=False, enabled=["image_caption", "text_embedding"], markers=_MARKERS
    )
    assert list(view) == ["image_caption", "text_embedding"]
    assert "qmd_export" not in view
    assert view == {"image_caption": "n/a", "text_embedding": "complete"}


@pytest.mark.parametrize("raw", [None, "", "not json", "[]", 42])
def test_unreadable_status_reads_as_no_keys(raw):
    view = enhancement_view(raw, has_images=True, enabled=["text_embedding"], markers={})
    assert view == {"text_embedding": "pending"}


def test_status_given_as_json_string():
    raw = '{"text_embedding": {"status": "complete"}}'
    view = enhancement_view(raw, has_images=True, enabled=["text_embedding"], markers={})
    assert view == {"text_embedding": "complete"}


def test_image_modules_match_repository_keys():
    assert IMAGE_MODULES == IMAGE_STATUS_KEYS


# ---------------------------------------------------------------------------
# has_images
# ---------------------------------------------------------------------------


def test_has_images_sql_is_exists_over_viewable_attachment_files():
    sql = has_images_sql("e")
    assert sql == (
        "EXISTS (SELECT 1 FROM attachment_files f WHERE f.entry_id = e.entry_id"
        f" AND {viewable_sql('f')})"
    )
    assert "attachments" not in sql.replace("attachment_files", "")
    assert "%s" not in sql


def test_has_images_sql_without_copy_state_is_false():
    assert has_images_sql("e", has_copy_state=False) == "FALSE"


@pytest.mark.parametrize("alias", ["f", "e;", "1x", "e.x", ""])
def test_has_images_sql_rejects_bad_alias(alias):
    with pytest.raises(ValueError):
        has_images_sql(alias)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [({"has_images": True}, True), ({"has_images": False}, False), ({}, False)],
)
def test_has_images_reads_the_selected_boolean(entry, expected):
    assert has_images(entry) is expected


# ---------------------------------------------------------------------------
# enhancement_state_sql
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", _ALL)
@pytest.mark.parametrize("state", ["failed", "stale"])
def test_sql_placeholders_match_params(module, state):
    sql, params = enhancement_state_sql(module, state, "e", marker=_MARKERS.get(module))
    assert sql.count("%s") == len(params)
    assert "%(" not in sql


@pytest.mark.parametrize("module", ["semantic_processor", "text_embedding", "qmd_export"])
def test_text_module_is_never_stale_in_sql(module):
    assert enhancement_state_sql(module, "stale", "e", marker="whatever") == ("FALSE", [])


def test_image_module_without_marker_is_never_stale_in_sql():
    assert enhancement_state_sql("image_embedding", "stale", "e", marker=None) == ("FALSE", [])


def test_image_stale_sql_requires_pictures_and_differing_marker():
    sql, params = enhancement_state_sql("image_caption", "stale", "x", marker="vis-a")
    assert "x.enhancement_status ? %s" in sql
    assert "->>'marker' IS DISTINCT FROM %s" in sql
    assert sql.rstrip(")").endswith(viewable_sql("f").rstrip(")"))
    assert has_images_sql("x") in sql
    assert params == ["image_caption", "image_caption", "vis-a"]


def test_image_failed_sql_excludes_stale_and_requires_pictures():
    sql, params = enhancement_state_sql("image_caption", "failed", "e", marker="vis-a")
    assert "NOT (e.enhancement_status->%s->>'marker' IS DISTINCT FROM %s)" in sql
    assert "->>'status' = 'failed'" in sql
    assert "->>'gave_up')::boolean" in sql
    assert "attempts" not in sql
    assert has_images_sql("e") in sql
    assert params == ["image_caption", "image_caption", "vis-a", "image_caption", "image_caption"]


def test_text_failed_sql_counts_attempts_and_ignores_pictures():
    sql, params = enhancement_state_sql("text_embedding", "failed", "e", marker="ignored")
    assert "attachment_files" not in sql
    assert "marker" not in sql
    assert "->>'attempts')::int, 0) >= %s" in sql
    assert params[-1] == MAX_ENHANCEMENT_ATTEMPTS
    assert "ignored" not in params


def test_image_sql_without_copy_state_never_matches():
    sql, _ = enhancement_state_sql(
        "image_caption", "failed", "e", marker="vis-a", has_copy_state=False
    )
    assert sql.endswith(" AND FALSE)")
    assert "attachment_files" not in sql


def test_sql_uses_the_alias():
    sql, _ = enhancement_state_sql("qmd_export", "failed", "ent")
    assert "ent.enhancement_status" in sql
    assert "e.enhancement_status" not in sql.replace("ent.enhancement_status", "")


@pytest.mark.parametrize(
    ("state", "alias"), [("pending", "e"), ("complete", "e"), ("failed", "e x")]
)
def test_sql_rejects_unknown_state_or_bad_alias(state, alias):
    with pytest.raises(ValueError):
        enhancement_state_sql("text_embedding", state, alias)


# ---------------------------------------------------------------------------
# Markers, enabled modules, context
# ---------------------------------------------------------------------------


def _config(**modules):
    return ARIELConfig.from_dict({**_DB, "enhancement_modules": modules})


def test_current_markers_none_when_unconfigured():
    assert current_markers(ARIELConfig.from_dict(_DB)) == {
        "image_caption": None,
        "image_embedding": None,
    }


def test_current_markers_caption_model_ignores_enabled_flag():
    config = _config(image_caption={"enabled": False, "model": {"model_id": "vis-a"}})
    assert current_markers(config)["image_caption"] == "vis-a"


def test_image_embedding_marker_off():
    assert image_embedding_marker(ARIELConfig.from_dict(_DB)) == (None, False)


def test_enabled_modules_in_execution_order_and_disabled_absent():
    config = _config(
        qmd_export={"enabled": True},
        text_embedding={"enabled": True},
        semantic_processor={"enabled": False},
    )
    enabled = enabled_modules(config)
    assert "semantic_processor" not in enabled
    assert enabled == [name for name in _ALL if name in enabled]
    assert enabled.index("text_embedding") < enabled.index("qmd_export")


def test_enhancement_context_from_config():
    config = _config(text_embedding={"enabled": True})
    ctx = enhancement_context(config, has_copy_state=False)
    assert isinstance(ctx, EnhancementContext)
    assert list(ctx.enabled) == enabled_modules(config)
    assert ctx.markers == current_markers(config)
    assert ctx.has_copy_state is False


def test_enhancement_context_positional_and_frozen():
    ctx = EnhancementContext({"image_caption": "m"}, ["image_caption"])
    assert ctx.has_copy_state is True
    with pytest.raises(AttributeError):
        ctx.enabled = []  # type: ignore[misc]
