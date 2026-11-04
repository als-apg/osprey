"""Unit tests for :mod:`osprey.services.ariel_search.enhancement.status`.

The Python decision table (:func:`enhancement_view`) as a parametrized case
table; the five-state SQL CASE (:func:`enhancement_state_case_sql`) evaluated
row for row against it by translating the SQL text with PostgreSQL's JSONB and
NULL semantics; the config-level markers (:func:`current_markers`); and the
re-run command (:func:`rerun_command`). Parity on a real store is pinned by the
integration tables.
"""

from __future__ import annotations

import json
import re

import pytest

from osprey.imaging.formats import viewable_sql
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import (
    IMAGE_STATUS_KEYS,
    MAX_ENHANCEMENT_ATTEMPTS,
)
from osprey.services.ariel_search.enhancement.qmd_export.exporter import (
    QMD_FORMAT_VERSION,
    resolve_mirror_path,
)
from osprey.services.ariel_search.enhancement.status import (
    IMAGE_MODULES,
    MARKER_MODULES,
    SQL_STATES,
    STATES,
    EnhancementContext,
    TextEmbeddingBuild,
    build_key_stale_sql,
    current_markers,
    current_text_table,
    enabled_modules,
    enhancement_context,
    enhancement_state,
    enhancement_state_case_sql,
    enhancement_state_sql,
    enhancement_view,
    filter_needs_build_keys,
    has_images,
    has_images_sql,
    image_embedding_marker,
    marker_stale_sql,
    qmd_export_marker,
    reembed_command,
    rerun_command,
    semantic_processor_marker,
    store_enhancement_context,
    text_embedding_build,
)
from osprey.services.ariel_search.enhancement.text_embedding.embedder import running_input_key
from osprey.services.ariel_search.vocabulary.entry import classify_build_keys

_DB = {"database": {"uri": "postgresql://localhost/test"}}
_ALL = ["image_caption", "semantic_processor", "text_embedding", "image_embedding", "qmd_export"]
_MARKERS = {
    "image_caption": "vis-a",
    "semantic_processor": "default",
    "image_embedding": "image_embeddings_x",
    "qmd_export": "1:mirror:none",
}


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
        "image-failed-wins-over-stale",
        "image_caption",
        {"status": "failed", "gave_up": True, "attempts": 3, "marker": "vis-old"},
        True,
        "failed",
    ),
    (
        "image-old-marker-no-pictures-na",
        "image_caption",
        {"status": "complete", "marker": "vis-old"},
        False,
        "n/a",
    ),
    (
        "text-without-current-marker-never-stale",
        "text_embedding",
        {"status": "complete", "marker": "anything", "md5": "x"},
        True,
        "complete",
    ),
    (
        "text-marker-differs-stale",
        "semantic_processor",
        {"status": "complete", "marker": "old-model"},
        True,
        "stale",
    ),
    (
        "text-marker-absent-complete",
        "semantic_processor",
        {"status": "complete"},
        True,
        "complete",
    ),
    (
        "text-marker-null-complete",
        "qmd_export",
        {"status": "complete", "marker": None},
        True,
        "complete",
    ),
    (
        "text-marker-current-complete",
        "qmd_export",
        {"status": "complete", "marker": "1:mirror:none"},
        True,
        "complete",
    ),
    (
        "text-failed-wins-over-stale",
        "semantic_processor",
        {"status": "failed", "attempts": 1, "marker": "old-model"},
        True,
        "failed",
    ),
    (
        "text-marker-differs-pending-stale",
        "qmd_export",
        {"status": "pending", "attempts": 1, "marker": "0:old:none"},
        True,
        "stale",
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
# The SQL form: one five-state CASE
# ---------------------------------------------------------------------------


def _pg_text(status, module, key):
    """``status->module->>key`` as PostgreSQL computes it on a JSONB value."""
    value = status.get(module)
    if not isinstance(value, dict):
        return None
    field = value.get(key)
    if field is None:
        return None
    if isinstance(field, str):
        return field
    return json.dumps(field)


def _pg_bool(status, module, key):
    text = _pg_text(status, module, key)
    if text is None:
        return False
    lowered = text.strip().lower()
    if lowered in ("true", "t", "yes", "y", "on", "1"):
        return True
    if lowered in ("false", "f", "no", "n", "off", "0"):
        return False
    raise ValueError(f"invalid input syntax for type boolean: {text!r}")


def _pg_int(status, module, key):
    text = _pg_text(status, module, key)
    return 0 if text is None else int(text)


_STATUS = r"e\.enhancement_status"
_LIT = r"('(?:[^']*)')"


def _evaluate_case(sql, params, status, has_pictures):
    """Evaluate the CASE :func:`enhancement_state_case_sql` builds, on one row.

    The SQL text itself is translated, operator by operator, into Python with
    PostgreSQL's JSONB and NULL semantics, so the test checks the SQL the store
    runs and not a second copy of the rules.
    """
    sql = sql.replace(has_images_sql("e"), "HAS")
    for param in params:
        assert "%s" in sql
        sql = sql.replace("%s", repr(param), 1)
    assert "%s" not in sql
    sql = re.sub(
        rf"COALESCE\(\({_STATUS}->{_LIT}->>'(\w+)'\)::boolean, false\)", r"_bool(S, \1, '\2')", sql
    )
    sql = re.sub(rf"COALESCE\(\({_STATUS}->{_LIT}->>'(\w+)'\)::int, 0\)", r"_int(S, \1, '\2')", sql)
    sql = re.sub(rf"COALESCE\({_STATUS} \? {_LIT}, false\)", r"(\1 in S)", sql)
    sql = re.sub(rf"{_STATUS}->{_LIT}->>'(\w+)'", r"_text(S, \1, '\2')", sql)
    # ``=`` and ``<>`` on NULL are NULL (falsy in a WHEN); IS DISTINCT FROM is not.
    operand = r"(_text\([^)]*\)|'[^']*')"
    sql = re.sub(rf"{operand} = {operand}", r"_eq(\1, \2)", sql)
    sql = re.sub(rf"{operand} <> {operand}", r"_ne(\1, \2)", sql)
    for pattern, python in (
        (r" IS DISTINCT FROM ", " != "),
        (r" IS NOT NULL", " is not None"),
        (r"\bAND\b", "and"),
        (r"\bOR\b", "or"),
        (r"\bNOT\b", "not"),
        (r"\bFALSE\b", "False"),
    ):
        sql = re.sub(pattern, python, sql)
    match = re.fullmatch(r"\(CASE (.*) ELSE '([^']*)' END\)", sql)
    assert match, sql
    names = {
        "S": status,
        "HAS": has_pictures,
        "_text": _pg_text,
        "_bool": _pg_bool,
        "_int": _pg_int,
        "_eq": lambda a, b: None if a is None or b is None else a == b,
        "_ne": lambda a, b: None if a is None or b is None else a != b,
    }
    for condition, value in re.findall(r"WHEN (.*?) THEN '([^']*)'", match.group(1)):
        if eval(condition, names):
            return value
    return match.group(2)


_STORED = [
    None,
    "complete",
    {},
    {"status": "complete"},
    {"status": "complete", "marker": None},
    {"status": "complete", "marker": "current"},
    {"status": "complete", "marker": "old"},
    {"status": "pending", "attempts": 1, "marker": "old"},
    {"status": "pending", "attempts": MAX_ENHANCEMENT_ATTEMPTS - 1},
    {"status": "pending", "attempts": MAX_ENHANCEMENT_ATTEMPTS},
    {"status": "pending", "attempts": str(MAX_ENHANCEMENT_ATTEMPTS)},
    {"status": "running", "marker": "current"},
    {"status": "failed", "attempts": 1},
    {"status": "failed", "attempts": 1, "marker": "old"},
    {"status": "failed", "attempts": 1, "marker": "current"},
    {"status": "complete", "gave_up": True, "marker": "old"},
    {"status": "complete", "gave_up": "true", "marker": "current"},
    {"status": "complete", "gave_up": False, "marker": "current"},
]

# One module per type: image, marker text, text without a marker.
_PARITY_MODULES = ["image_caption", "image_embedding", "semantic_processor", "qmd_export"]
_PARITY_MODULES.append("text_embedding")


@pytest.mark.parametrize("module", _PARITY_MODULES)
@pytest.mark.parametrize("marker", ["current", None])
def test_python_and_sql_agree_on_every_state(module, marker):
    seen = set()
    for stored in _STORED:
        status = {} if stored is None else {module: stored}
        for pictures in (True, False):
            expected = enhancement_state(module, status, has_images=pictures, marker=marker)
            sql, params = enhancement_state_case_sql(module, "e", marker=marker)
            assert _evaluate_case(sql, params, status, pictures) == expected, (stored, pictures)
            for state in STATES:
                predicate, predicate_params = enhancement_state_sql(
                    module, state, "e", marker=marker
                )
                if predicate == "FALSE":
                    assert state != expected
                    continue
                assert predicate == f"({sql} = %s)"
                assert predicate_params == [*params, state]
            seen.add(expected)
    # text_embedding's stale is its build key's, never a stored marker
    # (test_build_key_python_and_sql_agree_on_every_state).
    can_be_stale = marker is not None and module != "text_embedding"
    image = module in IMAGE_MODULES
    expected_states = {"pending", "failed", "complete"}
    expected_states |= {"stale"} if can_be_stale else set()
    expected_states |= {"n/a"} if image else set()
    assert seen == expected_states


@pytest.mark.parametrize("module", _ALL)
@pytest.mark.parametrize("state", STATES)
def test_sql_placeholders_match_params(module, state):
    sql, params = enhancement_state_sql(module, state, "e", marker=_MARKERS.get(module))
    assert sql.count("%s") == len(params)
    assert "%(" not in sql


@pytest.mark.parametrize("module", _ALL)
def test_stale_without_a_current_marker_is_false(module):
    assert enhancement_state_sql(module, "stale", "e", marker=None) == ("FALSE", [])


def test_marker_stale_sql_text_needs_a_present_differing_marker():
    sql, params = marker_stale_sql("semantic_processor", "e", marker="m")
    assert sql == (
        "(e.enhancement_status->%s->>'marker' IS NOT NULL"
        " AND e.enhancement_status->%s->>'marker' <> %s)"
    )
    assert params == ["semantic_processor", "semantic_processor", "m"]


def test_marker_stale_sql_image_counts_a_missing_marker():
    sql, params = marker_stale_sql("image_caption", "x", marker="vis-a")
    assert sql == "(x.enhancement_status->%s->>'marker' IS DISTINCT FROM %s)"
    assert params == ["image_caption", "vis-a"]


def test_marker_stale_sql_without_marker_is_false():
    assert marker_stale_sql("image_caption", "e", marker=None) == ("FALSE", [])


def test_image_case_requires_pictures():
    sql, _ = enhancement_state_case_sql("image_caption", "x", marker="vis-a")
    assert f"WHEN NOT {has_images_sql('x')} THEN 'n/a'" in sql
    assert "attempts" not in sql


def test_text_case_counts_attempts_and_ignores_pictures():
    sql, params = enhancement_state_case_sql("text_embedding", "e")
    assert "attachment_files" not in sql
    assert "marker" not in sql
    assert "->>'attempts')::int, 0) >= %s" in sql
    assert MAX_ENHANCEMENT_ATTEMPTS in params


def test_image_sql_without_copy_state_never_matches_a_picture():
    sql, _ = enhancement_state_sql(
        "image_caption", "failed", "e", marker="vis-a", has_copy_state=False
    )
    assert "WHEN NOT FALSE THEN 'n/a'" in sql
    assert "attachment_files" not in sql


def test_sql_uses_the_alias():
    sql, _ = enhancement_state_sql("qmd_export", "failed", "ent", marker="1:m:none")
    assert "ent.enhancement_status" in sql
    assert "e.enhancement_status" not in sql.replace("ent.enhancement_status", "")


@pytest.mark.parametrize(("state", "alias"), [("done", "e"), ("", "e"), ("failed", "e x")])
def test_sql_rejects_unknown_state_or_bad_alias(state, alias):
    with pytest.raises(ValueError):
        enhancement_state_sql("text_embedding", state, alias)


def test_filter_states_are_failed_and_stale():
    assert SQL_STATES == ("failed", "stale")
    assert set(SQL_STATES) <= set(STATES)


def test_context_state_sql_passes_its_marker():
    ctx = EnhancementContext(_MARKERS, _ALL)
    assert ctx.state_sql("qmd_export", "stale", "e") == enhancement_state_sql(
        "qmd_export", "stale", "e", marker="1:mirror:none"
    )


# ---------------------------------------------------------------------------
# Markers, enabled modules, context
# ---------------------------------------------------------------------------


def _config(vocabulary=None, **modules):
    raw = {**_DB, "enhancement_modules": modules}
    if vocabulary is not None:
        raw["vocabulary"] = vocabulary
    return ARIELConfig.from_dict(raw)


def _semantic(**settings):
    return _config(semantic_processor={"enabled": True, "provider": "cborg", **settings})


def test_current_markers_none_when_unconfigured():
    assert current_markers(ARIELConfig.from_dict(_DB)) == {
        "image_caption": None,
        "semantic_processor": None,
        "image_embedding": None,
        "qmd_export": None,
    }


def test_current_markers_covers_every_marker_module():
    config = _config(
        semantic_processor={"enabled": True, "provider": "cborg", "model": {"model_id": "m1"}},
        qmd_export={"enabled": True, "mirror_path": "mirror"},
    )
    markers = current_markers(config)
    assert set(markers) == set(MARKER_MODULES)
    assert markers["semantic_processor"] == "m1"
    assert markers["qmd_export"] == f"{QMD_FORMAT_VERSION}:mirror:none"


def test_current_markers_caption_model_ignores_enabled_flag():
    config = _config(image_caption={"enabled": False, "model": {"model_id": "vis-a"}})
    assert current_markers(config)["image_caption"] == "vis-a"


def test_image_embedding_marker_off():
    assert image_embedding_marker(ARIELConfig.from_dict(_DB)) == (None, False)


def test_semantic_marker_is_the_explicit_model_id():
    assert semantic_processor_marker(_semantic(model={"model_id": " sum-1 "})) == "sum-1"


@pytest.mark.parametrize("model", [None, {}, {"model_id": None}, {"model_id": ""}])
def test_semantic_marker_without_model_id_is_default(model):
    settings = {} if model is None else {"model": model}
    assert semantic_processor_marker(_semantic(**settings)) == "default"


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(lambda: _semantic(model="not-a-mapping"), id="model-not-mapping"),
        pytest.param(lambda: _semantic(model={"model_id": 42}), id="model-id-not-str"),
        pytest.param(
            lambda: _config(semantic_processor={"enabled": True, "model": {"model_id": "m"}}),
            id="no-provider",
        ),
        pytest.param(
            lambda: _config(semantic_processor={"enabled": False, "provider": "cborg"}),
            id="disabled",
        ),
    ],
)
def test_semantic_marker_none_on_broken_or_disabled_config(config):
    assert semantic_processor_marker(config()) is None


def test_marker_functions_swallow_any_config_error():
    class Broken:
        def is_enhancement_module_enabled(self, _name):
            return True

        def get_enhancement_module_config(self, _name):
            raise RuntimeError("broken")

        def entry_expander(self):
            raise RuntimeError("broken")

    assert semantic_processor_marker(Broken()) is None
    assert qmd_export_marker(Broken()) is None


def test_qmd_marker_is_version_authored_path_and_expander_key(monkeypatch):
    config = _config(qmd_export={"enabled": True, "mirror_path": "~/mirror"})
    assert qmd_export_marker(config) == f"{QMD_FORMAT_VERSION}:~/mirror:none"

    class Expander:
        key = "e1:0123456789abcdef"

    monkeypatch.setattr(config, "entry_expander", lambda: Expander())
    assert qmd_export_marker(config) == f"{QMD_FORMAT_VERSION}:~/mirror:e1:0123456789abcdef"


@pytest.mark.parametrize(
    "settings", [{"enabled": True}, {"enabled": True, "mirror_path": "  "}, {"enabled": False}]
)
def test_qmd_marker_none_without_mirror_path_or_disabled(settings):
    assert qmd_export_marker(_config(qmd_export=settings)) is None


def test_qmd_marker_none_when_expansion_on_and_vocabulary_unusable():
    config = _config(
        vocabulary={"enabled": True, "expand_entries": True},
        qmd_export={"enabled": True, "mirror_path": "mirror"},
    )
    assert qmd_export_marker(config) is None


def test_qmd_marker_same_for_one_authored_path_under_two_roots(tmp_path, monkeypatch):
    markers, roots = [], []
    for name in ("host", "terminal"):
        config_dir = tmp_path / name
        config_dir.mkdir()
        (config_dir / "config.yml").write_text("{}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(config_dir / "config.yml"))
        roots.append(resolve_mirror_path("mirror"))
        config = _config(qmd_export={"enabled": True, "mirror_path": "mirror"})
        ctx = enhancement_context(config, has_copy_state=True, input_key_column=False)
        markers.append(ctx.markers["qmd_export"])
    assert roots[0] != roots[1]
    assert markers[0] == markers[1] == f"{QMD_FORMAT_VERSION}:mirror:none"


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
    ctx = enhancement_context(config, has_copy_state=False, input_key_column=False)
    assert isinstance(ctx, EnhancementContext)
    assert list(ctx.enabled) == enabled_modules(config)
    assert ctx.markers == current_markers(config)
    assert ctx.has_copy_state is False


def test_enhancement_context_resolves_markers_once(monkeypatch):
    from osprey.services.ariel_search.enhancement import status

    calls = []
    real = status.semantic_processor_marker
    monkeypatch.setattr(
        status, "semantic_processor_marker", lambda config: calls.append(1) or real(config)
    )
    ctx = enhancement_context(
        _semantic(model={"model_id": "m"}), has_copy_state=True, input_key_column=False
    )
    for _ in range(3):
        ctx.view({"enhancement_status": {"semantic_processor": {"status": "complete"}}})
        ctx.state_sql("semantic_processor", "stale", "e")
    assert calls == [1]
    assert ctx.markers["semantic_processor"] == "m"


def test_enhancement_context_positional_and_frozen():
    ctx = EnhancementContext({"image_caption": "m"}, ["image_caption"])
    assert ctx.has_copy_state is True
    with pytest.raises(AttributeError):
        ctx.enabled = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The re-run command
# ---------------------------------------------------------------------------

_CTX = EnhancementContext(_MARKERS, _ALL)


@pytest.mark.parametrize(
    ("module", "state"),
    [
        (module, state)
        for module in _ALL
        for state in ("failed", "stale")
        # text_embedding stale is build-key stale, which names reembed (below).
        if (module, state) != ("text_embedding", "stale")
    ],
)
def test_rerun_command_enhances_one_module_for_one_entry(module, state):
    assert rerun_command(module, "e-1", state, _CTX) == (
        f"osprey ariel enhance --module {module} --entry e-1"
    )


@pytest.mark.parametrize("state", ["complete", "pending", "n/a"])
def test_rerun_command_none_for_states_with_nothing_to_rerun(state):
    assert rerun_command("semantic_processor", "e-1", state, _CTX) is None


def test_rerun_command_quotes_the_entry_id():
    assert rerun_command("qmd_export", "a b", "failed", _CTX) == (
        "osprey ariel enhance --module qmd_export --entry 'a b'"
    )


def test_rerun_command_text_embedding_stale_names_reembed():
    ctx = EnhancementContext(_MARKERS, _ALL, text_model={"name": "nomic", "dimension": 768})
    command = rerun_command("text_embedding", "e-1", "stale", ctx)
    assert command == reembed_command("nomic", 768)
    assert command == "osprey ariel reembed --model nomic --dimension 768 --force"


def test_rerun_command_text_embedding_stale_without_table_is_none():
    assert rerun_command("text_embedding", "e-1", "stale", _CTX) is None


# ---------------------------------------------------------------------------
# text_embedding: the build key (A4)
# ---------------------------------------------------------------------------

_RUNNING = "r1:e1:0123456789abcdef"
_TEXT_TABLE = "text_embeddings_nomic"
_BUILD_KEYS = [None, "r1", _RUNNING, "r1:e1:fedcba9876543210", "r2", "garbage"]


def _text_config(vocabulary=None, *, search_model=None, models=("nomic",), enabled=True):
    raw = {
        **_DB,
        "enhancement_modules": {
            "text_embedding": {
                "enabled": enabled,
                "models": [{"name": name, "dimension": 768} for name in models],
            }
        },
    }
    if search_model is not None:
        raw["search_modules"] = {"semantic": {"enabled": True, "model": search_model}}
    if vocabulary is not None:
        raw["vocabulary"] = vocabulary
    return ARIELConfig.from_dict(raw)


def _evaluate_build_key_case(sql, params, status, pictures, build_key):
    """:func:`_evaluate_case` with the build-key ``EXISTS`` read off ``build_key``.

    The row of the text table is the entry's, its stored key ``build_key`` (None:
    no row), so the fragment is "a row exists and its key differs".
    """
    fragment, _ = build_key_stale_sql("e", _TEXT_TABLE, _RUNNING)
    if fragment in sql:
        sql = sql.replace(fragment, f"({build_key is not None} and {build_key!r} != %s)")
    return _evaluate_case(sql, params, status, pictures)


def test_build_key_python_and_sql_agree_on_every_state():
    seen = set()
    for stored in _STORED:
        status = {} if stored is None else {"text_embedding": stored}
        for build_key in _BUILD_KEYS:
            expected = enhancement_state(
                "text_embedding", status, has_images=False, marker=_RUNNING, build_key=build_key
            )
            sql, params = enhancement_state_case_sql(
                "text_embedding", "e", marker=_RUNNING, text_table=_TEXT_TABLE
            )
            got = _evaluate_build_key_case(sql, params, status, False, build_key)
            assert got == expected, (stored, build_key)
            seen.add(expected)
    assert seen == {"pending", "failed", "stale", "complete"}


def test_build_key_stale_is_a_differing_row_key_and_failed_wins():
    complete = {"text_embedding": {"status": "complete"}}
    failed = {"text_embedding": {"status": "failed", "attempts": 1}}

    def state(status, build_key, marker=_RUNNING):
        return enhancement_state(
            "text_embedding", status, has_images=False, marker=marker, build_key=build_key
        )

    assert state(complete, "r1") == "stale"
    assert state(complete, _RUNNING) == "complete"
    assert state(complete, None) == "complete"  # no row: nothing built to be stale
    assert state(failed, "r1") == "failed"
    assert state(complete, "r1", marker=None) == "complete"  # no verdict


def test_build_key_stale_sql_reads_the_row_key_with_r1_for_null():
    sql, params = build_key_stale_sql("e", "text_embeddings_a:b", _RUNNING)
    assert sql == (
        'EXISTS (SELECT 1 FROM "text_embeddings_a:b" t WHERE t.entry_id = e.entry_id'
        " AND COALESCE(t.input_key, 'r1') <> %s)"
    )
    assert params == [_RUNNING]
    with pytest.raises(ValueError):
        build_key_stale_sql("t", _TEXT_TABLE, _RUNNING)


def test_build_key_stale_needs_both_table_and_running_key():
    for marker, table in ((None, _TEXT_TABLE), (_RUNNING, None), (None, None)):
        assert enhancement_state_sql(
            "text_embedding", "stale", "e", marker=marker, text_table=table
        ) == ("FALSE", [])
        sql, _ = enhancement_state_case_sql("text_embedding", "e", marker=marker, text_table=table)
        assert "'stale'" not in sql
    sql, params = enhancement_state_sql(
        "text_embedding", "stale", "e", marker=_RUNNING, text_table=_TEXT_TABLE
    )
    assert sql.count("%s") == len(params)
    assert build_key_stale_sql("e", _TEXT_TABLE, _RUNNING)[0] in sql


def test_build_key_is_never_owed():
    # marker_stale_sql feeds the owed-work query: re-embedding stays with reembed.
    assert marker_stale_sql("text_embedding", "e", marker=_RUNNING) == ("FALSE", [])


def test_text_embedding_build_key_running_key_and_reasons():
    plain = text_embedding_build(_text_config())
    assert plain == TextEmbeddingBuild(
        _TEXT_TABLE, {"name": "nomic", "dimension": 768}, running_input_key(None), None
    )
    assert text_embedding_build(_text_config(enabled=False)).unavailable == "disabled"
    assert text_embedding_build(_text_config(), "text_embeddings_other").unavailable == (
        "unconfigured"
    )
    unusable = text_embedding_build(_text_config({"enabled": True, "expand_entries": True}))
    assert (unusable.running_key, unusable.unavailable) == (None, "vocabulary")

    class Expander:
        key = "e1:0123456789abcdef"

    given = text_embedding_build(_text_config(), expander=(Expander(), False))
    assert given.running_key == _RUNNING == running_input_key(Expander())
    assert text_embedding_build(_text_config(), expander=(None, True)).running_key is None


def test_current_text_table_build_key_is_search_model_else_first_model():
    assert current_text_table(_text_config(models=("a", "b"))) == "text_embeddings_a"
    assert (
        current_text_table(_text_config(models=("a", "b"), search_model="b")) == "text_embeddings_b"
    )
    assert current_text_table(ARIELConfig.from_dict(_DB)) is None


def test_enhancement_context_build_key_only_with_a_verdict():
    config = _text_config()
    without = enhancement_context(config, has_copy_state=True, input_key_column=False)
    assert (without.text_table, without.text_key) == (None, None)
    assert without.text_model == {"name": "nomic", "dimension": 768}
    with_column = enhancement_context(config, has_copy_state=True, input_key_column=True)
    assert (with_column.text_table, with_column.text_key) == (_TEXT_TABLE, "r1")

    unusable = _text_config({"enabled": True, "expand_entries": True})
    ctx = enhancement_context(unusable, has_copy_state=True, input_key_column=True)
    assert (ctx.text_table, ctx.text_key) == (None, None)
    assert ctx.state_sql("text_embedding", "stale", "e") == ("FALSE", [])


def test_context_view_build_key_stale_count_equals_classify_build_keys():
    keys = {"a": "r1", "b": _RUNNING, "c": "r1:e1:fedcba9876543210", "d": "r2", "e": "x"}
    ctx = EnhancementContext(
        _MARKERS,
        ["text_embedding"],
        text_table=_TEXT_TABLE,
        text_key=_RUNNING,
        build_keys=keys,
    )
    entries = [
        {"entry_id": entry_id, "enhancement_status": {"text_embedding": {"status": "complete"}}}
        for entry_id in [*keys, "no-row"]
    ]
    stale = sum(ctx.view(entry)["text_embedding"] == "stale" for entry in entries)
    counts: dict[str, int] = {}
    for key in keys.values():
        counts[key] = counts.get(key, 0) + 1
    assert stale == sum(classify_build_keys(counts, _RUNNING).values()) == 4

    blind = EnhancementContext(_MARKERS, ["text_embedding"], build_keys=keys)
    assert all(blind.view(entry)["text_embedding"] == "complete" for entry in entries)


def test_context_state_sql_build_key_passes_table_and_running_key():
    ctx = EnhancementContext(_MARKERS, _ALL, text_table=_TEXT_TABLE, text_key=_RUNNING)
    assert ctx.state_sql("text_embedding", "stale", "e") == enhancement_state_sql(
        "text_embedding", "stale", "e", marker=_RUNNING, text_table=_TEXT_TABLE
    )


class _BuildKeyRepository:
    """Records the reads :func:`store_enhancement_context` makes."""

    def __init__(self, *, column=True, keys=None):
        self.column = column
        self.keys = keys or {}
        self.calls: list[tuple] = []

    async def schema_facts(self):
        from osprey.services.ariel_search.database.repository import SchemaFacts

        return SchemaFacts(True, True, True)

    async def text_input_key_column(self, table):
        self.calls.append(("probe", table))
        return self.column

    async def get_text_build_keys(self, table, entry_ids):
        self.calls.append(("keys", table, tuple(entry_ids)))
        return {i: self.keys[i] for i in entry_ids if i in self.keys}


async def test_store_context_build_key_one_batched_read_per_page():
    repo = _BuildKeyRepository(keys={"a": "r1:e1:old", "b": "r1"})
    ctx = await store_enhancement_context(_text_config(), repo, entry_ids=["a", "b", "c"])
    assert repo.calls == [("probe", _TEXT_TABLE), ("keys", _TEXT_TABLE, ("a", "b", "c"))]
    assert ctx.build_keys == {"a": "r1:e1:old", "b": "r1"}
    assert ctx.text_key == "r1"

    sql_only = _BuildKeyRepository()
    ctx = await store_enhancement_context(_text_config(), sql_only)
    assert sql_only.calls == [("probe", _TEXT_TABLE)]
    assert (ctx.text_table, ctx.text_key) == (_TEXT_TABLE, "r1")


async def test_store_context_build_key_without_verdict_reads_nothing():
    no_column = _BuildKeyRepository(column=False, keys={"a": "r9"})
    ctx = await store_enhancement_context(_text_config(), no_column, entry_ids=["a"])
    assert no_column.calls == [("probe", _TEXT_TABLE)]
    assert (ctx.text_key, dict(ctx.build_keys)) == (None, {})

    unusable = _BuildKeyRepository(keys={"a": "r9"})
    ctx = await store_enhancement_context(
        _text_config({"enabled": True, "expand_entries": True}), unusable, entry_ids=["a"]
    )
    assert unusable.calls == [("probe", _TEXT_TABLE)]
    assert ctx.text_key is None

    off = _BuildKeyRepository()
    ctx = await store_enhancement_context(_text_config(enabled=False), off, entry_ids=["a"])
    assert off.calls == []
    assert ctx.text_key is None

    unasked = _BuildKeyRepository()
    ctx = await store_enhancement_context(
        _text_config(), unasked, entry_ids=["a"], text_embedding=False
    )
    assert unasked.calls == []
    assert ctx.text_key is None


def test_filter_needs_build_keys_only_for_a_text_embedding_state():
    from osprey.services.ariel_search.filters import EntryFilter, filter_dimensions

    config = _text_config()
    dims = filter_dimensions(config)
    assert filter_needs_build_keys(
        EntryFilter.parse({"enhancement": ["text_embedding:stale"]}, dims)
    )
    assert not filter_needs_build_keys(EntryFilter.parse({"author": ["x"]}, dims))
