"""Unit tests for TriggerConfig dataclass and load_triggers() function."""

import textwrap

import pytest

from osprey.dispatch.trigger_config import (
    DispatcherConfig,
    TriggerConfig,
    load_triggers,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def write_yaml(tmp_path, content: str):
    """Helper: write YAML content to a temp file and return its path string."""
    p = tmp_path / "triggers.yml"
    p.write_text(textwrap.dedent(content))
    return str(p)


VALID_WEBHOOK_YAML = """\
    dispatcher:
      max_concurrent_runs: 3
      max_queue_depth: 50
      dispatch_target: http://localhost:8010/dispatch

    triggers:
      - name: beam-loss-alert
        source: webhook
        on_error:
          action: retry
          max_retries: 3
          backoff_sec: 5.0
        action:
          prompt: "Investigate the beam loss event: {payload}"
          allowed_tools:
            - get_pv
            - archiver_query
          skill: beam-diagnostics
"""


# ---------------------------------------------------------------------------
# Test 1: Valid webhook trigger parses all fields correctly
# ---------------------------------------------------------------------------


def test_valid_webhook_trigger_parses_all_fields(tmp_path):
    path = write_yaml(tmp_path, VALID_WEBHOOK_YAML)
    dispatcher_cfg, triggers = load_triggers(path)

    assert len(triggers) == 1
    t = triggers[0]

    assert isinstance(t, TriggerConfig)
    assert t.name == "beam-loss-alert"
    assert t.source == "webhook"

    assert t.on_error["action"] == "retry"
    assert t.on_error["max_retries"] == 3
    assert t.on_error["backoff_sec"] == 5.0

    assert t.action["prompt"] == "Investigate the beam loss event: {payload}"
    assert t.action["allowed_tools"] == ["get_pv", "archiver_query"]
    assert t.action.get("skill") == "beam-diagnostics"


# ---------------------------------------------------------------------------
# Test 2: Missing `name` raises ValueError
# ---------------------------------------------------------------------------


def test_missing_name_raises_value_error(tmp_path):
    yaml_content = """\
        dispatcher:
          max_concurrent_runs: 5
          max_queue_depth: 100
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - source: webhook
            on_error:
              action: drop
              max_retries: 0
              backoff_sec: 0.0
            action:
              prompt: "Handle event"
              allowed_tools: []
    """
    path = write_yaml(tmp_path, yaml_content)
    with pytest.raises(ValueError, match="name"):
        load_triggers(path)


# ---------------------------------------------------------------------------
# Test 3: Missing `action.prompt` raises ValueError
# ---------------------------------------------------------------------------


def test_missing_action_prompt_raises_value_error(tmp_path):
    yaml_content = """\
        dispatcher:
          max_concurrent_runs: 5
          max_queue_depth: 100
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: no-prompt-trigger
            source: webhook
            on_error:
              action: drop
              max_retries: 0
              backoff_sec: 0.0
            action:
              allowed_tools: []
    """
    path = write_yaml(tmp_path, yaml_content)
    with pytest.raises(ValueError, match="prompt"):
        load_triggers(path)


# ---------------------------------------------------------------------------
# Test 4: Unknown `source` type is accepted (forward-compatible)
# ---------------------------------------------------------------------------


def test_unknown_source_type_is_accepted(tmp_path):
    yaml_content = """\
        dispatcher:
          max_concurrent_runs: 5
          max_queue_depth: 100
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: future-source-trigger
            source: mqtt
            on_error:
              action: drop
              max_retries: 0
              backoff_sec: 0.0
            action:
              prompt: "Handle mqtt event"
              allowed_tools: []
    """
    path = write_yaml(tmp_path, yaml_content)
    _, triggers = load_triggers(path)

    assert len(triggers) == 1
    assert triggers[0].source == "mqtt"


# ---------------------------------------------------------------------------
# Test 5: `on_error` defaults to `drop` when omitted
# ---------------------------------------------------------------------------


def test_on_error_defaults_to_drop_when_omitted(tmp_path):
    yaml_content = """\
        dispatcher:
          max_concurrent_runs: 5
          max_queue_depth: 100
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: no-error-config
            source: webhook
            action:
              prompt: "Handle event"
              allowed_tools: []
    """
    path = write_yaml(tmp_path, yaml_content)
    _, triggers = load_triggers(path)

    assert len(triggers) == 1
    assert triggers[0].on_error["action"] == "drop"


# ---------------------------------------------------------------------------
# Test 6: `dispatcher` section parses max_concurrent_runs and max_queue_depth
#          with defaults (5, 100)
# ---------------------------------------------------------------------------


def test_dispatcher_config_parses_explicit_values(tmp_path):
    path = write_yaml(tmp_path, VALID_WEBHOOK_YAML)
    dispatcher_cfg, _ = load_triggers(path)

    assert isinstance(dispatcher_cfg, DispatcherConfig)
    assert dispatcher_cfg.max_concurrent_runs == 3
    assert dispatcher_cfg.max_queue_depth == 50
    assert dispatcher_cfg.dispatch_target == "http://localhost:8010/dispatch"


def test_dispatcher_config_defaults_when_omitted(tmp_path):
    yaml_content = """\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: minimal-trigger
            source: webhook
            action:
              prompt: "Handle event"
              allowed_tools: []
    """
    path = write_yaml(tmp_path, yaml_content)
    dispatcher_cfg, _ = load_triggers(path)

    assert dispatcher_cfg.max_concurrent_runs == 5
    assert dispatcher_cfg.max_queue_depth == 100


# ---------------------------------------------------------------------------
# Test 7: Duplicate trigger names raise (would silently overwrite at registry)
# ---------------------------------------------------------------------------


def test_duplicate_trigger_names_raise(tmp_path):
    yaml_content = """\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: dup
            source: webhook
            action:
              prompt: "first"
              allowed_tools: []
          - name: dup
            source: webhook
            action:
              prompt: "second"
              allowed_tools: []
    """
    path = write_yaml(tmp_path, yaml_content)
    with pytest.raises(ValueError, match="Duplicate trigger name"):
        load_triggers(path)


# ---------------------------------------------------------------------------
# Test 8: Missing/empty `source` raises (was silently accepted as "")
# ---------------------------------------------------------------------------


def test_missing_source_raises_value_error(tmp_path):
    yaml_content = """\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: no-source
            action:
              prompt: "Handle event"
              allowed_tools: []
    """
    path = write_yaml(tmp_path, yaml_content)
    with pytest.raises(ValueError, match="source"):
        load_triggers(path)


# ---------------------------------------------------------------------------
# Test 9: Empty / non-mapping YAML raises a clean error (not AttributeError)
# ---------------------------------------------------------------------------


def test_empty_yaml_raises_clean_error(tmp_path):
    path = write_yaml(tmp_path, "")
    with pytest.raises(ValueError, match="empty"):
        load_triggers(path)


def test_non_mapping_yaml_raises_clean_error(tmp_path):
    path = write_yaml(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_triggers(path)


# ---------------------------------------------------------------------------
# Test 10: `action.surface` and `action.surface_prompt` parse when present
# ---------------------------------------------------------------------------


def test_surface_and_surface_prompt_are_parsed_when_present(tmp_path):
    yaml_content = """\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: with-surface
            source: webhook
            action:
              prompt: "Handle event"
              allowed_tools: []
              surface: control-room
              surface_prompt: "Extra guidance appended to the system prompt."
    """
    path = write_yaml(tmp_path, yaml_content)
    _, triggers = load_triggers(path)

    assert len(triggers) == 1
    t = triggers[0]
    assert t.surface == "control-room"
    assert t.surface_prompt == "Extra guidance appended to the system prompt."


# ---------------------------------------------------------------------------
# Test 11: `action.surface` / `action.surface_prompt` are optional and default
#          to None; an otherwise-identical trigger parses the same as before
# ---------------------------------------------------------------------------


def test_surface_fields_default_to_none_when_absent(tmp_path):
    path = write_yaml(tmp_path, VALID_WEBHOOK_YAML)
    _, triggers = load_triggers(path)

    assert len(triggers) == 1
    t = triggers[0]
    assert t.surface is None
    assert t.surface_prompt is None

    # No behavior change for the rest of the fields: identical to the
    # pre-existing assertions in test_valid_webhook_trigger_parses_all_fields.
    assert t.name == "beam-loss-alert"
    assert t.source == "webhook"
    assert t.on_error["action"] == "retry"
    assert t.on_error["max_retries"] == 3
    assert t.on_error["backoff_sec"] == 5.0
    assert t.action["prompt"] == "Investigate the beam loss event: {payload}"
    assert t.action["allowed_tools"] == ["get_pv", "archiver_query"]
    assert t.action.get("skill") == "beam-diagnostics"


# ---------------------------------------------------------------------------
# Test 12: non-string `action.surface` / `action.surface_prompt` raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_key", ["surface", "surface_prompt"])
@pytest.mark.parametrize("bad_value", [1, ["not", "a", "string"], {"nope": True}])
def test_non_string_surface_field_raises_value_error(tmp_path, bad_key, bad_value):
    yaml_content = f"""\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: bad-surface
            source: webhook
            action:
              prompt: "Handle event"
              allowed_tools: []
              {bad_key}: {bad_value!r}
    """
    path = write_yaml(tmp_path, yaml_content)
    with pytest.raises(ValueError, match=bad_key):
        load_triggers(path)


# ---------------------------------------------------------------------------
# Test 13: only one of `surface` / `surface_prompt` present is fine
# ---------------------------------------------------------------------------


def test_only_surface_present_is_fine(tmp_path):
    yaml_content = """\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: only-surface
            source: webhook
            action:
              prompt: "Handle event"
              allowed_tools: []
              surface: control-room
    """
    path = write_yaml(tmp_path, yaml_content)
    _, triggers = load_triggers(path)

    assert triggers[0].surface == "control-room"
    assert triggers[0].surface_prompt is None


def test_only_surface_prompt_present_is_fine(tmp_path):
    yaml_content = """\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: only-surface-prompt
            source: webhook
            action:
              prompt: "Handle event"
              allowed_tools: []
              surface_prompt: "Extra guidance."
    """
    path = write_yaml(tmp_path, yaml_content)
    _, triggers = load_triggers(path)

    assert triggers[0].surface is None
    assert triggers[0].surface_prompt == "Extra guidance."
