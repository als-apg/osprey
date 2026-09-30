"""Unit tests for TriggerConfig dataclass and load_triggers() function."""

import textwrap
from datetime import time

import pytest

from osprey.dispatch.clock_schedule import ClockSchedule
from osprey.dispatch.trigger_config import (
    DEFAULT_MAX_CONCURRENT_RUNS,
    DEFAULT_MAX_QUEUE_DEPTH,
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
#          with the package defaults
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

    assert dispatcher_cfg.max_concurrent_runs == DEFAULT_MAX_CONCURRENT_RUNS
    assert dispatcher_cfg.max_queue_depth == DEFAULT_MAX_QUEUE_DEPTH


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
#          to None; an otherwise-identical trigger parses identically
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


# ---------------------------------------------------------------------------
# Test 14: `action.max_turns` is a typed field, refused at load time
# ---------------------------------------------------------------------------


def test_max_turns_is_parsed_when_present(tmp_path):
    yaml_content = """\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: with-ceiling
            source: webhook
            action:
              prompt: "Handle event"
              allowed_tools: []
              max_turns: 5
    """
    path = write_yaml(tmp_path, yaml_content)
    _, triggers = load_triggers(path)

    assert triggers[0].max_turns == 5


def test_max_turns_defaults_to_none_when_absent(tmp_path):
    path = write_yaml(tmp_path, VALID_WEBHOOK_YAML)
    _, triggers = load_triggers(path)

    assert triggers[0].max_turns is None


@pytest.mark.parametrize("bad_value", ["five", 0, -1, 2.5, True, ["3"]])
def test_an_unusable_max_turns_is_refused_when_the_file_is_loaded(tmp_path, bad_value):
    """A turn ceiling the worker would reject is a typo the author must see.

    The worker refuses it with a 422 at dispatch time — which is the moment an
    event fires, long after the file was written — so the file is where it is
    caught.
    """
    yaml_content = f"""\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: bad-ceiling
            source: webhook
            action:
              prompt: "Handle event"
              allowed_tools: []
              max_turns: {bad_value!r}
    """
    path = write_yaml(tmp_path, yaml_content)
    with pytest.raises(ValueError, match="max_turns"):
        load_triggers(path)


# ---------------------------------------------------------------------------
# A trigger may not name the dispatcher's own tools
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool",
    [
        "mcp__event_dispatcher__manual_fire",
        "mcp__event_dispatcher__list_triggers",
        "mcp__event_dispatcher__",
    ],
)
def test_a_trigger_naming_a_dispatcher_tool_is_refused_at_load(tmp_path, tool):
    """A dispatch job may not fire dispatch jobs, so it holds no dispatcher tool.

    The refusal names the trigger and the tool so the author can find both in
    the file in front of them.
    """
    yaml_content = f"""\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: recursive-trigger
            source: webhook
            action:
              prompt: "Handle event"
              allowed_tools:
                - get_pv
                - {tool}
    """
    path = write_yaml(tmp_path, yaml_content)

    with pytest.raises(ValueError) as excinfo:
        load_triggers(path)

    message = str(excinfo.value)
    assert "recursive-trigger" in message
    assert tool in message


def test_a_trigger_naming_a_dispatcher_tool_as_a_bare_string_is_refused(tmp_path):
    """``allowed_tools`` written as one scalar is still read as a tool name."""
    yaml_content = """\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: scalar-trigger
            source: webhook
            action:
              prompt: "Handle event"
              allowed_tools: mcp__event_dispatcher__manual_fire
    """
    path = write_yaml(tmp_path, yaml_content)

    with pytest.raises(ValueError) as excinfo:
        load_triggers(path)

    message = str(excinfo.value)
    assert "scalar-trigger" in message
    assert "mcp__event_dispatcher__manual_fire" in message


def test_a_tool_that_merely_mentions_the_dispatcher_elsewhere_is_allowed(tmp_path):
    """Only the server prefix is refused, not any name containing it."""
    yaml_content = """\
        dispatcher:
          dispatch_target: http://localhost:8010/dispatch

        triggers:
          - name: fine-trigger
            source: webhook
            action:
              prompt: "Handle event"
              allowed_tools:
                - mcp__controls__event_dispatcher_status
                - get_pv
    """
    path = write_yaml(tmp_path, yaml_content)

    _, triggers = load_triggers(path)

    assert triggers[0].action["allowed_tools"] == [
        "mcp__controls__event_dispatcher_status",
        "get_pv",
    ]


# ---------------------------------------------------------------------------
# Clock schedule of a cron trigger
# ---------------------------------------------------------------------------


def test_a_clock_schedule_is_parsed_onto_the_trigger(tmp_path):
    path = write_yaml(
        tmp_path,
        """\
        triggers:
          - name: morning-report
            source: cron
            source_config:
              at: ["07:45", "17:00"]
              days: [mon, tue, wed, thu, fri]
            action:
              prompt: "Summarise the night"
        """,
    )
    _, triggers = load_triggers(path)

    assert triggers[0].schedule == ClockSchedule(
        times=(time(7, 45), time(17, 0)), days=frozenset({0, 1, 2, 3, 4})
    )


def test_an_unreadable_clock_schedule_is_refused_when_the_file_is_loaded(tmp_path):
    path = write_yaml(
        tmp_path,
        """\
        triggers:
          - name: evening-report
            source: cron
            source_config:
              at: [17:00]
            action:
              prompt: "Summarise the day"
        """,
    )
    with pytest.raises(ValueError, match="evening-report"):
        load_triggers(path)


@pytest.mark.parametrize("interval", [3600, 0])
def test_an_interval_trigger_loads_without_a_schedule(tmp_path, interval):
    path = write_yaml(
        tmp_path,
        f"""\
        triggers:
          - name: hourly
            source: cron
            source_config:
              interval_sec: {interval}
            action:
              prompt: "tick"
        """,
    )
    _, triggers = load_triggers(path)

    assert triggers[0].schedule is None


def test_a_non_cron_trigger_is_not_read_for_a_schedule(tmp_path):
    path = write_yaml(
        tmp_path,
        """\
        triggers:
          - name: hook
            source: webhook
            source_config:
              at: 5
            action:
              prompt: "handle {payload}"
        """,
    )
    _, triggers = load_triggers(path)

    assert triggers[0].schedule is None


@pytest.mark.parametrize(
    ("field_name", "line"),
    [
        ("action", "action: do it"),
        ("on_error", "on_error: retry"),
        ("source_config", 'source_config: "interval_sec: 60"'),
        ("source_config", "source_config: [60]"),
    ],
    ids=["action", "on_error", "source_config-string", "source_config-list"],
)
def test_a_trigger_part_that_is_not_a_mapping_is_refused_by_name(tmp_path, field_name, line):
    parts = {
        "action": "action: {prompt: tick}",
        "on_error": "on_error: {action: drop}",
        "source_config": "source_config: {interval_sec: 60}",
    }
    parts[field_name] = line
    body = "\n".join(f"    {p}" for p in parts.values())
    path = write_yaml(
        tmp_path,
        f"triggers:\n  - name: beam-loss\n    source: cron\n{body}\n",
    )

    with pytest.raises(ValueError, match=f"'beam-loss' field '{field_name}' must be a mapping"):
        load_triggers(path)


def test_a_trigger_entry_that_is_not_a_mapping_is_refused_by_index(tmp_path):
    path = write_yaml(tmp_path, "triggers: [beam-loss]\n")

    with pytest.raises(ValueError, match="index 0"):
        load_triggers(path)


@pytest.mark.parametrize(
    ("field_name", "content"),
    [
        ("dispatcher", "dispatcher: x\ntriggers: []\n"),
        ("triggers", "triggers: {a: 1}\n"),
        ("triggers", "triggers: x\n"),
    ],
    ids=["dispatcher", "triggers-mapping", "triggers-string"],
)
def test_a_misshapen_dispatcher_or_triggers_block_is_refused(tmp_path, field_name, content):
    path = write_yaml(tmp_path, content)

    with pytest.raises(ValueError, match=f"field '{field_name}' must be"):
        load_triggers(path)


def test_a_blank_source_config_is_empty(tmp_path):
    path = write_yaml(
        tmp_path,
        """\
        triggers:
          - name: tick
            source: cron
            source_config:
            action:
              prompt: "tick"
          - name: watch
            source: epics_ca
            source_config:
            action:
              prompt: "watch"
        """,
    )
    _, triggers = load_triggers(path)

    assert [t.source_config for t in triggers] == [{}, {}]


def _surface_tools_yaml(line: str) -> str:
    return (
        "triggers:\n"
        "  - name: deploy-bot\n"
        "    source: webhook\n"
        "    action:\n"
        "      prompt: handle it\n"
        "      allowed_tools: [read_pv]\n"
        f"{line}"
    )


def test_surface_tools_is_parsed_when_present(tmp_path):
    path = write_yaml(
        tmp_path,
        _surface_tools_yaml("      surface_tools: [read_pv, mcp__osprey_workspace__list_files]\n"),
    )
    _, triggers = load_triggers(path)

    assert triggers[0].surface_tools == ["read_pv", "mcp__osprey_workspace__list_files"]


@pytest.mark.parametrize("line", ["", "      surface_tools:\n"], ids=["absent", "blank"])
def test_surface_tools_is_none_when_absent_or_blank(tmp_path, line):
    path = write_yaml(tmp_path, _surface_tools_yaml(line))
    _, triggers = load_triggers(path)

    assert triggers[0].surface_tools is None


def test_an_empty_surface_tools_loads_and_narrows_nothing(tmp_path):
    path = write_yaml(tmp_path, _surface_tools_yaml("      surface_tools: []\n"))
    _, triggers = load_triggers(path)

    assert triggers[0].surface_tools == []


@pytest.mark.parametrize(
    "value",
    ["get_pv", "5", "{a: 1}", "[get_pv, 1]"],
    ids=["string", "number", "mapping", "list-with-non-string"],
)
def test_a_surface_tools_that_is_not_a_list_of_strings_is_refused(tmp_path, value):
    path = write_yaml(tmp_path, _surface_tools_yaml(f"      surface_tools: {value}\n"))

    with pytest.raises(ValueError, match="'deploy-bot' field 'action.surface_tools'"):
        load_triggers(path)
