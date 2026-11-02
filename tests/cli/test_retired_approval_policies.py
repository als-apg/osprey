"""A per-tool policy for a governed tool is dropped at the single parse point.

The write switch's slider (off / asked / allowed) decides whether a governed
write asks, so ``approval.tools.<tool>`` for one of those tools is read for no
answer. For one release window the parse point drops it with one warning per
document naming the slider and the three consequences; every other tool keeps
its policy. The governed set is the framework's, plus the tools the same
document's own ``mcp_servers`` hand to both write hooks.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest
import yaml

from osprey.cli import build_profile_document as doc
from osprey.cli.build_profile_document import (
    _parse_profile_document,
    governed_policy_tools,
    normalize_retired_approval_policies,
)
from osprey.cli.build_profile_resolve import resolve_build_document
from osprey.registry.mcp import governed_write_tools, resolve_servers
from osprey.utils.config_writer import config_update_fields

_LOGGER_NAME = doc.__name__

#: What every warning must name: the slider, and each of the three consequences.
_SLIDER = "slider"
_CONSEQUENCES = ("`skip`", "`execute: always`", "`approval.enabled: false`")


@pytest.fixture(autouse=True)
def _fresh_state():
    doc._WARNED_RETIRED_KEYS.clear()
    yield
    doc._WARNED_RETIRED_KEYS.clear()


def _parse(text: str, source: str = "profile.yml"):
    return _parse_profile_document(textwrap.dedent(text), source)


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER_NAME and r.levelno == logging.WARNING
    ]


def _policy_warnings(caplog) -> list[str]:
    return [line for line in _warnings(caplog) if "approval.tools." in line]


def _rendered_approval(overlay: str, scratch: Path) -> dict:
    """The ``approval`` section the build writes for a profile extending the preset.

    *overlay* is the rest of that profile file, read through the same parse
    point as every profile document.
    """
    path = scratch / "profile.yml"
    (scratch / "data").mkdir(exist_ok=True)
    path.write_text(
        "name: Child\nextends: control-assistant\ndata: data\n" + textwrap.dedent(overlay),
        encoding="utf-8",
    )
    loaded = resolve_build_document(path, None, (), ())
    target = scratch / "rendered-config.yml"
    target.write_text("{}\n", encoding="utf-8")
    config_update_fields(target, loaded.profile.config)
    section = (yaml.safe_load(target.read_text(encoding="utf-8")) or {}).get("approval")
    return section if isinstance(section, dict) else {}


# ── The set ─────────────────────────────────────────────────────────────────


def test_the_framework_set_is_the_registry_producers() -> None:
    servers = resolve_servers(
        {"servers": {name: {"enabled": True} for name in ("bluesky", "phoebus", "ariel")}},
        {"event_dispatcher_wired": True, "phoebus_agent_access": "read_write"},
    )
    shorts = {tool.rsplit("__", 1)[-1] for tool in governed_write_tools(servers)}
    assert shorts == set(governed_policy_tools({}))
    assert {"channel_write", "execute", "execute_file", "pyaml_measure"} <= shorts
    assert {"queue_add", "queue_start"} <= shorts
    for kept in ("entry_create", "setup_patch", "manual_fire", "phoebus_drive", "queue_stop"):
        assert kept not in shorts


# ── The gate ────────────────────────────────────────────────────────────────


def test_a_governed_policy_builds_with_one_warning_and_renders_no_key(
    caplog, tmp_path: Path
) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    approval = _rendered_approval("config:\n  approval.tools.channel_write: always\n", tmp_path)

    lines = _policy_warnings(caplog)
    assert len(lines) == 1
    assert "`approval.tools.channel_write`" in lines[0]
    assert _SLIDER in lines[0]
    for consequence in _CONSEQUENCES:
        assert consequence in lines[0]
    assert "osprey init --revalidate" in lines[0]
    tools = approval.get("tools") or {}
    assert "channel_write" not in tools
    assert tools.get("entry_create") == "always"


def test_both_shipped_presets_build_silently(caplog) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    for preset in ("hello-world", "control-assistant"):
        resolve_build_document(None, preset, (), ())
    assert _policy_warnings(caplog) == []


# ── Spellings ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "config:\n  approval.tools.channel_write: always\n  approval.tools.entry_create: always\n",
        "config:\n  approval.tools:\n    channel_write: always\n    entry_create: always\n",
        "config:\n  approval:\n    tools:\n      channel_write: always\n      entry_create: always\n",
        "config:\n  approval:\n    tools.channel_write: always\n    tools.entry_create: always\n",
    ],
    ids=["dotted", "mixed", "nested", "inner-dotted"],
)
def test_every_spelling_is_dropped_and_its_neighbour_kept(text: str, caplog) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    raw = _parse(text)

    flat = _flat(raw["config"])
    assert "approval.tools.channel_write" not in flat
    assert flat["approval.tools.entry_create"] == "always"
    assert len(_policy_warnings(caplog)) == 1


def test_a_mapping_the_removal_empties_goes_with_it() -> None:
    raw = _parse(
        """\
        config:
          approval:
            enabled: true
            tools:
              channel_write: skip
              execute: selective
        """
    )
    assert raw["config"] == {"approval": {"enabled": True}}


def test_several_tools_in_one_document_warn_once_naming_each(caplog) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    _parse(
        """\
        config:
          approval.tools.channel_write: always
          approval.tools.execute: selective
          approval.tools.queue_add: skip
        """
    )
    lines = _policy_warnings(caplog)
    assert len(lines) == 1
    for tool in ("channel_write", "execute", "queue_add"):
        assert f"`approval.tools.{tool}`" in lines[0]


def test_a_repeated_parse_of_one_source_warns_once(caplog) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    text = "config:\n  approval.tools.execute: always\n"
    _parse(text)
    _parse(text)
    assert len(_policy_warnings(caplog)) == 1
    _parse(text, source="other.yml")
    assert len(_policy_warnings(caplog)) == 2


def test_a_document_with_no_governed_policy_is_untouched_and_quiet(caplog) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    text = """\
        config:
          approval.enabled: false
          approval.default_policy: always
          approval.tools.entry_create: always
          approval.tools.setup_patch: always
          approval.tools.manual_fire: always
          approval.tools.phoebus_drive: always
        """
    raw = _parse(text)
    assert raw == yaml.safe_load(textwrap.dedent(text))
    assert _policy_warnings(caplog) == []


def test_a_tier_file_reads_the_framework_set(caplog, tmp_path: Path) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    tier = tmp_path / "tiers" / "operators.yml"
    tier.parent.mkdir()
    tier.write_text(
        "config:\n  approval.tools.pyaml_measure: skip\n  approval.tools.entry_create: skip\n",
        encoding="utf-8",
    )
    raw = doc._read_profile_document(tier)
    assert raw["config"] == {"approval.tools.entry_create": "skip"}
    assert len(_policy_warnings(caplog)) == 1


# ── A facility's own write tools ────────────────────────────────────────────

_FACILITY = """\
    mcp_servers:
      plc:
        command: python3
        args: [-m, plc]
{extra}
        permissions:
          allow: [mcp__plc__read_input, set_output, mcp__plc__.*]
    config:
      approval.tools.set_output: always
      approval.tools.read_input: skip
    """


def _facility(extra: str) -> str:
    return _FACILITY.format(extra=textwrap.indent(textwrap.dedent(extra), " " * 8).rstrip())


def test_a_declared_write_tool_has_both_hooks_and_its_policy_is_dropped(caplog) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    raw = _parse(_facility("writes: [set_output]\n"))

    assert raw["config"] == {"approval.tools.read_input": "skip"}
    lines = _policy_warnings(caplog)
    assert len(lines) == 1
    assert "`approval.tools.set_output`" in lines[0]
    assert _SLIDER in lines[0]


def test_a_whole_server_pair_governs_every_tool_it_permits(caplog) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    raw = _parse(_facility("hooks:\n  pre_tool_use: [writes_check, approval]\n"))

    assert raw["config"] == {}
    assert len(_policy_warnings(caplog)) == 1


@pytest.mark.parametrize("presets", [["writes_check"], ["approval"], ["limits"]])
def test_a_server_missing_either_hook_keeps_its_policies(presets: list[str], caplog) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    raw = _parse(_facility(f"hooks:\n  pre_tool_use: {presets}\n"))

    assert raw["config"] == {
        "approval.tools.set_output": "always",
        "approval.tools.read_input": "skip",
    }
    assert _policy_warnings(caplog) == []


def test_another_documents_server_does_not_govern_this_one(caplog) -> None:
    """Only the same raw document's servers count; a tier declares none."""
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    raw = _parse("config:\n  approval.tools.set_output: always\n")
    assert raw["config"] == {"approval.tools.set_output": "always"}
    assert _policy_warnings(caplog) == []


def test_the_normalizer_alone_leaves_a_document_without_config_alone() -> None:
    raw = {"name": "x", "mcp_servers": {"plc": {"writes": ["set_output"]}}}
    assert normalize_retired_approval_policies(raw, "x.yml") == {
        "name": "x",
        "mcp_servers": {"plc": {"writes": ["set_output"]}},
    }


def test_a_facility_profile_builds_with_its_own_policy_dropped(caplog, tmp_path: Path) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    approval = _rendered_approval(
        """\
        mcp_servers:
          plc:
            command: python3
            args: [-m, plc]
            writes: [set_output]
        config:
          approval.tools.set_output: always
          approval.tools.read_input: skip
        """,
        tmp_path,
    )

    lines = _policy_warnings(caplog)
    assert len(lines) == 1
    assert "`approval.tools.set_output`" in lines[0]
    tools = approval.get("tools") or {}
    assert "set_output" not in tools
    assert tools.get("read_input") == "skip"


def _flat(mapping: dict, prefix: str = "") -> dict:
    out: dict = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(_flat(value, f"{dotted}."))
        else:
            out[dotted] = value
    return out
