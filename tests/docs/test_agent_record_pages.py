"""The operator pages of the agent record against the code they describe.

An operator learns from these pages how long the record keeps an entry, what
``osprey health`` says about it, which host commands ask it directly, what the
archive holds and what ``osprey reset --purge-audit`` destroys. Each of those is
a number, a name or a scope the code owns, so this module holds the pages to
the code's own constants and commands rather than to a copy of them.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey.agent_record.config import DEFAULT_RETENTION_DAYS
from osprey.agent_record.service import REFUSALS_WINDOW, STOPPED_AFTER
from osprey.bridges.core.history import RECORD_INCOMPLETE, RECORD_UNAVAILABLE
from osprey.bridges.core.pipeline import HISTORY_RECORD_KEY
from osprey.cli.agent_record_cmd import agent_record_group
from osprey.health.core.agent_record import BACKLOG_BEHIND_S, EXPORT_LAG_ALLOWED_DAYS
from osprey.services.archive.run import SOURCE_TABLE
from osprey.utils.workspace import ARCHIVE_DIR_RELPATH, PRE_UPGRADE_AUDIT_RELPATH, SPOOL_DIR_RELPATH

_DOCS = Path(__file__).resolve().parents[2] / "docs/source"
_AGENT_RECORD = _DOCS / "how-to/health-and-monitoring/agent-record.rst"
_HEALTH_INDEX = _DOCS / "how-to/health-and-monitoring/index.rst"
_AUDIT_TRAIL = _DOCS / "reference/contracts/audit-trail.rst"
_BRIDGE_DISPATCH = _DOCS / "reference/contracts/bridge-dispatch.rst"
_CLI = _DOCS / "reference/cli.rst"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _flat(path: Path) -> str:
    """The page with every run of whitespace collapsed to one space."""
    return " ".join(_text(path).split())


def _section(path: Path, title: str) -> str:
    """The body of the section headed *title*, up to the next heading of any level."""
    lines = _text(path).splitlines()
    underline = re.compile(r"^([=\-~^\"'`#*+])\1{2,}$")
    for i, line in enumerate(lines):
        if line.strip() == title and i + 1 < len(lines) and underline.match(lines[i + 1].strip()):
            body: list[str] = []
            for j in range(i + 2, len(lines)):
                nxt = lines[j + 1].strip() if j + 1 < len(lines) else ""
                if lines[j].strip() and underline.match(nxt) and len(nxt) >= len(lines[j].strip()):
                    break
                body.append(lines[j])
            return " ".join(" ".join(body).split())
    raise AssertionError(f"{path.name} has no section headed {title!r}")


def _minutes(delta) -> int:
    return int(delta.total_seconds() // 60)


# --- agent-record.rst: retention -------------------------------------------------


def test_agent_record_page_states_the_default_retention_window() -> None:
    page = _flat(_AGENT_RECORD)
    assert f"default {DEFAULT_RETENTION_DAYS}" in page or (
        f"``{DEFAULT_RETENTION_DAYS}``" in page
    ), "agent-record.rst must state the default of agent_record.retention_days"
    assert "agent_record.retention_days" in page


def test_agent_record_page_says_retention_waits_for_the_archive() -> None:
    section = _section(_AGENT_RECORD, "How long the record keeps an entry")
    assert "agent_record.retention_days" in section
    assert "archive holds" in section, "retention removes only days the archive holds"
    assert "removes nothing" in section, "an export outage longer than the window removes nothing"
    assert "arrival day" in section, "the window counts arrival days"


def test_agent_record_page_explains_the_retention_floor() -> None:
    section = _section(_AGENT_RECORD, "How long the record keeps an entry")
    assert "stays out" in section, "a removed day must stay out of the record"
    assert "transcript" in section, "the floor is what keeps a rewritten transcript from returning"


# --- agent-record.rst: the health row ----------------------------------------------


def test_agent_record_page_describes_the_health_row() -> None:
    section = _section(_AGENT_RECORD, "Check it")
    for row in (
        "agent_record_backend",
        "agent_record_source.<identity>",
        "agent_record_transcripts.<identity>",
        "agent_record_export",
    ):
        assert f"``{row}``" in section, f"the health section must name the {row} row"
    for word in ("stopped", "idle", "backlog", "refused", "export"):
        assert word in section, f"the health section must say what {word!r} means"


def test_agent_record_page_health_numbers_match_the_code() -> None:
    section = _section(_AGENT_RECORD, "Check it")
    stopped = _minutes(STOPPED_AFTER)
    assert f"{stopped} minutes" in section, f"stopped is {stopped} minutes without a commit"
    backlog = int(BACKLOG_BEHIND_S)
    assert f"{backlog // 60} minutes" in section or f"{backlog} seconds" in section
    refusals_h = int(REFUSALS_WINDOW.total_seconds() // 3600)
    assert f"{refusals_h} hours" in section
    lag = {1: "one closed day", 2: "two closed days"}[EXPORT_LAG_ALLOWED_DAYS]
    assert f"more than {lag}" in section, "the export row warns past the allowed lag"


def test_agent_record_page_names_the_host_commands() -> None:
    section = _section(_AGENT_RECORD, "Check it")
    for verb in ("health", "export"):
        assert verb in agent_record_group.commands, f"osprey agent-record {verb} is not registered"
        assert f"osprey agent-record {verb}" in section


# --- agent-record.rst: what the archive holds -----------------------------------------


def test_archive_holds_only_what_the_archive_service_copies() -> None:
    table = _section(_AGENT_RECORD, "Where the record lives")
    assert "transcripts" in table
    assert set(SOURCE_TABLE) == {"terminal_agent_data", "dispatch", "bluesky"}, (
        "the archive service's kinds moved: revisit the table on agent-record.rst"
    )
    permissions = _section(_AGENT_RECORD, "Permissions")
    assert "copies transcripts" not in permissions, "the archive service copies no transcripts"
    assert f"``{ARCHIVE_DIR_RELPATH}/``" in permissions


def test_pre_upgrade_files_are_on_disk_unarchived_and_purge_only() -> None:
    section = _section(_AGENT_RECORD, "Audit files from before the record")
    assert f"``{PRE_UPGRADE_AUDIT_RELPATH}/``" in section
    assert "not archived" in section
    assert "osprey reset --purge-audit" in section
    assert "only" in section


def test_health_index_card_says_record_and_archive() -> None:
    page = _flat(_HEALTH_INDEX)
    assert "Every store" not in page, "the record is one store; the card must not list several"
    assert "archive" in page


# --- audit-trail.rst ---------------------------------------------------------------------


def test_audit_trail_is_record_plus_archive() -> None:
    page = _flat(_AUDIT_TRAIL)
    assert "``audit`` entry" in page
    assert "``surface``" in page
    assert f"``{SPOOL_DIR_RELPATH}/<identity>/``" in page
    assert "agent_record/entries.parquet" in page
    assert "agent_record.retention_days" in page
    assert "Nothing rotates or expires it" not in page
    assert "audit.tool_call.enabled" not in page, "audit.tool_call is a retired key"
    assert "OSPREY_AUDIT_WRITER" not in page
    # The pre-upgrade tree is named only as what reset keeps and purges.
    for line in _text(_AUDIT_TRAIL).splitlines():
        if f"{PRE_UPGRADE_AUDIT_RELPATH}/<" in line:
            raise AssertionError(
                f"audit-trail.rst still files decisions under the old tree: {line}"
            )


def test_audit_trail_keeps_its_anchors() -> None:
    text = _text(_AUDIT_TRAIL)
    for anchor in (
        "reference-audit-trail",
        "audit-trail-files",
        "audit-trail-record",
        "audit-trail-identity-keys",
        "audit-trail-attribution",
        "audit-trail-login-refusals",
        "audit-trail-tool-call",
        "audit-trail-identity-ladder",
    ):
        assert f".. _{anchor}:" in text, f"other pages link to {anchor}"


# --- bridge-dispatch.rst -----------------------------------------------------------------


def test_bridge_dispatch_documents_the_conversation_record_key() -> None:
    section = _section(_BRIDGE_DISPATCH, f"``{HISTORY_RECORD_KEY}``")
    for value in (RECORD_UNAVAILABLE, RECORD_INCOMPLETE):
        assert f'``"{value}"``' in section, f"{HISTORY_RECORD_KEY} value {value!r} is undocumented"
    assert "conversation_so_far" in section
    assert "Absent" in section or "absent" in section


def test_bridge_dispatch_does_not_claim_a_bridge_copy() -> None:
    assert "The bridge's own copy keeps every answer" not in _flat(_BRIDGE_DISPATCH)


# --- cli.rst: reset ----------------------------------------------------------------------


def test_cli_reset_section_matches_the_purge_scope() -> None:
    section = _section(_CLI, "osprey reset")
    for path in (
        f"{PRE_UPGRADE_AUDIT_RELPATH}/",
        "var/agent_record/",
        f"{SPOOL_DIR_RELPATH}/",
        f"{ARCHIVE_DIR_RELPATH}/<day>/agent_record/",
        "var/state/",
    ):
        assert f"``{path}``" in section, f"the reset section must name {path}"
    assert "agent_record_clickhouse_data" in section
    assert "Destroy ``var/audit/`` as well. It is kept by default." not in section
    assert "refuses" in section and "host" in section, "reset refuses while a host session runs"
