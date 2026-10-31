"""The retired telemetry stores are named only where a reason is written down.

OSPREY no longer deploys OpenObserve, nor the ClickHouse telemetry store, its
OTLP gateway and its reader accounts that replaced it for a while: one agent
record holds every fact, and the per-identity ``var/audit`` ledger it absorbed
is gone with them. Each retired vocabulary is a :class:`WordGroup` — a pattern
and the files allowed to match it, each with its reason (:data:`GROUPS`).
Everything else that names a retired store is a leftover.

The reasons a file may keep a retired word are few:

* the code that refuses a retired key, and the tests that prove the refusal;
* the config-key manifest, whose ``deleted:`` rows keep a key from coming back;
* a docs page that lists removed keys;
* the guards themselves, which have to spell what they guard against;
* for ``var/audit`` only, the pre-upgrade tree a deployment may still hold:
  the reset code that keeps and purges it, its tests and reset docs, and the
  frozen render that predates the record.

``OSPREY_AUDIT_DIR`` has no reason at all: only the guard names it.

Two subset assertions per group rather than one equality:

* ``found ⊆ ALLOWED`` — nothing names a retired store without a reason.
* ``ALLOWED ⊆ found`` — no reason outlives the words it excuses.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where the words are looked for. Tracked files only: a compiled cache or a
#: local render is not the tree.
SEARCHED = ("src", "packages", "plugins", "scripts", "tests", "docs", ".github")

GUARD = "tests/test_retired_store_words.py"
GUARD_REASON = "holds the patterns and every ALLOWED list"


@dataclass(frozen=True)
class WordGroup:
    """One retired vocabulary: what it matches, and where it may appear."""

    name: str
    #: Matched case-insensitively.
    pattern: str
    allowed: dict[str, str]
    #: Spellings the pattern must catch, and near-misses it must not.
    matches: tuple[str, ...]
    non_matches: tuple[str, ...]


#: OpenObserve: its name, its variables, the two variables the agent read its
#: address from, and the config and code names of its organisation setting.
OPENOBSERVE = WordGroup(
    name="openobserve",
    pattern=(
        r"openobserve|ZO_[A-Z_]+|OSPREY_TELEMETRY_(URL|ORG)|telemetry_org|store_org"
        r"|telemetryUrlForRun"
    ),
    allowed={
        GUARD: GUARD_REASON,
        "src/osprey/profiles/config_key_manifest.yml": (
            "the retired keys are recorded under deleted: with their orphan_sites, so "
            "none of them can come back"
        ),
        "tests/cli/test_service_existence.py": (
            "proves a service named openobserve is refused by the ordinary unknown-service check"
        ),
        "tests/deployment/web_terminals/golden/pre_audit_roles/docker-compose.web.yml": (
            "a frozen render that predates the audit-and-roles feature, compared line "
            "by line and never regenerated"
        ),
        "tests/deployment/web_terminals/test_auth_off_baseline_pin.py": (
            "names the two carrier lines the frozen baseline has and today's render must not"
        ),
    },
    matches=(
        "OpenObserve",
        "ZO_ROOT_USER_PASSWORD",
        "ZO_INGEST_SA_TOKEN",
        "OSPREY_TELEMETRY_URL",
        "OSPREY_TELEMETRY_ORG",
        "telemetry_org",
        "store_org",
        "telemetryUrlForRun",
    ),
    non_matches=("telemetry_store", "ZOOM", "OSPREY_TELEMETRY_DIGEST_STORE"),
)

#: The ClickHouse telemetry store, its gateway, its tables and reader accounts,
#: the history page's old key names and wording, and the ``var/audit`` ledger's
#: provisioning. Whole words only; the bare word ``clickhouse`` is the record's
#: own backend and is not guarded.
STORE = WordGroup(
    name="store",
    pattern=(
        r"\b(telemetry[_-]store|telemetry[_-]gateway|otel_logs|audit_ledger"
        r"|reader_port|reader_origin|READER_TABLES_VAR_PREFIX|reader_tables"
        r"|ensure_audit_dir)\b"
        r"|/telemetry/play\b|(?<![A-Za-z_.-])/play\b"
        r"|\breaders['’] origin"
    ),
    allowed={
        GUARD: GUARD_REASON,
        "src/osprey/agent_record/config.py": (
            "the refusal code: RETIRED_CONFIG_KEYS and RETIRED_DEPLOYED_SERVICES name each "
            "retired key and service so a profile that spells one is refused"
        ),
        "tests/cli/test_retired_keys.py": "proves every retired key and service is refused",
        "tests/e2e/test_agent_record_e2e.py": (
            "FR6's profile: carries a retired key so the replay e2e proves the refusal, "
            "then the edit, build and up it directs"
        ),
        "src/osprey/profiles/config_key_manifest.yml": (
            "the retired keys are recorded under deleted:, so none of them can come back"
        ),
        "tests/agent_record/test_writer_inventory.py": (
            "a guard: lists ensure_audit_dir among the writers no production module may name"
        ),
        "tests/deployment/web_terminals/test_lint.py": (
            "a guard: proves no history lint message speaks of the readers' origin"
        ),
    },
    matches=(
        "services.telemetry_store",
        "telemetry-store",
        "TELEMETRY_GATEWAY",
        "telemetry-gateway",
        "otel_logs",
        "audit_ledger",
        "modules.web_terminals.reader_port",
        "reader_origin",
        "READER_ORIGIN = 1",
        "READER_TABLES_VAR_PREFIX",
        "reader_tables",
        "ensure_audit_dir()",
        "GET /telemetry/play",
        'location = "/play"',
        "/play/",
        "http://127.0.0.1:8123/play",
        "the readers' origin",
        "the readers’ origin",
    ),
    non_matches=(
        "clickhouse",
        "telemetry_storage",
        "test_audit_ledgers_are_not_copied",
        "history_origin",
        "history_port",
        "/playwright",
        "replay",
        "/api/replay",
        "a reader's origin",
        "OSPREY_TELEMETRY_DIGEST_STORE",
    ),
)

#: The per-identity ledger directory. It survives only as the pre-upgrade tree
#: (FR7): kept by ``osprey reset``, removed by ``--purge-audit``, protected from
#: agent code, never written.
VAR_AUDIT = WordGroup(
    name="var-audit",
    pattern=r"\bvar/audit\b",
    allowed={
        GUARD: GUARD_REASON,
        "tests/agent_record/test_writer_inventory.py": (
            "a guard: no writer leaves a file under the pre-upgrade tree"
        ),
        "src/osprey/deployment/reset.py": (
            "FR7: reset keeps the pre-upgrade tree and --purge-audit removes it"
        ),
        "src/osprey/cli/reset_cmd.py": "FR7: the reset verb's help names what it keeps and purges",
        "tests/deployment/test_reset_scoping.py": (
            "FR7: proves reset keeps the pre-upgrade tree and the purge removes it"
        ),
        "tests/deployment/test_reset_keeps_record_volume.py": (
            "FR7: proves reset keeps the pre-upgrade tree beside the record"
        ),
        "tests/cli/test_reset_verb.py": "FR7: pins the reset verb's help text",
        "tests/docs/test_agent_record_pages.py": (
            "FR7: pins what the reset docs say about the pre-upgrade tree"
        ),
        "docs/source/reference/cli.rst": "FR7 reset docs: what osprey reset keeps and purges",
        "docs/source/how-to/health-and-monitoring/agent-record.rst": (
            "FR7 reset docs: the pre-upgrade files stay on disk, unarchived, until --purge-audit"
        ),
        "docs/source/reference/contracts/audit-trail.rst": (
            "FR7 reset docs: the audit files written before the record, and how reset treats them"
        ),
        "tests/deployment/web_terminals/golden/pre_audit_roles/docker-compose.web.yml": (
            "a frozen render of a pre-upgrade deployment, compared line by line and never "
            "regenerated"
        ),
        "tests/deployment/web_terminals/test_auth_off_baseline_pin.py": (
            "names the pre-upgrade bind lines the frozen baseline has and today's render must not"
        ),
    },
    matches=("var/audit", "./var/audit/alice", "VAR/AUDIT/"),
    non_matches=("var/auditor", "var/agent_record", "myvar/audit"),
)

#: The variable that once named the ledger directory to a container.
AUDIT_DIR_VARIABLE = WordGroup(
    name="audit-dir-variable",
    pattern=r"\bOSPREY_AUDIT_DIR\b",
    allowed={GUARD: GUARD_REASON},
    matches=("OSPREY_AUDIT_DIR", "${OSPREY_AUDIT_DIR}"),
    non_matches=("OSPREY_AUDIT_IDENTITY", "OSPREY_AUDIT_DIRS"),
)

GROUPS: tuple[WordGroup, ...] = (OPENOBSERVE, STORE, VAR_AUDIT, AUDIT_DIR_VARIABLE)


@cache
def _tracked_texts() -> tuple[tuple[str, str], ...]:
    """Every text file git knows under :data:`SEARCHED`, tracked or new and not ignored."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *SEARCHED],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        timeout=60,
    )
    texts: list[tuple[str, str]] = []
    for name in listed.stdout.decode("utf-8").split("\0"):
        path = REPO_ROOT / name
        if not name or not path.is_file():
            continue  # deleted in the working tree
        try:
            texts.append((name, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue  # a binary file names nothing
    return tuple(texts)


def files_naming(group: WordGroup) -> set[str]:
    """The searched files whose text matches the group's pattern."""
    pattern = re.compile(group.pattern, re.IGNORECASE)
    return {name for name, text in _tracked_texts() if pattern.search(text)}


_BY_NAME = pytest.mark.parametrize("group", GROUPS, ids=lambda group: group.name)


@_BY_NAME
def test_nothing_names_a_retired_store_without_a_reason(group: WordGroup) -> None:
    strays = sorted(files_naming(group) - set(group.allowed))
    assert strays == [], (
        f"file(s) match the retired {group.name} words with no entry in its ALLOWED — "
        "remove the words, or add the file with its reason:\n  " + "\n  ".join(strays)
    )


@_BY_NAME
def test_every_reason_still_has_its_words(group: WordGroup) -> None:
    stale = sorted(set(group.allowed) - files_naming(group))
    assert stale == [], (
        f"{group.name} ALLOWED entries whose file no longer matches — drop the entry:\n  "
        + "\n  ".join(stale)
    )


@_BY_NAME
def test_every_reason_is_written_down(group: WordGroup) -> None:
    assert group.allowed
    assert all(reason.strip() for reason in group.allowed.values())


@_BY_NAME
def test_the_pattern_matches_each_retired_word(group: WordGroup) -> None:
    pattern = re.compile(group.pattern, re.IGNORECASE)
    for word in group.matches:
        assert pattern.search(word), word
    for word in group.non_matches:
        assert not pattern.search(word), word


def test_only_the_guard_names_the_audit_dir_variable() -> None:
    assert AUDIT_DIR_VARIABLE.allowed.keys() == {GUARD}
