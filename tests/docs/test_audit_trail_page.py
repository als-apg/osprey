"""The audit-trail contract page against the code that writes the trail.

``reference/contracts/audit-trail.rst`` is the shape a reader of the agent
record relies on: which surface files a decision, which fields a record has,
which words a field may hold, which keys ride ``detail``. Every one of those is
a name the code owns, so this module holds the page's tables to the code's own
declarations (the :class:`~osprey.audit.envelope.AuditEnvelope` dataclass,
:func:`osprey.audit.tool_call.build_record`, the surface constants and the
hooks' own names) rather than to a copy of them.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from functools import cache
from pathlib import Path

import pytest

from osprey.audit import envelope, posture, tool_call
from osprey.audit.call import DETAIL_FACTS
from osprey.interfaces.common_middleware import (
    AUDIT_ACCOUNT_KEY,
    AUDIT_EXPECTED_ACCOUNT_KEY,
    AUDIT_OIDC_SUBJECT_KEY,
)
from osprey.jupyter_kernel import _channel_token
from osprey.services.auth_sidecar import audit as sidecar_audit
from osprey.services.bluesky_bridge.preflight import _verdict_detail
from osprey.services.watchdog import verdict
from tests._by_path import load_module

_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _ROOT / "docs/source/reference/contracts/audit-trail.rst"
_HOOKS = _ROOT / "src/osprey/templates/claude_code/claude/hooks"
_CODE_ROOTS = (_ROOT / "src/osprey", _ROOT / "packages")

hook_log = load_module("osprey_hook_log_audit_page", _HOOKS / "osprey_hook_log.py")

_LITERAL = re.compile(r"``([^`]+)``")
_ROW = re.compile(r"^   \* - ", re.M)


# --- reading the page ------------------------------------------------------------------------


def _text() -> str:
    return _PAGE.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _section(title: str) -> str:
    """The body under *title*, up to the next heading of any level."""
    lines = _text().splitlines()
    start = lines.index(title)
    body: list[str] = []
    for i in range(start + 2, len(lines) - 1):
        nxt = lines[i + 1]
        if lines[i] and nxt and set(nxt) <= set("=-~^") and len(nxt) >= len(lines[i]):
            break
        body.append(lines[i])
    return "\n".join(body)


def _rows(section: str) -> list[tuple[str, str]]:
    """Each ``list-table`` row of *section* as ``(first cell, rest)``, header excluded."""
    rows = []
    for chunk in _ROW.split(section)[1:]:
        first, _, rest = chunk.partition("\n     - ")
        rows.append((_flat(first), _flat(rest)))
    return rows[1:] if rows else rows


def _first_table(section: str) -> str:
    """*section* up to its second ``list-table``, so only the first table is read."""
    parts = section.split(".. list-table::")
    return ".. list-table::".join(parts[:2])


def _names(cell: str) -> set[str]:
    return set(_LITERAL.findall(cell))


def _record_rows() -> dict[str, str]:
    """The field rows of "The record", one entry per field a row names."""
    rows: dict[str, str] = {}
    for first, rest in _rows(_first_table(_section("The record"))):
        for name in _names(first):
            rows[name] = rest
    return rows


def _surface_rows() -> list[tuple[str, str]]:
    return _rows(_section("Entries, by surface"))


def _page_surfaces() -> set[str]:
    """Every surface the "Entries, by surface" table names, aliases included."""
    names: set[str] = set()
    for first, rest in _surface_rows():
        names |= _names(first)
        if first == "``<server>``":
            names |= _names(rest)
    names |= {"tool_call"}  # the tool-call section's own surface, pinned below
    return {n for n in names if "<" not in n}


# --- reading the code ------------------------------------------------------------------------


@cache
def _trees() -> tuple[tuple[Path, ast.Module], ...]:
    out = []
    for root in _CODE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.relative_to(root).parts:
                continue
            try:
                out.append((path, ast.parse(path.read_text(encoding="utf-8"))))
            except (SyntaxError, UnicodeDecodeError):
                continue
    return tuple(out)


@cache
def _string_literals() -> frozenset[str]:
    return frozenset(
        node.value
        for _, tree in _trees()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


_SURFACE_NAME = re.compile(r"(?:^|_)SURFACE(?:_|$)")


def _surface_constants(tree: ast.Module) -> set[str]:
    """String values of module-level ``*SURFACE*`` constants in *tree*."""
    found = set()
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and _SURFACE_NAME.search(target.id)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                found.add(value.value)
    return found


def _hook_names(tree: ast.Module) -> set[str]:
    """The hook names a hook module passes to ``emit_audit``, module constants resolved."""
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "emit_audit"
            and node.args
        ):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                names.add(arg.value)
            elif isinstance(arg, ast.Name):
                names.add(constants[arg.id])
            else:
                raise AssertionError(f"emit_audit with an unreadable hook name: {ast.unparse(arg)}")
    return names


def _hook_surfaces() -> set[str]:
    names: set[str] = set()
    for path in sorted(_HOOKS.glob("osprey_*.py")):
        names |= _hook_names(ast.parse(path.read_text(encoding="utf-8")))
    return {hook_log.audit_surface(name) for name in names}


#: ``*SURFACE*`` constants that name something other than an audit surface, by value.
_NOT_AUDIT_SURFACES: dict[str, str] = {
    "surface": "agent_record.contract.SURFACE_FIELD: the field's name, not a surface",
    "terminal": "spend attribution, the session handoff and the agent turn: a client kind",
    "dispatch": "models.spend_attribution: which kind of process spent model tokens",
    "service": "models.spend_attribution: which kind of process spent model tokens",
    "local": "models.spend_attribution: which kind of process spent model tokens",
    "chat": "web_terminal.session_handoff: which client holds a session",
    "served": "virtual_accelerator model surface: a serving mode",
    "model": "virtual_accelerator model surface: a serving mode",
    "execute": "code_review.contract: the kind of code a review reads",
    "notebook": "code_review.contract: the kind of code a review reads",
    "python_executor": "target_eligibility: the client an in-flight marker names",
    "write_switch": "osprey_connectors.write_authorization: who approved a write, not a surface",
    "hook_": "osprey_hook_log.AUDIT_SURFACE_PREFIX: the prefix the hook surfaces are built from",
}


# --- the surfaces ----------------------------------------------------------------------------


@pytest.mark.parametrize("surface", sorted(_page_surfaces() - _hook_surfaces()))
def test_each_surface_the_page_names_is_a_literal_in_the_code(surface: str) -> None:
    assert surface in _string_literals(), (
        f"audit-trail.rst names the surface {surface!r}, which no code under src/osprey or "
        "packages spells"
    )


def test_the_hook_surfaces_are_the_ones_the_hooks_compose() -> None:
    """The hooks spell no surface: ``audit_surface`` builds it from each hook's own name."""
    on_page = {n for n in _page_surfaces() if n.startswith(hook_log.AUDIT_SURFACE_PREFIX)}
    assert on_page == _hook_surfaces()
    assert {"hook_writes_check", "hook_code_review"} <= on_page


def test_every_audit_surface_constant_is_on_the_page() -> None:
    declared = set().union(*(_surface_constants(tree) for _, tree in _trees()))
    missing = declared - _page_surfaces() - set(_NOT_AUDIT_SURFACES)
    assert not missing, f"surface constants audit-trail.rst does not list: {sorted(missing)}"
    stale = set(_NOT_AUDIT_SURFACES) - declared
    assert not stale, f"_NOT_AUDIT_SURFACES entries no constant declares: {sorted(stale)}"


def test_a_surface_constant_off_the_page_is_caught() -> None:
    planted = ast.parse('SURFACE_PLANTED = "planted_surface"\nOTHER = "x"\n')
    assert _surface_constants(planted) == {"planted_surface"}
    assert "planted_surface" not in _page_surfaces()
    assert "planted_surface" not in _string_literals()


def test_a_planted_hook_name_composes_a_surface_the_page_lacks() -> None:
    planted = ast.parse('_HOOK = "no-such-hook"\nemit_audit(_HOOK, {}, "refused", "s", "r")\n')
    assert {hook_log.audit_surface(n) for n in _hook_names(planted)} == {"hook_no_such_hook"}
    assert "hook_no_such_hook" not in _page_surfaces()


# --- the record ------------------------------------------------------------------------------


def test_the_record_table_is_the_envelope_fields() -> None:
    fields = {f.name for f in dataclasses.fields(envelope.AuditEnvelope)}
    assert set(_record_rows()) == fields


def test_a_planted_field_is_not_an_envelope_field() -> None:
    fields = {f.name for f in dataclasses.fields(envelope.AuditEnvelope)}
    assert "planted_field" not in fields
    assert set(_record_rows()) | {"planted_field"} != fields


def test_the_posture_words_are_the_two_postures() -> None:
    assert _names(_record_rows()["posture"]) == {posture.POSTURE_SANDBOX, posture.POSTURE_WRITES}


def test_the_posture_source_words_are_the_closed_set() -> None:
    assert _names(_record_rows()["posture_source"]) == set(envelope.POSTURE_SOURCES)


def test_the_decision_words_are_the_canonical_spellings() -> None:
    expected = {
        *envelope.DECISIONS,
        envelope.DECISION_APPROVED,
        envelope.DECISION_DENIED,
        hook_log.AUDIT_DECISION_ASK,
    }
    assert _names(_record_rows()["decision"]) == expected


def test_the_watchdog_decision_words_are_the_watchdogs() -> None:
    match = re.search(r"``safety_watchdog`` has its own too: (.*?)\.", _flat(_text()))
    assert match is not None
    expected = {getattr(verdict, name) for name in verdict.__all__ if name.startswith("DECISION_")}
    assert _names(match.group(1)) == expected


def test_the_payload_bounds_are_the_envelopes() -> None:
    page = _flat(_section("What the trail is not"))
    assert f"({envelope.MAX_SOURCE_CHARS} characters" in page
    assert f"cut at {envelope.SESSION_VALUE_CHARS} characters" in page
    tokens = " ".join(f"{token}=" for token in envelope.SESSION_VALUE_TOKENS)
    assert tokens.replace("= ", "=<n> ") + "<text>" in page
    assert envelope.SURFACE_EXECUTOR == "executor"


# --- detail keys -----------------------------------------------------------------------------


def test_the_login_keys_are_the_middlewares() -> None:
    rows = _rows(_section("The login behind the record"))
    on_page = {name for first, _ in rows for name in _names(first)}
    expected = {AUDIT_ACCOUNT_KEY, AUDIT_EXPECTED_ACCOUNT_KEY, AUDIT_OIDC_SUBJECT_KEY}
    assert on_page == {f"{key}=" for key in expected}


def test_the_attribution_keys_are_written_by_the_writers() -> None:
    rows = _rows(_first_table(_section("Who the control system saw")))
    on_page = {name for first, _ in rows for name in _names(first)}
    assert on_page == {"ca_user=", "ca_host=", "owner=", "channel=", "channels"}
    # ``ca_user``, ``ca_host`` and ``owner`` ride a default record's detail as noted facts.
    assert {"ca_user", "ca_host", "owner"} <= set(DETAIL_FACTS)
    assert {"from_target", "to_target"} <= set(DETAIL_FACTS)
    # A notebook kernel record names the channel it wrote or was refused.
    assert _channel_token("SR:C01:X") == "channel=SR:C01:X"


def test_the_preflight_detail_tokens_are_the_preflights() -> None:
    detail = _verdict_detail(
        "lane-a", "sim", ["B", "A"], owner="alice", unresponsive=0, unchecked=0, limit=1024
    )
    written = {token.partition("=")[0] + "=" for token in detail.split()}
    page = _flat(_section("Who the control system saw"))
    for token in ("lane=", "target=", "owner=", "addresses=", "digest=", "channels=", "truncated="):
        assert f"``{token}``" in page
        assert token in written


# --- why a login was refused -----------------------------------------------------------------


def _sidecar_reasons() -> set[str]:
    return {getattr(sidecar_audit, n) for n in dir(sidecar_audit) if n.startswith("REASON_")}


def test_each_principal_refusal_reason_is_the_sidecars() -> None:
    rows = _rows(_section("Why a login was refused"))
    on_page = {name for first, _ in rows for name in _names(first)}
    assert on_page == {
        "hosted_domain_mismatch",
        "unverified_email",
        "no_covering_principal",
        "unsafe_asserted_identity",
    }
    assert on_page <= _sidecar_reasons()
    assert "planted_reason" not in _sidecar_reasons()


@pytest.mark.parametrize("reason", ["bad_credential", "credential_unevaluable", "no_card"])
def test_each_reason_the_prose_names_is_the_sidecars(reason: str) -> None:
    assert f"``{reason}``" in _section("Why a login was refused")
    assert reason in _sidecar_reasons()


# --- the tool-call record --------------------------------------------------------------------


def _tool_call_rows() -> dict[str, str]:
    rows: dict[str, str] = {}
    for first, rest in _rows(_section("The tool-call record")):
        for name in _names(first):
            rows[name] = rest
    return rows


def test_the_tool_call_table_is_the_built_record() -> None:
    record = tool_call.build_record(
        ts="t",
        actor="a",
        posture="sandbox",
        posture_source="process",
        session="s",
        session_id=None,
        tool_use_id=None,
        server="srv",
        subject="mcp__srv__tool",
        approval=None,
        target=None,
        generation=None,
        facts={},
        duration_ms=0.0,
    )
    assert set(_tool_call_rows()) == set(record)
    assert record["surface"] == tool_call.SURFACE_TOOL_CALL
    assert f"always ``{tool_call.SURFACE_TOOL_CALL}``" in _flat(_section("The tool-call record"))


def test_the_approval_keys_are_the_middlewares() -> None:
    tree = ast.parse((_ROOT / "src/osprey/mcp_server/audit_middleware.py").read_text("utf-8"))
    stamps = [
        {k.value for k in node.keys if isinstance(k, ast.Constant)}
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(isinstance(k, ast.Constant) and k.value == "outcome" for k in node.keys)
    ]
    assert stamps == [{"outcome", "approver", "permission_mode", "position"}]
    cell = _tool_call_rows()["approval"]
    assert all(f'"{key}"' in cell for key in stamps[0])
