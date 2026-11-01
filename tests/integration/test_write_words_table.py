"""The write words: one source per word, and no surface that speaks a config key.

D4 makes two package tables the source of every machine word an operator
reads: ``osprey_connectors.write_target.KIND_BANNERS`` (the banner beside a
machine) and ``osprey_connectors.refusals.KIND_WORDS`` (the words a refusal
adds), with the refusal phrases and reasons beside them. D5 makes
``osprey.agent_runner.tool_names.SHELL_TOOLS`` the one shell set. The hooks
and the chip JS cannot import the package, so they restate the tables; every
restated copy is pinned here to its source, and the framework modules that
can import are pinned to hold the source object itself.

The table also says which spellings are vocabulary (words an operator reads,
free to change with the source) and which are contract (machine-matched
spellings that never change): the hook's ``WRITES OFF`` lead is vocabulary,
``refusal_reason="WRITES_DISABLED"`` and the ``[osprey-refusal: <reason>]``
trailer are contract.

The word lint runs over the strings the refusal, tool-error, chip, approval
and agent-context surfaces emit, and finds no config key, file or build
command in any of them. Build warnings, lints and docs are not surfaces here:
PR-3 and PR-9 require them to name the key and the command.
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from osprey.agent_runner import tool_names
from osprey_connectors import refusals, write_target
from osprey_connectors.control_system import _base as connector_base
from osprey_connectors.errors import (
    ChannelNotFoundError,
    ChannelWriteBlockedError,
    RefusalReason,
)

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "osprey"
PACKAGE_SRC = REPO / "packages" / "osprey-connectors" / "src" / "osprey_connectors"
HOOKS_DIR = SRC / "templates" / "claude_code" / "claude" / "hooks"
HOOKS_PACKAGE = "osprey.templates.claude_code.claude.hooks"
JS_DIR = SRC / "interfaces" / "web_terminal" / "static" / "js"
FACTS_JS = JS_DIR / "control-target-facts.js"

#: The chip's JS, whose string literals are the chip surface.
CHIP_JS = (
    "control-target-chip.js",
    "control-target-facts.js",
    "control-target-popover.js",
    "control-target-popover-switch.js",
    "control-target-rows.js",
    "control-target-lab-bar.js",
)

KINDS = ("real", "rehearsal", "simulator")


def _import_hook(name: str) -> ModuleType:
    """Import a hook in-process, stripping the ``sys.path`` entry it adds."""
    hooks_dir = os.path.abspath(str(HOOKS_DIR))
    try:
        return importlib.import_module(f"{HOOKS_PACKAGE}.{name}")
    finally:
        sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry) != hooks_dir]


# ── the table ────────────────────────────────────────────────────────────────

#: Every word the write surfaces share, whether it is vocabulary or contract,
#: where its single source lives, and where it is restated.
WORDS_TABLE: dict[str, dict[str, Any]] = {
    "machine banners": {
        "class": "vocabulary",
        "source": "osprey_connectors.write_target.KIND_BANNERS",
        "copies": (
            "osprey_approval._KIND_WORDS (approval card)",
            "osprey_control_context.MACHINE_WORDS (agent context)",
            "control-target-facts.js KIND_BANNERS (chip)",
            "control-target-facts.js KIND_WORDS (chip; the banner's name)",
        ),
    },
    "refusal kind words": {
        "class": "vocabulary",
        "source": "osprey_connectors.refusals.KIND_WORDS",
        "copies": ("osprey_target_state.KIND_WORDS (hook twin)",),
    },
    "refusal phrases": {
        "class": "vocabulary",
        "source": "osprey_connectors.refusals._PHRASES",
        "copies": (
            "osprey_target_state._PHRASES (hook twin)",
            "osprey_writes_check._READONLY_RUN_PHRASE / _RECORD_UNAVAILABLE_PHRASE",
        ),
    },
    "WRITES OFF": {
        "class": "vocabulary",
        "source": "osprey_writes_check._DENY_HEADER",
        "copies": (),
    },
    "refusal reasons": {
        "class": "contract",
        "source": "osprey_connectors.refusals.REASONS",
        "copies": ("osprey_target_state.REASONS (hook twin)",),
    },
    "refusal trailer": {
        "class": "contract",
        "source": "osprey_connectors.refusals.refusal_text",
        "copies": ("osprey_target_state.refusal_text (hook twin)",),
    },
    "store path markers": {
        "class": "contract",
        "source": "osprey_connectors.refusals.STORE_PATH_MARKERS",
        "copies": (
            "osprey_target_state.STORE_PATH_MARKERS (hook twin)",
            "osprey_approval._FALLBACK_STORE_PATH_MARKERS",
        ),
    },
    "WRITES_DISABLED": {
        "class": "contract",
        "source": "osprey_connectors.errors.RefusalReason",
        "copies": ("control_system._base._verdict_refusal refusal_reason",),
    },
    "shell set": {
        "class": "contract",
        "source": "osprey.agent_runner.tool_names.SHELL_TOOLS",
        "copies": (
            "osprey.deployment.web_terminals.roster (imports it)",
            "osprey.cli.templates.claude_code (imports it)",
        ),
    },
}


def test_the_table_classes_each_word_once():
    assert {entry["class"] for entry in WORDS_TABLE.values()} == {"vocabulary", "contract"}
    assert WORDS_TABLE["WRITES OFF"]["class"] == "vocabulary"
    assert WORDS_TABLE["WRITES_DISABLED"]["class"] == "contract"


# ── machine banners ──────────────────────────────────────────────────────────


def test_the_banner_source_names_every_kind():
    assert tuple(write_target.KIND_BANNERS) == KINDS
    assert write_target.KIND_BANNERS["simulator"] == "Simulator — nothing moves"
    assert all("LIVE MACHINE" not in words for words in write_target.KIND_BANNERS.values())


def test_the_approval_card_restates_the_banners():
    approval = _import_hook("osprey_approval")
    assert approval._KIND_WORDS == write_target.KIND_BANNERS


def test_agent_context_restates_the_banners():
    context = _import_hook("osprey_control_context")
    assert context.MACHINE_WORDS == write_target.KIND_BANNERS


def _js_kind_constants(text: str) -> dict[str, str]:
    return dict(re.findall(r"export const (KIND_[A-Z]+) = '([a-z]+)';", text))


def _js_kind_table(name: str) -> dict[str, str]:
    """The ``{kind: words}`` table *name* in the chip facts, read as text."""
    text = FACTS_JS.read_text(encoding="utf-8")
    constants = _js_kind_constants(text)
    body = re.search(rf"export const {name} = \{{(.*?)\}};", text, re.S)
    assert body, f"{name} not found in {FACTS_JS.name}"
    entries = re.findall(r"\[(KIND_[A-Z]+)\]:\s*(['\"])(.*?)\2,", body.group(1))
    return {constants[key]: words for key, _quote, words in entries}


def test_the_chip_restates_the_banners():
    assert _js_kind_table("KIND_BANNERS") == write_target.KIND_BANNERS


def test_the_chip_names_each_machine_as_its_banner_does():
    names = {kind: banner.split(" — ")[0] for kind, banner in write_target.KIND_BANNERS.items()}
    assert _js_kind_table("KIND_WORDS") == names


def test_the_framework_reads_the_banner_source_itself():
    from osprey.deployment import status_display
    from osprey.interfaces.web_terminal.routes import websocket
    from osprey.mcp_server.control_system import connector_host_manager

    assert connector_host_manager.KIND_BANNERS is write_target.KIND_BANNERS
    assert websocket.KIND_BANNERS is write_target.KIND_BANNERS
    assert websocket.refusal_phrase is refusals.refusal_phrase
    source = Path(status_display.__file__).read_text(encoding="utf-8")
    assert "from osprey_connectors.write_target import KIND_BANNERS" in source


#: Files allowed to spell a banner as a value: the source and its pinned copies.
_BANNER_HOMES = {
    PACKAGE_SRC / "write_target.py",
    HOOKS_DIR / "osprey_approval.py",
    HOOKS_DIR / "osprey_control_context.py",
}


def _non_docstring_strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_no_other_python_module_spells_a_banner():
    banners = set(write_target.KIND_BANNERS.values())
    offenders = []
    for root in (SRC, PACKAGE_SRC):
        for path in root.rglob("*.py"):
            if path in _BANNER_HOMES:
                continue
            for value in _non_docstring_strings(path):
                if any(banner in value for banner in banners):
                    offenders.append(f"{path.relative_to(REPO)}: {value!r}")
    assert offenders == []


def test_no_other_js_module_spells_a_banner():
    banners = set(write_target.KIND_BANNERS.values())
    offenders = [
        str(path.relative_to(REPO))
        for path in JS_DIR.rglob("*.js")
        if path != FACTS_JS
        and any(banner in _strip_js_comments(path.read_text("utf-8")) for banner in banners)
    ]
    assert offenders == []


# ── refusal words ────────────────────────────────────────────────────────────


def test_the_twin_restates_the_refusal_words():
    twin = _import_hook("osprey_target_state")
    assert twin.KIND_WORDS == refusals.KIND_WORDS
    assert twin._PHRASES == refusals._PHRASES
    assert twin.REASONS == refusals.REASONS
    assert tuple(twin.STORE_PATH_MARKERS) == refusals.STORE_PATH_MARKERS


@pytest.mark.parametrize("reason", refusals.REASONS)
@pytest.mark.parametrize("kind", (*KINDS, None))
def test_the_twin_speaks_each_refusal_as_the_package_does(reason, kind):
    twin = _import_hook("osprey_target_state")
    assert twin.refusal_text(reason, kind) == refusals.refusal_text(reason, kind)
    assert refusals.refusal_text(reason, kind).endswith(f"\n[osprey-refusal: {reason}]")


def test_the_writes_check_fallback_phrases_restate_the_package():
    check = _import_hook("osprey_writes_check")
    assert check._READONLY_RUN_PHRASE == refusals.refusal_phrase("readonly_run")
    assert check._RECORD_UNAVAILABLE_PHRASE == refusals.refusal_phrase("record_unavailable")
    assert check._fallback_text("readonly_run") == refusals.refusal_text("readonly_run")
    assert check._fallback_text("x") == refusals.refusal_text("record_unavailable")


def test_the_approval_store_markers_restate_the_package():
    approval = _import_hook("osprey_approval")
    assert approval._FALLBACK_STORE_PATH_MARKERS == refusals.STORE_PATH_MARKERS


def test_writes_off_is_the_deny_lead():
    check = _import_hook("osprey_writes_check")
    assert "WRITES OFF" in check._DENY_HEADER
    assert "WRITES OFF" not in "".join(refusals._PHRASES.values())


# ── contract spellings ───────────────────────────────────────────────────────


def _refused(reason: str | None, kind: str | None = None, allowed: bool = False) -> Any:
    verdict = SimpleNamespace(allowed=allowed, reason=reason, kind=kind)
    return connector_base._verdict_refusal("SR:TEST:SP", 1.0, verdict)


def test_writes_disabled_is_the_refusal_contract():
    assert RefusalReason("WRITES_DISABLED") is RefusalReason.WRITES_DISABLED
    for reason in refusals.REASONS:
        result = _refused(reason, "real")
        assert result.refusal_reason == "WRITES_DISABLED"
        assert "WRITES_DISABLED" not in result.error_message
    assert _refused(None, allowed=True).refusal_reason == "WRITES_DISABLED"


# ── shell set ────────────────────────────────────────────────────────────────


def test_the_shell_set_has_one_declaration():
    from osprey.cli.templates import claude_code
    from osprey.deployment.web_terminals import roster

    assert tool_names.SHELL_TOOLS == ("Bash", "Monitor")
    assert roster.SHELL_TOOLS is tool_names.SHELL_TOOLS
    assert claude_code.SHELL_TOOLS is tool_names.SHELL_TOOLS


def _literal_strings(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return {
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
    return set()


def test_no_other_module_restates_the_shell_set():
    shell = set(tool_names.SHELL_TOOLS)
    home = Path(tool_names.__file__).resolve()
    offenders = []
    for root in (SRC, PACKAGE_SRC):
        for path in root.rglob("*.py"):
            if path.resolve() == home:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if shell <= _literal_strings(node):
                    offenders.append(f"{path.relative_to(REPO)}:{getattr(node, 'lineno', '?')}")
    assert offenders == []


# ── word lint ────────────────────────────────────────────────────────────────

#: What an operator-facing write string may not name. Reason slugs such as
#: ``tier_never`` are contract (the trailer, the agent-context word) and pass.
FORBIDDEN: dict[str, tuple[re.Pattern[str], ...]] = {
    "config key": (
        re.compile(
            r"\b(?:control_system|mcp_servers|claude_code|virtual_accelerator|permissions"
            r"|epics)\.[A-Za-z_]+"
        ),
        re.compile(
            r"\b(?:control_system|writes_enabled|write_tools|login_tiers|remove_deny"
            r"|live_standin|writes_tier)\b"
        ),
    ),
    "file": (
        re.compile(r"\b[\w-]+\.(?:ya?ml|json|toml|env|py|sh|cfg|ini)\b"),
        re.compile(r"\bcontrol_target\b"),
        re.compile(r"(?:^|[\s(])~?/[\w.-]+/"),
    ),
    "build command": (
        re.compile(r"\bosprey\s+(?:build|init|deploy|config|vendor)\b", re.I),
        re.compile(r"--revalidate\b"),
        re.compile(r"\b(?:re)?build\b", re.I),
        re.compile(r"\bredeploy\b", re.I),
    ),
}


def lint(text: str) -> list[str]:
    """The forbidden categories *text* hits, empty when it is clean."""
    return [
        f"{category}: {pattern.pattern}"
        for category, patterns in FORBIDDEN.items()
        for pattern in patterns
        if pattern.search(text)
    ]


@pytest.mark.parametrize(
    "planted",
    [
        "Set control_system.writes to allowed.",
        "Writes are off; see writes_enabled.",
        "Edit profile.yml and retry.",
        "The record lives under control_target.",
        "Run osprey build, then try again.",
        "Run osprey init --revalidate.",
        "Rebuild and redeploy the container.",
    ],
)
def test_the_lint_catches_each_category(planted):
    assert lint(planted)


def _refusal_surface() -> list[str]:
    twin = _import_hook("osprey_target_state")
    check = _import_hook("osprey_writes_check")
    texts = []
    for reason in refusals.REASONS:
        for kind in (*KINDS, None):
            texts.append(refusals.refusal_text(reason, kind))
            texts.append(twin.refusal_text(reason, kind))
            texts.append(check._DENY_HEADER + check._text(reason, kind))
    texts += [check._fallback_text("readonly_run"), check._fallback_text("record_unavailable")]
    return texts


def _monitor_refusals(monkeypatch) -> list[str]:
    texts = []
    for in_server in (True, False):
        monkeypatch.setattr(connector_base, "_in_mcp_server_process", lambda v=in_server: v)
        texts += [_refused(r, k).error_message for r in refusals.REASONS for k in (*KINDS, None)]
    texts.append(_refused(None, allowed=True).error_message)
    return texts


def _tool_error_surface(monkeypatch) -> list[str]:
    from osprey.mcp_server.control_system import error_handling

    texts = []
    errors: list[BaseException] = [
        TimeoutError("deadline"),
        ChannelNotFoundError("SR:NOPE"),
        RuntimeError("boom"),
        ChannelWriteBlockedError("SR:TEST:SP", "CONTROL_SYSTEM_REFUSED", "denied"),
        ChannelWriteBlockedError("SR:TEST:SP", "LIMITS", "over"),
        ChannelWriteBlockedError("SR:TEST:SP", None, "refused"),
    ]
    errors += [
        ChannelWriteBlockedError("SR:TEST:SP", "WRITES_DISABLED", message)
        for message in _monitor_refusals(monkeypatch)
    ]
    for banner in (*write_target.KIND_BANNERS.values(), None):
        identity = {"name": "live", "label": banner, "endpoint": "gw:5064"} if banner else None
        monkeypatch.setattr(error_handling, "describe_active_target", lambda i=identity: i)
        for exc in errors:
            _type, message, suggestions, _details = error_handling._envelope(
                exc, "channel_write", "control_system"
            )
            texts += [message, *suggestions]
    return texts


def _strip_js_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/|(?<![:'\"\\])//[^\n]*", "", text, flags=re.S)


_JS_LITERAL = re.compile(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"|`(?:[^`\\]|\\.)*`", re.S)


def _chip_surface() -> list[str]:
    from osprey.interfaces.web_terminal import control_context_owner as owner

    texts = []
    for name in CHIP_JS:
        source = _strip_js_comments((JS_DIR / name).read_text(encoding="utf-8"))
        texts += [m.group(0)[1:-1] for m in _JS_LITERAL.finditer(source) if " " in m.group(0)]
    texts.append(owner.owned_elsewhere_message(None))
    texts.append(
        owner.owned_elsewhere_message(
            SimpleNamespace(kind=owner.OWNER_WEB_TERMINAL, pid=42, port=8080)
        )
    )
    return texts


def _approval_surface(monkeypatch) -> list[str]:
    approval = _import_hook("osprey_approval")
    texts = [approval._TARGET_BASELINE_LINE, approval._REAL_SWITCH_WARNING]
    texts += list(approval._KIND_WORDS.values())
    for kind in KINDS:
        fake = SimpleNamespace(
            selected_target=lambda record: "live",
            target_metadata=lambda record, target: {
                "endpoint": "gw:5064",
                "probe_channel": "SR:PROBE",
            },
            published_machine_kind=lambda target, hook_input=None, k=kind: k,
        )
        monkeypatch.setattr(approval, "_target_state", fake)
        record = {"target": "live"}
        texts.append(approval._target_line(None, record=record))
        texts += approval._describe_control_target_set(
            {"target": "live"}, {}, None, read_record=lambda r=record: r
        )
    texts += approval._describe_control_target_set({}, {}, None)
    texts += approval._describe_control_target_set(
        {"target": "live"}, {}, None, read_record=lambda: None
    )
    fake_none = SimpleNamespace(target_metadata=lambda record, target: None)
    monkeypatch.setattr(approval, "_target_state", fake_none)
    texts += approval._describe_control_target_set(
        {"target": "live"}, {}, None, read_record=lambda: {"target": "live"}
    )
    return texts


def _agent_context_surface() -> list[str]:
    context = _import_hook("osprey_control_context")
    writes = {
        **{f"t-{reason}": f"{context.WRITES_REFUSED}:{reason}" for reason in refusals.REASONS},
        "asked": context.WRITES_ASKED,
        "allowed": context.WRITES_ALLOWED,
        "off": context.WRITES_OFF,
    }
    state = {
        "target": "live",
        "generation": 3,
        "writes": writes,
        "machines": dict(context.MACHINE_WORDS),
    }
    previous = {
        "target": "va",
        "generation": 2,
        "writes": dict.fromkeys(writes, context.WRITES_OFF),
        "machines": dict.fromkeys(context.MACHINE_WORDS, context.UNKNOWN),
    }
    return [context.render_block(state), context.render_block(state, previous)]


SURFACES = ("refusal", "tool-error", "chip", "approval", "agent-context")


@pytest.mark.parametrize("surface", SURFACES)
def test_no_surface_names_a_config_key_file_or_build_command(surface, monkeypatch):
    texts = {
        "refusal": lambda: _refusal_surface() + _monitor_refusals(monkeypatch),
        "tool-error": lambda: _tool_error_surface(monkeypatch),
        "chip": _chip_surface,
        "approval": lambda: _approval_surface(monkeypatch),
        "agent-context": _agent_context_surface,
    }[surface]()
    assert texts, f"the {surface} surface emitted nothing to lint"
    hits = {text: lint(text) for text in texts if lint(text)}
    assert hits == {}
