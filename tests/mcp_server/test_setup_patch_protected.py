"""`setup_patch` must refuse the protected set before it reads anything.

``setup_patch`` is the one MCP tool through which the agent edits its own
``config.yml`` and ``.mcp.json`` — the two files that carry the write gate, the
approval gate, the paths the safety layers derive their allow and deny areas
from, and the agent's own rendered permission surface. A tool that can set
those keys is a tool that can un-gate itself, so the refusal here is not a
warning: it fires on ``(file, key_path)`` alone, before the project root is
resolved and before the target is opened.

Two properties are under test throughout, and both are load-bearing:

* **Nothing moves.** Every refusal leaves both patchable files byte-identical.
  Byte comparison rather than a parsed one, because a round-trip that rewrote
  the file with the same values would still be a write to a file the agent may
  not write.
* **Existence-independence.** The same key is refused whether or not the target
  exists or parses. A gate that consulted the filesystem first would answer
  ``not_found`` for a protected key on an unbuilt render — handing a caller a
  file-existence oracle, and making the refusal contingent on state that has
  nothing to do with what makes the key protected.

The refusal also has to leave a trace: one ``protected-writes.jsonl`` record
per attempt, and one activity emit, so a refused edit is visible to the
operator afterwards and not only to the agent that saw the error.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from osprey.cli.profile_conventions import RESERVED_PATH_CHANNELS, is_protected_key
from osprey.services.python_executor import refusal_audit
from osprey.services.python_executor.refusal_audit import PROTECTED_WRITES_LOG_FILENAME
from osprey_connectors.config import RUNTIME_WRITE_PATH_KEYS
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

SETUP_MOD = "osprey.mcp_server.workspace.tools.setup"

#: A value distinctive enough that no coincidental substring can hide a leak of
#: it into the error message, the audit record or the activity feed.
SENTINEL = "qqzzSENTINELvalue77"


def _get_setup_patch():
    from osprey.mcp_server.workspace.tools.setup import setup_patch

    return get_tool_fn(setup_patch)


@pytest.fixture
def render(tmp_path, monkeypatch):
    """A render holding both patchable files, with config and audit zone pinned.

    ``resolve_config_path`` is patched in the tool's own namespace (the tool
    resolves the project root as its parent), and the audit zone is redirected
    through :func:`refusal_audit.audit_dir` — the single seam that module
    documents for exactly this, so the test never has to stand up a real repo.
    """
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "control_system": {"type": "mock", "writes_enabled": False},
                "agent_data": {"base_dir": "./_agent_data"},
                "ui": {"theme": "light"},
            }
        )
    )
    (tmp_path / ".mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"demo": {"command": "python", "env": {"API_KEY": "old"}}}},
            indent=2,
        )
        + "\n"
    )
    monkeypatch.setattr(refusal_audit, "audit_dir", lambda: tmp_path / "var" / "audit")
    with patch(f"{SETUP_MOD}.resolve_config_path", return_value=tmp_path / "config.yml"):
        yield tmp_path


def _snapshot(root: Path) -> dict[str, bytes]:
    """Raw bytes of every patchable file that exists, keyed by name."""
    return {
        name: (root / name).read_bytes()
        for name in ("config.yml", ".mcp.json")
        if (root / name).is_file()
    }


def _audit_records(root: Path) -> list[dict]:
    path = root / "var" / "audit" / PROTECTED_WRITES_LOG_FILENAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _notify():
    """Patch the activity emit in the tool's namespace (``patch`` gives an AsyncMock)."""
    return patch(f"{SETUP_MOD}.notify_agent_activity_async")


#: One case per reason a key lands in the protected set, spelled as the tool
#: receives it. ``control_system`` and ``services`` are the ancestor rule:
#: writing the parent rewrites the protected descendant beneath it.
PROTECTED_CASES = [
    pytest.param("config.yml", "control_system.writes_enabled", id="write-gate"),
    pytest.param("config.yml", "agent_data.base_dir", id="agent-data-root"),
    pytest.param("config.yml", "control_system.limits_checking.enabled", id="limits-gate"),
    pytest.param("config.yml", "approval.mode", id="approval-gate"),
    pytest.param("config.yml", "claude_code.permissions.deny", id="permission-surface"),
    pytest.param("config.yml", "control_system", id="ancestor-block"),
    pytest.param("config.yml", "services", id="ancestor-of-runtime-write-path"),
    pytest.param(".mcp.json", "mcpServers.demo.command", id="mcp-command"),
    pytest.param(".mcp.json", "mcpServers.demo.env.API_KEY", id="mcp-credential"),
    pytest.param(".mcp.json", "mcpServers.demo", id="mcp-ancestor"),
]


@pytest.mark.unit
@pytest.mark.parametrize("file,key_path", PROTECTED_CASES)
async def test_protected_key_is_refused(render, file, key_path):
    """The envelope names the key, the file it did not touch, and the real channel."""
    fn = _get_setup_patch()
    with _notify(), assert_raises_error(error_type="protected_key") as ctx:
        await fn(file=file, key_path=key_path, value=SENTINEL)

    message = ctx["envelope"]["error_message"]
    assert key_path in message, message
    assert file in message, message
    assert "unchanged" in message.lower(), message
    assert RESERVED_PATH_CHANNELS[file] in message, (
        f"the refusal must send the operator to the channel that DOES own the key: {message}"
    )
    assert ctx["envelope"]["suggestions"], "a refusal the agent cannot act on is a dead end"
    assert SENTINEL not in json.dumps(ctx["envelope"]), "the rejected value must not be echoed"


@pytest.mark.unit
@pytest.mark.parametrize("key_path", RUNTIME_WRITE_PATH_KEYS)
async def test_every_runtime_write_path_key_is_refused(render, key_path):
    """Repointing a runtime-write path moves what a safety layer treats as writable.

    Parametrized over the tuple itself rather than a copy of it: a key added to
    ``RUNTIME_WRITE_PATH_KEYS`` later must be covered here without an edit.
    """
    assert is_protected_key("config.yml", key_path), (
        "precondition: the protected table unpacks RUNTIME_WRITE_PATH_KEYS"
    )
    fn = _get_setup_patch()
    with _notify(), assert_raises_error(error_type="protected_key"):
        await fn(file="config.yml", key_path=key_path, value=SENTINEL)


@pytest.mark.unit
@pytest.mark.parametrize("file,key_path", PROTECTED_CASES)
async def test_refusal_leaves_both_files_byte_identical(render, file, key_path):
    """Nothing is rewritten — not even round-tripped back to the same values."""
    before = _snapshot(render)
    fn = _get_setup_patch()
    with _notify(), assert_raises_error(error_type="protected_key"):
        await fn(file=file, key_path=key_path, value=SENTINEL)

    assert _snapshot(render) == before


@pytest.mark.unit
async def test_refusal_writes_one_audit_record(render):
    """One ``protected-writes.jsonl`` line, naming the surface, key and channel."""
    fn = _get_setup_patch()
    with _notify(), assert_raises_error(error_type="protected_key"):
        await fn(file="config.yml", key_path="control_system.writes_enabled", value=SENTINEL)

    (record,) = _audit_records(render)
    assert record["surface"] == "setup_patch"
    assert record["target_file"] == "config.yml"
    assert record["key_or_path"] == "control_system.writes_enabled"
    assert record["channel"] == RESERVED_PATH_CHANNELS["config.yml"], (
        "the log and the refusal message must name the channel the same way"
    )
    assert record["reason"] == "protected_key"
    assert SENTINEL not in json.dumps(record), "the audit trail is not a place for the value"


@pytest.mark.unit
async def test_each_refusal_appends_its_own_record(render):
    """The log counts attempts — a retried refusal is a second line, not an overwrite."""
    fn = _get_setup_patch()
    for key_path in ("control_system.writes_enabled", "agent_data.base_dir"):
        with _notify(), assert_raises_error(error_type="protected_key"):
            await fn(file="config.yml", key_path=key_path, value="x")

    assert [r["key_or_path"] for r in _audit_records(render)] == [
        "control_system.writes_enabled",
        "agent_data.base_dir",
    ]


@pytest.mark.unit
async def test_refusal_emits_activity(render):
    """The operator sees the attempt in the feed, with the file and key but no value."""
    fn = _get_setup_patch()
    with _notify() as notify, assert_raises_error(error_type="protected_key"):
        await fn(file=".mcp.json", key_path="mcpServers.demo.env.API_KEY", value=SENTINEL)

    notify.assert_called_once()
    call = notify.call_args
    assert call.args[:2] == ("setup_patch", "config")
    detail = call.kwargs["detail"]
    assert "blocked" in detail.lower(), f"a refusal must not read like a landed patch: {detail}"
    assert ".mcp.json" in detail and "mcpServers.demo.env.API_KEY" in detail
    reported = [str(arg) for arg in call.args] + [str(v) for v in call.kwargs.values()]
    for piece in reported:
        assert SENTINEL not in piece, f"value leaked into the activity feed: {piece!r}"


@pytest.mark.unit
async def test_an_unprotected_key_still_patches(render):
    """The gate is a gate, not a wall: an ordinary key goes through untouched."""
    fn = _get_setup_patch()
    with _notify() as notify:
        result = extract_response_dict(
            await fn(file="config.yml", key_path="ui.theme", value="dark")
        )

    assert result["before"] == "light"
    assert result["after"] == "dark"
    assert yaml.safe_load((render / "config.yml").read_text())["ui"]["theme"] == "dark"
    assert _audit_records(render) == [], "a landed patch is not a refusal"
    assert notify.call_args.kwargs["detail"] == "config.yml: ui.theme"


@pytest.mark.unit
async def test_an_unprotected_mcp_key_still_patches(render):
    """The `.mcp.json` branch is gated on the same table, not on the file."""
    fn = _get_setup_patch()
    with _notify():
        result = extract_response_dict(
            await fn(file=".mcp.json", key_path="mcpServers.demo.disabled", value="true")
        )

    assert result["after"] is True
    on_disk = json.loads((render / ".mcp.json").read_text())
    assert on_disk["mcpServers"]["demo"]["disabled"] is True


@pytest.mark.unit
async def test_a_protected_key_is_refused_even_when_the_file_is_missing(render):
    """Existence-independence: no ``not_found`` oracle in front of the gate."""
    (render / ".mcp.json").unlink()

    fn = _get_setup_patch()
    with _notify(), assert_raises_error(error_type="protected_key"):
        await fn(file=".mcp.json", key_path="mcpServers.demo.env.API_KEY", value=SENTINEL)

    assert not (render / ".mcp.json").exists(), "a refusal must not create the file either"
    assert len(_audit_records(render)) == 1


@pytest.mark.unit
async def test_a_protected_key_is_refused_even_when_the_file_is_unparseable(render):
    """A broken target cannot downgrade the refusal into an ``internal_error``."""
    (render / ".mcp.json").write_text("{ this is not json")
    before = _snapshot(render)

    fn = _get_setup_patch()
    with _notify(), assert_raises_error(error_type="protected_key"):
        await fn(file=".mcp.json", key_path="mcpServers.demo.command", value=SENTINEL)

    assert _snapshot(render) == before


@pytest.mark.unit
async def test_the_file_allowlist_still_answers_first(render):
    """A file no writer may patch is a validation error, whatever the key says."""
    fn = _get_setup_patch()
    with _notify(), assert_raises_error(error_type="validation_error"):
        await fn(file=".claude/settings.json", key_path="permissions.deny", value="[]")

    assert _audit_records(render) == []


@pytest.mark.unit
@pytest.mark.parametrize("key_path", ["", "control_system..writes_enabled", "/control_system"])
async def test_key_path_validation_still_answers_first(render, key_path):
    """A malformed path is not a protected key — it is not a key at all."""
    fn = _get_setup_patch()
    with _notify(), assert_raises_error(error_type="validation_error"):
        await fn(file="config.yml", key_path=key_path, value="x")

    assert _audit_records(render) == []


@pytest.mark.unit
async def test_a_broken_audit_recorder_does_not_rescue_the_write(render):
    """Reporting is best-effort; the refusal is not.

    If the audit write could turn a refusal into a traceback, an agent could
    make the gate fail open by making the audit zone unwritable.
    """
    before = _snapshot(render)
    fn = _get_setup_patch()
    with (
        # Intercepts only because setup.py imports the function at CALL time
        # (lazy import inside the tool). If that import is ever hoisted to
        # module top, the broken_recorder.called assertion below goes red
        # instead of this test passing vacuously with the fault never injected.
        patch.object(
            refusal_audit, "record_protected_refusal", side_effect=OSError("read-only")
        ) as broken_recorder,
        _notify(),
        assert_raises_error(error_type="protected_key"),
    ):
        await fn(file="config.yml", key_path="control_system.writes_enabled", value=SENTINEL)

    assert broken_recorder.called, "fault was never injected — the guard went untested"
    assert _snapshot(render) == before


@pytest.mark.unit
async def test_a_broken_activity_emit_does_not_rescue_the_write(render):
    """Same for the feed: an unreachable Web Terminal must not mask the refusal."""
    before = _snapshot(render)
    fn = _get_setup_patch()
    with (
        patch(f"{SETUP_MOD}.notify_agent_activity_async", side_effect=RuntimeError("no socket")),
        assert_raises_error(error_type="protected_key"),
    ):
        await fn(file="config.yml", key_path="control_system.writes_enabled", value=SENTINEL)

    assert _snapshot(render) == before
    assert len(_audit_records(render)) == 1, "the durable record is written before the alert"
