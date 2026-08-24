"""Configuration and Claude setup routes."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from osprey.cli.profile_conventions import (
    RESERVED_PATH_CHANNELS,
    is_protected_key,
    protected_view,
)
from osprey.interfaces.web_terminal.claude_code_files import (
    PROFILE_EDIT_NOTICE,
    ClaudeCodeFileService,
    ProtectedWriteError,
)
from osprey.interfaces.web_terminal.routes.agent_activity import record_activity

logger = logging.getLogger(__name__)

router = APIRouter()

#: The file the config panel patches. Spelled once: it is both the name the
#: protected-set table is keyed by and the name a refusal shows the operator,
#: and the two must not be able to disagree.
_CONFIG_FILE = "config.yml"

#: Machine-ish reason recorded for a refused protected key. The same string
#: ``setup_patch`` records, so one query over ``protected-writes.jsonl`` finds
#: every protected-key refusal regardless of which surface produced it.
_PROTECTED_KEY_REASON = "protected_key"

#: Sentinel for "this key path is not in the document at all", so an absent key
#: and a key whose value happens to be ``None`` are two different states. A
#: protected key deleted by a whole-document PUT must read as a change, and
#: ``dict.get(key)`` alone cannot tell the two apart.
_ABSENT = object()

#: How many changed protected keys a single refusal names before it summarizes
#: the rest. A PATCH body is small enough that this never fires; a PUT replaces
#: the whole document, so deleting one block can change dozens of keys at once
#: and the detail would stop being readable. Every changed key still reaches
#: ``protected-writes.jsonl`` -- the cap trims the message, never the audit.
_MAX_NAMED_KEYS = 10

#: What a successful write reports when this process may not write the render
#: zone. Addressed to the operator, so it says what happens next rather than
#: naming the marker that caused it: the config edit landed, the artifacts
#: derived from it follow on the restart. The panel renders this string
#: verbatim -- one wording, spelled server-side, so the two surfaces cannot
#: drift into telling an operator two different stories about the same write.
_RENDER_READONLY_DETAIL = "derived artifacts re-render on container restart"

# Config sections the Form view offers for editing. Deliberately narrower than
# the file: infra/build sections (api, cli, deploy, modules, file_paths, ...)
# are left to the Raw YAML view. Every name here must match a real top-level
# key -- an entry that matches nothing renders nothing, silently, which is how
# "python_execution" (the section is called "execution") kept the execution
# settings out of the form.
_AGENT_CONFIG_SECTIONS = [
    "control_system",
    "archiver",
    "approval",
    "claude_code",
    "channel_finder",
    "ariel",
    "logbook",
    "facility_knowledge",
    "execution",
    "artifact_server",
    "screen_capture",
    "hooks",
]


def _require_config_panel(request: Request) -> None:
    """Refuse with 403 when this deployment has taken the Config panel away.

    ``web.config_panel.enabled: false`` is a TIER boundary, not a cosmetic one:
    ``config.yml`` and ``.claude/`` are what the agent's permission surface is
    rendered from, so a tier that may not edit them may not reach these routes
    at all. Hiding the tab is the other half and cannot be the only half — a
    client-side gate is undone by typing the URL.

    Called FIRST in every handler on both surfaces, ahead of the protected-set
    logic and ahead of anything that reads or writes a file. The two gates are
    independent and answer different questions: this one is "may this
    deployment's operators use this panel at all", the protected set is "may
    this particular key be written by anyone through it". A disabled panel
    therefore never reports a protected-key refusal, because it never gets as
    far as having a key to judge.

    The refusal names the key, so an operator who meets it knows which switch
    produced it rather than suspecting a broken deployment.

    Args:
        request: Incoming request carrying ``app.state``.

    Raises:
        HTTPException: 403 when the panel is disabled for this deployment.
    """
    if not getattr(request.app.state, "config_panel_enabled", True):
        raise HTTPException(
            status_code=403,
            detail=(
                "the Config panel is disabled for this deployment (web.config_panel.enabled: false)"
            ),
        )


def _project_dir(request: Request) -> Path | None:
    """The project directory this panel is pointed at, or ``None``.

    ``project_cwd`` is the answer whenever the app carries one; the config's own
    directory stands in otherwise, which is what a flat project looks like.

    This is the RENDER directory -- the tree holding ``config.yml`` and the
    ``.claude/`` artifacts -- and it is the right anchor for exactly one thing:
    the artifact regen, which re-renders that tree. It is emphatically NOT the
    anchor for the state zone; see :func:`_backup_config`.

    Args:
        request: The incoming request, for ``app.state``.

    Returns:
        The project directory, or ``None`` when the app names neither.
    """
    project_cwd = getattr(request.app.state, "project_cwd", None)
    config_path: Path | None = request.app.state.config_path
    return Path(project_cwd) if project_cwd else (config_path.parent if config_path else None)


def _backup_config(config_path: Path) -> Path:
    """Copy ``config.yml`` into the state zone and return where it landed.

    The scheme *and* its anchor live in
    :func:`~osprey.utils.config_writer.write_config_backup`, which every config
    writer in the tree shares. This route passes nothing but the file, on
    purpose: a relative ``agent_data.base_dir`` anchors on the deployment repo
    ROOT that ``config_path`` sits in, never on :func:`_project_dir`.

    The two differ in a container, and that difference was a live 500. The
    render is ``/app/<project>/build`` while the state zone the image creates
    and chowns is ``/app/<project>/var/agent_data``; anchoring the backup on the
    render made this route mkdir ``<render>/var`` inside the root-owned render
    zone, so every admin ``PATCH /api/config`` failed with ``PermissionError``
    before a byte reached ``config.yml``. A panel pointed at another project
    still backs up into *that* project's zone, because ``config_path`` is that
    project's config.

    Args:
        config_path: The config file to back up.

    Returns:
        Path to the backup that was written.
    """
    from osprey.utils.config_writer import write_config_backup

    return write_config_backup(config_path)


def _regen_if_drift(request: Request) -> list[str]:
    """Re-render Claude Code artifacts if config.yml drifted from them.

    Called after a successful config write so safety-critical fields (e.g. the
    writes_enabled kill-switch baked into settings.json's permissions.deny) take
    effect on the next terminal restart, which respawns the agent and re-reads
    the on-disk ``.claude/`` artifacts. Fails open: a regen error must never undo
    a config write that already succeeded. ``regen_if_drift`` no-ops when the
    project has no rendered ``.claude/`` to re-sync.
    """
    project_dir = _project_dir(request)
    if project_dir is None:
        return []
    try:
        from osprey.cli.templates.manager import TemplateManager

        return TemplateManager().regen_if_drift(project_dir)
    except Exception:  # noqa: BLE001 — config write already succeeded; never raise here
        logger.warning("Claude Code artifact regen after config write failed", exc_info=True)
        return []


def _regen_after_write(request: Request) -> tuple[list[str], str | None]:
    """Re-render after a successful write, or explain why this process will not.

    Strictly the success path: both write handlers reach here only once the
    panel gate, the protected-set gate and the write itself are behind them, so
    nothing about a refusal changes shape.

    In a privilege-split container the render zone is root-owned and the root
    entrypoint renders it before dropping to the non-root app user. The admin
    image is the one where the panel's write to ``config.yml`` still lands --
    ``config.yml`` and ``.claude/`` are separate concerns for ownership -- and
    the regen that normally follows would then be this process trying to move a
    tree it cannot write. So it is skipped, and the skip is *reported*.

    That second half is the load-bearing one. ``regenerated: []`` alone is
    exactly what a render with nothing to do returns, so an operator who just
    edited a render-shaping key would read it as "already in effect" when the
    derived artifacts are in fact still the ones the image was built with. The
    detail is what closes that gap, and the panel shows it verbatim.

    Args:
        request: The incoming request, for ``app.state``.

    Returns:
        ``(regenerated, detail)`` -- the re-rendered paths and ``None`` on a
        writable render zone; ``([], detail)`` when this process may not write it.
    """
    if getattr(request.app.state, "render_zone_readonly", False):
        logger.info(
            "Render zone is read-only (OSPREY_RENDER_ZONE_READONLY=1); "
            "config write landed but the Claude Code artifact regen was skipped — "
            "%s",
            _RENDER_READONLY_DETAIL,
        )
        return [], _RENDER_READONLY_DETAIL
    return _regen_if_drift(request), None


@router.get("/api/config")
async def get_config(request: Request):
    """Return agent-relevant config sections as structured JSON + raw YAML."""
    _require_config_panel(request)
    config_path: Path | None = request.app.state.config_path
    if not config_path or not config_path.exists():
        raise HTTPException(status_code=404, detail="No config.yml found")

    raw = config_path.read_text(encoding="utf-8")
    try:
        full_config = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Invalid YAML: {e}") from e

    # Extract only agent-relevant sections
    sections = {}
    for key in _AGENT_CONFIG_SECTIONS:
        if key in full_config:
            sections[key] = full_config[key]

    return {
        "sections": sections,
        "raw": raw,
        "path": str(config_path),
    }


def _refuse_protected_keys(request: Request, keys: list[str]) -> HTTPException:
    """Record and report a refused config write, and return the 403 to raise.

    Shared by both write surfaces on ``/api/config``: PATCH names the protected
    keys its body aimed at, PUT names the protected keys its replacement
    document would have changed. One refusal path, so the two cannot drift into
    telling an operator different stories about the same rule, and one query
    over ``protected-writes.jsonl`` still finds every refusal either produced.

    The machine-readable parts stay ``setup_patch``'s to the letter -- the same
    ``reason``, the same ``surface``, the same ``BLOCKED a protected config key``
    phrase in the feed -- so one query still spans every surface. Only the
    human sentence is wider than ``setup_patch``'s: that tool takes a key list
    and can honestly say a key may not be *set*, while a whole-document PUT can
    also delete one or reshape the block around it, and a refusal that named
    only setting would misdescribe two of the three.

    Ordering is ``setup_patch``'s, for the same reason: the durable record is
    written before the activity frame, so a refusal survives even when the feed
    is unreachable. Both reporting steps are guarded and the refusal itself sits
    outside the guards -- reporting is best-effort, refusing is not. An
    unwritable audit zone must degrade the trail, never turn a 403 into the 500
    an escaping exception would produce, which is the one shape an operator
    could mistake for a gate that failed open.

    The frame is stamped in-process through :func:`record_activity`, the same
    way the claude-setup refusals are: this handler already holds the request,
    so publishing the attempt needs no HTTP round trip back into the panel and
    therefore no panel token.

    Args:
        request: The incoming request, for the activity ring on ``app.state``.
        keys: Every protected key the request would have written, dotted for
            display. A PATCH lists them in the order the body sent them; a PUT
            lists them sorted, since a document diff has no request order.

    Returns:
        The :class:`HTTPException` the caller raises. Returned rather than
        raised so the refusal reads as a ``raise`` at the gate itself.
    """
    channel = RESERVED_PATH_CHANNELS[_CONFIG_FILE]
    # Every key is audited below; only the operator-facing text is capped.
    shown = keys[:_MAX_NAMED_KEYS]
    overflow = len(keys) - len(shown)
    named = ", ".join(f"`{key}`" for key in shown)
    if overflow:
        named += f" and {overflow} more"

    for key in keys:
        try:
            from osprey.services.python_executor.refusal_audit import record_protected_refusal

            record_protected_refusal(
                surface="http_config",
                target_file=_CONFIG_FILE,
                key_or_path=key,
                channel=channel,
                reason=_PROTECTED_KEY_REASON,
            )
        except Exception:  # noqa: BLE001 -- audit is best-effort; the refusal is not
            logger.warning("Could not record the protected-key refusal for audit", exc_info=True)

    # Keys only, never values: config values are secrets, and the activity ring
    # is persistent and served over HTTP. The wording is `setup_patch`'s, so one
    # phrase in the feed covers a refused config write whichever surface saw it.
    detail = f"BLOCKED a protected config key — {_CONFIG_FILE}: {', '.join(shown)}"
    if overflow:
        detail += f" and {overflow} more"
    try:
        record_activity(
            request, "config_patch_refused", {"kind": "config", "detail": detail[:1024]}
        )
    except Exception:  # noqa: BLE001 -- the feed is best-effort; the refusal is not
        logger.warning("Could not report the protected-key refusal to the feed", exc_info=True)

    subject = "is a protected key" if len(shown) == 1 and not overflow else "are protected keys"
    return HTTPException(
        status_code=403,
        detail=(
            f"{named} {subject} in {_CONFIG_FILE}: part of the safety surface this "
            f"agent runs under, so no agent-side writer may change it -- not by "
            f"setting it, not by deleting it, not by reshaping the block it lives in. "
            f"{_CONFIG_FILE} is unchanged -- no field in this request was applied. "
            f"The change belongs to {channel} -- make it there, then re-run "
            f"`osprey build` to re-render the project."
        ),
    )


def _as_document(parsed: Any) -> Mapping[str, Any]:
    """Coerce a parsed YAML document to the mapping the protected view walks.

    Anything that is not a mapping -- an empty file, a bare scalar, a top-level
    list -- carries no keys at all, so it reads as the empty document. That is
    the conservative reading in both directions: as the *incoming* document it
    holds no protected key, and every protected key the file currently has is
    therefore missing from it, which is exactly the deletion the diff refuses.

    Args:
        parsed: Whatever ``yaml.safe_load`` returned.

    Returns:
        The document as a mapping, or ``{}`` when it is not one.
    """
    return parsed if isinstance(parsed, Mapping) else {}


def _current_document(config_path: Path) -> Mapping[str, Any]:
    """The config.yml on disk, parsed, as the baseline a PUT is diffed against.

    Args:
        config_path: The project's ``config.yml``.

    Returns:
        The parsed document, or ``{}`` when the file does not exist yet -- a
        PUT that creates the file is then diffed against an empty baseline, so
        it may write anything unprotected and no protected key at all.

    Raises:
        HTTPException: 500 when the file exists but does not parse. The gate
            fails *closed*: without a readable baseline there is nothing to
            compare against, and a whole-document replace that cannot be
            checked is precisely the write that must not go through.
    """
    if not config_path.exists():
        return {}
    try:
        return _as_document(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    except yaml.YAMLError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"The current {_CONFIG_FILE} does not parse, so this replacement "
                f"cannot be checked against it: {e}. Repair the file on disk first "
                f"-- {_CONFIG_FILE} is unchanged."
            ),
        ) from e


def _changed_protected_keys(current: Mapping[str, Any], incoming: Mapping[str, Any]) -> list[str]:
    """Protected keys that differ between two whole documents, dotted for display.

    A writer that replaces a whole file has no patch to inspect, so it is judged
    by comparing :func:`~osprey.cli.profile_conventions.protected_view` of the
    old document against the same view of the new one. That single comparison
    covers every class of change at once, because the view is a flat mapping and
    mappings compare by keys *and* values: a key added, a key deleted, a value
    changed, a list widened, and a subtree reshaped either way (dict to scalar
    makes the flattened children vanish while the parent appears as a leaf;
    scalar to dict does the reverse). Anything outside the protected set is
    absent from both views, so an ordinary edit compares equal and is allowed.

    The view is keyed by *segment tuple*, never by a dotted string, and that is
    load-bearing rather than stylistic. A raw key may itself contain a ``.``, so
    a dotted view is not injective: a top-level key literally named
    ``"control_system.writes_enabled"`` renders to the same string as the real
    nested key, and a document that dropped the nested one while adding the flat
    one would compare *equal* to the original -- the gate would wave through the
    removal of the write gate. Tuples cannot collide that way. Segments are
    joined with ``.`` only here, at the edge, to name a key to an operator.

    Args:
        current: The document on disk.
        incoming: The document the request would write.

    Returns:
        Every changed protected key as a dotted path, sorted. Empty exactly when
        the two protected views are equal, which is the condition the PUT gate
        is specified in terms of -- ``dict.__eq__`` compares the same key set
        and the same per-key values this walk does.
    """
    before = protected_view(_CONFIG_FILE, current)
    after = protected_view(_CONFIG_FILE, incoming)
    return sorted(
        ".".join(key_path)
        for key_path in before.keys() | after.keys()
        if before.get(key_path, _ABSENT) != after.get(key_path, _ABSENT)
    )


class ConfigUpdate(BaseModel):
    raw: str


@router.put("/api/config")
async def put_config(body: ConfigUpdate, request: Request):
    """Validate YAML, back up config.yml, and write updated config.

    The Raw YAML view's write path: it replaces the whole document verbatim,
    comments and all. That makes it the widest write surface onto the file that
    carries the write gate, the approval gate, the agent's rendered permission
    surface and the paths the safety layers derive their allow and deny areas
    from -- and the only one that could change any of them without naming them,
    simply by handing over different bytes. So it is gated the way PATCH is, by
    the same protected set and with the same refusal, but on a document diff
    rather than a key list: the replacement must leave every protected key
    exactly as it found it. See :func:`_changed_protected_keys` for what that
    comparison catches and why it is done on segment tuples.

    The check sits between the parse and the *first* thing that touches disk --
    ahead of the backup, which is itself a write derived from a file this
    request may turn out not to be allowed to replace.

    An edit that touches only unprotected keys goes through untouched, and still
    re-renders the ``.claude/`` artifacts through :func:`_regen_after_write`. That
    matters more here than it reads: since the protected set took the render's
    inputs away from PATCH, this is the only surface left whose writes can move
    a rendered artifact at all.

    Note:
        One residual is known and deliberate. For a protected pattern that is
        *not* a trailing-wildcard family -- ``config.yml`` has three:
        ``artifacts.hooks``, ``simulation.state_dir`` and the channel-finder
        ``...feedback.store_path`` -- the flatten descends past the pattern's
        depth when the node there is a mapping, and the children come back
        unprotected. ``artifacts.hooks`` is covered anyway by the ``artifacts.*``
        family beside it. The other two both name a *path*, and neither can be
        made to point somewhere by planting a block there -- but for different
        reasons, and the difference is worth stating because only one of them is
        the tidy one:

        * ``simulation.state_dir`` is read through ``dotted_config_str``, which
          answers ``None`` for anything that is not a non-empty string. A block
          there reads as unset, so the runtime takes its default.
        * ``...feedback.store_path`` is read *raw* --
          ``server_context.py`` does ``feedback_config["store_path"]`` behind a
          truthiness guard that a non-empty mapping passes. It reaches
          ``resolve_cf_state_path``, ``Path(dict)`` raises ``TypeError``, and the
          ``except Exception`` around the initialization swallows it. The
          feedback store is therefore left *disabled*, not defaulted.

        Both outcomes are inert for this gate's purpose -- neither repoints a
        write at an attacker-named location, which is what the key is protected
        for -- and the second is a degradation of an optional store, logged at
        warning level. Neither ratchets, either: every transition that would put
        a real value back (mapping to scalar, absent to scalar) makes the
        protected leaf appear, which this diff refuses. Sealing such a node as a
        whole value cannot be done here without over-refusing: through the
        public matcher a node like ``services`` is indistinguishable from a
        genuine exact-depth key, being protected only as an *ancestor* of one,
        and sealing it would refuse every unprotected edit under ``services.*``.
        ``test_put_protected_families_are_descent_safe_or_known_inert`` is the
        tripwire: it fails if a fourth exact-depth pattern is ever added, so the
        judgement is re-made rather than inherited.
    """
    _require_config_panel(request)
    config_path: Path | None = request.app.state.config_path
    if not config_path:
        raise HTTPException(status_code=404, detail="No config.yml found")

    # Validate YAML parses cleanly
    try:
        parsed = yaml.safe_load(body.raw)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {e}") from e

    try:
        changed = _changed_protected_keys(_current_document(config_path), _as_document(parsed))
    except RecursionError as e:
        # A self-referential YAML anchor (``a: &x {b: *x}``) parses fine and then
        # walks forever. The gate already fails closed on it -- the walk raises
        # before anything is written -- so this only makes the refusal legible:
        # a 422 naming the document, rather than the bare 500 an escaping
        # RecursionError produces, which is the one shape an operator could
        # mistake for a gate that broke rather than one that held.
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid YAML: the document refers to itself, so it cannot be "
                f"checked against the protected set. {_CONFIG_FILE} is unchanged."
            ),
        ) from e
    if changed:
        raise _refuse_protected_keys(request, changed)

    # Backup existing config -- into the state zone, not beside the render.
    # See osprey.utils.config_writer.CONFIG_BACKUP_DIRNAME for why.
    if config_path.exists():
        backup_path = _backup_config(config_path)
        logger.info("Config backed up to %s", backup_path)

    # Write new config (fsync to ensure data is on disk before restart)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(body.raw)
        f.flush()
        os.fsync(f.fileno())
    logger.info("Config updated at %s", config_path)

    regenerated, detail = _regen_after_write(request)
    payload: dict[str, Any] = {
        "status": "ok",
        "requires_restart": True,
        "regenerated": regenerated,
    }
    # Only present when there is something to say. An always-present key would
    # make the panel decide when to show its banner; absence is the signal.
    if detail:
        payload["detail"] = detail
    return payload


class ConfigPatch(BaseModel):
    updates: dict[str, object]


@router.patch("/api/config")
async def patch_config(body: ConfigPatch, request: Request):
    """Apply structured field updates to config.yml, preserving comments.

    Accepts dot-notation keys (e.g. ``"channel_finder.mode": "hierarchical"``).
    Uses ruamel.yaml round-trip mode so comments, ordering, and formatting
    in the YAML file are retained.

    A key in the protected set is refused with 403 before anything on disk is
    touched. These are the keys that carry the write gate, the approval gate,
    the agent's rendered permission surface and the paths the safety layers
    derive their allow and deny areas from: a surface that can set them is a
    surface that can un-gate the agent, so this one may not, whoever is driving
    it. The refusal covers the whole request -- see :func:`_refuse_protected_keys`.
    """
    _require_config_panel(request)
    from osprey.utils.config_writer import config_update_fields

    config_path: Path | None = request.app.state.config_path
    if not config_path or not config_path.exists():
        raise HTTPException(status_code=404, detail="No config.yml found")

    if not body.updates:
        raise HTTPException(status_code=422, detail="No updates provided")

    # The protected set is consulted here -- before the backup, which is itself a
    # write derived from a file this request may turn out not to be allowed to
    # touch. Matching is dotted because that is exactly how these updates are
    # applied: `config_update_fields` splits each key on `.`, so what is matched
    # is precisely what would be written, and the ancestor rule covers a body
    # that aims at a parent block to reach a protected leaf beneath it.
    #
    # All-or-nothing, and checked across the whole body first: a PATCH is one
    # request, so one protected key refuses all of it rather than letting the
    # cosmetic half land and reporting the rest as an error. A partial apply
    # would leave the operator to work out which half took effect.
    protected = [key for key in body.updates if is_protected_key(_CONFIG_FILE, key)]
    if protected:
        raise _refuse_protected_keys(request, protected)

    # Backup before mutation -- into the state zone, not beside the render.
    backup_path = _backup_config(config_path)

    try:
        config_update_fields(config_path, body.updates)
    except Exception as e:
        # Restore backup on failure
        shutil.copy2(backup_path, config_path)
        logger.error("Config patch failed, restored backup: %s", e)
        raise HTTPException(status_code=500, detail=f"Config update failed: {e}") from e

    logger.info("Config patched (%d fields) at %s", len(body.updates), config_path)
    regenerated, detail = _regen_after_write(request)
    payload: dict[str, Any] = {
        "status": "ok",
        "requires_restart": True,
        "fields_updated": len(body.updates),
        "regenerated": regenerated,
    }
    if detail:
        payload["detail"] = detail
    return payload


# ---- Hook Debug Endpoints ---- #


@router.get("/api/hooks/debug-status")
async def get_hook_debug_status(request: Request):
    """Return current hooks.debug state from config.yml."""
    config_path: Path | None = request.app.state.config_path
    if not config_path or not config_path.exists():
        return {"enabled": False}

    try:
        full_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        enabled = bool(full_config.get("hooks", {}).get("debug", False))
        return {"enabled": enabled}
    except Exception:
        return {"enabled": False}


@router.get("/api/hooks/debug-log")
async def get_hook_debug_log(request: Request, limit: int = 50):
    """Return recent hook debug log entries from hook_debug.jsonl."""
    import json

    project_cwd = getattr(request.app.state, "project_cwd", None)
    if not project_cwd:
        return {"entries": []}

    log_path = Path(project_cwd) / ".claude" / "hooks" / "hook_debug.jsonl"
    if not log_path.exists():
        return {"entries": []}

    try:
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        # Take the last N entries (most recent)
        recent = lines[-limit:] if len(lines) > limit else lines
        entries = []
        for line in reversed(recent):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return {"entries": entries}
    except Exception:
        logger.warning("Failed to read hook debug log", exc_info=True)
        return {"entries": []}


# ---- Claude Setup Endpoints ---- #


class ClaudeSetupSaveRequest(BaseModel):
    path: str
    content: str


@router.get("/api/claude-setup")
async def get_claude_setup(request: Request):
    """Read all Claude Code integration files from the project directory.

    Each file carries ``read_only``; the panel-level ``notice`` says why some
    of them are, so an operator meets that fact while reading rather than only
    when a save comes back 403.
    """
    _require_config_panel(request)
    service = ClaudeCodeFileService(Path(request.app.state.project_cwd))
    return {"files": service.list_files(), "notice": PROFILE_EDIT_NOTICE}


@router.put("/api/claude-setup")
async def save_claude_setup(request: Request, body: ClaudeSetupSaveRequest):
    """Save an existing Claude Code file.

    A write into the protected set is refused with 403 and the owning channel
    in the detail, and is *published* as agent activity: an attempt to rewrite
    the framework that constrains the agent is exactly the kind of move a
    watching operator should see happen, not something they have to go and
    find in the audit log afterwards.
    """
    _require_config_panel(request)
    service = ClaudeCodeFileService(Path(request.app.state.project_cwd))
    try:
        return service.write_file(body.path, body.content)
    except ProtectedWriteError as e:
        raise _refuse_protected_write(request, e) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _refuse_protected_write(request: Request, exc: ProtectedWriteError) -> HTTPException:
    """Publish a refused Claude-setup write as agent activity, and build its 403.

    The PUT and the POST answer a protected-set refusal identically on purpose
    -- dropping a new file into a reserved subtree is the same move as
    rewriting one already there, so it must not be the quieter of the two.
    Returning the 403 rather than raising it keeps both handlers spelling the
    refusal as one statement, so publishing it and answering it cannot come
    apart.

    Args:
        request: Incoming request, carrying ``app.state``.
        exc: The refusal, carrying the path and the channel that owns it.

    Returns:
        The 403 the caller raises.
    """
    record_activity(
        request,
        "claude_setup_refused",
        {"kind": "config", "detail": f"{exc.rel_path}: {exc.channel}"[:1024]},
    )
    return HTTPException(status_code=403, detail=str(exc))


@router.post("/api/claude-setup")
async def create_claude_setup(request: Request, body: ClaudeSetupSaveRequest):
    """Create a new Claude Code file in an allowed .claude/ subdirectory.

    A creation inside the protected set is surfaced exactly as the PUT surfaces
    a refused rewrite -- 403 naming the owning channel, and published as agent
    activity. Dropping a new file into a reserved subtree is the same move as
    rewriting one already there, so it must not be the quieter of the two.
    """
    _require_config_panel(request)
    service = ClaudeCodeFileService(Path(request.app.state.project_cwd))
    try:
        return service.create_file(body.path, body.content)
    except ProtectedWriteError as e:
        raise _refuse_protected_write(request, e) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
