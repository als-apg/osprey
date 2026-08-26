"""Protected-set refusals, filed on the unified ledger.

Five writers guard the framework the agent runs under — the ``setup_patch``
tool, the config routes, the Claude-setup panel, the scaffold gallery and the
container-start restore — and each of them can refuse a write aimed at a
protected key or a reserved path. They all refuse for the same reason and an
operator reads them together, so they file through one funnel rather than five,
exactly as they did when the answer was ``protected-writes.jsonl``. What
changed is where a record lands: :mod:`osprey.audit.writer` routes it to
``var/audit/<identity>/<surface>.jsonl``, so a refusal is filed under the
identity that made it and can be joined with every other decision that identity
took, instead of into one deployment-wide file no per-user mount could isolate.

**The surface is the file name**, which is why the five spellings live here as
constants rather than as a literal at each site: the name is what an operator
greps for and what the per-surface ledger is called, and two writers that
disagree about it would split one question across two files.

**Claiming the call.** Four of the five refuse the operation the caller asked
for: the tool raises, the routes answer 403. Each of those runs *inside* an
outer audit layer (the MCP middleware, or ``HttpAuditMiddleware``) that would
otherwise file a second record for the same decision, so they record through
:func:`~osprey.audit.dedup.record_and_mark` and the outer layer defers — see
:mod:`osprey.audit.dedup` for why the marker carries the decision itself. The
fifth does not: the container-start restore *skips* one poisoned store record
and lets the request or the startup walk continue, so it is not the decision
its caller is awaiting, and claiming it would suppress the outer layer's
perfectly correct record of an operation that did happen. That site passes
``claim=False``.

The marker is only ever seen by a layer that awaited this call on the same
task; a recorder reached through a worker thread (a ``def`` route, a sync tool
body) is invisible to it, which is why every route reaching the claiming sites
is ``async def``. :mod:`osprey.audit.dedup` states the rule; the retirement
handoff lists which route reaches which site.

**Nothing here may cost the refusal it records.** The writer swallows its own
errors, and the two lazy imports below are wrapped for the same reason: an
unwritable audit zone, or an import that failed, must degrade the trail rather
than turn a 403 into a 500 that reads like the gate malfunctioned.
"""

from __future__ import annotations

from pathlib import Path

from osprey.audit import posture
from osprey.utils.logger import get_logger

logger = get_logger("protected_write_audit")

__all__ = [
    "MAX_CHANNEL_CHARS",
    "PROTECTED_SURFACES",
    "SURFACE_CLAUDE_SETUP",
    "SURFACE_HTTP_CONFIG",
    "SURFACE_SCAFFOLD_GALLERY",
    "SURFACE_SCAFFOLD_RESTORE",
    "SURFACE_SETUP_PATCH",
    "record_protected_refusal",
]

#: The ``setup_patch`` MCP tool, refusing a protected key in a patchable file.
SURFACE_SETUP_PATCH = "setup_patch"

#: ``PUT``/``PATCH /api/config``, refusing the protected keys a body would set.
SURFACE_HTTP_CONFIG = "http_config"

#: The Claude-setup panel, refusing a save aimed at a profile-owned file.
SURFACE_CLAUDE_SETUP = "claude_setup"

#: The scaffold gallery, refusing a write or delete inside the protected set.
SURFACE_SCAFFOLD_GALLERY = "scaffold_gallery"

#: The ownership-store restore, refusing to install one stored body. Its own
#: surface rather than the gallery's: the two arrive by different routes and at
#: different privilege levels, and the restore's are the ones worth waking up
#: for.
SURFACE_SCAFFOLD_RESTORE = "scaffold_restore"

#: Every surface that files here. Documentation and a test's roster, not a
#: gate — the writer accepts any surface, and a sixth writer should be added
#: here rather than kept out of the vocabulary an operator reads.
PROTECTED_SURFACES: tuple[str, ...] = (
    SURFACE_SETUP_PATCH,
    SURFACE_HTTP_CONFIG,
    SURFACE_CLAUDE_SETUP,
    SURFACE_SCAFFOLD_GALLERY,
    SURFACE_SCAFFOLD_RESTORE,
)

#: The posture vocabulary and its readers are :mod:`osprey.audit.posture`'s;
#: the names are re-exported here because ``tests/audit/test_two_key_fixture.py``
#: pins them through this module.
POSTURE_ENV_VAR = posture.POSTURE_ENV_VAR
POSTURE_SOURCE_ENV_VAR = posture.POSTURE_SOURCE_ENV_VAR
POSTURE_SESSION_ENV_VAR = posture.POSTURE_SESSION_ENV_VAR
SANDBOX_MODE = posture.SANDBOX_MODE
POSTURE_SANDBOX = posture.POSTURE_SANDBOX
POSTURE_WRITES = posture.POSTURE_WRITES

#: Characters of the owning channel kept inside ``detail``. The channel is a
#: human phrase ("the build profile this project was rendered from"), and the
#: bound is here rather than left to the envelope's silent ``detail``
#: truncation so that a long channel cannot push the target file out of the
#: record it shares.
MAX_CHANNEL_CHARS = 200

#: Where the refused target is named inside ``detail``. The envelope's
#: top-level ``subject`` is what was protected — the dotted key, or the path
#: when the whole file is the target — so the file the write was aimed at and
#: the channel that owns it ride along beside it.
TARGET_KEY = "target"
CHANNEL_KEY = "channel"


def _detail(target_file: str, channel: str) -> str:
    """The supplementary context one refusal carries: what, and whose.

    ``target=`` first so that the bounded channel — the field that can be a
    sentence — cannot crowd it out.
    """
    channel_text = channel[:MAX_CHANNEL_CHARS] if channel else ""
    return f"{TARGET_KEY}={target_file} {CHANNEL_KEY}={channel_text}"


def record_protected_refusal(
    *,
    surface: str,
    target_file: str,
    key_or_path: str,
    channel: str,
    reason: str,
    claim: bool = True,
    posture_source: str | None = None,
) -> Path | None:
    """File one protected-set refusal. Returns the ledger path, or ``None``.

    Args:
        surface: Which writer refused — one of :data:`PROTECTED_SURFACES`.
        target_file: The file the write was aimed at, as the caller names it.
        key_or_path: What inside it was protected — a dotted config key, or the
            project-relative path when the whole file is the target. This is
            the record's ``subject``.
        channel: The channel that owns the target, named the same way the
            refusal message names it, so the record and the message agree.
        reason: Short machine-ish reason (``protected_key``, ``reserved path``).
        claim: Whether this refusal *is* the decision on the call an outer
            audit layer is awaiting. ``True`` (the default) marks it, so the
            MCP middleware or ``HttpAuditMiddleware`` defers instead of filing
            a second record for the same decision. Pass ``False`` where the
            caller carries on afterwards — the refusal is then about something
            the operation skipped, not about the operation, and the outer
            layer's own record is still true.
        posture_source: How the posture this refusal was decided under was
            established, when the *surface* knows better than the environment
            does. The HTTP surfaces pass
            :data:`~osprey.audit.envelope.POSTURE_SOURCE_APP`: the Web Terminal
            server is nobody's session child, so the env ladder would call a
            refused request a bare process while ``HttpAuditMiddleware`` files
            ``app`` for the same request. Left ``None`` — the MCP tools and the
            container-start restore — the ladder answers, because those really
            do run under whatever posture their process inherited. Not a
            per-surface constant inside this funnel: ``scaffold_restore``
            arrives from the entrypoint and ``setup_patch`` from MCP, so only
            the caller knows.

    Returns:
        The path the record landed in, or ``None`` when it was not durably
        stored. Callers do not branch on it; a test can tell "wrote" from "gave
        up quietly" without reading the directory.

    Never raises: see the module docstring.
    """
    try:
        from osprey.audit import dedup, writer
        from osprey.audit.envelope import DECISION_REFUSED

        fields = {
            "surface": surface,
            "posture": posture.posture(),
            "posture_source": posture.posture_source(posture_source),
            "session": posture.posture_session(),
            "subject": key_or_path,
            "detail": _detail(target_file, channel),
        }
        if claim:
            return dedup.record_and_mark(decision=DECISION_REFUSED, reason=reason, **fields)
        return writer.record(decision=DECISION_REFUSED, reason=reason, **fields)
    except Exception:
        logger.warning("Could not record a protected-write refusal", exc_info=True)
        return None
