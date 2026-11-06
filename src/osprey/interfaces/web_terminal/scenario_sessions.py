"""Lay a deployment's seeded scenario sessions down into the hub's own stores.

A deploy that activates scenarios holding a ``sessions`` block writes their
resolved conversations to ``scenario-sessions.json`` in this identity's state
directory (:func:`osprey.simulation.apply.write_scenario_sessions`). The deploy
cannot write the stores those conversations belong in: transcripts, the session
ledger, the session meta, the transcript map and the artifact store are this
hub's, sited by its own app state. So the hub lays them down itself, once, at
startup, right after the ledger backfill and before it serves a request.

The lay-down only fills what is absent, which makes it safe to repeat: a key
interrupted part-way is completed by the next startup without duplicating a
row or an artifact, and a transcript the operator has since extended is never
rewritten. Nothing is ever overwritten or deleted but the seed file itself,
which goes once every key in it is complete.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.interfaces.web_terminal import session_meta, transcript_map
from osprey.stores import session_ledger
from osprey_connectors.safety_file import atomic_replace

if TYPE_CHECKING:
    from osprey.stores.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

__all__ = ["LayDownReport", "lay_down_scenario_sessions", "seed_file_path"]

#: Keys of a seed-file session that :func:`from_dict` does not read.
_SESSION_EXTRA_KEYS = ("scenario", "key", "artifacts")

#: The CLI version authored lines claim when the deployment pins none.
_UNKNOWN_CLI_VERSION = "unknown"


@dataclass(frozen=True)
class LayDownReport:
    """What one lay-down did.

    Attributes:
        laid: Keys every store now holds completely.
        incomplete: Keys a step failed for; the seed file stays for them.
        seed_removed: Whether the seed file was deleted.
    """

    laid: tuple[str, ...] = field(default=())
    incomplete: tuple[str, ...] = field(default=())
    seed_removed: bool = False


def seed_file_path() -> Path | None:
    """Where this identity's seed file lives, or ``None`` when no state dir resolves."""
    from osprey.simulation.apply import SCENARIO_SESSIONS_FILENAME

    state_dir = session_meta.ledger_state_dir()
    return None if state_dir is None else state_dir / SCENARIO_SESSIONS_FILENAME


def lay_down_scenario_sessions(app, store: ArtifactStore | None = None) -> LayDownReport:
    """Fill the hub's stores from the seed file, then delete it once all is complete.

    For each seeded session (key K): author each part's transcript into the
    project's transcript directory when the file is absent, append one ledger
    row per part when the ledger holds none for K, mark each part's meta as
    seeded when unmarked, point K at its last part's transcript when it has
    several and the map holds nothing for K, and save each artifact the store
    does not already hold under its stated id.

    Args:
        app: The hub; ``app.state`` supplies the project cwd, the store root
            (``panel_status_root``), the config path, the meta and the map.
        store: The artifact store to save into; defaults to the one under
            ``app.state.panel_status_root``, built without a gallery launch.

    Returns:
        What was laid down. Never raises: every failure is logged and leaves the
        seed file for the next startup.
    """
    try:
        return _lay_down(app, store)
    except Exception:
        logger.warning("The scenario-session lay-down failed", exc_info=True)
        return LayDownReport()


def _lay_down(app, store: ArtifactStore | None) -> LayDownReport:
    path = seed_file_path()
    if path is None or not path.exists():
        return LayDownReport()
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        logger.warning("The scenario-session seed file %s could not be read: %s", path, exc)
        return LayDownReport()
    sessions = document.get("sessions") if isinstance(document, dict) else None
    if not isinstance(sessions, list):
        logger.warning("The scenario-session seed file %s holds no sessions list", path)
        return LayDownReport()

    state_dir = path.parent
    if store is None:
        root = getattr(app.state, "panel_status_root", None)
        if root is not None:
            from osprey.stores.artifact_store import ArtifactStore

            store = ArtifactStore(workspace_root=Path(root), auto_launch=False)
    cli_version = _cli_version(app)

    laid: list[str] = []
    incomplete: list[str] = []
    for index, raw in enumerate(sessions):
        key = raw.get("key") if isinstance(raw, dict) else None
        label = key if isinstance(key, str) and key else f"sessions[{index}]"
        try:
            _lay_down_session(app, raw, state_dir, store, cli_version)
        except Exception:
            logger.warning(
                "Scenario session %s from %s was not laid down completely",
                label,
                path,
                exc_info=True,
            )
            incomplete.append(label)
        else:
            laid.append(label)

    removed = False
    if not incomplete:
        try:
            path.unlink(missing_ok=True)
            removed = True
        except OSError:
            logger.warning("The laid-down seed file %s could not be removed", path, exc_info=True)
    if laid:
        logger.info("Laid down %d scenario session(s) from %s", len(laid), path)
    return LayDownReport(laid=tuple(laid), incomplete=tuple(incomplete), seed_removed=removed)


def _cli_version(app) -> str:
    """The agent CLI version authored lines claim: the deployment's pin, else ``unknown``."""
    config_path = getattr(app.state, "config_path", None)
    if config_path is None:
        return _UNKNOWN_CLI_VERSION
    try:
        from osprey.agent_runner.launcher import pinned_cli_version
        from osprey.interfaces.web_terminal.app import _load_claude_code_config

        return pinned_cli_version(_load_claude_code_config(config_path)) or _UNKNOWN_CLI_VERSION
    except Exception:
        logger.warning("The agent CLI pin could not be read", exc_info=True)
        return _UNKNOWN_CLI_VERSION


def _lay_down_session(
    app,
    raw: dict[str, Any],
    state_dir: Path,
    store: ArtifactStore | None,
    cli_version: str,
) -> None:
    """Fill every store for one seeded session; raise when any step did not land."""
    from osprey_connectors.simulation.sessions import from_dict

    key = raw["key"]
    scenario = raw["scenario"]
    if not isinstance(key, str) or not key:
        raise ValueError(f"a seeded session needs a key, got {key!r}")
    parts_raw = raw["parts"]
    transcript_ids = [part["transcript_id"] for part in parts_raw]
    # Each id names a file under the transcript dir: only a canonical UUID may.
    for tid in transcript_ids:
        if not isinstance(tid, str) or str(uuid.UUID(tid)) != tid:
            raise ValueError(f"session {key}: transcript id {tid!r} is not a canonical UUID")
    plain = {k: v for k, v in raw.items() if k not in _SESSION_EXTRA_KEYS}
    plain["parts"] = [{k: v for k, v in p.items() if k != "transcript_id"} for p in parts_raw]
    resolved = from_dict(plain, prefix=f"session {key}")
    parts = list(zip(resolved.parts, transcript_ids, strict=True))

    # 1. Transcripts, each written only when absent.
    cwd = getattr(app.state, "project_cwd", None)
    if not cwd:
        raise RuntimeError("the hub has no project directory to write transcripts for")
    from osprey.agent_runner.transcript_author import author_transcript
    from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

    sessions_dir = SessionDiscovery(cwd)._resolve_sessions_dir()
    for part, tid in parts:
        target = sessions_dir / f"{tid}.jsonl"
        if target.exists():
            continue
        lines = author_transcript(part, session_id=tid, cwd=str(cwd), cli_version=cli_version)
        _write_transcript(target, lines)

    # 2. Ledger rows, one per part, only when the ledger holds none for the key.
    held_keys = session_ledger.load(state_dir).keys()
    if key not in held_keys:
        rows = [(key, tid, part.turns[0].at if part.turns else resolved.at) for part, tid in parts]
        if not session_ledger.append_many(state_dir, rows):
            raise RuntimeError(f"the ledger rows of {key} could not be written")

    # 3. Meta, marked only when unmarked.
    for part, tid in parts:
        if session_meta.get(app, tid).get("scenario") is None:
            title = part.part.title or resolved.session.title
            session_meta.mark_seeded(app, tid, scenario, title)

    # 4. The map, for a key that moved past its first transcript.
    if len(transcript_ids) > 1 and key not in transcript_map._mapping(app):
        transcript_map.set(app, key, transcript_ids[-1])

    # 5. Artifacts, each saved only when its stated id is not yet held.
    artifacts = raw.get("artifacts") or []
    if not artifacts:
        return
    if store is None:
        raise RuntimeError(f"no artifact store to save the artifacts of {key} into")
    for artifact in artifacts:
        if store.get_entry(artifact["entry_id"]) is not None:
            continue
        store.save_file(
            base64.b64decode(artifact["content_b64"]),
            artifact["filename"],
            artifact["artifact_type"],
            artifact["title"],
            mime_type=artifact["mime_type"],
            tool_source=artifact["tool_source"],
            metadata={"transcript_id": artifact["transcript_id"], "scenario": scenario},
            run_id="",
            origin="",
            session_id=key,
            kind=artifact.get("kind", ""),
            at=datetime.fromisoformat(artifact["at"]),
            entry_id=artifact["entry_id"],
        )


def _write_transcript(target: Path, lines: list[dict[str, Any]]) -> None:
    """Write *lines* to *target* whole, its mtime set to the last line's time.

    Owner-only, as the agent CLI keeps its own transcripts.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(line) + "\n" for line in lines).encode("utf-8")
    atomic_replace(target, data, mode=0o600, sync="best_effort")
    stamps = [line["timestamp"] for line in lines if isinstance(line.get("timestamp"), str)]
    if stamps:
        moment = datetime.fromisoformat(stamps[-1]).timestamp()
        os.utime(target, (moment, moment))
