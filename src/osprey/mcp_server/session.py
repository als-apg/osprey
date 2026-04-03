"""Session metadata collection for logbook entries."""

import logging

logger = logging.getLogger("osprey.mcp_server.session")


def gather_session_metadata(created_via: str) -> dict:
    """Collect session metadata for logbook entries.

    Returns a dict with 8 fields, all with graceful ``None`` fallback:
    ``session_id``, ``transcript_path``, ``session_start_time``,
    ``git_branch``, ``git_commit_short``, ``operator``, ``model_name``,
    ``created_via``.

    Args:
        created_via: Caller identifier (e.g. ``"ariel-mcp"``, ``"gallery-compose"``).
    """
    import json as _json
    import os
    import subprocess

    from osprey.utils.workspace import resolve_workspace_root

    meta: dict = {"created_via": created_via}

    # --- Transcript-derived fields ---
    session_id: str | None = None
    transcript_path: str | None = None
    session_start_time: str | None = None
    try:
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        project_dir = resolve_workspace_root().parent
        reader = TranscriptReader(project_dir)
        current = reader.find_current_transcript()
        if current is not None:
            transcript_path = str(current)
            # Read first few lines to extract sessionId and timestamp
            with open(current) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    if session_id is None and entry.get("sessionId"):
                        session_id = entry["sessionId"]
                    if session_start_time is None and entry.get("timestamp"):
                        session_start_time = entry["timestamp"]
                    if session_id is not None and session_start_time is not None:
                        break
    except Exception as exc:
        logger.warning("Session metadata: transcript read failed (non-fatal): %s", exc)

    # Fallback for session_id
    if session_id is None:
        session_id = os.environ.get("OSPREY_SESSION_ID")

    meta["session_id"] = session_id
    meta["transcript_path"] = transcript_path
    meta["session_start_time"] = session_start_time

    # --- Git fields ---
    git_branch: str | None = None
    git_commit_short: str | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            git_branch = result.stdout.strip() or None
    except Exception as exc:
        logger.warning("Session metadata: git branch failed (non-fatal): %s", exc)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            git_commit_short = result.stdout.strip() or None
    except Exception as exc:
        logger.warning("Session metadata: git commit failed (non-fatal): %s", exc)

    meta["git_branch"] = git_branch
    meta["git_commit_short"] = git_commit_short

    # --- Operator ---
    operator: str | None = None
    try:
        operator = os.environ.get("USER") or os.getlogin()
    except Exception as exc:
        logger.warning("Session metadata: operator lookup failed (non-fatal): %s", exc)
    meta["operator"] = operator

    # --- Model name ---
    model_name: str | None = None
    try:
        project_dir_for_settings = resolve_workspace_root().parent
        settings_path = project_dir_for_settings / ".claude" / "settings.json"
        if settings_path.exists():
            with open(settings_path) as fh:
                settings = _json.load(fh)
            model_name = settings.get("model")
    except Exception as exc:
        logger.warning("Session metadata: settings.json read failed (non-fatal): %s", exc)
    if model_name is None:
        model_name = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("CLAUDE_MODEL")
    meta["model_name"] = model_name

    return meta
