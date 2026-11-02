#!/usr/bin/env python3
"""
---
name: Watchdog Trigger
description: Hands each operator prompt to the deployment's watchdogs without delaying the prompt
summary: Every prompt an operator submits is judged by the enabled watchdogs, in a detached process the agent never waits for
event: UserPromptSubmit
---

## Flow

```
hook fires ──► OSPREY_WATCHDOG_SKIP or OSPREY_DISPATCH_RUN set? ──YES──► exit 0, silent
                    │
                    NO
                    ▼
               read stdin JSON ──► no prompt or no session id ──► exit 0, silent
                    │
                    ▼
               mint submission_id, write the 0600 submission file
                    │
                    ▼
               one detached child of this file ──► exit 0, silent
                    │
                    ▼
child:         OSPREY_WEB_PORT and OSPREY_SESSION_ID both set?
                    │                     │
                   YES                    NO
                    ▼                     ▼
               POST /api/watchdog/prompt  record entry, record-only
                    │         │
                    2xx      any failure
                    │         ▼
                    │        record entry --session-key
                    ▼         ▼
               the submission file is gone
```

## Details

A watchdog judges what an operator typed. Headless runs carry
``OSPREY_WATCHDOG_SKIP`` in their environment and dispatch jobs carry
``OSPREY_DISPATCH_RUN``; both prompts come from code, so neither is judged. Every
other session is watched: a web terminal, ``osprey chat``, and a bare ``claude``
in a built project.

The hook runs on the operator's prompt, so the parent does only the cheap part
and imports the rest inside the functions that need it. The child holds no pipe
of the parent's, because Claude Code waits for the hook's stdout to close.

``submission_id`` names one submission, so an identical prompt typed twice is
judged twice. A retry or a fallback carries the same id, which is what lets the
server count one submission once when a delayed answer and a fallback both land.

In a web session, any failure of the POST falls back to the record entry with
the session key, so an unreachable or flooded intake cannot leave the session
unheld. The submission file is unlinked by whichever process consumes it last;
the child's final unlink covers a record entry that could not run.

The POST carries ``Authorization: Bearer <OSPREY_PANEL_TOKEN>`` when the token
holds a non-blank value. The call is written here because this file declares no
sibling in the build catalog and must work shipped alone. The opener ignores
proxies: the call stays on loopback, and neither the prompt nor the token may leave
the machine because the host sets ``http_proxy``.

The record entry needs the osprey package, and it runs under this interpreter:
the render substitutes the project's Python into every hook's command. It runs
with ``-P`` because the working directory is normally the project, where a watched
agent can create files; a package planted there must not replace the entry. The
flag, unlike ``PYTHONSAFEPATH``, stops at that one process, so hooks under any
agent the entry starts still import their peers from their own directory.

stdlib-only — must NOT import osprey, yaml, requests, or any third-party lib, and
must stay 3.9-safe. Fails open on any error: writes nothing to stdout and exits 0.
"""

import json
import os
import sys

#: Set by the headless entry points in ``osprey.agent_runner``.
SKIP_ENV_VAR = "OSPREY_WATCHDOG_SKIP"

#: The mark of a dispatch job. Mirrors ``posture_store.DISPATCH_RUN_ENV_VAR``.
DISPATCH_RUN_ENV_VAR = "OSPREY_DISPATCH_RUN"

#: The argument that makes a run of this file the detached child.
_CHILD_FLAG = "--child"

#: The prefix of a submission file, for anyone reading the temp directory.
_SUBMISSION_PREFIX = "osprey_watchdog_"

_INTAKE_ROUTE = "/api/watchdog/prompt"

#: Seconds the POST may take. Long enough for a loopback call to a busy server;
#: a slower answer falls back to the record entry.
_POST_TIMEOUT = 2


def _env(name):
    """Return the stripped value of variable *name*, blank when unset."""
    return (os.environ.get(name) or "").strip()


def _unlink(path):
    """Remove *path* if it is still there."""
    import contextlib

    with contextlib.suppress(OSError):
        os.unlink(path)


def _read_prompt():
    """Return ``(prompt, record_session_id)`` from stdin, or ``None``."""
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        return None
    prompt = payload.get("prompt")
    session_id = payload.get("session_id")
    if not isinstance(prompt, str) or not prompt:
        return None
    if not isinstance(session_id, str) or not session_id:
        return None
    return prompt, session_id


def _write_submission(prompt, record_session_id):
    """Write one submission to a fresh 0600 file and return its path."""
    import tempfile
    import uuid

    submission = {
        "session_key": _env("OSPREY_SESSION_ID") or None,
        "record_session_id": record_session_id,
        "submission_id": str(uuid.uuid4()),
        "prompt": prompt,
    }
    descriptor, path = tempfile.mkstemp(prefix=_SUBMISSION_PREFIX, suffix=".json")
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(submission, stream)
    return path


def _spawn_child(path):
    """Start the detached child that owns the submission at *path*."""
    import subprocess

    subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), _CHILD_FLAG, path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _post_submission(path, port):
    """Return whether intake accepted the submission at *path*."""
    try:
        import urllib.request

        with open(path, "rb") as stream:
            body = stream.read()
        headers = {"Content-Type": "application/json"}
        token = _env("OSPREY_PANEL_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{_INTAKE_ROUTE}",
            data=body,
            headers=headers,
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        opener.open(request, timeout=_POST_TIMEOUT).close()
    except Exception:
        return False
    return True


def _run_record_entry(path, session_key):
    """Run the record entry over the submission at *path* and wait for it."""
    import subprocess

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    argv = [
        sys.executable,
        "-P",
        "-m",
        "osprey.services.watchdog",
        "record",
        "--project-dir",
        project_dir,
        "--submission-file",
        path,
    ]
    if session_key:
        argv += ["--session-key", session_key]
    subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _run_child(path):
    """Deliver the submission at *path*, then make sure the file is gone."""
    try:
        port = _env("OSPREY_WEB_PORT")
        session_key = _env("OSPREY_SESSION_ID")
        if not port or not session_key:
            _run_record_entry(path, None)
        elif not _post_submission(path, port):
            _run_record_entry(path, session_key)
    finally:
        _unlink(path)


def _run_hook():
    """Hand the prompt on stdin to a detached child, unless it is not an operator's."""
    if os.environ.get(SKIP_ENV_VAR) or os.environ.get(DISPATCH_RUN_ENV_VAR):
        return
    submission = _read_prompt()
    if submission is None:
        return
    path = _write_submission(*submission)
    try:
        _spawn_child(path)
    except Exception:
        _unlink(path)


def main(argv=None):
    """Run the hook, or the child when *argv* asks for it. Always returns 0."""
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) == 2 and args[0] == _CHILD_FLAG:
            _run_child(args[1])
        else:
            _run_hook()
    except Exception:
        # Last-resort fail-open: a watchdog must never block a prompt.
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
