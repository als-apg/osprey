"""Shared, stdlib-only reader for the control-system target-state file.

Not a hook: this is the frontmatter-less library that hooks and the status line
import to answer one question — *which control-system target is this session
pointed at?* The controls MCP server is the single writer
(``osprey.mcp_server.control_system.target_state``); everything here reads, and
reads read-only. Stale files are the writer's to sweep; a reader that deleted
them would be a second opinion about identity.

Hooks run outside the osprey venv, so every path through this module is standard
library only — no ``osprey`` import is required to succeed, no PyYAML, no third
party. The one optional ``osprey`` import (the agent-data base dir) is wrapped
and falls back to the literal the path contract fixes.

Path contract
-------------
Restated from the writer's docstring, in stdlib terms::

    <repo_root>/var/agent_data/control_target/target_state_<server_pid>.json

* ``repo_root`` comes from :func:`osprey_hook_log.get_repo_root` — the repo, not
  the render: ``build/`` is disposable and ``data/`` is checksummed.
* the base dir is the framework DEFAULT ``var/agent_data``. A project that
  overrides ``agent_data.base_dir`` moves the directory somewhere this reader
  does not look; the reader then reports the baseline fallback, which is the
  documented fail-closed outcome rather than a wrong target.
* one file per server process, discovered by the glob ``target_state_*.json``.

Multi-session resolution (CC-3, fail-closed)
--------------------------------------------
Two Claude Code sessions can share one checkout, so the directory can hold two
live state files that disagree. The tiebreak is parentage: each record carries
the ``owner_ppid`` the server captured at start — the Claude Code process that
spawned it. This reader walks its own ancestor PID chain and selects the unique
live file whose ``owner_ppid`` is on that chain.

Zero matches or more than one both resolve to the **deployment baseline**
fallback. Ambiguity is not broken by guessing: naming the wrong target is worse
than naming none, because the caller renders "deployment baseline (state
unavailable)" and the operator knows to look, whereas a confidently wrong
``Target:`` line is a safety claim nobody audits.

Return contract (shared with the status-line consumer)
------------------------------------------------------
:func:`read_session_target` returns a dict whose five keys are ALWAYS present::

    {
      "target": "va" | "live" | None,
      "generation": int | None,
      "display": {"label": str, "endpoint": str, "real_machine": bool} | None,
      "fallback": None | "baseline",
      "reason": None | "no state" | "ambiguous" | "unreadable",
    }

``fallback`` is the explicit sentinel: falsy on success, ``"baseline"`` when the
caller must render the deployment baseline. Callers branch on it and never on a
missing key. There is no third outcome and no exception path — every failure
mode (absent directory, dead server, corrupt JSON, schema drift, an unwalkable
process tree) arrives as the same baseline marker, differing only in ``reason``.

:func:`read_session_record` is the same call with the projection left off: it
hands back the writer's RAW record for callers that need metadata the contract
dict cannot carry — the approval prompt previewing the DESTINATION of a
prospective switch, which is by definition not the selected target. Both go
through :func:`_select_record`, so the fail-closed selection rules exist once.
A caller that walked the state directory itself would eventually walk it
differently; the liveness filter is the easiest of these rules to leave out, and
leaving it out means a crashed server's stale file answering for a live one.

::

    read_session_target() / read_session_record()
        |
        v
    _select_record()
        |
        v
    resolve state dir --(missing)--> baseline / "no state"
        |
        v
    glob target_state_*.json
        |
        +--> dead server_pid   --> ignore (never delete)
        +--> unreadable/corrupt --> ignore, remember
        |
        v
    ancestor pid chain (/proc/<pid>/stat, else `ps -o ppid= -p`)
        |
        v
    owner_ppid on chain?  --0--> baseline / "no state" | "unreadable"
        |                 --2+-> baseline / "ambiguous"
        |
        1
        v
    {target, generation, display}

This module never writes to stdout and never raises into a hook.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import get_repo_root  # noqa: E402

# The framework DEFAULT agent-data root, imported rather than spelled out here so
# the two cannot drift apart. The fallback covers the ordinary case — a hook
# running with osprey off the path — where the literal from the path contract is
# the right answer, not a guess.
try:
    from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR as _AGENT_DATA_BASE_DIR
except Exception:  # pragma: no cover - hooks must never crash the agent
    _AGENT_DATA_BASE_DIR = "var/agent_data"

#: Fixed subdirectory of the agent-data root. Mirrors ``STATE_DIR_NAME`` on the
#: writer; part of the greppable path contract.
STATE_DIR_NAME = "control_target"

STATE_FILE_PREFIX = "target_state_"
STATE_FILE_SUFFIX = ".json"
STATE_FILE_GLOB = f"{STATE_FILE_PREFIX}*{STATE_FILE_SUFFIX}"

#: Value of the ``fallback`` key when the caller must render the deployment
#: baseline. Falsy (``None``) means the record was resolved.
FALLBACK_BASELINE = "baseline"

#: The three ``reason`` values that accompany :data:`FALLBACK_BASELINE`.
REASON_NO_STATE = "no state"
REASON_AMBIGUOUS = "ambiguous"
REASON_UNREADABLE = "unreadable"

#: Bound on the ancestor walk. A process tree deeper than this, or one with a
#: cycle, is pathological; stopping is a "no match", which is fail-closed.
MAX_ANCESTOR_HOPS = 64

#: Seconds to wait for the ``ps`` fallback. A hook that blocked on a wedged
#: process table would stall the agent, so the wait is short and a timeout is
#: simply the end of the chain.
PS_TIMEOUT_S = 5

__all__ = [
    "FALLBACK_BASELINE",
    "MAX_ANCESTOR_HOPS",
    "REASON_AMBIGUOUS",
    "REASON_NO_STATE",
    "REASON_UNREADABLE",
    "STATE_DIR_NAME",
    "STATE_FILE_GLOB",
    "STATE_FILE_PREFIX",
    "STATE_FILE_SUFFIX",
    "ancestor_pids",
    "baseline_result",
    "is_baseline",
    "parent_pid",
    "read_session_record",
    "read_session_target",
    "read_state_file",
    "resolve_state_dir",
    "selected_target",
    "target_metadata",
]


# -- result construction ---------------------------------------------------


def baseline_result(reason=REASON_NO_STATE):
    """The explicit baseline-fallback marker, with every contract key present.

    Callers render "Target: deployment baseline (state unavailable)" from this.
    ``reason`` is advisory detail for a debug line, never a second signal: the
    only thing a caller must branch on is ``fallback``.
    """
    return {
        "target": None,
        "generation": None,
        "display": None,
        "fallback": FALLBACK_BASELINE,
        "reason": reason,
    }


def is_baseline(result):
    """Whether *result* is the baseline fallback rather than a resolved target.

    Tolerant of a caller handing back anything at all, because the contract's
    whole point is that no reader of this module has to defend itself.
    """
    if not isinstance(result, dict):
        return True
    return bool(result.get("fallback"))


# -- paths -----------------------------------------------------------------


def resolve_state_dir(hook_input=None):
    """Directory holding every server's state file, or ``None`` if unresolvable.

    Not created here — a reader that created state directories would leave
    litter in every repo a hook ever ran in.
    """
    try:
        repo_root = get_repo_root(hook_input)
        if not repo_root:
            return None
        return os.path.join(repo_root, _AGENT_DATA_BASE_DIR, STATE_DIR_NAME)
    except Exception:  # pragma: no cover - defensive; get_repo_root is total
        return None


def _pid_from_filename(name):
    """PID encoded in a state file's name, or ``None`` if it is not a number."""
    if not name.startswith(STATE_FILE_PREFIX) or not name.endswith(STATE_FILE_SUFFIX):
        return None
    stem = name[len(STATE_FILE_PREFIX) : -len(STATE_FILE_SUFFIX)]
    try:
        return int(stem)
    except ValueError:
        return None


def _list_state_files(directory):
    """State-file paths in *directory*, sorted. Empty on any filesystem trouble."""
    try:
        names = sorted(
            n
            for n in os.listdir(directory)
            if n.startswith(STATE_FILE_PREFIX) and n.endswith(STATE_FILE_SUFFIX)
        )
    except OSError:
        return []
    return [os.path.join(directory, n) for n in names]


# -- liveness --------------------------------------------------------------


def _is_process_alive(pid):
    """Whether *pid* names a running process.

    ``os.kill(pid, 0)`` sends no signal and only asks the kernel whether the
    process exists. ``PermissionError`` means it exists but belongs to another
    user, so it counts as ALIVE — treating an unreachable owner as dead would
    discard live state. Non-positive PIDs address process *groups* and are
    rejected without calling ``os.kill`` at all.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:  # pragma: no cover - platform oddity; assume alive
        return True
    return True


# -- ancestor pid chain ----------------------------------------------------


def _ppid_from_proc(pid):
    """Parent PID from ``/proc/<pid>/stat`` (Linux), or ``None``.

    The ``comm`` field is parenthesized and may itself contain spaces and
    parentheses, so the fields are taken after the LAST ``)``: they begin with
    ``state`` and ``ppid``.
    """
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8", errors="replace") as handle:
            data = handle.read()
    except (OSError, TypeError, ValueError):
        return None
    try:
        fields = data[data.rindex(")") + 1 :].split()
        return int(fields[1])
    except (ValueError, IndexError):
        return None


def _ppid_from_ps(pid):
    """Parent PID from ``ps -o ppid= -p <pid>`` (macOS and any POSIX), or ``None``.

    ``ps`` is the only portable answer where ``/proc`` does not exist. It is
    stdlib-reachable through :mod:`subprocess`, bounded by
    :data:`PS_TIMEOUT_S`, and every failure — missing binary, non-zero exit,
    timeout, unparseable output — is simply "no parent".
    """
    try:
        completed = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(int(pid))],
            capture_output=True,
            text=True,
            timeout=PS_TIMEOUT_S,
            check=False,
        )
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        return int((completed.stdout or "").strip())
    except (AttributeError, ValueError):
        return None


def parent_pid(pid):
    """Parent of *pid*, or ``None`` when the chain cannot be walked further.

    ``/proc`` first because it is a file read rather than a process spawn; the
    ``ps`` fallback carries macOS, where ``/proc`` does not exist and the first
    attempt fails immediately and cheaply.
    """
    ppid = _ppid_from_proc(pid)
    if ppid is None:
        ppid = _ppid_from_ps(pid)
    return ppid


def ancestor_pids(start_pid=None, max_hops=MAX_ANCESTOR_HOPS):
    """This process's ancestor PID chain, nearest first.

    The chain INCLUDES *start_pid* itself: a server re-parented onto the hook's
    own process would still be this session's, and including it costs nothing
    because a PID cannot be both this process and another session's parent.

    The walk stops at PID 1, at ``max_hops``, on a repeat (a cycle can only come
    from a lying process table), or the moment a parent cannot be determined.
    Any failure mid-walk simply ends the chain — a short chain yields no match,
    which is the fail-closed answer.
    """
    try:
        current = os.getpid() if start_pid is None else int(start_pid)
    except (TypeError, ValueError):
        return []

    chain = []
    for _ in range(max(0, int(max_hops))):
        if current <= 1:
            break
        chain.append(current)
        parent = parent_pid(current)
        if parent is None or parent <= 1 or parent in chain:
            break
        current = parent
    return chain


# -- record reading --------------------------------------------------------


def read_state_file(path):
    """Load one state file, or ``None`` if absent, unreadable, or corrupt.

    Never raises. ``ValueError`` covers ``JSONDecodeError`` and
    ``UnicodeDecodeError`` alike; a non-dict payload is corruption too.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _coerce_display(record, target):
    """Display metadata for *target*, degraded rather than missing.

    Schema-tolerant by construction: a record written by an older or newer
    writer, or one whose ``targets`` mapping lacks the selected target, still
    yields the three keys a caller renders. An empty ``label`` degrades to the
    target NAME — ``va`` and ``live`` are truthful minimal labels — so the
    rendered line never has a blank where an identity belongs.
    """
    targets = record.get("targets")
    meta = targets.get(target) if isinstance(targets, dict) else None
    if not isinstance(meta, dict):
        meta = {}
    return {
        "label": str(meta.get("label") or "") or target,
        "endpoint": str(meta.get("endpoint") or ""),
        "real_machine": bool(meta.get("real_machine", False)),
    }


def selected_target(record):
    """The usable ``target`` string on *record*, or ``None`` if it has none.

    A record whose ``target`` is absent or empty is corruption in the one field
    that cannot be defaulted: there is no safe guess between ``live`` and
    ``va``. Exported because a caller holding a raw record (see
    :func:`read_session_record`) has to answer the same question the same way.
    """
    if not isinstance(record, dict):
        return None
    target = record.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def target_metadata(record, target):
    """The RAW per-target metadata mapping on *record*, or ``None``.

    The counterpart to :func:`_coerce_display` for callers that must tell a
    metadata key that is ABSENT from one that is present and false — schema
    drift from an older or newer writer, where a coerced default would state a
    machine identity the record never claimed. Returns the writer's own mapping,
    untouched; ``None`` when there is none for *target*.
    """
    if not isinstance(record, dict):
        return None
    targets = record.get("targets")
    if not isinstance(targets, dict):
        return None
    meta = targets.get(target)
    return meta if isinstance(meta, dict) else None


def _resolve_record(record, target):
    """Project one selected record onto the success half of the contract.

    ``generation`` is soft — a missing or unparseable one degrades to ``0``
    rather than discarding an otherwise good target.
    """
    try:
        generation = int(record.get("generation", 0))
    except (TypeError, ValueError):
        generation = 0

    return {
        "target": target,
        "generation": generation,
        "display": _coerce_display(record, target),
        "fallback": None,
        "reason": None,
    }


# -- public entry point ----------------------------------------------------


def read_session_target(hook_input=None):
    """Resolve this session's control-system target. Never raises.

    A thin projection of :func:`read_session_record` onto the shared contract
    dict — a resolved ``{target, generation, display}`` or the explicit baseline
    fallback. The selection itself lives in one place (:func:`_select_record`)
    so no caller can end up with a differently-selected record; see the module
    docstring for the contract and the fail-closed rule.

    Args:
        hook_input: The parsed hook stdin payload, used only to resolve the repo
            root. Optional, because hooks that read no stdin still need an
            answer.

    Returns:
        dict: keys ``target``, ``generation``, ``display``, ``fallback``,
        ``reason`` — always all five.
    """
    try:
        record, target, reason = _select_record(hook_input)
        if record is None:
            return baseline_result(reason)
        return _resolve_record(record, target)
    except Exception:
        # The contract's last line of defence: a hook that raised here would
        # take the agent's turn down over a status line.
        return baseline_result(REASON_UNREADABLE)


def read_session_record(hook_input=None):
    """The RAW state record this session resolves to, or ``None``. Never raises.

    The same selection as :func:`read_session_target`, because it is literally
    the same call — identical liveness, parentage, ambiguity and corrupt-target
    rules. What differs is what comes back: the writer's own record, so a caller
    can read metadata the projected contract dict cannot carry.

    That caller is the approval prompt, which previews the DESTINATION of a
    prospective target switch: destination metadata is by definition not the
    selected target's ``display``. Handing the record out here is what keeps the
    fail-closed selection rules in one place — a caller that re-walked the
    directory itself would sooner or later walk it differently (skipping the
    liveness filter, say, and preferring a crashed server's stale file).

    ``None`` for every baseline outcome. A caller that must name the reason asks
    :func:`read_session_target` for it.
    """
    try:
        record, _target, _reason = _select_record(hook_input)
        return record
    except Exception:
        return None


def _select_record(hook_input):
    """Select this session's state record: ``(record, target, reason)``.

    The single authority on WHICH record answers for this session. Exactly one
    of ``record`` and ``reason`` is set: on success ``(record, target, None)``
    with *target* already validated non-empty, on every failure
    ``(None, None, <reason>)``.
    """
    directory = resolve_state_dir(hook_input)
    if not directory:
        return None, None, REASON_NO_STATE

    paths = _list_state_files(directory)
    if not paths:
        return None, None, REASON_NO_STATE

    live_records = []
    saw_unreadable = False
    for path in paths:
        pid = _pid_from_filename(os.path.basename(path))
        if pid is not None and not _is_process_alive(pid):
            # Dead owner: ignore the file, never delete it. Sweeping belongs to
            # the writer; this reader is read-only by design.
            continue
        record = read_state_file(path)
        if record is None:
            saw_unreadable = True
            continue
        if pid is None and not _is_process_alive(record.get("server_pid")):
            continue
        live_records.append(record)

    if not live_records:
        return None, None, (REASON_UNREADABLE if saw_unreadable else REASON_NO_STATE)

    chain = set(ancestor_pids())
    matches = []
    for record in live_records:
        try:
            owner_ppid = int(record.get("owner_ppid"))
        except (TypeError, ValueError):
            continue
        if owner_ppid in chain:
            matches.append(record)

    if len(matches) > 1:
        return None, None, REASON_AMBIGUOUS
    if not matches:
        return None, None, (REASON_UNREADABLE if saw_unreadable else REASON_NO_STATE)

    record = matches[0]
    target = selected_target(record)
    if target is None:
        return None, None, REASON_UNREADABLE
    return record, target, None
