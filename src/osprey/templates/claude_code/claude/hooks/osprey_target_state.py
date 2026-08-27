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

Write posture (a second question, answered from config)
-------------------------------------------------------
Hooks that gate writes need one more answer that identity alone cannot give:
does this deployment ARM writes for the target a call names? That is a config
question, and its authority is ``osprey_connectors.types`` —
:func:`type_writes_enabled` and :func:`target_writes_enabled`. Hooks cannot
import it, so :func:`writes_posture`, :func:`session_types` and
:func:`most_restrictive_posture` restate it here in stdlib terms, once, for
every hook that asks: two hooks mirroring the same rules separately is two ways
for one deployment to be described.

The mirrored rules, on the ``control_system:`` section:

* ``connector.<type>.writes_enabled`` is a tri-state. Absent — no connector
  table, no block for the type, a block that is not a mapping, or one without
  the leaf — inherits ``control_system.writes_enabled``. Literally ``True``
  arms. Any other value leaves writes unarmed and does NOT fall back to the
  deployment-wide key;
* ``<type>`` is one whole key, never a path: a custom connector's dotted module
  path names a single block;
* ``va`` resolves to the virtual accelerator; ``live`` resolves to the section's
  own type when that type is not simulated, else to the single non-simulated
  key under ``connector``. Zero or more than one is underivable, and an
  underivable target answers the deployment-wide key.

A caller holding no target of its own asks a prior question: which targets can a
session on THIS deployment reach at all? :func:`session_types` answers it,
restating ``osprey_connectors.types.session_posture`` — both targets only where
the deployment renders the target switch, and otherwise the single type
``control_system.type`` builds, read by TYPE under the baseline target that
names it. Iterating the two target names instead would answer for a machine no
session here ever reaches: a mock deployment carrying one ``epics`` block
resolves ``live`` to that block, while the connector the runtime built is the
mock.

What this module adds on top of the framework's booleans is a THIRD state:
``None``, for a section that expresses no posture at all — no deployment-wide
key and no per-type key anywhere. That is the shape every deployment had before
the per-type key existed, and a hook must leave it exactly as it found it rather
than reading silence as a refusal.

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

#: The two session targets, spelled as the state file and the config spell them.
TARGET_LIVE = "live"
TARGET_VA = "va"

#: Connector types that serve a machine nobody has to be careful around, and the
#: type ``resolve_control_system_type`` falls back to when a section names none.
#: They are why ``live`` cannot simply be "whatever the config selects": a
#: deployment whose baseline is one of these has not said what its real machine
#: is. Literals rather than an import, like everything else in this module.
MOCK_TYPE = "mock"
VIRTUAL_ACCELERATOR_TYPE = "virtual_accelerator"
SIMULATED_TYPES = (MOCK_TYPE, VIRTUAL_ACCELERATOR_TYPE)

#: The write-posture key, as a LEAF: it is looked up both directly on the
#: ``control_system:`` section and inside one already-resolved connector block,
#: whose own key is the connector type in full.
WRITES_ENABLED_LEAF = "writes_enabled"

__all__ = [
    "FALLBACK_BASELINE",
    "MAX_ANCESTOR_HOPS",
    "MOCK_TYPE",
    "REASON_AMBIGUOUS",
    "REASON_NO_STATE",
    "REASON_UNREADABLE",
    "SIMULATED_TYPES",
    "STATE_DIR_NAME",
    "STATE_FILE_GLOB",
    "STATE_FILE_PREFIX",
    "STATE_FILE_SUFFIX",
    "TARGET_LIVE",
    "TARGET_VA",
    "VIRTUAL_ACCELERATOR_TYPE",
    "WRITES_ENABLED_LEAF",
    "ancestor_pids",
    "baseline_result",
    "is_baseline",
    "most_restrictive_posture",
    "parent_pid",
    "read_session_record",
    "read_session_target",
    "read_state_file",
    "resolve_state_dir",
    "selected_target",
    "session_types",
    "target_metadata",
    "target_type",
    "type_posture",
    "writes_posture",
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


# -- write posture ---------------------------------------------------------


def _resolved_type(section):
    """The connector type ``control_system.type`` selects — the mock when absent.

    The factory's documented fail-closed default, restated: a missing section, a
    section that is not a mapping, and a bare ``type:`` (which YAML gives as
    ``None``) all name no type, and a deployment that named none gets the mock.
    """
    declared = section.get("type") if isinstance(section, dict) else None
    return str(declared) if declared else MOCK_TYPE


def _live_type(section):
    """The connector type that reaches this deployment's real machine, or ``None``.

    ``None`` wherever the framework's own ``_live_type`` raises: a section whose
    declared type is simulated and whose connector table holds no single
    non-simulated block has never said what ``live`` means here, and there is
    nothing to infer it from. A section that is not a mapping resolves to the
    mock, which is the factory's documented fail-closed default.
    """
    declared = _resolved_type(section)
    if declared not in SIMULATED_TYPES:
        return declared

    connector = section.get("connector") if isinstance(section, dict) else None
    if not isinstance(connector, dict):
        return None
    candidates = [key for key in connector if isinstance(key, str) and key not in SIMULATED_TYPES]
    return candidates[0] if len(candidates) == 1 else None


def target_type(section, target):
    """The connector type *target* selects, or ``None`` when it selects none.

    An unknown target and an underivable ``live`` are the same answer for the
    same reason: there is no per-type block to consult because there is no type.

    Public because a refusal has to NAME the block its answer came from, and a
    hook re-deriving the mapping to spell that key is a hook whose message can
    drift away from its own decision.
    """
    if target == TARGET_VA:
        return VIRTUAL_ACCELERATOR_TYPE
    if target == TARGET_LIVE:
        return _live_type(section)
    return None


def _baseline_target(section):
    """The target *section* describes when nobody has switched."""
    return TARGET_VA if _resolved_type(section) == VIRTUAL_ACCELERATOR_TYPE else TARGET_LIVE


def _switch_capable(section):
    """Whether a session on this deployment can be pointed at either target.

    The stdlib restatement of ``osprey_connectors.types.switch_capable``, whose
    three conditions are mirrored here in order: both targets resolve to a type
    at all; the deployment's OWN type is what its baseline target resolves back
    to, which is what keeps a mock that happens to carry an ``epics`` block out
    of the two-target world; and both types carry a non-empty connector block,
    since that block is what a connector is configured from.
    """
    if not isinstance(section, dict):
        return False
    types_by_target = {target: target_type(section, target) for target in (TARGET_LIVE, TARGET_VA)}
    if any(name is None for name in types_by_target.values()):
        return False
    if types_by_target[_baseline_target(section)] != _resolved_type(section):
        return False
    connector = section.get("connector")
    if not isinstance(connector, dict):
        return False
    return all(
        isinstance(connector.get(name), dict) and bool(connector.get(name))
        for name in types_by_target.values()
    )


def session_types(section):
    """``{target: connector type}`` for the targets a session here can REACH.

    The stdlib restatement of ``osprey_connectors.types.session_posture``'s
    reachable-target rule, and what a caller with no target of its own iterates
    instead of the two target names. Both targets on a deployment that renders
    the switch; otherwise the one type ``control_system.type`` builds, under the
    baseline target that names it — by type on purpose, because ``live`` is the
    switch's own derivation and without the switch it can name a machine the
    built connector is not.

    Never raises: every section is at minimum one baseline target holding the
    mock, which is what the factory would build from it.
    """
    if _switch_capable(section):
        return {target: target_type(section, target) for target in (TARGET_LIVE, TARGET_VA)}
    return {_baseline_target(section): _resolved_type(section)}


def _global_posture(section):
    """``control_system.writes_enabled`` alone — explicitly ``True`` or nothing."""
    return isinstance(section, dict) and section.get(WRITES_ENABLED_LEAF) is True


def _states_posture(section):
    """Whether *section* says anything at all about write posture."""
    if not isinstance(section, dict):
        return False
    if WRITES_ENABLED_LEAF in section:
        return True
    connector = section.get("connector")
    if not isinstance(connector, dict):
        return False
    return any(
        isinstance(block, dict) and WRITES_ENABLED_LEAF in block for block in connector.values()
    )


def type_posture(section, connector_type):
    """Whether *section* arms writes for one connector TYPE. Never raises.

    :func:`writes_posture` below the target-to-type step, for the callers that
    already hold a type: :func:`session_types` hands out types, and a refusal
    naming a key must name the block the answer was read from.

    ``True`` or ``False`` only. The third state belongs to
    :func:`writes_posture`, because a section that states no posture anywhere
    states none for any type and the distinction is not a per-type one.
    """
    connector = section.get("connector") if isinstance(section, dict) else None
    block = connector.get(connector_type) if isinstance(connector, dict) else None
    if not isinstance(block, dict) or WRITES_ENABLED_LEAF not in block:
        return _global_posture(section)
    return block[WRITES_ENABLED_LEAF] is True


def writes_posture(section, target):
    """Whether *section* arms writes for one session *target*. Never raises.

    ``True`` armed, ``False`` not armed, and ``None`` for a section that states
    no posture anywhere — see the module docstring for why silence is its own
    answer rather than a refusal, and for the rules the ``True``/``False`` half
    mirrors from ``osprey_connectors.types``.

    Args:
        section: The ``control_system:`` config section. A caller holding a
            whole rendered config passes ``config.get("control_system")``.
        target: The session target, ``"live"`` or ``"va"``.
    """
    if not _states_posture(section):
        return None
    connector_type = target_type(section, target)
    if connector_type is None:
        return _global_posture(section)
    return type_posture(section, connector_type)


def most_restrictive_posture(section):
    """:func:`type_posture` ANDed over the REACHABLE targets. Never raises.

    The answer for a caller that could not identify which target a call would
    act on. Armed only where every target a session here could be pointed at is
    armed, so an unidentifiable call on a deployment that armed one of two is
    treated as the unarmed one — a guess between them could be a guess in
    favour of hardware.

    The set ANDed over is :func:`session_types` rather than the two target
    names. Without the switch there is only one target to be uncertain between,
    and ANDing in a ``live`` no session here can select would leave a
    simulator-armed deployment unarmed on the strength of a machine it does not
    have — while telling the operator to flip a key that is deliberately false.

    ``None`` when the section states no posture at all, which is target-blind
    and therefore the same ``None`` :func:`writes_posture` returns.
    """
    if not _states_posture(section):
        return None
    return all(
        type_posture(section, connector_type) is True
        for connector_type in session_types(section).values()
    )
