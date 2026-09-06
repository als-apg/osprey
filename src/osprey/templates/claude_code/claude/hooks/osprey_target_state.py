"""Shared, stdlib-only reader for the deployment's control-context record.

Not a hook: this is the frontmatter-less library that the hooks import to
answer two questions — *which control-system target is this deployment pointed
at?* and *may a write reach it?* The record is written by whichever process
owns it (the web terminal, else a controls MCP server); everything here reads,
and reads read-only. Stale files are the owner's to sweep; a reader that
deleted them would be a second opinion about identity.

Hooks run outside the osprey venv, so every path through this module is standard
library only — no ``osprey`` import is required to succeed, no PyYAML, no third
party. The one optional ``osprey`` import (the agent-data base dir) is wrapped
and falls back to the literal the path contract fixes.

Path contract
-------------
Restated from the writer's docstring, in stdlib terms::

    <agent_data_root>/control_target/control_context.json
    <agent_data_root>/control_target/server_<server_pid>.json

* ``agent_data_root`` resolves by ONE rule, the same one the writer and the
  connector-side reader use: the ``OSPREY_AGENT_DATA_ROOT`` stamp when it names
  a non-blank path, else ``<repo_root>/var/agent_data``;
* ``repo_root`` comes from :func:`osprey_hook_log.get_repo_root` — the repo, not
  the render: ``build/`` is disposable and ``data/`` is checksummed;
* the DERIVED base dir is the framework default ``var/agent_data``. A project
  that overrides ``agent_data.base_dir`` and does not stamp the root moves the
  directory somewhere this reader does not look; the reader then reports the
  baseline fallback, which is the documented fail-closed outcome rather than a
  wrong target — and :func:`posture_unknown` turns that same silence into a
  refusal for every process the stamp did not reach;
* ONE record for the whole deployment, and one report per controls server,
  discovered by the glob ``server_*.json``.

There is no per-session record and no ancestor walk. The target, its generation
and the operator's narrowings are properties of the DEPLOYMENT: two Claude Code
sessions sharing a checkout read one file and get one answer, and a hook that
tried to tell them apart would be inventing a distinction the writer does not
make.

Record contract
---------------
Restated from ``osprey_connectors.control_context``, whose parser this one
mirrors — a filter here that admitted a record that one rejects would let a
hook answer for a machine no other reader believes in::

    {"schema": 1, "target": "live"|"va"|"standin", "generation": int >= 0,
     "posture": {target: "sandbox"}, "owner": {...}|null, "last_switch": {...}|null}

``schema``, ``target`` and ``generation`` are the record's IDENTITY: any of
them absent, mistyped or unrecognised and there is no record at all, because
every one of them is a field no reader can default. ``posture`` degrades on its
own to "nothing narrowed". ``owner`` and ``last_switch`` are deliberately not
carried out of the parse: which process may WRITE the record is the owner's
business, and hooks never evaluate convergence — a hook decides about the tool
call in front of it, at the generation the record states, and a switch landing
mid-call is the executor's and the kernel's to catch.

Return contract
---------------
:func:`read_target` returns a dict whose four keys are ALWAYS present::

    {
      "target": "va" | "live" | "standin" | None,
      "generation": int | None,
      "fallback": None | "baseline",
      "reason": None | "no state" | "unreadable",
    }

``fallback`` is the explicit sentinel: falsy on success, ``"baseline"`` when the
caller must render the deployment baseline. Callers branch on it and never on a
missing key. There is no third outcome and no exception path — every failure
mode (absent directory, absent record, corrupt JSON, schema drift) arrives as
the same baseline marker, differing only in ``reason``.

How a target is SPOKEN OF
-------------------------
The record names which machine the deployment points at; it does not carry the
label, the endpoint or the ``real_machine`` claim a prompt renders. Those are
the writer's own rendering of the deployment's config, published by every
controls server into its report (``targets``), and :func:`read_target_view`
folds them onto the record so a prompt describes one read of one target rather
than assembling an identity of its own. Reports are read from the LIVE servers
only, the rule every other reader of that directory follows, and any of them
answers because they all render one config. With no live server there is no
metadata and the caller renders its explicit "state unavailable" line — which
is what it rendered before this record existed, for the same reason: nobody is
there to say what the target IS.

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
* ``va`` resolves to the virtual accelerator and ``standin`` to the live
  stand-in — the same answer on every deployment, because those are machines a
  deployment stands up for itself. ``live`` resolves to the section's own type
  when that type is neither simulated nor a stand-in, else to the single such
  key under ``connector``. Zero or more than one is underivable, and an
  underivable target answers the deployment-wide key.

A caller holding no target of its own asks a prior question: which targets can a
session on THIS deployment reach at all? :func:`session_types` answers it,
restating ``osprey_connectors.types.session_posture`` — the deployment's
CONFIGURED targets where it renders the target switch, and otherwise the single
type ``control_system.type`` builds, read by TYPE under the baseline target that
names it. Iterating the target vocabulary instead would answer for a machine no
session here ever reaches: a mock deployment carrying one ``epics`` block
resolves ``live`` to that block, while the connector the runtime built is the
mock, and a deployment with no ``live_standin`` block would grow a ``standin``
slot for a soft IOC nobody stood up.

What this module adds on top of the framework's booleans is a THIRD state:
``None``, for a section that expresses no posture at all — no deployment-wide
key and no per-type key anywhere. That is the shape every deployment had before
the per-type key existed, and a hook must leave it exactly as it found it rather
than reading silence as a refusal.

Recorded posture (a third question, answered from the record)
-------------------------------------------------------------
An operator narrows one control target from the control-target chip in the
header: "stand-in is read-only, leave the virtual accelerator alone". That
narrowing is the record's ``posture`` field and is enforced at WRITE time
rather than delivered by respawning the agent, which is what lets a flip land
on a session already mid-conversation.

``osprey_connectors.session_store`` is the canonical reader. Hooks cannot import
it, so its rules are restated here — a change there is a change here.

**1. Whose narrowing it is.** The deployment's. It is keyed by TARGET and by
nothing else: an operator taking the ring away takes it away from every agent
on this deployment, and a hook that asked whether the narrowing was addressed
to *its own* session would let a session the operator never saw write to the
machine they just closed.

**2. What the shapes mean.** ``posture`` is either a per-target object
(``{"live": "sandbox"}``) or the bare string ``"sandbox"``, which narrows every
target in :data:`CONTROL_TARGETS`. Bare ``"writes"`` is dropped — the writes
posture is the *absence* of a narrowing, never a stored assertion, so nothing
in this field can ever widen. Anything else — an unknown value, a non-string
key, a leaf nobody recognises — is dropped rather than honoured: what survives
this filter decides whether a real machine is written to, so a hand-edited or
future-version entry must not reach the decision.

**3. How a lookup combines.** :func:`effective_writes_for` is the whole rule::

    ceiling AND not readonly run AND (recorded posture != sandbox)

The ceiling is the deployment's own posture, read through this module's own
predicates and never re-derived: :func:`writes_posture` for the target the
record states, and :func:`most_restrictive_posture` when it states none. That
second half is the one place this restatement is deliberately STRICTER than the
module it restates, which takes the union across configured targets for a caller
that holds no target at all — right for a roster describing a deployment, wrong
for a gate, which must not be handed the more permissive of two answers it
cannot choose between. The record can only narrow the ceiling. With no target
resolved the MOST RESTRICTIVE entry wins: any sandbox refuses.

**4. When a change is seen.** Every read re-reads the file. The canonical reader
caches its parse against ``(st_mtime_ns, st_size, st_ino)`` because it lives in
a long-running process; a hook is a fresh process per tool call, where reading
is already the freshest answer there is. A missing record is not an empty
posture, though — see :func:`posture_unknown`.

This module never writes to stdout and never raises into a hook.
"""

from __future__ import annotations

import json
import os
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

#: The record's filename, mirroring ``control_context.RECORD_FILENAME``.
RECORD_FILENAME = "control_context.json"

#: The payload version this reader understands. There is exactly one: hooks and
#: record change together, and a deployment is regenerated rather than migrated,
#: so a payload spelling anything else is not this record.
SCHEMA_VERSION = 1

#: One report per controls server, beside the record. Read for the per-target
#: display metadata only — the identity itself is the record's.
REPORT_FILE_PREFIX = "server_"
REPORT_FILE_SUFFIX = ".json"
REPORT_FILE_GLOB = f"{REPORT_FILE_PREFIX}*{REPORT_FILE_SUFFIX}"

#: Value of the ``fallback`` key when the caller must render the deployment
#: baseline. Falsy (``None``) means the record was resolved.
FALLBACK_BASELINE = "baseline"

#: The two ``reason`` values that accompany :data:`FALLBACK_BASELINE`. There is
#: no ambiguity value: one deployment has one record, and a reader that could
#: not decide which of two files answered belonged to the pid-keyed model this
#: one replaces.
REASON_NO_STATE = "no state"
REASON_UNREADABLE = "unreadable"

#: The three control targets, spelled as the record and the config spell them.
#: A target names a MACHINE — the facility's own, the virtual accelerator, the
#: stand-in soft IOC a deployment runs for itself — and not a connector type.
TARGET_LIVE = "live"
TARGET_VA = "va"
TARGET_STANDIN = "standin"

#: The target vocabulary, in the framework's order. The machines that CAN exist,
#: never the ones a given deployment has: what a session here can reach is
#: :func:`session_types`, and a caller looping this constant would hand every
#: deployment a slot for every machine anybody could run.
CONTROL_TARGETS = [TARGET_LIVE, TARGET_VA, TARGET_STANDIN]

#: Connector types that serve a machine nobody has to be careful around, and the
#: type ``resolve_control_system_type`` falls back to when a section names none.
#: They are why ``live`` cannot simply be "whatever the config selects": a
#: deployment whose baseline is one of these has not said what its real machine
#: is. Literals rather than an import, like everything else in this module.
MOCK_TYPE = "mock"
VIRTUAL_ACCELERATOR_TYPE = "virtual_accelerator"
SIMULATED_TYPES = (MOCK_TYPE, VIRTUAL_ACCELERATOR_TYPE)

#: The live stand-in's own connector type, and the tuple of types that serve it.
#: Served by the EPICS connector but keyed apart from ``epics``, so that the
#: facility's authored block stays the one thing ``live`` can mean. Reachable
#: only through the ``standin`` target: a stand-in is a machine in its own
#: right, never a candidate for a deployment's live one.
LIVE_STANDIN_TYPE = "live_standin"
STANDIN_TYPES = (LIVE_STANDIN_TYPE,)

#: The target each self-standing machine's type is the baseline of. A type
#: absent from this table describes the facility's own machine, hence ``live``.
_BASELINE_TARGETS = {
    VIRTUAL_ACCELERATOR_TYPE: TARGET_VA,
    LIVE_STANDIN_TYPE: TARGET_STANDIN,
}

#: The write-posture key, as a LEAF: it is looked up both directly on the
#: ``control_system:`` section and inside one already-resolved connector block,
#: whose own key is the connector type in full.
WRITES_ENABLED_LEAF = "writes_enabled"

#: The agent-data root stamp a session child carries. Read by NAME rather than
#: imported from ``osprey.audit.posture``, which declares it for the stamping
#: side: hooks are stdlib-only and must not grow an ``osprey`` import to learn
#: one string. Its absence is half of :func:`posture_unknown`.
AGENT_DATA_ROOT_ENV_VAR = "OSPREY_AGENT_DATA_ROOT"

#: This session's audit id. It indexes nothing any more — the posture it used to
#: key is the deployment's — and it is read only by the write-approval stamp,
#: which records WHO approved beside what was approved.
POSTURE_SESSION_ENV_VAR = "OSPREY_POSTURE_SESSION"

#: The session-wide execution mode, and the one value of it that sandboxes.
#: A value comparison, never a presence check: ``readwrite`` is the writes
#: posture of a readwrite execution, and a presence check would sandbox on it.
EXECUTION_MODE_ENV_VAR = "OSPREY_EXECUTION_MODE"
SANDBOX_MODE = "readonly"

#: The narrowing value — the only one that ever refuses anything — and the
#: un-narrowed one, recorded by some writers and meaningful to none: it is
#: dropped on parse so the record holds narrowings and nothing else.
POSTURE_SANDBOX = "sandbox"
POSTURE_WRITES = "writes"

#: The two values a posture entry may spell. Everything else is dropped.
VALID_POSTURES = (POSTURE_SANDBOX, POSTURE_WRITES)

__all__ = [
    "AGENT_DATA_ROOT_ENV_VAR",
    "CONTROL_TARGETS",
    "EXECUTION_MODE_ENV_VAR",
    "FALLBACK_BASELINE",
    "LIVE_STANDIN_TYPE",
    "MOCK_TYPE",
    "POSTURE_SANDBOX",
    "POSTURE_SESSION_ENV_VAR",
    "POSTURE_WRITES",
    "REASON_NO_STATE",
    "REASON_UNREADABLE",
    "RECORD_FILENAME",
    "REPORT_FILE_GLOB",
    "REPORT_FILE_PREFIX",
    "REPORT_FILE_SUFFIX",
    "SANDBOX_MODE",
    "SCHEMA_VERSION",
    "SIMULATED_TYPES",
    "STANDIN_TYPES",
    "STATE_DIR_NAME",
    "TARGET_LIVE",
    "TARGET_STANDIN",
    "TARGET_VA",
    "VALID_POSTURES",
    "VIRTUAL_ACCELERATOR_TYPE",
    "WRITES_ENABLED_LEAF",
    "agent_data_root",
    "baseline_result",
    "effective_writes_for",
    "is_baseline",
    "is_readonly_run",
    "most_restrictive_posture",
    "parse_posture",
    "parse_record",
    "posture_unknown",
    "read_json_file",
    "read_record",
    "read_target",
    "read_target_view",
    "record_path",
    "recorded_posture",
    "resolve_state_dir",
    "selected_target",
    "session_key",
    "session_types",
    "target_metadata",
    "target_posture",
    "target_sandboxed",
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


def agent_data_root(hook_input=None):
    """The agent-data root the record and the reports share.

    Rule 1 of the path contract: the :data:`AGENT_DATA_ROOT_ENV_VAR` stamp when
    it names a non-blank path, else ``<repo_root>/var/agent_data``. ``None``
    when neither answers — a session whose repo root cannot be resolved has no
    record, which is a different thing from having an empty one.

    The stamp comes first because it is the only answer that is right when a
    project moved ``agent_data.base_dir``: the derivation below is the framework
    DEFAULT, and a reader that preferred it would look in a directory nobody
    writes while a live narrowing sat somewhere else.
    """
    try:
        stamped = (os.environ.get(AGENT_DATA_ROOT_ENV_VAR) or "").strip()
        if stamped:
            return os.path.expanduser(stamped)
        repo_root = get_repo_root(hook_input)
        if not repo_root:
            return None
        return os.path.join(repo_root, _AGENT_DATA_BASE_DIR)
    except Exception:  # pragma: no cover - defensive; get_repo_root is total
        return None


def resolve_state_dir(hook_input=None):
    """Directory holding the record, the server reports and the approval stamps.

    ``None`` when the root is unresolvable. Not created here — a reader that
    created state directories would leave litter in every repo a hook ever ran
    in.
    """
    root = agent_data_root(hook_input)
    return None if root is None else os.path.join(root, STATE_DIR_NAME)


def record_path(hook_input=None):
    """The record's path, or ``None`` when the root is unresolvable.

    The same file the canonical reader's ``record_path()`` names; a record the
    owner writes in one directory and a hook looks for in another is a
    narrowing that silently never applies.
    """
    directory = resolve_state_dir(hook_input)
    return None if directory is None else os.path.join(directory, RECORD_FILENAME)


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


# -- record reading --------------------------------------------------------


def read_json_file(path):
    """Load one JSON object, or ``None`` if absent, unreadable, or corrupt.

    Never raises. ``ValueError`` covers ``JSONDecodeError`` and
    ``UnicodeDecodeError`` alike; a non-dict payload is corruption too.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _usable_int(value, minimum):
    """*value* as an int of at least *minimum*, or ``None``.

    ``bool`` is excluded on purpose: it is an ``int`` in Python, and a payload
    carrying ``true`` where a generation belongs states nothing about one.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def parse_posture(value):
    """One record's ``posture`` as a ``{target: posture}`` map of NARROWINGS.

    Rule 2 of the restated posture contract, and the same filter
    ``session_store.parse_store`` applies to the same field: what survives here
    decides whether a real machine is written to, so an entry the two filters
    disagree about is a narrowing that silently does not apply. Returns an empty
    map for anything that narrows nothing.
    """
    if isinstance(value, str):
        if value == POSTURE_SANDBOX:
            # The bare string the session-wide posture wrote before targets
            # existed: it narrowed everything, so it narrows every target.
            return dict.fromkeys(CONTROL_TARGETS, POSTURE_SANDBOX)
        # Bare "writes" — and every unknown string — narrows nothing.
        return {}
    if isinstance(value, dict):
        return {
            target: posture
            for target, posture in value.items()
            if isinstance(target, str) and posture == POSTURE_SANDBOX
        }
    return {}


def parse_record(raw):
    """Decode a record into ``{target, generation, posture}``, or ``None``.

    Accepts the decoded JSON object or the raw text of one. The identity fields
    are all-or-nothing (see the module docstring): a schema this reader does not
    know, a target outside the vocabulary, and a missing or negative generation
    each mean there is no record here, because none of the three can be guessed
    at and a guess would be a safety claim. ``posture`` degrades on its own to
    "nothing narrowed". Never raises.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, dict):
        return None
    if _usable_int(raw.get("schema"), SCHEMA_VERSION) != SCHEMA_VERSION:
        return None
    target = raw.get("target")
    if target not in CONTROL_TARGETS:
        return None
    generation = _usable_int(raw.get("generation"), 0)
    if generation is None:
        return None
    return {
        "target": target,
        "generation": generation,
        "posture": parse_posture(raw.get("posture")),
    }


def _read_record(hook_input=None):
    """``(record, reason)`` for the deployment's record. Never raises.

    Exactly one of the two is set. The reason separates "there is no record"
    from "there is one and it cannot be read", which is the difference between
    a deployment nobody has started yet and one whose record something
    corrupted; both refuse, and an operator has to be able to tell them apart.
    """
    path = record_path(hook_input)
    if not path:
        return None, REASON_NO_STATE
    try:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
    except OSError:
        return None, REASON_NO_STATE
    except ValueError:  # UnicodeDecodeError: a record written as not-UTF-8
        return None, REASON_UNREADABLE
    record = parse_record(raw)
    if record is None:
        return None, REASON_UNREADABLE
    return record, None


def read_record(hook_input=None):
    """The deployment's control-context record, or ``None``. Never raises.

    ``{target, generation, posture}`` — the three things a hook answers from.
    ``None`` for every degraded outcome: no agent-data root, no record, a
    schema this reader does not know, or identity fields it cannot read. A
    caller that must name which of those happened asks :func:`read_target` for
    the reason.
    """
    try:
        record, _reason = _read_record(hook_input)
        return record
    except Exception:  # pragma: no cover - defensive; every step above is total
        return None


def selected_target(record):
    """The usable ``target`` string on *record*, or ``None`` if it has none.

    Exported because a caller holding a record (see :func:`read_target_view`)
    has to answer the same question the same way. Tolerant of any input at all,
    so no caller has to defend itself against the shape it was handed.
    """
    if not isinstance(record, dict):
        return None
    target = record.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def read_target(hook_input=None):
    """Resolve the control target this deployment is pointed at. Never raises.

    A projection of :func:`read_record` onto the shared contract dict — a
    resolved ``{target, generation}`` or the explicit baseline fallback. See the
    module docstring for the contract and the fail-closed rule.

    Args:
        hook_input: The parsed hook stdin payload, used only to resolve the repo
            root. Optional, because hooks that read no stdin still need an
            answer.

    Returns:
        dict: keys ``target``, ``generation``, ``fallback``, ``reason`` — always
        all four.
    """
    try:
        record, reason = _read_record(hook_input)
        if record is None:
            return baseline_result(reason)
        return {
            "target": record["target"],
            "generation": record["generation"],
            "fallback": None,
            "reason": None,
        }
    except Exception:
        # The contract's last line of defence: a hook that raised here would
        # take the agent's turn down over one line of a prompt.
        return baseline_result(REASON_UNREADABLE)


# -- how a target is spoken of ---------------------------------------------


def _report_paths(directory):
    """Report-file paths in *directory*, sorted. Empty on any trouble at all."""
    try:
        names = sorted(
            n
            for n in os.listdir(directory)
            if n.startswith(REPORT_FILE_PREFIX) and n.endswith(REPORT_FILE_SUFFIX)
        )
    except OSError:
        return []
    return [os.path.join(directory, n) for n in names]


def _pid_from_report_name(name):
    """PID encoded in a report file's name, or ``None`` if it is not a number."""
    stem = name[len(REPORT_FILE_PREFIX) : -len(REPORT_FILE_SUFFIX)]
    try:
        return int(stem)
    except ValueError:
        return None


def _published_targets(hook_input=None):
    """Per-target display metadata as the live controls servers publish it.

    The first live report that states any is the answer, and there is nothing
    to choose between: every server renders this block from the one config the
    deployment was built with. A dead server's report is ignored and never
    deleted — sweeping belongs to the writer, and a reader that deleted files
    would be a second opinion about which servers exist.

    ``{}`` when no live server has published metadata, which is the deployment
    with nothing running: the caller then says the identity is unavailable
    rather than assembling one of its own.
    """
    directory = resolve_state_dir(hook_input)
    if not directory:
        return {}
    for path in _report_paths(directory):
        pid = _pid_from_report_name(os.path.basename(path))
        if pid is None or not _is_process_alive(pid):
            continue
        report = read_json_file(path)
        if report is None:
            continue
        targets = report.get("targets")
        if isinstance(targets, dict) and targets:
            return targets
    return {}


def read_target_view(hook_input=None):
    """The record with the servers' per-target metadata folded in, or ``None``.

    The same record :func:`read_record` returns, plus a ``targets`` mapping —
    the identity the deployment's own writer publishes for each machine, read
    through :func:`_published_targets`. It is a VIEW assembled here and not a
    shape any writer stores: the record says which target, the reports say what
    that target IS, and a prompt naming a machine needs both from one read so
    its "where you are" and "where you would be" lines cannot straddle a switch.

    ``None`` when there is no readable record, exactly as :func:`read_record`.
    An empty ``targets`` is the ordinary answer with no controls server running
    and is not a failure of the read. Never raises.
    """
    try:
        record = read_record(hook_input)
        if record is None:
            return None
        view = dict(record)
        view["targets"] = _published_targets(hook_input)
        return view
    except Exception:  # pragma: no cover - defensive; every step above is total
        return None


def target_metadata(record, target):
    """The RAW per-target metadata mapping on *record*, or ``None``.

    For callers that must tell a metadata key that is ABSENT from one that is
    present and false — a report from an older or newer writer, where a coerced
    default would state a machine identity nobody claimed. Returns the writer's
    own mapping, untouched; ``None`` when there is none for *target*, which is
    also what a record read without :func:`read_target_view` yields.
    """
    if not isinstance(record, dict):
        return None
    targets = record.get("targets")
    if not isinstance(targets, dict):
        return None
    meta = targets.get(target)
    return meta if isinstance(meta, dict) else None


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
    declared type cannot be the real machine, and whose connector table holds no
    single block that can be, has never said what ``live`` means here, and there
    is nothing to infer it from. A section that is not a mapping resolves to the
    mock, which is the factory's documented fail-closed default.

    Neither a simulated type nor a stand-in one can be that machine, and the
    exclusion holds on BOTH sides of the derivation — the baseline it starts
    from, and the candidate blocks it falls back to. A stand-in is a machine the
    deployment stands up itself; counting it would either answer ``live`` with
    the stand-in, or make ``live`` ambiguous on exactly the deployments that run
    the stand-in beside the block naming their facility's own machine.
    """
    never_live = SIMULATED_TYPES + STANDIN_TYPES
    declared = _resolved_type(section)
    if declared not in never_live:
        return declared

    connector = section.get("connector") if isinstance(section, dict) else None
    if not isinstance(connector, dict):
        return None
    candidates = [key for key in connector if isinstance(key, str) and key not in never_live]
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
    if target == TARGET_STANDIN:
        return LIVE_STANDIN_TYPE
    if target == TARGET_LIVE:
        return _live_type(section)
    return None


def _baseline_target(section):
    """The target *section* describes when nobody has switched.

    ``va`` for a virtual accelerator, ``standin`` for the live stand-in, and
    ``live`` for everything else — including a mock deployment, whose ``live``
    may well be underivable, because ``live`` is still the target its section
    describes.
    """
    return _BASELINE_TARGETS.get(_resolved_type(section), TARGET_LIVE)


def _switch_capable(section):
    """Whether this deployment gives a session more than one target to point at.

    The stdlib restatement of ``osprey_connectors.types.switch_capable``, whose
    two conditions are mirrored here in order: the deployment's OWN type is
    what its baseline target resolves back to, which is what keeps a mock that
    happens to carry an ``epics`` block out of the multi-target world; and at
    least two targets are configured (:func:`_configured_targets`, the same
    enumeration every roster walks).

    Which two is deliberately not asked, in step with the framework: a
    stand-in beside a simulator with no live machine authored is exactly the
    switching world, and demanding the ``live``/``va`` pair would deny it.
    """
    if not isinstance(section, dict):
        return False
    if target_type(section, _baseline_target(section)) != _resolved_type(section):
        return False
    return len(_configured_targets(section)) >= 2


def _configured_targets(section):
    """The targets a session on this deployment can actually be POINTED at.

    The stdlib restatement of ``osprey_connectors.types.configured_targets``:
    the deployment's baseline first — a session sits on it whether or not the
    config wrote a block for the connector ``control_system.type`` builds — then
    every other target in :data:`CONTROL_TARGETS` order whose type resolves and
    whose ``control_system.connector.<type>`` block is present and non-empty,
    since that block is what a connector is configured from.

    Read from the config rather than looped over :data:`CONTROL_TARGETS`,
    because which machines exist in the vocabulary and which ones a deployment
    stood up are different questions. Looping the constant would grow a
    ``standin`` slot on every deployment with no ``live_standin`` block at all —
    a machine that is not there, described as if it were.

    Never raises and never empty: a section that is missing or malformed still
    has a baseline, and that one target is what such a deployment is on.
    """
    baseline = _baseline_target(section)
    connector = section.get("connector") if isinstance(section, dict) else None
    targets = [baseline]
    for target in CONTROL_TARGETS:
        if target == baseline:
            continue
        connector_type = target_type(section, target)
        if connector_type is None:
            continue
        block = connector.get(connector_type) if isinstance(connector, dict) else None
        if isinstance(block, dict) and block:
            targets.append(target)
    return targets


def session_types(section):
    """``{target: connector type}`` for the targets a session here can REACH.

    The stdlib restatement of ``osprey_connectors.types.session_posture``'s
    reachable-target rule, and what a caller with no target of its own iterates
    instead of the target vocabulary. Every :func:`_configured_targets` target
    on a deployment that renders the switch; otherwise the one type
    ``control_system.type`` builds, under the baseline target that names it — by
    type on purpose, because ``live`` is the switch's own derivation and without
    the switch it can name a machine the built connector is not.

    Never raises: every section is at minimum one baseline target holding the
    mock, which is what the factory would build from it.
    """
    if _switch_capable(section):
        return {target: target_type(section, target) for target in _configured_targets(section)}
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
    """Whether *section* arms writes for one control *target*. Never raises.

    ``True`` armed, ``False`` not armed, and ``None`` for a section that states
    no posture anywhere — see the module docstring for why silence is its own
    answer rather than a refusal, and for the rules the ``True``/``False`` half
    mirrors from ``osprey_connectors.types``.

    Args:
        section: The ``control_system:`` config section. A caller holding a
            whole rendered config passes ``config.get("control_system")``.
        target: The control target — one of :data:`CONTROL_TARGETS`.
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

    The set ANDed over is :func:`session_types` rather than the target
    vocabulary. Without the switch there is only one target to be uncertain between,
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


# -- recorded posture ------------------------------------------------------


def recorded_posture(hook_input=None):
    """The narrowings the record carries — ``{target: "sandbox"}``.

    Empty when there is no readable record. That is not the same claim as
    "nothing was narrowed": :func:`posture_unknown` is the predicate that tells
    the two apart, and it refuses on exactly the shape where an empty answer
    here could be hiding one. Never raises.
    """
    record = read_record(hook_input)
    if record is None:
        return {}
    posture = record.get("posture")
    return dict(posture) if isinstance(posture, dict) else {}


def _posture_for(posture, target):
    """The recorded posture for one target inside a *posture* map.

    ``"sandbox"`` or ``None``. A *target* of ``None`` takes the MOST RESTRICTIVE
    entry — any sandbox answers sandbox — because a caller that cannot say which
    machine it is about must not be granted the most permissive answer.
    """
    if not isinstance(posture, dict) or not posture:
        return None
    if target:
        return posture.get(target)
    return POSTURE_SANDBOX if POSTURE_SANDBOX in posture.values() else None


def target_posture(target, hook_input=None):
    """The recorded posture for *target*, or ``None``. Never raises.

    The counterpart of ``session_store.target_posture``: one target, one
    answer, read from the deployment's record rather than from anything this
    process happens to carry.
    """
    return _posture_for(recorded_posture(hook_input), target)


def target_sandboxed(hook_input, target):
    """Whether the operator narrowed this target. Never raises.

    The record half of :func:`effective_writes_for` on its own, for a caller
    that already knows writes are refused and needs to say WHICH refusal it is:
    a narrowing is lifted on the header chip, an unarmed deployment in
    config.yml, and telling an operator the wrong one sends them to a control
    that will not move.
    """
    return target_posture(target, hook_input) == POSTURE_SANDBOX


def is_readonly_run():
    """Whether this process is a read-only run.

    A value comparison, never a presence check — the same semantics as
    ``osprey_connectors.control_system.base.is_readonly_run`` and the executor's
    posture clamp: only the exact ``"readonly"`` string sandboxes a session.
    """
    return os.environ.get(EXECUTION_MODE_ENV_VAR) == SANDBOX_MODE


def session_key():
    """This session's audit id, or ``None`` when it carries none.

    It keys nothing: the posture is the deployment's, and this string only says
    WHO a write-approval stamp belongs to. ``None`` for a bare ``claude`` and
    for every process the web terminal did not spawn.
    """
    return (os.environ.get(POSTURE_SESSION_ENV_VAR) or "").strip() or None


def posture_unknown(hook_input=None):
    """Whether the posture cannot be read where a narrowing would be.

    Both at once, and nothing less:

    * :data:`AGENT_DATA_ROOT_ENV_VAR` is NOT stamped — the directory below was
      derived from the framework default rather than handed over, and a project
      that moved ``agent_data.base_dir`` moved it out from under this reader;
    * there is no readable record in that directory — the evidence that the
      derivation found the right directory after all is missing.

    Deliberately fail-closed for an unstamped process on a deployment whose
    record has never been written: a readwrite call from a bare ``claude`` or a
    dispatch worker is refused until the first controls server has published,
    and succeeds on retry afterwards. The alternative reads a directory nobody
    writes as "nothing was narrowed", which is the one wrong answer that ends
    at a machine.

    Stamped, or with a readable record, the record means exactly what it says
    and nothing is refused on its account. Never raises.
    """
    if (os.environ.get(AGENT_DATA_ROOT_ENV_VAR) or "").strip():
        return False
    return read_record(hook_input) is None


def effective_writes_for(hook_input, section, target):
    """Whether a write may proceed here and now. Never raises.

    Rule 3 of the restated posture contract, spelled once::

        ceiling AND not readonly run AND (recorded posture != sandbox)

    The ceiling is :func:`writes_posture` for the target the record states, and
    :func:`most_restrictive_posture` when it states none — the fail-closed half,
    and the one place this restatement is stricter than the module it restates
    (see the module docstring). The record can only narrow it.

    Args:
        hook_input: The hook's stdin payload, for repo-root resolution.
        section: The ``control_system:`` config section.
        target: The control target, or ``None`` when it could not be resolved.

    Returns:
        ``True`` only when the deployment arms this machine, the process is not
        a read-only run, and the operator has not narrowed it.
    """
    ceiling = writes_posture(section, target) if target else most_restrictive_posture(section)
    if ceiling is not True:
        return False
    if is_readonly_run():
        return False
    return not target_sandboxed(hook_input, target)
