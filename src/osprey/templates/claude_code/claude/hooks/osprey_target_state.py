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

    <state_dir>/control_context.json
    <state_dir>/server_<server_pid>.json

* ``state_dir`` resolves by ONE ladder, the same one the writer and the
  connector-side reader use: ``OSPREY_CONTROL_CONTEXT_DIR`` when it names a
  non-blank path — the directory a container has bound for its own state,
  already per-identity — else
  ``<agent_data_root>/control_target/<acting_identity()>``;
* ``agent_data_root`` resolves by ONE rule, the same one the writer and the
  connector-side reader use: the ``OSPREY_AGENT_DATA_ROOT`` stamp when it names
  a non-blank path, else ``<repo_root>/<agent_data.base_dir>``;
* ``repo_root`` comes from :func:`osprey_hook_log.get_repo_root` — the repo, not
  the render: ``build/`` is disposable and ``data/`` is checksummed;
* the DERIVED base dir is ``agent_data.base_dir`` read from the config the hooks
  load, so a relocated root is followed. The derivation can still diverge from
  the framework's where the hook's raw YAML read and the framework's
  ``${VAR}``-resolving loader disagree, or where the two anchors differ; the
  reader then reports the baseline fallback, which is the documented
  fail-closed outcome rather than a wrong target — and :func:`posture_unknown`
  turns that same silence into a refusal for every process the stamp did not
  reach;
* ``acting_identity()`` restates the ladder :mod:`osprey_connectors.identity`
  owns, for the reason every constant here is restated. A hook that resolved a
  different identity from the writer would read a narrowing somebody else
  published, or read none where one applies;
* ONE record per identity, and one report per controls server, discovered by
  the glob ``server_*.json``.

There is no per-session record and no ancestor walk. The target, its generation
and the narrowings are properties of the deployment as ONE identity holds it:
two Claude Code sessions sharing a checkout and an identity read one file and
get one answer, and a hook that tried to tell those two apart would be
inventing a distinction the writer does not make.

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

Which targets a deployment has (a second question, answered from config)
-------------------------------------------------------------------------
Hooks that gate writes need two answers identity alone cannot give: which
connector type does a target select here, and which targets can a session on
this deployment reach at all? Both are config questions, and their authority is
``osprey_connectors.types``. Hooks cannot import it, so :func:`target_type` and
:func:`session_types` restate it here in stdlib terms, once, for every hook that
asks: two hooks mirroring the same rules separately is two ways for one
deployment to be described.

The mirrored rules, on the ``control_system:`` section:

* ``<type>`` is one whole key, never a path: a custom connector's dotted module
  path names a single block;
* ``va`` resolves to the virtual accelerator and ``standin`` to the live
  stand-in — the same answer on every deployment, because those are machines a
  deployment stands up for itself. ``live`` resolves to the section's own type
  when that type is neither simulated nor a stand-in, else to the single such
  key under ``connector``. Zero or more than one is underivable, and an
  underivable target selects no type.

A caller holding no target of its own asks which targets a session on THIS
deployment can reach. :func:`session_types` answers it, restating the
reachable set of ``osprey_connectors.types.most_restrictive_limits_posture`` —
the deployment's CONFIGURED targets where it renders the target switch, and
otherwise the single type ``control_system.type`` builds, read by TYPE under
the baseline target that names it. Iterating the target vocabulary instead
would answer for a machine no session here ever reaches: without the switch, ``live`` names a block the
connector the runtime built never serves, and a deployment with no
``live_standin`` block would grow a ``standin`` slot for a soft IOC nobody
stood up.

The write verdict (a third question, answered from config and the record)
-------------------------------------------------------------------------
The person arms one machine from the control-target chip in the header. That is
the record's ``switch`` field, ``{target: "asked" | "allowed"}``, and it is
enforced at WRITE time rather than delivered by respawning the agent, which is
what lets a flip land on a session already mid-conversation. A stored ``"on"``,
the retired spelling, reads as ``"asked"``; no other value arms.

``osprey_connectors.posture_store.write_verdict`` is the canonical verdict.
Hooks cannot import it, so :func:`write_verdict` restates it here term for term
— a change there is a change here, and ``tests/hooks/test_write_verdict_twin.py``
drives one case table through both.

**1. The tier.** :func:`tier_writes` reads ``control_system.writes``: only the
literals ``asked`` and ``allowed`` arm. A section that states no tier is
``never``; a retired ``writes_enabled`` leaf, flat or in a connector block, is
never read, in step with the package's ``types.tier_writes``. The tier is a
ceiling on the position as well: an allowed verdict's ``position`` is the stored
word capped by it (by the owner's rung under a bound tree).

**2. Whose switch it is.** The owner's, resolved by :func:`current_owner`: this
host's own record, or the owner's record in a bound read-only tree.

**3. What the record's ``posture`` field is.** A mirror of the switch kept for
old readers — ``sandbox`` for every machine not switched on — and never an
input to a write decision. A file written before the mirror may still carry an
old narrowing; :func:`parse_posture` still parses it (``{"live": "sandbox"}``,
or the bare string ``"sandbox"`` for every target in :data:`CONTROL_TARGETS`;
anything else is dropped) so that :func:`parse_record` admits exactly the
records the canonical parser admits, and nothing that decides a write reads it.
An old narrowing with the machine armed is allowed, on both sides.

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
from osprey_hook_log import (
    agent_data_root_at,
    get_repo_root,
    is_write_call,
    is_write_tool,
    load_hook_config,
    load_osprey_config,
    write_tools,
)

#: Fixed subdirectory of the agent-data root, holding one directory per
#: identity. Mirrors ``STATE_DIR_NAME`` on the writer; part of the greppable
#: path contract.
STATE_DIR_NAME = "control_target"

#: The container's own state directory, bound in by compose. Top rung of the
#: path contract and already the per-identity directory, so neither fixed hop
#: is appended to it. Mirrors ``posture_store.CONTROL_CONTEXT_DIR_ENV_VAR``;
#: read by NAME for the reason every other marker here is, and never confused
#: with ``OSPREY_CONTROL_CONTEXT_TREE``, which names every identity's
#: directory and therefore none of them.
CONTROL_CONTEXT_DIR_ENV_VAR = "OSPREY_CONTROL_CONTEXT_DIR"

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

#: Connector types that serve a machine nobody has to be careful around; the
#: simulator is also the type ``resolve_control_system_type`` falls back to when
#: a section names none. They are why ``live`` cannot simply be "whatever the
#: config selects": a deployment whose baseline is one of these has not said
#: what its real machine is. Literals rather than an import, like everything
#: else in this module.
VIRTUAL_ACCELERATOR_TYPE = "virtual_accelerator"
SIMULATED_TYPES = (VIRTUAL_ACCELERATOR_TYPE,)

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

#: The agent-data root stamp a session child carries. Read by NAME rather than
#: imported from ``osprey.audit.posture``, which declares it for the stamping
#: side: hooks are stdlib-only and must not grow an ``osprey`` import to learn
#: one string. Its absence is half of :func:`posture_unknown`.
AGENT_DATA_ROOT_ENV_VAR = "OSPREY_AGENT_DATA_ROOT"

#: The identity ladder, restated from ``osprey_connectors.identity`` — the
#: multi-user deployment's per-container user, then the identity of a container
#: that hosts no single user, then the local account, then an honest floor.
#: Never the hostname; see that module for why, and for why
#: ``OSPREY_AUDIT_IDENTITY`` may never join a scrub list. Restated rather than
#: imported for the reason every other constant here is: a hook runs outside the
#: osprey venv, where neither package exists. ``osprey_hook_log`` carries the
#: same four names for the audit directory; the three copies are one contract
#: and must answer alike on every environment they are asked about.
TERMINAL_USER_ENV = "OSPREY_TERMINAL_USER"
AUDIT_IDENTITY_ENV = "OSPREY_AUDIT_IDENTITY"
IDENTITY_ENV_LADDER = (TERMINAL_USER_ENV, AUDIT_IDENTITY_ENV)
UNKNOWN_IDENTITY = "unknown"

# What disqualifies a value from being the one path component an identity is.
_PATH_SEPARATORS = ("/", "\\", "\0")
_RESERVED_NAMES = (".", "..")

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
    "AUDIT_IDENTITY_ENV",
    "CONTROL_CONTEXT_DIR_ENV_VAR",
    "CONTROL_CONTEXT_TREE_ENV_VAR",
    "CONTROL_OWNER_ENV_VAR",
    "CONTROL_TARGETS",
    "CONTROL_TREE_MARKER_NAME",
    "DISPATCH_RUN_ENV_VAR",
    "EXECUTION_MODE_ENV_VAR",
    "FALLBACK_BASELINE",
    "IDENTITY_ENV_LADDER",
    "KIND_WORDS",
    "LAUNCH_POSTURE_ALL_TARGETS",
    "LAUNCH_POSTURE_ENV_VAR",
    "LIVE_STANDIN_TYPE",
    "LOGIN_TIERS_LEAF",
    "NO_OWNER",
    "POSTURE_SANDBOX",
    "POSTURE_SESSION_ENV_VAR",
    "POSTURE_WRITES",
    "REASONS",
    "REASON_NO_STATE",
    "REASON_UNREADABLE",
    "RECORD_FILENAME",
    "RECORD_READ_LIMIT",
    "REPORT_FILE_GLOB",
    "REPORT_FILE_PREFIX",
    "REPORT_FILE_SUFFIX",
    "SANDBOX_MODE",
    "SCHEMA_VERSION",
    "SIMULATED_TYPES",
    "STANDIN_TYPES",
    "STATE_DIR_NAME",
    "STORE_PATH_MARKERS",
    "SWITCH_ALLOWED",
    "SWITCH_ASKED",
    "SWITCH_OFF",
    "TARGET_LIVE",
    "TARGET_STANDIN",
    "TARGET_VA",
    "TERMINAL_USER_ENV",
    "TIER_ALLOWED",
    "TIER_ASKED",
    "TIER_NEVER",
    "UNKNOWN_IDENTITY",
    "UNRESOLVED",
    "VALID_POSTURES",
    "VIRTUAL_ACCELERATOR_TYPE",
    "WRITES_LEAF",
    "NoOwner",
    "WriteVerdict",
    "acting_identity",
    "agent_data_root",
    "baseline_result",
    "current_owner",
    "hook_write_target",
    "is_baseline",
    "is_readonly_run",
    "launch_pin_term",
    "machine_kind",
    "owner_tier_term",
    "parse_posture",
    "parse_record",
    "parse_switch",
    "parse_switch_record",
    "published_machine_kind",
    "posture_unknown",
    "read_json_file",
    "read_record",
    "read_record_status",
    "read_target",
    "read_target_view",
    "read_tree_record",
    "record_path",
    "refusal_phrase",
    "refusal_text",
    "resolve_state_dir",
    "selected_target",
    "session_key",
    "session_types",
    "switch_position",
    "target_metadata",
    "target_type",
    "tier_allows",
    "tier_writes",
    "write_verdict",
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


# -- identity --------------------------------------------------------------


def _usable(value):
    """Return *value* stripped if it can serve as an identity, else ``""``.

    An identity is used as a name and as one directory component, so a rung
    only counts when its value works as both. Blank is the unset case spelled
    differently — a rendered-but-empty ``environment:`` entry. A separator or a
    relative name is refused because the same string becomes a directory under
    the state root, where ``../elsewhere`` would read another identity's
    narrowing. The refusal stays narrow on purpose: a stricter allowlist would
    turn a real account name into :data:`UNKNOWN_IDENTITY`.
    """
    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if not candidate or candidate in _RESERVED_NAMES:
        return ""
    if any(separator in candidate for separator in _PATH_SEPARATORS):
        return ""
    return candidate


def acting_identity():
    """Who this process is, per :data:`IDENTITY_ENV_LADDER` and then the account.

    The same answer the record's writer resolved when it chose which directory
    to write into: a hook that resolved a different one would read a narrowing
    somebody else's session published, or read none where one applies.

    Reads the environment on every call, because the markers are set per
    process by compose and by the entrypoint. Never raises:
    :func:`getpass.getuser` fails for a uid with no passwd entry, which is
    ordinary in a slim image rather than exceptional, and every failure lands
    on :data:`UNKNOWN_IDENTITY`. ``getpass`` is imported inside the try, the
    same way ``osprey_hook_log`` imports it: the module-level imports are what
    a hook pays on every tool call, and this rung is reached only by the
    sessions no container marker names.
    """
    for env_name in IDENTITY_ENV_LADDER:
        candidate = _usable(os.environ.get(env_name))
        if candidate:
            return candidate
    try:
        import getpass

        return _usable(getpass.getuser()) or UNKNOWN_IDENTITY
    except Exception:  # pragma: no cover - hooks must never crash the agent
        return UNKNOWN_IDENTITY


# -- paths -----------------------------------------------------------------


def agent_data_root(hook_input=None):
    """The agent-data root the record and the reports share.

    Rule 1 of the path contract: the :data:`AGENT_DATA_ROOT_ENV_VAR` stamp when
    it names a non-blank path, else ``<repo_root>/<agent_data.base_dir>`` read
    from the config the hooks load. ``None`` when neither answers — a session
    whose repo root cannot be resolved has no record, which is a different thing
    from having an empty one.

    The stamp comes first because it is the root the writer was handed. The
    derivation below follows a relocated ``agent_data.base_dir``, but it can
    still diverge from the framework's where the hook's raw YAML read and the
    framework's ``${VAR}``-resolving loader disagree, or where the anchors
    differ.
    """
    try:
        stamped = (os.environ.get(AGENT_DATA_ROOT_ENV_VAR) or "").strip()
        if stamped:
            return os.path.expanduser(stamped)
        repo_root = get_repo_root(hook_input)
        if not repo_root:
            return None
        return agent_data_root_at(repo_root, load_osprey_config(hook_input))
    except Exception:  # pragma: no cover - defensive; get_repo_root is total
        return None


def resolve_state_dir(hook_input=None):
    """This identity's record, server reports and approval stamps live here.

    The path contract's ladder: :data:`CONTROL_CONTEXT_DIR_ENV_VAR` when a
    container carries the bind — nothing is appended to it, because a bind
    target is already one identity's directory — else
    :func:`acting_identity` under :data:`STATE_DIR_NAME` under
    :func:`agent_data_root`.

    Blank, whitespace-only and non-absolute values all name no bind, the same
    rule the canonical reader applies: a bind is a path in the container's own
    filesystem, so a relative one would resolve against whatever directory the
    tool call happened to start in — a directory no writer creates. Expansion
    is ``os.path.expanduser``, which unlike the ``pathlib`` method does not
    raise for a ``~`` naming a user this container has no passwd entry for.

    ``None`` when the root is unresolvable. Not created here — a reader that
    created state directories would leave litter in every repo a hook ever ran
    in.
    """
    bound = os.path.expanduser((os.environ.get(CONTROL_CONTEXT_DIR_ENV_VAR) or "").strip())
    if bound and os.path.isabs(bound):
        return bound
    root = agent_data_root(hook_input)
    return None if root is None else os.path.join(root, STATE_DIR_NAME, acting_identity())


def record_path(hook_input=None):
    """The record's path, or ``None`` when the root is unresolvable.

    The same file the canonical reader's ``record_path()`` names; a record the
    owner writes in one directory and a hook looks for in another is a
    narrowing that silently never applies.
    """
    directory = resolve_state_dir(hook_input)
    return None if directory is None else os.path.join(directory, RECORD_FILENAME)


# -- liveness --------------------------------------------------------------

_PROC_ROOT = "/proc"


def _is_thread_id(pid):
    """Whether the kernel knows *pid* as a thread of some other process.

    Linux draws PIDs and thread ids from one number space, and ``/proc/<tid>``
    opens by number though no listing shows it. A process is its own thread
    group leader, so a ``Tgid`` that differs from *pid* means a thread. ``False``
    when ``/proc`` says nothing, which is every platform without one.
    """
    try:
        with open(f"{_PROC_ROOT}/{pid}/status", "rb") as status:
            for line in status:
                if line.startswith(b"Tgid:"):
                    return int(line.split()[1]) != pid
    except (OSError, ValueError, IndexError):
        return False  # no /proc, or one that does not read like Linux's: no claim either way
    return False


def _is_process_alive(pid):
    """Whether *pid* names a running process.

    ``os.kill(pid, 0)`` sends no signal and only asks the kernel whether the
    process exists. ``PermissionError`` means it exists but belongs to another
    user, so it counts as ALIVE — treating an unreachable owner as dead would
    discard live state. Non-positive PIDs address process *groups* and are
    rejected without calling ``os.kill`` at all. A thread id also answers
    ``os.kill`` and is no process (:func:`_is_thread_id`).
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
        pass  # another user's process still exists; the thread check below applies to it too
    except OSError:  # pragma: no cover - platform oddity; assume alive
        return True
    return not _is_thread_id(pid)


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
    ``posture_store.parse_posture_value`` applies to the same field: what survives here
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


def _published_targets_in(directory):
    """The first live server's per-target metadata in *directory*, or ``{}``."""
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
    return _published_targets_in(directory)


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


# -- which targets a deployment has ----------------------------------------


def _resolved_type(section):
    """The connector type ``control_system.type`` selects — the simulator when absent.

    The factory's documented fail-closed default, restated: a missing section, a
    section that is not a mapping, and a bare ``type:`` (which YAML gives as
    ``None``) all name no type, and a deployment that named none gets the
    simulator, served in process.
    """
    declared = section.get("type") if isinstance(section, dict) else None
    return str(declared) if declared else VIRTUAL_ACCELERATOR_TYPE


def _live_type(section):
    """The connector type that reaches this deployment's real machine, or ``None``.

    ``None`` wherever the framework's own ``_live_type`` raises: a section whose
    declared type cannot be the real machine, and whose connector table holds no
    single block that can be, has never said what ``live`` means here, and there
    is nothing to infer it from. A section that is not a mapping resolves to the
    simulator, which is the factory's documented fail-closed default.

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

    ``va`` for the simulator in either venue, ``standin`` for the live
    stand-in, and ``live`` for everything else.
    """
    return _BASELINE_TARGETS.get(_resolved_type(section), TARGET_LIVE)


def _switch_capable(section):
    """Whether this deployment gives a session more than one target to point at.

    The stdlib restatement of ``osprey_connectors.types.switch_capable``: at
    least two targets are configured (:func:`_configured_targets`, the same
    enumeration every roster walks).

    Which two is deliberately not asked, in step with the framework: a
    stand-in beside a simulator with no live machine authored is exactly the
    switching world, and demanding the ``live``/``va`` pair would deny it.
    """
    if not isinstance(section, dict):
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

    The stdlib restatement of the reachable-target rule of
    ``osprey_connectors.types.most_restrictive_limits_posture``, and what a
    caller with no target of its own iterates instead of the target vocabulary. Every :func:`_configured_targets` target
    on a deployment that renders the switch; otherwise the one type
    ``control_system.type`` builds, under the baseline target that names it — by
    type on purpose, because ``live`` is the switch's own derivation and without
    the switch it can name a machine the built connector is not.

    Never raises: every section is at minimum one baseline target holding the
    simulator, which is what the factory would build from it.
    """
    if _switch_capable(section):
        return {target: target_type(section, target) for target in _configured_targets(section)}
    return {_baseline_target(section): _resolved_type(section)}


# -- process and record state ----------------------------------------------


def is_readonly_run():
    """Whether this process is a read-only run.

    A value comparison, never a presence check — the same semantics as
    ``osprey_connectors.posture_env.is_readonly_run`` and the executor's
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

    * :data:`AGENT_DATA_ROOT_ENV_VAR` is NOT stamped — the root was derived from
      config rather than handed over;
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


# -- the write verdict -----------------------------------------------------
#
# The stdlib twin of ``posture_store.write_verdict``: the same terms in the same
# order, so a write the connector refuses is refused here for the same reason
# and in the same words. Every name below restates a package symbol and says
# which; ``tests/hooks/test_write_verdict_twin.py`` drives the shared case table
# through both and compares the word tables key by key. Nothing in this section
# reads the record's ``posture`` field: the switch is the state, and an old
# narrowing left in a file refuses nothing.

#: The stamped owner of a dispatch job's work. Mirrors
#: ``posture_store.CONTROL_OWNER_ENV_VAR``; a blank value is the unset case.
CONTROL_OWNER_ENV_VAR = "OSPREY_CONTROL_OWNER"

#: The mark of a dispatch job. Mirrors ``posture_store.DISPATCH_RUN_ENV_VAR``;
#: only ``"1"`` marks one.
DISPATCH_RUN_ENV_VAR = "OSPREY_DISPATCH_RUN"

#: The read-only bind of every login's control-context directory, carried by a
#: container that holds no chip of its own. Mirrors
#: ``posture_store.CONTROL_CONTEXT_TREE_ENV_VAR``. Its PRESENCE makes the
#: process owned-or-nothing; its value is read only by the tree reader.
CONTROL_CONTEXT_TREE_ENV_VAR = "OSPREY_CONTROL_CONTEXT_TREE"

#: The marker ``osprey build`` writes at the tree's root. Mirrors
#: ``posture_store.CONTROL_TREE_MARKER_NAME``: a bind without it is an
#: unprovisioned tree, which refuses rather than reads as "nothing narrowed".
CONTROL_TREE_MARKER_NAME = ".osprey-control-tree"

#: The executor's launch-time pin, ``"<target>=<posture>"``. Mirrors
#: ``posture_store.LAUNCH_POSTURE_ENV_VAR`` and its all-targets spelling.
LAUNCH_POSTURE_ENV_VAR = "OSPREY_LAUNCH_POSTURE"
LAUNCH_POSTURE_ALL_TARGETS = "*"

#: The most a tree record may be and still be read. Mirrors
#: ``posture_store._RECORD_READ_LIMIT``.
RECORD_READ_LIMIT = 64 * 1024

#: The tier key's leaf, its three values, and the login→tier map's leaf. Mirror
#: ``types._WRITES_LEAF``/``_TIER_*`` and ``posture_store._LOGIN_TIERS_LEAF``.
WRITES_LEAF = "writes"
TIER_NEVER = "never"
TIER_ASKED = "asked"
TIER_ALLOWED = "allowed"
LOGIN_TIERS_LEAF = "login_tiers"

#: Lowest to highest, which is what caps a position. Mirrors ``posture_store._LOGIN_TIERS``.
_LOGIN_TIERS = (TIER_NEVER, TIER_ASKED, TIER_ALLOWED)

#: The positions a machine's switch can be in, and the retired spelling read as
#: asked. Mirror ``control_context._SWITCH_*``.
SWITCH_OFF = "off"
SWITCH_ASKED = "asked"
SWITCH_ALLOWED = "allowed"
_SWITCH_LEGACY_ON = "on"

#: The kind of machine each target names by its name alone. Mirrors
#: ``types._MACHINE_KINDS``: ``standin`` is ``real`` by name, and only the
#: published report (:func:`published_machine_kind`) calls it the rehearsal.
_MACHINE_KINDS = {TARGET_LIVE: "real", TARGET_STANDIN: "real", TARGET_VA: "simulator"}

#: The report key a target's published kind sits under. Mirrors ``types._MACHINE_KIND_KEY``.
_MACHINE_KIND_KEY = "machine_kind"

# What makes an owner name unusable as one directory of the tree. Mirrors
# ``posture_store._TREE_UNSAFE``.
_TREE_UNSAFE = frozenset({"/", "\\", "\x00"})

#: The closed set of refusal reasons. Mirrors ``refusals.REASONS``, in order.
REASONS = (
    "readonly_run",
    "tier_never",
    "target_unknown",
    "owner_unknown",
    "owner_unscheduled",
    "owner_foreign",
    "record_unavailable",
    "switch_off",
    "launch_pinned",
    "switching",
    "stopped_since",
)

# Mirrors ``refusals._PHRASES`` word for word.
_PHRASES = {
    "readonly_run": (
        "Writes are refused because the deployment is running read-only; "
        "the chip cannot lift this — ask the deployment admin."
    ),
    "tier_never": (
        "Writes are refused because this login's write tier does not allow them; "
        "ask the deployment admin."
    ),
    "target_unknown": (
        "Writes are refused because the machine this connector writes to is not known; "
        "retry, then ask the deployment admin."
    ),
    "owner_unknown": (
        "Writes are refused because this login's write tier is not known; ask the deployment admin."
    ),
    "owner_unscheduled": (
        "Writes are refused because this scheduled or triggered job runs for no login, "
        "so no switch covers it; ask the deployment admin."
    ),
    "owner_foreign": (
        "Writes are refused because this machine's writes belong to another login; "
        "only that login can write here."
    ),
    "record_unavailable": (
        "Writes are refused because the write switch cannot be read; "
        "retry, then ask the deployment admin."
    ),
    "switch_off": "Writes are off; turn it on from the chip in the web terminal.",
    "launch_pinned": (
        "Writes are refused because this run was started with writes held off; "
        "run it again without the pin."
    ),
    "switching": (
        "Writes are refused because the write gateway has not yet followed the switch; "
        "retry, then ask the deployment admin."
    ),
    "stopped_since": (
        "Writes are off because a stop came after the turn-on began; "
        "turn it on again from the chip."
    ),
}

#: The words a machine kind adds to a phrase. Mirrors ``refusals.KIND_WORDS``.
KIND_WORDS = {
    "real": "the live machine",
    "rehearsal": "the rehearsal machine",
    "simulator": "the simulator",
}

#: Path spellings of the posture store. Mirrors ``refusals.STORE_PATH_MARKERS``.
STORE_PATH_MARKERS = (STATE_DIR_NAME, RECORD_FILENAME)


def refusal_phrase(reason, kind=None):
    """The phrase for *reason*; a known *kind* names the machine it concerns.

    Mirrors ``refusals.refusal_phrase``: ``ValueError`` for a reason outside
    :data:`REASONS` — the one thing in this module that raises, because a
    caller passing an unknown reason is a bug, not a degraded state.
    """
    try:
        phrase = _PHRASES[reason]
    except (KeyError, TypeError):
        raise ValueError(f"unknown refusal reason: {reason!r}") from None
    words = KIND_WORDS.get(kind) if isinstance(kind, str) else None
    return f"{phrase} (Concerns {words}.)" if words else phrase


def refusal_text(reason, kind=None, *, detail=None):
    """The refusal a write path returns: the phrase, then ``[osprey-refusal: <reason>]``.

    Mirrors ``refusals.refusal_text``: *detail* follows the phrase on its line,
    the trailer is its own last line, and a reason outside :data:`REASONS`
    raises ``ValueError``.
    """
    phrase = refusal_phrase(reason, kind)
    if detail:
        phrase = f"{phrase} {detail}"
    return f"{phrase}\n[osprey-refusal: {reason}]"


class NoOwner:
    """The absence of an owner: one instance, compared by identity only.

    Mirrors ``posture_store.NoOwner``. ``None`` already means "the caller named
    nobody, walk the ladder", and every string is a name some deployment may
    give an account, so neither can say "this work belongs to nobody".
    """

    __slots__ = ()

    def __str__(self):
        return "<no owner>"

    __repr__ = __str__


#: The answer every owner-less path gives. See :class:`NoOwner`.
NO_OWNER = NoOwner()


class WriteVerdict:
    """Whether one write may go ahead, and the one reason when it may not.

    The same five fields as ``posture_store.WriteVerdict``: *allowed*, *reason*
    (a member of :data:`REASONS` or ``None``), *target* (verbatim when it was a
    string), *kind* (``real``/``rehearsal``/``simulator`` or ``None``) and
    *position* (``asked``/``allowed`` when allowed, ``None`` on every refusal).
    A plain slotted class rather than a dataclass: a hook pays every
    module-level import on every tool call.
    """

    __slots__ = ("allowed", "reason", "target", "kind", "position")

    def __init__(self, allowed, reason, target, kind, position=None):
        self.allowed = allowed
        self.reason = reason
        self.target = target
        self.kind = kind
        self.position = position

    def __eq__(self, other):
        if not isinstance(other, WriteVerdict):
            return NotImplemented
        return (self.allowed, self.reason, self.target, self.kind, self.position) == (
            other.allowed,
            other.reason,
            other.target,
            other.kind,
            other.position,
        )

    # Defining __eq__ already sets __hash__ to None: a verdict is unhashable.

    def __repr__(self):
        return (
            f"WriteVerdict(allowed={self.allowed!r}, reason={self.reason!r}, "
            f"target={self.target!r}, kind={self.kind!r}, position={self.position!r})"
        )


def tier_writes(section):
    """The login tier's write ceiling: ``never``, ``asked`` or ``allowed``. Never raises.

    Mirrors ``types.tier_writes``: only the literals ``"asked"`` and ``"allowed"``
    in the ``writes`` leaf arm, matched exactly. An absent leaf, any other value
    (the switch's retired ``"on"`` among them), and a section that is not a mapping
    are ``never``; the retired ``writes_enabled`` leaves are not read.
    """
    value = section.get(WRITES_LEAF) if isinstance(section, dict) else None
    if isinstance(value, str) and value in (TIER_ASKED, TIER_ALLOWED):
        return value
    return TIER_NEVER


def tier_allows(section):
    """Whether the tier arms anything: :func:`tier_writes` is not ``never``."""
    return tier_writes(section) != TIER_NEVER


def machine_kind(section, target):
    """The kind of machine *target* names here, or ``None`` when it does not resolve.

    Mirrors ``types.machine_kind``; :func:`target_type` is ``None`` exactly where
    the package's ``resolve_target`` raises. Never raises.
    """
    try:
        if target_type(section, target) is None:
            return None
        return _MACHINE_KINDS.get(target)
    except Exception:
        return None


def published_machine_kind(target, *, state_dir=None, hook_input=None):
    """The kind of machine *target* is, as the controls server publishes it.

    Mirrors ``types.published_machine_kind``: ``standin`` is ``rehearsal`` only
    when the first live server's report says so for it — no live report, no
    key, or any other value reads ``real``. Every other target answers from the
    name table (``va`` the simulator, ``live`` and anything unknown the real
    machine). *state_dir* names the reports' directory; ``None`` resolves it
    through :func:`resolve_state_dir` from *hook_input*. Never raises.
    """
    if target != TARGET_STANDIN:
        return _MACHINE_KINDS.get(target, "real") if isinstance(target, str) else "real"
    try:
        directory = resolve_state_dir(hook_input) if state_dir is None else os.fspath(state_dir)
        if not directory:
            return "real"
        meta = _published_targets_in(directory).get(TARGET_STANDIN)
    except Exception:
        return "real"
    if isinstance(meta, dict) and meta.get(_MACHINE_KIND_KEY) == "rehearsal":
        return "rehearsal"
    return "real"


def launch_pin_term(target):
    """The launch pin as a verdict term: ``"none"``, ``"pinned"`` or ``"unreadable"``.

    Mirrors ``posture_store.launch_pin_term``: no stamp is ``none``; a
    ``<t>=sandbox`` stamp pins *target* when ``<t>`` is it, is ``*``, or
    *target* is empty; a ``<t>=writes`` stamp is ``none``; anything unparseable
    is ``unreadable``, which refuses like a pin.
    """
    text = (os.environ.get(LAUNCH_POSTURE_ENV_VAR) or "").strip()
    if not text:
        return "none"
    stamped, separator, launch_posture = text.partition("=")
    stamped = stamped.strip()
    launch_posture = launch_posture.strip()
    if (
        not separator
        or launch_posture not in VALID_POSTURES
        or (stamped != LAUNCH_POSTURE_ALL_TARGETS and stamped not in CONTROL_TARGETS)
    ):
        return "unreadable"
    if launch_posture != POSTURE_SANDBOX:
        return "none"
    if stamped in (LAUNCH_POSTURE_ALL_TARGETS, target) or not target:
        return "pinned"
    return "none"


def _tree_is_bound():
    """Whether a control-context tree is bound at all, usable or not.

    Mirrors ``posture_store._tree_is_bound``: presence of a non-blank value.
    """
    return bool((os.environ.get(CONTROL_CONTEXT_TREE_ENV_VAR) or "").strip())


def _control_context_tree():
    """The tree bind as an absolute path, or ``None`` when it names none.

    Mirrors ``posture_store._control_context_tree``: stripped, expanded with
    ``os.path.expanduser``, and ``None`` unless absolute.
    """
    raw = (os.environ.get(CONTROL_CONTEXT_TREE_ENV_VAR) or "").strip()
    if not raw:
        return None
    expanded = os.path.expanduser(raw)
    return expanded if os.path.isabs(expanded) else None


def current_owner(owner=None):
    """The owner whose record governs a write, by the package's ladder minus one rung.

    Mirrors ``posture_store.current_owner`` without the context-variable rung,
    which only a queueserver worker sets: an explicit *owner* (a string is
    stripped; :data:`NO_OWNER` or any other value is returned as is), else
    :data:`CONTROL_OWNER_ENV_VAR` when non-blank, else :data:`NO_OWNER` for a
    dispatch run (:data:`DISPATCH_RUN_ENV_VAR` is ``"1"``) or when a tree is
    bound, else :func:`acting_identity`. Never raises.
    """
    if isinstance(owner, str):
        return owner.strip()
    if owner is not None:
        return owner
    stamped = os.environ.get(CONTROL_OWNER_ENV_VAR)
    if isinstance(stamped, str) and stamped.strip():
        return stamped.strip()
    if _is_dispatch_run():
        return NO_OWNER
    if _tree_is_bound():
        return NO_OWNER
    return acting_identity()


def _is_dispatch_run():
    """Whether this process is a dispatch job. Mirrors ``posture_store._is_dispatch_run``."""
    return os.environ.get(DISPATCH_RUN_ENV_VAR) == "1"


def owner_tier_term(section, owner):
    """The owner's tier from the login→tier map: ``skip``, a tier word, or ``unknown``.

    Mirrors ``posture_store.owner_tier_term``: the rung applies when the
    section carries ``login_tiers`` or a tree is bound; a map that is not one
    render's well-formed output, an owner that is not a non-empty string, or an
    owner the map does not name is ``unknown``. Logins match exactly. Never
    raises.
    """
    try:
        carries = isinstance(section, dict) and LOGIN_TIERS_LEAF in section
        if not carries:
            return "unknown" if _tree_is_bound() else "skip"
        login_tiers = section[LOGIN_TIERS_LEAF]
        if not isinstance(login_tiers, dict) or not all(
            isinstance(login, str) and login and isinstance(tier, str) and tier in _LOGIN_TIERS
            for login, tier in login_tiers.items()
        ):
            return "unknown"
        if not isinstance(owner, str) or not owner:
            return "unknown"
        tier = login_tiers.get(owner)
    except Exception:
        return "unknown"
    return tier if tier in _LOGIN_TIERS else "unknown"


def _switch_word(value):
    """One stored ``switch`` value as the word it arms with, or ``None`` for off.

    Mirrors ``control_context._switch_word``: an exact match, no folding and no
    trimming. The legacy ``"on"`` is ``"asked"``.
    """
    if not isinstance(value, str):
        return None
    if value == _SWITCH_LEGACY_ON:
        return SWITCH_ASKED
    if value in (SWITCH_ASKED, SWITCH_ALLOWED):
        return value
    return None


def parse_switch(value):
    """One record's ``switch`` as ``{target: "asked" | "allowed"}``.

    Mirrors ``control_context._parse_switch``: string keys are kept whether or
    not they name a known target, ``"on"`` reads as ``"asked"``, and any other
    value is dropped. Anything that is not a mapping is ``{}``.
    """
    if not isinstance(value, dict):
        return {}
    parsed = {}
    for key, state in value.items():
        word = _switch_word(state)
        if isinstance(key, str) and word is not None:
            parsed[key] = word
    return parsed


def switch_position(record, target):
    """Where the person has put *target*: ``off``, ``asked`` or ``allowed``. Never raises.

    Mirrors ``control_context.switch_position``, over the dict
    :func:`parse_switch_record` builds. No record, a record with no switch, a
    target absent from the map and an unknown value all read as off; a
    hand-built record still spelling ``"on"`` reads as asked, as a parsed one does.
    """
    switch = record.get("switch") if isinstance(record, dict) else None
    if not isinstance(switch, dict) or not isinstance(target, str):
        return SWITCH_OFF
    return _switch_word(switch.get(target)) or SWITCH_OFF


def parse_switch_record(raw):
    """A record as ``{target, generation, posture, switch}``, or ``None``.

    :func:`parse_record` with the switch beside the three fields it answers —
    a separate function so that one's shape, which hooks rely on, stays as it
    is. The same all-or-nothing identity filter. Never raises.
    """
    try:
        record = parse_record(raw)
        if record is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            raw = json.loads(raw)
        record["switch"] = parse_switch(raw.get("switch"))
        return record
    except Exception:
        return None


def read_record_status(hook_input=None):
    """``(record, status)`` for this host's record, where status tells why there is none.

    Mirrors ``control_context.read_record_status``: ``"ok"`` with the record
    (switch included); ``"missing"`` for no file at the path; ``"unreadable"``
    for every other way of having none — an unresolvable root, an OS or decode
    failure, a payload this version cannot read. :func:`_read_record` folds a
    missing file and an unreadable one together, which a write verdict must
    not. Never raises.
    """
    try:
        path = record_path(hook_input)
        if not path:
            return None, "unreadable"
        try:
            with open(path, encoding="utf-8") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return None, "missing"
        except (OSError, ValueError):
            return None, "unreadable"
        record = parse_switch_record(raw)
        return (None, "unreadable") if record is None else (record, "ok")
    except Exception:  # pragma: no cover - defensive; every step above is total
        return None, "unreadable"


def _read_no_follow(path):
    """*path*'s text, refusing a symbolic link at its last component and an over-long file.

    Mirrors ``posture_store._read_no_follow``: ``O_NOFOLLOW`` makes the kernel
    refuse a link at the open (``ELOOP``), and a file past
    :data:`RECORD_READ_LIMIT` raises ``EFBIG`` rather than being truncated into
    a parse. Raises ``OSError`` or ``UnicodeDecodeError``.
    """
    import errno

    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(fd, encoding="utf-8", closefd=False) as handle:
            text = handle.read(RECORD_READ_LIMIT + 1)
    finally:
        os.close(fd)
    if len(text) > RECORD_READ_LIMIT:
        raise OSError(errno.EFBIG, "the file is larger than a record may be")
    return text


def read_tree_record(tree, owner):
    """``(record, unavailable)`` for *owner*'s record in the read-only *tree*.

    Mirrors ``posture_store._read_tree_context``, and fails closed the same
    way: *unavailable* is ``True`` for a tree that is not a directory, not
    readable, carries no marker (or a link or non-file there), an owner that
    is not one directory name, a link at the owner's directory or record, and
    every failure to read or parse a record that is THERE. Exactly one absence
    is an answer — no directory or no record for the owner — and returns
    ``(None, False)``: an owner who has switched nothing on. Never raises.
    """
    import stat

    try:
        if not os.path.isdir(tree):
            return None, True
        if not os.access(tree, os.R_OK | os.X_OK):
            return None, True
        try:
            marker_mode = os.lstat(os.path.join(tree, CONTROL_TREE_MARKER_NAME)).st_mode
        except OSError:
            return None, True
        if stat.S_ISLNK(marker_mode) or not stat.S_ISREG(marker_mode):
            return None, True
        if (
            not isinstance(owner, str)
            or owner.strip() in {"", ".", ".."}
            or _TREE_UNSAFE & set(owner)
        ):
            return None, True
        owner_dir = os.path.join(tree, owner)
        try:
            owner_mode = os.lstat(owner_dir).st_mode
        except FileNotFoundError:
            return None, False
        except OSError:
            return None, True
        if stat.S_ISLNK(owner_mode):
            return None, True
        try:
            text = _read_no_follow(os.path.join(owner_dir, RECORD_FILENAME))
        except FileNotFoundError:
            return None, False
        except Exception:
            return None, True
        record = parse_switch_record(text)
        return (None, True) if record is None else (record, False)
    except Exception:  # pragma: no cover - defensive; refuse rather than allow
        return None, True


def _write_refusal(section, target, owner, ignore_launch_pin, hook_input):
    """The first term of :func:`write_verdict` that refuses, else the position.

    Answers ``(reason, None)`` or ``(None, position)``. May raise.
    """
    if is_readonly_run():
        return "readonly_run", None
    if not tier_allows(section):
        return "tier_never", None
    if target_type(section, target) is None:
        return "target_unknown", None

    resolved = current_owner(owner)
    if resolved is NO_OWNER and _is_dispatch_run():
        return "owner_unscheduled", None
    if resolved is NO_OWNER or not isinstance(resolved, str) or not resolved.strip():
        return "owner_unknown", None
    # Mirrors the package: a host never lends its login's switch to another owner.
    if not _tree_is_bound() and resolved.strip() != acting_identity():
        return "owner_foreign", None

    rung = owner_tier_term(section, resolved)
    if rung == "unknown":
        return "owner_unknown", None
    if rung == TIER_NEVER:
        return "tier_never", None

    if _tree_is_bound():
        tree = _control_context_tree()
        if tree is None:
            return "record_unavailable", None
        record, unavailable = read_tree_record(tree, resolved)
        if unavailable:
            return "record_unavailable", None
    else:
        record, status = read_record_status(hook_input)
        if status == "unreadable":
            return "record_unavailable", None
        if status != "ok":
            record = None

    stored = switch_position(record, target)
    if stored == SWITCH_OFF:
        return "switch_off", None

    if not ignore_launch_pin and launch_pin_term(target) != "none":
        return "launch_pinned", None
    ceiling = rung if _tree_is_bound() else tier_writes(section)
    return None, min(stored, ceiling, key=_LOGIN_TIERS.index)


def write_verdict(section, target, *, owner=None, ignore_launch_pin=False, hook_input=None):
    """The verdict on one write to *target*: one total order, first refusal wins.

    The stdlib twin of ``posture_store.write_verdict``, term for term:
    ``readonly_run``, ``tier_never`` (the section's tier), ``target_unknown``,
    ``owner_unknown`` (the ladder found nobody; ``owner_unscheduled`` instead
    for a dispatch run), ``owner_foreign`` (no tree
    bound and an owner that is not :func:`acting_identity`), the login-tier rung
    (``owner_unknown`` / ``tier_never``), ``record_unavailable`` (the tree
    record under a bound tree, else this host's record — absent is not
    unavailable), ``switch_off`` (:func:`switch_position` reads off for
    *target*; the record's ``posture`` field is not read, so an old ``sandbox``
    narrowing with the machine armed is allowed), and ``launch_pinned`` unless
    *ignore_launch_pin*. An allowed verdict's ``position`` is the stored word
    capped by the ceiling: the section's tier on a host, the owner's rung under
    a bound tree.

    Args:
        section: The ``control_system:`` config section, as loaded.
        target: The control target the write lands on.
        owner: Whose record governs the write; ``None`` walks
            :func:`current_owner`.
        ignore_launch_pin: Skip the last term.
        hook_input: The hook's stdin payload, for repo-root resolution of the
            host record.

    Returns:
        A :class:`WriteVerdict`. Never raises: an exception in any term
        refuses as ``record_unavailable``, as the package does.
    """
    named = target if isinstance(target, str) else None
    kind = machine_kind(section, target)
    if kind is not None:
        kind = published_machine_kind(target, hook_input=hook_input)
    try:
        reason, position = _write_refusal(section, target, owner, ignore_launch_pin, hook_input)
    except Exception:
        reason, position = "record_unavailable", None
    return WriteVerdict(
        allowed=reason is None, reason=reason, target=named, kind=kind, position=position
    )


# -- which machine a hooked call writes to ---------------------------------
#
# One classifier for both PreToolUse hooks: ``osprey_writes_check`` denies
# exactly where ``osprey_approval`` defers, so the two must agree on which
# machine a call writes to before either asks the verdict about it.


class _Unresolved:
    """A write whose machine could not be placed: one instance, compared by identity.

    Distinct from ``None``, which :func:`hook_write_target` keeps for a call
    that writes nothing. A write nobody can place is refused, never waved
    through as if it were a read.
    """

    __slots__ = ()

    def __str__(self):
        return "<unresolved>"

    __repr__ = __str__


#: The answer for a write whose machine cannot be placed. See :class:`_Unresolved`.
UNRESOLVED = _Unresolved()

#: The hook_config key naming the tools a plan lane addresses, as short names.
_LANE_ADDRESSED_KEY = "lane_addressed_tools"

#: The hook_config key naming the ``control_system.write_tools`` entries the
#: render appended to ``write_tools``: the deployment's own write tools.
_FACILITY_EXTRAS_KEY = "control_system_write_tools"

#: The hook_config key naming the facility (custom-server) write set, fully
#: qualified: one exact name per tool a declaring server gates, and the
#: whole-server matcher ``mcp__<name>__.*`` of a server with no list. Rendered
#: verbatim from ``osprey.registry.mcp.facility_write_tools``, its one producer.
_FACILITY_WRITES_KEY = "facility_write_tools"

#: The wildcard a whole-server ``writes_check`` matcher ends in.
_SERVER_MATCHER_WILDCARD = ".*"

#: The service key of the default lane, and every service key that can name a
#: lane, in render order. Restated from ``osprey.bluesky_bridge_connection``
#: for the reason every literal here is: a hook cannot import the framework.
_LANE_ONE = "bluesky"
_LANE_KEYS = (_LANE_ONE, "bluesky_va", "bluesky_live", "bluesky_standin")


def _is_facility_tool(tool_name, hook_config):
    """Whether *tool_name* is a write tool the deployment declared, not the framework.

    Two producers, both read from the render: the ``facility_write_tools`` set
    (exact names for a server that declares ``writes:``, the whole-server
    matcher for one that does not), and a ``control_system.write_tools`` entry
    the render appended. An extra that names a framework tool was not appended,
    and stays a framework tool. A render from before that key carried no exact
    names, so there the whole-server matchers in ``write_tools`` are the set.
    """
    declared = hook_config.get(_FACILITY_WRITES_KEY)
    if isinstance(declared, list):
        if is_write_tool(tool_name, declared):
            return True
    else:
        for entry in hook_config.get("write_tools") or ():
            if (
                isinstance(entry, str)
                and entry.endswith(_SERVER_MATCHER_WILDCARD)
                and is_write_tool(tool_name, [entry])
            ):
                return True
    extras = hook_config.get(_FACILITY_EXTRAS_KEY) or ()
    return tool_name in [extra for extra in extras if isinstance(extra, str)]


def _lane_block(config, lane_key):
    """A lane's ``services.<lane>`` block, or ``{}``."""
    services = config.get("services") if isinstance(config, dict) else None
    block = services.get(lane_key) if isinstance(services, dict) else None
    return block if isinstance(block, dict) else {}


def _rendered_lanes(config):
    """``[(lane key, the target it serves)]`` in render order.

    Restates ``osprey_approval._rendered_lanes``: a lane block that declares no
    ``target`` serves the deployment baseline, and lane one is reported even
    when the config does not name it.
    """
    section = config.get("control_system") if isinstance(config, dict) else None
    baseline = _baseline_target(section)
    lanes = []
    for key in _LANE_KEYS:
        block = _lane_block(config, key)
        if not block:
            continue
        declared = block.get("target")
        lanes.append((key, declared if isinstance(declared, str) and declared else baseline))
    if not any(key == _LANE_ONE for key, _ in lanes):
        lanes.insert(0, (_LANE_ONE, baseline))
    return lanes


def _lane_machine(config, tool_input):
    """The target the call's lane serves, or :data:`UNRESOLVED`.

    A call naming no lane binds to the one lane a single-lane deployment has.
    No lane on a two-lane render, a lane the render does not carry, or a lane
    serving no known target cannot be placed.
    """
    lanes = _rendered_lanes(config)
    lane = tool_input.get("lane") if isinstance(tool_input, dict) else None
    if not lane:
        target = lanes[0][1] if len(lanes) == 1 else None
    else:
        target = next((served for key, served in lanes if key == lane), None)
    return target if target in CONTROL_TARGETS else UNRESOLVED


def _session_machine(config, hook_input):
    """The target the session is pointed at, or :data:`UNRESOLVED`.

    The record's target when it is readable. With no record at all, the
    baseline: nobody has switched, and with no record no switch is on, so the
    verdict refuses whichever target is named. :data:`UNRESOLVED` when the
    record cannot be read, and when the root was guessed and holds no record
    (:func:`posture_unknown`), where an absent record proves nothing.
    """
    result = read_target(hook_input)
    if not is_baseline(result):
        return result.get("target")
    if result.get("reason") != REASON_NO_STATE or posture_unknown(hook_input):
        return UNRESOLVED
    section = config.get("control_system") if isinstance(config, dict) else None
    return _baseline_target(section)


def hook_write_target(config, tool_name, short_name, hook_input):
    """The machine a hooked call writes to. Never raises.

    * ``None`` for a call that writes nothing: a tool outside the render's
      ``write_tools``, or a readonly execution.
    * :data:`TARGET_LIVE` for a facility write tool (:func:`_is_facility_tool`):
      the deployment declared it, and only the real machine's switch covers it.
      A declaring server's other tools are in no write list, so they are
      reads, and answer while ``live`` is off.
    * The lane's target for a lane-addressed tool (:func:`_lane_machine`).
    * The session's target for every other framework write tool
      (:func:`_session_machine`).

    :data:`UNRESOLVED` wherever a write cannot be placed, including any failure
    while answering: a write nobody can place is refused, never read as a read.

    Args:
        config: The loaded ``config.yml``, for the lane map and the baseline.
        tool_name: The full tool name from the hook payload.
        short_name: *tool_name* without its server prefix.
        hook_input: The hook's stdin payload: the call's ``tool_input`` and
            the repo root for the record.
    """
    try:
        tool_input = hook_input.get("tool_input") if isinstance(hook_input, dict) else None
        if not is_write_tool(tool_name, write_tools()):
            return None
        if not is_write_call(tool_name, tool_input, short_name):
            return None
        hook_config = load_hook_config()
        if _is_facility_tool(tool_name, hook_config):
            return TARGET_LIVE
        if short_name in tuple(hook_config.get(_LANE_ADDRESSED_KEY) or ()):
            return _lane_machine(config, tool_input)
        return _session_machine(config, hook_input)
    except Exception:
        return UNRESOLVED
