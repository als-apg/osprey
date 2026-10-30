"""Read-time attribution: the person an entry belongs to, from trusted sources only.

The person is never stored. The reader computes it for every entry it reads,
from three things only:

**The source identity.** An entry from a web terminal (its transcript volume,
its spool directory or its OTLP login) belongs to that terminal's user, and an
entry from the host projects mount belongs to ``deployment.host_identity``. The
identity was fixed at ingest from where the entry was found, so it is already
trusted.

**The trusted on-behalf list.** The sidecar's ``on_behalf_of`` names the person
on every audit surface, and a rendered queueserver's on surface ``preflight``
only (:func:`osprey.agent_record.identities.keep_on_behalf_of`). Ingest already
drops every other sender's value; the rule is applied again here, so an entry
that reaches the reader by any other path still cannot name a person.

**The run entries.** A dispatch identity's session belongs to the requester of
the first ``run:<run_id>:accept`` entry naming that session from the same
(mapped) dispatch identity, ordered by arrival. Because the join is computed
at read time over whatever run entries exist, an entry that landed before its
run's accept entry is attributed as soon as the accept entry is present, with
no update. A run with no requester (a scheduled run) names no person, and an
accept entry that is not its session's first names no person either. Shared-mode
workers are mapped to one identity through
:func:`osprey.agent_record.identities.spool_source_identity`, so the join holds
in both workspace modes.

Nothing else an entry carries counts: body fields, OTLP attributes and an
untrusted ``on_behalf_of`` are never read as a person. Every other source
(bridges, the history front, the operator, readers, other services) has no
person of its own.

The module holds no I/O and stores nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from osprey.agent_record.contract import (
    KIND_RUN,
    RUN_ACCEPT,
    RUN_KEY_PREFIX,
    SOURCE_DISPATCH,
    SOURCE_HOST,
    SOURCE_TERMINAL,
    SURFACE_FIELD,
    Entry,
    accept_session,
    arrival_order,
    run_id_of,
)
from osprey.agent_record.identities import (
    IdentityClasses,
    TrustedOnBehalf,
    keep_on_behalf_of,
    spool_source_identity,
    trusted_on_behalf,
)

#: The body field a run accept entry names its requester in.
REQUESTER_FIELD = "requester"

#: The source classes whose identity is the person.
_PERSONAL_CLASSES = frozenset({SOURCE_TERMINAL, SOURCE_HOST})


def is_run_accept(entry: Entry) -> bool:
    """Whether *entry* is a ``run:<run_id>:accept`` entry with a whole body."""
    if entry.kind != KIND_RUN or entry.chunk.total != 1:
        return False
    parts = entry.key.split(":")
    return len(parts) == 3 and parts[0] == RUN_KEY_PREFIX and parts[2] == RUN_ACCEPT


def _requester(body: Mapping[str, Any]) -> str | None:
    value = body.get(REQUESTER_FIELD)
    if isinstance(value, str) and value.strip():
        return value
    return None


@dataclass(frozen=True)
class Attribution:
    """The person of every entry, for one read.

    Build it with :meth:`build` from the deployment config and the run entries
    the read can see, then ask :meth:`person` per entry.
    """

    config: Mapping[str, Any]
    trusted: TrustedOnBehalf
    #: ``(dispatch identity, session_id)`` → requester (``None``: no person).
    sessions: Mapping[tuple[str, str], str | None]
    #: ``(dispatch identity, run_id)`` → requester, for the runs that won their session.
    runs: Mapping[tuple[str, str], str | None]
    _class: IdentityClasses = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_class", IdentityClasses(self.config))

    @classmethod
    def build(cls, config: Mapping[str, Any] | None, run_entries: Iterable[Entry]) -> Attribution:
        """Attribution over *run_entries*; entries that are not run accepts are skipped.

        Each session's requester comes from the first accept entry, by arrival,
        that names it under a dispatch identity. Later accepts naming the same
        session name no one.
        """
        config = dict(config or {})
        attribution = cls(config, trusted_on_behalf(config), {}, {})
        sessions: dict[tuple[str, str], str | None] = {}
        runs: dict[tuple[str, str], str | None] = {}
        accepts = sorted((e for e in run_entries if is_run_accept(e)), key=arrival_order)
        for accept in accepts:
            if attribution._class(accept.identity) != SOURCE_DISPATCH:
                continue
            identity = attribution._dispatch_identity(accept.identity)
            session = accept_session(accept)
            run_id = run_id_of(accept)
            if not session or (identity, session) in sessions:
                continue
            requester = _requester(accept.body)
            sessions[(identity, session)] = requester
            runs[(identity, run_id)] = requester
        object.__setattr__(attribution, "sessions", sessions)
        object.__setattr__(attribution, "runs", runs)
        return attribution

    def _dispatch_identity(self, identity: str) -> str:
        return spool_source_identity(self.config, identity)

    def person(self, entry: Entry) -> str | None:
        """The person *entry* belongs to, or ``None`` when no trusted source names one."""
        if entry.on_behalf_of:
            surface = entry.body.get(SURFACE_FIELD)
            if keep_on_behalf_of(
                self.trusted, entry.identity, surface if isinstance(surface, str) else None
            ):
                return entry.on_behalf_of
        source_class = self._class(entry.identity)
        if source_class in _PERSONAL_CLASSES:
            return entry.identity
        if source_class != SOURCE_DISPATCH:
            return None
        identity = self._dispatch_identity(entry.identity)
        if entry.kind == KIND_RUN:
            # A run entry belongs to its run's requester, and only when that run
            # won its session: a refused accept names no one.
            return self.runs.get((identity, run_id_of(entry)))
        if entry.session_id and (identity, entry.session_id) in self.sessions:
            return self.sessions[(identity, entry.session_id)]
        if entry.run_id:
            return self.runs.get((identity, entry.run_id))
        return None
