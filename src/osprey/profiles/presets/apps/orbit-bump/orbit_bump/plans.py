"""The plan store: exactly one current plan per owner, in memory, behind one lock.

A plan is the HTTP contract's ``Plan`` dict. The store owns its ``id``, ``owner``,
``status`` and ``phase``; every other field is the caller's. Plans handed out are
deep copies, so nothing outside the lock ever mutates a stored plan. The store lives
for the process: a restart leaves no plans.

Status transitions (the contract's table)::

    ready          -> applying | superseded | discarded
    applying       -> ready | applied | partially_applied | discarded
    not_applicable -> superseded | discarded
    applied        -> superseded | discarded
    partially_applied -> superseded | discarded
    superseded, discarded: terminal

``applying -> applying`` is allowed as a field update (``phase``, row outcomes).
"""

from __future__ import annotations

import copy
import itertools
import threading
import uuid
from typing import Any

READY = "ready"
NOT_APPLICABLE = "not_applicable"
APPLYING = "applying"
APPLIED = "applied"
PARTIALLY_APPLIED = "partially_applied"
SUPERSEDED = "superseded"
DISCARDED = "discarded"

#: Every plan status the contract names.
STATUSES = (READY, NOT_APPLICABLE, APPLYING, APPLIED, PARTIALLY_APPLIED, SUPERSEDED, DISCARDED)
#: The statuses a freshly computed plan may carry.
INITIAL = (READY, NOT_APPLICABLE)
#: The statuses that are never current: a plan in one has left its owner.
GONE = (SUPERSEDED, DISCARDED)

#: status -> the statuses it may move to.
TRANSITIONS: dict[str, frozenset[str]] = {
    READY: frozenset({APPLYING, SUPERSEDED, DISCARDED}),
    APPLYING: frozenset({APPLYING, READY, APPLIED, PARTIALLY_APPLIED, DISCARDED}),
    NOT_APPLICABLE: frozenset({SUPERSEDED, DISCARDED}),
    APPLIED: frozenset({SUPERSEDED, DISCARDED}),
    PARTIALLY_APPLIED: frozenset({SUPERSEDED, DISCARDED}),
    SUPERSEDED: frozenset(),
    DISCARDED: frozenset(),
}

#: Fields only the store writes.
_OWNED_FIELDS = frozenset({"id", "owner", "status"})

# A per-process prefix plus a counter: unique within the process, opaque to callers.
_PREFIX = uuid.uuid4().hex[:12]
_COUNTER = itertools.count(1)


class PlanStoreError(ValueError):
    """A plan-store refusal; the message names the plan and the reason."""


class PlanNotFound(PlanStoreError):
    """No plan has this id."""


class PlanConflict(PlanStoreError):
    """The plan's status does not allow the asked move."""


class PlanNotCurrent(PlanStoreError):
    """The id is not the owner's current plan; ``current_id`` names the one that is."""

    def __init__(self, plan_id: str, owner: str, current_id: str | None) -> None:
        self.plan_id = plan_id
        self.owner = owner
        self.current_id = current_id
        current = f"the current plan is {current_id}" if current_id else "there is no current plan"
        super().__init__(f"plan {plan_id} is not {owner}'s current plan; {current}")


def _new_id() -> str:
    return f"{_PREFIX}-{next(_COUNTER)}"


class PlanStore:
    """One current plan per owner, every plan kept by id, all moves under one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._plans: dict[str, dict[str, Any]] = {}
        self._current: dict[str, str] = {}

    # -- queries -----------------------------------------------------------------

    def current(self, owner: str) -> dict[str, Any] | None:
        """The owner's current plan, or None when there is none or it was discarded."""
        with self._lock:
            plan = self._current_plan(owner)
            return copy.deepcopy(plan) if plan is not None else None

    def get(self, plan_id: str) -> dict[str, Any]:
        """The plan with this id, whatever its status; raises PlanNotFound."""
        with self._lock:
            return copy.deepcopy(self._lookup(plan_id))

    # -- moves -------------------------------------------------------------------

    def set_current(self, owner: str, plan: dict[str, Any]) -> dict[str, Any]:
        """Store ``plan`` as the owner's current one under a new id and return it.

        The previous current plan becomes ``superseded``; refused while it is
        ``applying``. ``plan["status"]`` must be ``ready`` or ``not_applicable``.
        """
        status = plan.get("status")
        if status not in INITIAL:
            raise PlanStoreError(f"a new plan must be {' or '.join(INITIAL)}, not {status!r}")
        stored = copy.deepcopy(plan)
        stored.pop("phase", None)
        with self._lock:
            previous = self._current_plan(owner)
            if previous is not None:
                if previous["status"] == APPLYING:
                    raise PlanConflict(
                        f"{owner}'s plan {previous['id']} is applying; "
                        "a new plan waits until it finishes"
                    )
                self._move(previous, SUPERSEDED)
            stored["id"] = _new_id()
            stored["owner"] = owner
            self._plans[stored["id"]] = stored
            self._current[owner] = stored["id"]
            return copy.deepcopy(stored)

    def discard(self, owner: str) -> dict[str, Any] | None:
        """Discard the owner's current plan and return it; None when there is none.

        Refused while it is ``applying``.
        """
        with self._lock:
            plan = self._current_plan(owner)
            if plan is None:
                return None
            if plan["status"] == APPLYING:
                raise PlanConflict(
                    f"{owner}'s plan {plan['id']} is applying; it cannot be discarded now"
                )
            self._move(plan, DISCARDED)
            return copy.deepcopy(plan)

    def discard_for_orm_change(self, model: str) -> list[str]:
        """Discard every owner's ``ready``/``not_applicable`` plan on ``model``.

        Returns the discarded ids. Applying and terminal plans are left as they are.
        """
        with self._lock:
            hit = [
                plan
                for plan in (self._plans[pid] for pid in self._current.values())
                if plan.get("model") == model and plan["status"] in INITIAL
            ]
            for plan in hit:
                self._move(plan, DISCARDED)
            return [plan["id"] for plan in hit]

    def claim(self, plan_id: str, owner: str) -> dict[str, Any]:
        """Move the owner's current plan ``ready -> applying`` (phase ``writing``).

        Raises PlanNotCurrent for any id that is not the owner's current plan, and
        PlanConflict when the plan is not ``ready`` (a concurrent claim included).
        """
        with self._lock:
            current_id = self._current.get(owner)
            if current_id != plan_id:
                raise PlanNotCurrent(plan_id, owner, current_id)
            plan = self._plans[plan_id]
            if plan["status"] != READY:
                raise PlanConflict(f"plan {plan_id} is {plan['status']}, not ready")
            self._move(plan, APPLYING, phase="writing")
            return copy.deepcopy(plan)

    def release(self, plan_id: str) -> dict[str, Any]:
        """Move a claimed plan ``applying -> ready`` (a refusal before any write)."""
        with self._lock:
            plan = self._lookup(plan_id)
            if plan["status"] != APPLYING:
                raise PlanConflict(f"plan {plan_id} is {plan['status']}, not applying")
            self._move(plan, READY)
            return copy.deepcopy(plan)

    def transition(self, plan_id: str, status: str, **fields: Any) -> dict[str, Any]:
        """Move a plan to ``status`` per the contract's table and set ``fields`` on it.

        ``phase`` survives only while the plan is ``applying``. Moving to
        ``superseded`` or ``discarded`` drops the plan from its owner.
        """
        if status not in STATUSES:
            raise PlanStoreError(f"unknown plan status {status!r}; one of {', '.join(STATUSES)}")
        owned = sorted(_OWNED_FIELDS & fields.keys())
        if owned:
            raise PlanStoreError(f"the store owns {', '.join(owned)}; transition cannot set it")
        with self._lock:
            plan = self._lookup(plan_id)
            if status not in TRANSITIONS[plan["status"]]:
                raise PlanConflict(f"plan {plan_id} cannot move from {plan['status']} to {status}")
            self._move(plan, status, **fields)
            return copy.deepcopy(plan)

    # -- internals (lock held) ---------------------------------------------------

    def _lookup(self, plan_id: str) -> dict[str, Any]:
        try:
            return self._plans[plan_id]
        except KeyError:
            raise PlanNotFound(f"no plan {plan_id}") from None

    def _current_plan(self, owner: str) -> dict[str, Any] | None:
        plan_id = self._current.get(owner)
        if plan_id is None:
            return None
        plan = self._plans[plan_id]
        return None if plan["status"] in GONE else plan

    def _move(self, plan: dict[str, Any], status: str, **fields: Any) -> None:
        plan.update(copy.deepcopy(fields))
        plan["status"] = status
        if status != APPLYING:
            plan.pop("phase", None)
        if status in GONE and self._current.get(plan["owner"]) == plan["id"]:
            del self._current[plan["owner"]]
