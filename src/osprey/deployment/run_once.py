"""Services that run once: what a deployment declares, and what a start makes of it.

A start that returns has handed every service to compose to run in the
background, and checks that each one has a running container. A service that
runs once (a migration, a seed, a one-shot import) has none afterwards, by
design. It says so with a compose label on its service::

    services:
      seed-store:
        labels:
          osprey.run-once: must-succeed   # or: may-fail

``must-succeed``
    It counts as started when one of its containers exited 0. Any other end
    fails the start, naming the service and its exit code.
``may-fail``
    It counts as started when one of its containers exited, whatever the code.
    A non-zero exit is reported by name and code, and the start succeeds.

A service another one waits on with ``condition: service_completed_successfully``
is run-once by compose's own declaration and counts as ``must-succeed``. A
service that declares nothing is never taken for run-once, whatever its
restart policy or exit code: a long-running service that stopped cleanly is a
service that is not running.

The check sees the moment the start returns. A run-once service still running
then counts as running; its later exit is not seen. A deployment that needs the
start to wait for it makes another service depend on it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "MAY_FAIL",
    "MUST_SUCCEED",
    "RUN_ONCE_LABEL",
    "VALUES",
    "declarations",
    "declaration_errors",
    "exit_code",
    "service_labels",
]

#: The compose label a service declares itself run-once with.
RUN_ONCE_LABEL = "osprey.run-once"
#: A run-once service whose non-zero exit fails the start.
MUST_SUCCEED = "must-succeed"
#: A run-once service whose non-zero exit is reported and does not fail the start.
MAY_FAIL = "may-fail"
#: The values the label takes.
VALUES = (MUST_SUCCEED, MAY_FAIL)
#: Restart policies under which the runtime starts an exited container again.
_RERUNNING = frozenset({"always", "unless-stopped"})
_EXITED = re.compile(r"^exited\s*\((-?\d+)\)", re.IGNORECASE)


def service_labels(spec: Any) -> dict[str, str]:
    """A compose service's ``labels``, in either compose spelling, as a mapping."""
    labels = spec.get("labels") if isinstance(spec, Mapping) else None
    if isinstance(labels, Mapping):
        return {str(k): "" if v is None else str(v) for k, v in labels.items()}
    if isinstance(labels, list):
        found: dict[str, str] = {}
        for item in labels:
            key, _, value = str(item).partition("=")
            found[key] = value
        return found
    return {}


def _waited_on(services: Mapping[str, Any]) -> set[str]:
    """The services another one waits on with ``condition: service_completed_successfully``."""
    waited: set[str] = set()
    for spec in services.values():
        depends = spec.get("depends_on") if isinstance(spec, Mapping) else None
        if not isinstance(depends, Mapping):
            continue
        for name, condition in depends.items():
            if (
                isinstance(condition, Mapping)
                and condition.get("condition") == "service_completed_successfully"
            ):
                waited.add(str(name))
    return waited


def declarations(services: Mapping[str, Any]) -> dict[str, str]:
    """Each run-once service of *services* and how it may end: :data:`MUST_SUCCEED`
    or :data:`MAY_FAIL`. A value the label does not take is left out here and
    refused by :func:`declaration_errors`."""
    found = {str(name): MUST_SUCCEED for name in _waited_on(services) if name in services}
    for name, spec in services.items():
        value = service_labels(spec).get(RUN_ONCE_LABEL)
        if value in VALUES:
            found[str(name)] = value
    return found


def declaration_errors(services: Mapping[str, Any]) -> list[str]:
    """Everything wrong with the run-once declarations of rendered *services*.

    Three things are refused: a label value other than :data:`VALUES`; a
    run-once service with a restart policy that starts it again after it
    exits; and ``may-fail`` on a service another one waits on, because compose
    then fails the services that wait on it when it exits non-zero.
    """
    errors: list[str] = []
    waited = _waited_on(services)
    for name, spec in services.items():
        labels = service_labels(spec)
        if RUN_ONCE_LABEL not in labels:
            continue
        value = labels[RUN_ONCE_LABEL]
        if value not in VALUES:
            errors.append(
                f"service {name!r} sets {RUN_ONCE_LABEL}={value!r}; the label takes "
                f"{' or '.join(VALUES)}."
            )
            continue
        restart = str(spec.get("restart") or "no") if isinstance(spec, Mapping) else "no"
        if restart in _RERUNNING:
            errors.append(
                f"service {name!r} is run-once ({RUN_ONCE_LABEL}={value}) but has "
                f"restart: {restart}, which starts it again each time it exits. Use "
                "restart: no or on-failure, or drop the label."
            )
        if value == MAY_FAIL and name in waited:
            errors.append(
                f"service {name!r} is {RUN_ONCE_LABEL}={MAY_FAIL}, but another service waits "
                "on it with condition: service_completed_successfully, so compose fails "
                f"that service when it exits non-zero. Declare it {MUST_SUCCEED}, or let "
                "nothing wait on it."
            )
    return errors


def exit_code(row: Mapping[str, Any]) -> int | None:
    """The exit code of an exited container's ``ps`` row, in either runtime's
    spelling, or ``None`` when the row is no exited container or does not say."""
    code = row.get("ExitCode")
    state = str(row.get("State") or "").lower()
    status = str(row.get("Status") or "")
    if state not in ("", "exited") and not status.lower().startswith("exited"):
        return None
    if isinstance(code, int) and not isinstance(code, bool) and state == "exited":
        return code
    match = _EXITED.match(status.strip())
    return int(match.group(1)) if match else None
