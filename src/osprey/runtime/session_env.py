"""The process environment a control session refuses, and the part it drops.

:meth:`~osprey.runtime.session.ControlSession.open` calls
:func:`check_environment` first: a session must not run with a second source
of truth for what the control-context record already answers. It then calls
:func:`scrub_epics_environment`, so the connector reaches the gateways the
config names and never ones a parent process exported.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

#: Variables a session process must not inherit. Each is a second source of
#: truth for something the live record already answers — the launch pin, the
#: read-only tree, an owner claim — and a session that ran with one would be
#: governed by whichever reader happened to look first.
FORBIDDEN_ENV = (
    "OSPREY_CONTROL_CONTEXT_TREE",
    "OSPREY_CONTROL_OWNER",
    "OSPREY_LAUNCH_POSTURE",
)

#: Every variable with this prefix is a target stamp (target, generation,
#: refusal). A stamp belongs to a run that was launched *for* a target; a
#: session reads the record instead.
FORBIDDEN_ENV_PREFIX = "OSPREY_CONTROL_TARGET"

#: Inherited Channel Access / PV Access settings are dropped before the
#: connector is built, so the gateways it reaches are the ones the config names
#: for its target and never the ones a parent process happened to export.
SCRUBBED_ENV_PREFIX = "EPICS_"


class SessionEnvironmentError(RuntimeError):
    """The process environment is one a control session must not run in.

    Raised by :meth:`~osprey.runtime.session.ControlSession.open` before
    anything is read or built. It is a deployment mistake, not a transient
    condition: a caller exits with
    :data:`~osprey.runtime.session.EXIT_START_REFUSED` rather than retrying.

    Attributes:
        problems: One sentence per variable that is wrong, in a stable order.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__(
            "Refusing to open a control session in this environment: " + "; ".join(self.problems)
        )


def check_environment(environ: Mapping[str, str]) -> None:
    """Raise :class:`SessionEnvironmentError` unless *environ* suits a session."""
    from osprey_connectors.identity import AUDIT_IDENTITY_ENV
    from osprey_connectors.posture_store import CONTROL_CONTEXT_DIR_ENV_VAR

    def blank(name: str) -> bool:
        return not (environ.get(name) or "").strip()

    problems = [
        f"{name} is not set"
        for name in (CONTROL_CONTEXT_DIR_ENV_VAR, AUDIT_IDENTITY_ENV, "OSPREY_CONFIG")
        if blank(name)
    ]
    if not blank("OSPREY_CONFIG") and environ.get("OSPREY_CONFIG") != environ.get("CONFIG_FILE"):
        problems.append("OSPREY_CONFIG and CONFIG_FILE do not name the same file")
    present = [name for name in FORBIDDEN_ENV if name in environ]
    present += sorted(name for name in environ if name.startswith(FORBIDDEN_ENV_PREFIX))
    problems.extend(f"{name} must not be set" for name in present)
    if problems:
        raise SessionEnvironmentError(problems)


def scrub_epics_environment() -> None:
    """Drop every inherited ``EPICS_*`` variable from this process."""
    for name in [name for name in os.environ if name.startswith(SCRUBBED_ENV_PREFIX)]:
        del os.environ[name]
