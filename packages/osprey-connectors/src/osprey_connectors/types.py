"""Connector type constants, and the fallbacks a factory applies to them.

Single source of truth for built-in connector type name strings.
Custom connectors use dotted module paths (e.g., 'mypackage.TangoConnector')
and don't need constants here.

The resolvers at the bottom are here rather than in
:mod:`osprey_connectors.factory` so that a guard deciding what a deployment
*will build* can share them with the factory that builds it. A guard which
re-implements "what does this config select" is a guard that can disagree with
the answer, and the disagreement is a bypass rather than a discrepancy — see
:mod:`osprey_connectors.honesty`.

:func:`resolve_target` extends that to the run-time question "which machine is
this session pointed at". It is the same shape of answer for the same reason:
several holders follow a session target — the connector-host process, its child,
an executor sandbox — and a holder that translates ``live`` into a connector type
privately can route a tool call somewhere the roster never claimed. The target is
an *argument* here, never something this module reads from the environment: an
environment-reading override would make the honesty predicate describe a
deployment the process is no longer running.
"""

from typing import Any

# -- Control system connector types (have implementations) --
MOCK = "mock"
EPICS = "epics"
VIRTUAL_ACCELERATOR = "virtual_accelerator"
DOOCS = "doocs"

# -- Archiver connector types --
MOCK_ARCHIVER = "mock_archiver"
EPICS_ARCHIVER = "epics_archiver"
MONGODB_ARCHIVER = "mongodb_archiver"
DOOCS_ARCHIVER = "doocs_archiver"

# -- CLI choice lists (only types with implementations) --
CLI_CONTROL_SYSTEM_TYPES = [MOCK, EPICS, VIRTUAL_ACCELERATOR, DOOCS]
CLI_ARCHIVER_TYPES = [MOCK_ARCHIVER, EPICS_ARCHIVER, MONGODB_ARCHIVER, DOOCS_ARCHIVER]

# -- Session targets --
# What a *session* can be pointed at, as opposed to what a *config* selects.
# Deliberately two names and not one per connector type: a target names a
# machine ("the real one" / "the simulated one"), and which connector type
# reaches that machine is this module's job to answer, not the caller's.
TARGET_LIVE = "live"
TARGET_VA = "va"
CONTROL_TARGETS = [TARGET_LIVE, TARGET_VA]

#: Types that serve a machine nobody has to be careful around. They are the
#: reason ``live`` cannot simply be "whatever the config selects": a deployment
#: whose baseline is one of these has not yet said what its real machine is.
_SIMULATED_TYPES = (MOCK, VIRTUAL_ACCELERATOR)


def _resolve_type(section: Any, fallback: str) -> str:
    """The connector type a factory builds from a config section.

    A section that is missing, is not a mapping, or carries no usable ``type``
    resolves to *fallback* — the factory's documented fail-closed default, which
    it announces with a ``… is not set; defaulting to …`` warning. Empty and
    ``None`` count as absent (YAML gives ``None`` for a bare ``type:``); any
    other value is returned as written, so a typo reaches the factory's
    "Unknown … type" error rather than being quietly rounded to something.
    """
    value = section.get("type") if isinstance(section, dict) else None
    return str(value) if value else fallback


def resolve_archiver_type(section: Any) -> str:
    """The archiver an ``archiver:`` config section actually selects.

    Absent means the mock: a config that says nothing about its archiver is a
    config that gets the synthesizing one, which is the fact the honesty rule is
    really about.
    """
    return _resolve_type(section, MOCK_ARCHIVER)


def resolve_control_system_type(section: Any) -> str:
    """The control system a ``control_system:`` config section actually selects."""
    return _resolve_type(section, MOCK)


def resolve_target(section: Any, target: Any) -> str:
    """The connector type a session *target* selects, in a given deployment.

    The returned type is also the key of the block the factory reads its
    settings from — ``control_system.connector.<type>`` — so one string answers
    both "what do I build" and "where are its gateways".

    ``va`` is the same answer everywhere: a virtual accelerator is a virtual
    accelerator regardless of what the deployment was built for. ``live`` is
    deployment-specific and is the half that can refuse:

    - When the section's own type is a real control system, that is the live
      machine, and it is returned as written — including a value this module
      does not recognize, which reaches the factory's "Unknown … type" error
      exactly as :func:`resolve_control_system_type` already lets it.
    - When the section's own type is simulated (or absent, which resolves to the
      mock), the deployment has not named its real machine there, so the live
      type is taken from the connector table: exactly one non-simulated block
      configured means the deployment has said which machine it means. None, or
      more than one, raises.

    Nothing is inferred from the absence of evidence. A deployment that has
    never been told about its real machine gets an error naming what it would
    have to configure, because the alternative — resolving ``live`` to the one
    connector type that talks to hardware because it is the usual one — is a
    tool call arriving at a real machine on the strength of a guess.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes.
        target: The session target, one of :data:`CONTROL_TARGETS`. An argument,
            never read from the environment, and never defaulted.

    Returns:
        The connector type, which doubles as the connector sub-block key.

    Raises:
        ValueError: If *target* is not a known target (including ``None`` and
            blank), or if it is ``live`` on a deployment with no single
            derivable live control system.
    """
    if target == TARGET_VA:
        return VIRTUAL_ACCELERATOR
    if target == TARGET_LIVE:
        return _live_type(section)
    raise ValueError(
        f"Unknown control target {target!r}. Valid targets are "
        f"{TARGET_LIVE!r} and {TARGET_VA!r}, spelled exactly. A session target "
        "is always stated, never defaulted — there is no target a caller gets "
        "by saying nothing, because the one it would get could be the real "
        "machine."
    )


def switch_capable(section: Any) -> bool:
    """Whether a deployment can be pointed at either target.

    The predicate itself, in the module that already owns "which connector type
    does a target mean here". Two layers ask the question and must never answer
    it differently: the controls server decides from it whether its tools are
    served by a connector-host child, and the build decides from it whether the
    agent's frozen safety rule describes a switchable machine at all. A build
    that promises a switch the runtime will not perform is worse than one that
    never mentions it.

    Three conditions, none of which is sufficient alone:

    1. **Both targets resolve** to a connector type at all
       (:func:`resolve_target`, which raises rather than guess a live machine).
    2. **The deployment's own control system is one of the two targets.**
       ``resolve_target`` answers ``live`` for configs with no business
       switching: a ``mock`` deployment that happens to carry an ``epics`` block
       resolves ``live`` to ``epics``, and treating that as switchable would
       point a session at a real machine the config never selected. Requiring
       the baseline target to resolve back to ``control_system.type`` rules it
       out.
    3. **Both types have a non-empty connector block**, since that block is what
       a connector is configured from.

    Deliberately *not* checked: ``probe_channel``, the gateways table, the
    operator acknowledgment. Those decide whether a target may be switched *to*
    — a per-switch question answered with a reason the operator is told. This is
    the coarser question of whether the deployment is in the two-target world.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes. Callers holding a whole
            rendered config pass ``config.get("control_system")``.

    Returns:
        ``True`` when both targets are configured and consistent. Never raises:
        every malformed, partial or contradictory config is simply not capable.
    """
    if not isinstance(section, dict):
        return False
    try:
        types_by_target = {
            target: resolve_target(section, target) for target in (TARGET_LIVE, TARGET_VA)
        }
    except ValueError:
        return False

    declared = resolve_control_system_type(section)
    baseline = TARGET_VA if declared == VIRTUAL_ACCELERATOR else TARGET_LIVE
    if types_by_target[baseline] != declared:
        return False

    connector = section.get("connector")
    if not isinstance(connector, dict):
        return False
    return all(
        isinstance(connector.get(name), dict) and bool(connector.get(name))
        for name in types_by_target.values()
    )


def _live_type(section: Any) -> str:
    """The control system type that reaches this deployment's real machine."""
    baseline = resolve_control_system_type(section)
    if baseline not in _SIMULATED_TYPES:
        return baseline

    connector = section.get("connector") if isinstance(section, dict) else None
    candidates: list[str] = []
    if isinstance(connector, dict):
        candidates = sorted(
            key for key in connector if isinstance(key, str) and key not in _SIMULATED_TYPES
        )
    if len(candidates) == 1:
        return candidates[0]

    found = ", ".join(repr(name) for name in candidates) if candidates else "none"
    raise ValueError(
        f"Target {TARGET_LIVE!r} has no control system on this deployment: "
        f"'control_system.type' resolves to {baseline!r}, which is simulated, "
        f"and the live blocks under 'control_system.connector' are: {found}. "
        "Exactly one non-simulated connector block is what names the real "
        f"machine (an {EPICS!r} or {DOOCS!r} block, say); configure it there. "
        "It is not inferred, so that no session reaches a real machine this "
        "config never described."
    )
