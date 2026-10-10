"""The one connector pairing a deployment may not have: invented history.

A virtual accelerator served from its container serves channels that move for
modelled reasons — a corrector is stepped, the orbit responds, and the numbers a
tool reads back are answers to what the deployment actually did — and its
recorder writes that history as it happens. The live stand-in makes the same
kind of claim from a soft IOC the deployment runs for itself. Neither machine
existed before this deployment stood it up, so neither has a past anybody
recorded beyond what its own recorder keeps — they are the
:data:`~osprey_connectors.types.INVENTED_HISTORY_TYPES`, reached over a
network, and every question below is asked of that whole set rather than of the
virtual accelerator alone. The mock archiver answers a history query the other
way round: it synthesizes a plausible-looking series at read time, for
questions nobody recorded the answer to. Configured together they produce an
agent whose past is fiction and whose present is not, with nothing connecting
the two — so the fiction can never be caught by disagreeing with the machine it
claims to describe.

Refused at every moment a deployment can acquire the pairing: ``osprey build``
writes the config, ``osprey up`` stands the services up, the MCP server reads
whatever ``config.yml`` it is pointed at — including one hand-edited long after
the build — and, at run time, a session asks to be pointed at the simulator.
Each site raises in its own vocabulary and names its own fix. What they share,
and what lives here, is the question they ask and the reason they ask it.

**The question is asked twice, because the two kinds of config are read by
different readers, and a guard must resolve a key exactly as the reader it
guards resolves it — otherwise the divergence *is* the bypass.**

- A build profile's ``config:`` block reaches the rendered project through the
  emitter, which honors the dotted spelling (``archiver.type:``, the canonical
  one) *and* a nested mapping — both land on the same rendered leaf. Both are
  therefore live, and :func:`pairing_in_profile` fails closed when they
  disagree: whichever one wins, the profile has stated the archive twice and is
  free to be wrong once.
- A rendered ``config.yml`` is read by :class:`~osprey_connectors.config.ConfigBuilder`
  and by ``MCPServerConfig``, and both walk *nested sections only* — a top-level
  ``archiver.type:`` line there sets nothing at all. So
  :func:`pairing_in_rendered_config` resolves nested-only. A flat line is not
  evidence of an archiver; it is an inert line, and one this module names in its
  message rather than silently reading as "unset", because someone who typed it
  deserves to be told why it did nothing.

Both readers fall back to the mock archiver when ``archiver.type`` is absent or
blank (see the factory's ``… is not set; defaulting to …`` warnings), so *unset
counts as the mock archiver* at every site. That is the fallback the rule is
really about: the common way into the pairing is not naming the mock archiver,
it is naming nothing.

The venue is part of the question. The simulator served from its container
has a recorder that writes its real history, so a synthesized archive beside
it is fiction. The simulator served in process has no recorder, so a
synthesized archive is the only archive it can have, and the chip says
"Simulator": that pairing claims nothing it cannot back up and is not refused.
:func:`_invents_history` asks it through
:func:`~osprey_connectors.types.talks_to_network`, the same wire fact every
other check reads, so the type and the ``serving`` leaf are judged together and
never the type word alone. A control system that states no type is the
simulator in process, so it invents nothing either.

The run-time question is the same question asked one step early. A session that
asks to be pointed at the virtual accelerator, or at the stand-in, has not
changed anything yet, so the config's own ``control_system.type`` is not the
type to judge — the pairing to refuse is the one the switch *would* create.
:func:`pairing_for_target` judges that one, taking the prospective control
system from :func:`~osprey_connectors.types.resolve_target` while the archiver
still comes from the config, because the switch changes the machine and leaves
the archive exactly where it was. The target is resolved through that shared
resolver and never here: a guard that translates ``va`` privately is guarding a
deployment other than the one the switch will produce, which is the same
divergence-is-the-bypass this module refuses on the config keys. A
target is judged as the section it builds from — the deployment's own, with
its type replaced by the target's — so on a deployment whose own type is a
real machine, ``va`` is the served container and is judged served.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import (
    INVENTED_HISTORY_TYPES,
    MOCK_ARCHIVER,
    SERVING_KEY,
    SERVING_LEAF,
    VIRTUAL_ACCELERATOR,
    resolve_archiver_type,
    resolve_control_system_type,
    resolve_target,
    talks_to_network,
)

#: Why the pairing is refused, in one sentence pair every site shares so the
#: explanation cannot drift between them. Each site supplies its own fix.
VA_MOCK_ARCHIVER_WHY = (
    "a virtual accelerator and the live stand-in are machines this deployment "
    "stands up for itself, with no past anybody recorded, while the mock archiver "
    "synthesizes history at read time. Paired, the agent reports a past that never "
    "happened and nothing can catch it — the one failure a simulated facility "
    "exists to make visible rather than to have."
)

_CONTROL_SYSTEM_TYPE = "control_system.type"
_ARCHIVER_TYPE = "archiver.type"


def _invents_history(control_system_section: Any) -> bool:
    """Whether the machine *control_system_section* selects has no recorded past.

    A type in :data:`~osprey_connectors.types.INVENTED_HISTORY_TYPES` that is
    reached over a network: the simulator served from its container and the
    live stand-in. The simulator served in process dials nothing and has no
    recorder, so it is not one. A ``serving`` value no reader accepts is judged
    served, the refusing side; the build names that value on its own.
    """
    if resolve_control_system_type(control_system_section) not in INVENTED_HISTORY_TYPES:
        return False
    try:
        return talks_to_network(control_system_section)
    except ValueError:
        return True


def _target_section(control_system_section: Any, connector_type: str) -> dict[str, Any]:
    """The section a target builds from: the deployment's own, with its type replaced."""
    section = control_system_section if isinstance(control_system_section, dict) else {}
    return {**section, "type": connector_type}


class _Absent:
    """Sentinel for "this spelling does not set the key", distinct from a key
    set to ``None`` — which YAML produces for a bare ``archiver.type:`` and which
    the factory resolves to the mock."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<absent>"


_ABSENT = _Absent()


@dataclass(frozen=True)
class ArchiverPairing:
    """What a config says about its archive, judged the way its reader reads it."""

    is_invented_history: bool
    """Whether this config gives a simulated machine an archiver that invents
    its past — the thing all three sites refuse."""

    archiver_phrase: str
    """How to name the config's archiver back to whoever wrote it, ready to drop
    into a message after "archiver.type is …". Says *unset* when unset, and says
    so about an inert flat line rather than pretending the key was never
    written."""


def pairing_in_profile(config: Any) -> ArchiverPairing:
    """Judge a build profile's ``config:`` block.

    Both spellings are live here — the emitter honors the dotted key and a
    nested mapping alike — so both are read, and a disagreement between them
    fails closed. Refusing an ambiguous profile is the same stance
    ``va_archiver_errors`` already takes on a duplicated connection key: one
    fact with two homes is free to disagree, and the build should not be the
    thing that picks a winner.

    Args:
        config: The profile's resolved ``config:`` block.

    Returns:
        The verdict and the phrase naming its archiver.
    """
    control_system = _spellings(config, _CONTROL_SYSTEM_TYPE, nested_only=False)
    serving = _spellings(config, SERVING_KEY, nested_only=False)
    archiver = _spellings(config, _ARCHIVER_TYPE, nested_only=False)

    # Each (type, serving) combination is judged as the section those spellings
    # would render into. When two spellings of one key both exist, either may be
    # the one that lands, so any combination inventing history is enough to
    # refuse.
    invents_history = any(
        _invents_history(_profile_section(type_value, serving_value))
        for type_value in control_system or [_ABSENT]
        for serving_value in serving or [_ABSENT]
    )
    is_mock = not archiver or any(
        resolve_archiver_type({"type": value}) == MOCK_ARCHIVER for value in archiver
    )

    return ArchiverPairing(
        is_invented_history=invents_history and is_mock,
        archiver_phrase=_profile_phrase(archiver),
    )


def pairing_in_rendered_config(config: Any) -> ArchiverPairing:
    """Judge a rendered ``config.yml`` — the deploy config and the MCP server's.

    Nested sections only, because that is all either reader honors: a top-level
    ``archiver.type:`` line in this file is read by nothing, so treating it as a
    statement about the archiver would excuse the very config it fails to
    configure.

    Args:
        config: The raw config mapping, as loaded from ``config.yml``.

    Returns:
        The verdict and the phrase naming its archiver — which calls out an
        inert flat line when one is what misled the writer.
    """
    # The sections handed to the resolvers are the very objects the MCP server
    # hands the factory (``MCPServerConfig.control_system`` / ``.archiver`` are
    # ``raw.get(section)``), resolved by the factory's own functions. There is no
    # second opinion to diverge from: this *is* what the deployment will build.
    control_system = _sections(config).get("control_system")
    return _rendered_pairing(config, _invents_history(control_system))


def pairing_for_target(config: Any, target: str) -> ArchiverPairing:
    """Judge the pairing a session *target* would create in a rendered config.

    The same nested-only reading of the same file as
    :func:`pairing_in_rendered_config`, asked one step early: the control system
    is not the one the config selects but the one *target* selects, because a
    session pointed at the virtual accelerator gets a virtual accelerator — and
    one pointed at ``standin`` gets the stand-in — whatever the deployment was
    built for. The archiver is still the config's
    own, since pointing a session somewhere else does not move the archive —
    which is exactly how a deployment honest at build time acquires the pairing
    at run time.

    The prospective type comes from
    :func:`~osprey_connectors.types.resolve_target` and is never worked out
    here, so this predicate and the switch it guards cannot disagree about where
    a target lands.

    Args:
        config: The raw config mapping, as loaded from ``config.yml``.
        target: The control target being asked for, one of
            :data:`~osprey_connectors.types.CONTROL_TARGETS`.

    Returns:
        The verdict and the phrase naming the config's archiver, ready for a
        refusal that quotes :data:`VA_MOCK_ARCHIVER_WHY`.

    Raises:
        ValueError: Propagated from :func:`~osprey_connectors.types.resolve_target`
            when *target* is unknown, or is ``live`` on a deployment with no
            derivable live control system. An underivable target has no pairing
            to judge, and answering "allowed" for one would report a session as
            honest that cannot be established at all; callers establish that a
            target exists before asking whether it may be used.
    """
    control_system = _sections(config).get("control_system")
    target_section = _target_section(control_system, resolve_target(control_system, target))
    return _rendered_pairing(config, _invents_history(target_section))


def _sections(config: Any) -> dict[Any, Any]:
    """The config's top-level sections, or none when it is not a mapping."""
    return config if isinstance(config, dict) else {}


def _profile_section(type_value: Any, serving_value: Any) -> dict[str, Any]:
    """The one-key ``control_system:`` section a profile's spellings render to."""
    section: dict[str, Any] = {}
    if type_value is not _ABSENT:
        section["type"] = type_value
    if serving_value is not _ABSENT:
        section["connector"] = {VIRTUAL_ACCELERATOR: {SERVING_LEAF: serving_value}}
    return section


def _rendered_pairing(config: Any, invents_history: bool) -> ArchiverPairing:
    """The verdict on a rendered config, given whether the control system to
    judge it against — the one it selects, or the one a target would select —
    has no recorded past."""
    archiver_type = resolve_archiver_type(_sections(config).get("archiver"))

    return ArchiverPairing(
        is_invented_history=invents_history and archiver_type == MOCK_ARCHIVER,
        archiver_phrase=_rendered_phrase(
            _spellings(config, _ARCHIVER_TYPE, nested_only=True),
            _flat_value(config, _ARCHIVER_TYPE),
        ),
    )


def _spellings(config: Any, dotted: str, *, nested_only: bool) -> list[Any]:
    """The values this config sets for *dotted*, in every spelling that is live.

    Returned exactly as written, never normalized: the resolvers in
    :mod:`osprey_connectors.types` are what turn a value into a decision, and
    tidying one on the way in (stripping whitespace, say) would decide something
    about it that the factory does not — a padded type name is a lookup failure
    there, and must stay one here.
    """
    if not isinstance(config, dict):
        return []
    values = [_nested_value(config, dotted)]
    if not nested_only:
        values.append(_flat_value(config, dotted))
    return [value for value in values if value is not _ABSENT]


def _flat_value(config: Any, dotted: str) -> Any:
    """The value of the whole dotted key written as one top-level key."""
    if not isinstance(config, dict) or dotted not in config:
        return _ABSENT
    return config[dotted]


def _nested_value(config: Any, dotted: str) -> Any:
    """The value of *dotted* walked as nested sections."""
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _ABSENT
        node = node[part]
    return node


def _stated(values: list[Any]) -> list[str]:
    """The values that actually say something, as text for a message."""
    return sorted({str(value) for value in values if value})


def _profile_phrase(archiver: list[Any]) -> str:
    if not archiver:
        return f"unset (which the connector factory resolves to {MOCK_ARCHIVER!r})"
    stated = _stated(archiver)
    if len(stated) > 1:
        return (
            f"spelled twice and differently ({' and '.join(repr(v) for v in stated)}) — "
            f"both spellings reach the same rendered key, and which one lands "
            f"depends on which comes last in this profile, so one of them is wrong"
        )
    if not stated:
        return f"blank (which the connector factory resolves to {MOCK_ARCHIVER!r})"
    return repr(stated[0])


def _rendered_phrase(archiver: list[Any], flat: Any) -> str:
    stated = _stated(archiver)
    if stated:
        return repr(stated[0])
    unset = f"unset (which the connector factory resolves to {MOCK_ARCHIVER!r})"
    if flat is _ABSENT:
        return unset
    return (
        f"{unset}. This file does carry a top-level '{_ARCHIVER_TYPE}: "
        f"{flat}' line, but config.yml is read as nested sections, so "
        f"that line configures nothing — the archiver is whatever the "
        f"'archiver:' section says, and there is none"
    )
