"""The one connector pairing a deployment may not have: invented history.

A virtual accelerator serves channels that move for modelled reasons — a
corrector is stepped, the orbit responds, and the numbers a tool reads back are
answers to what the deployment actually did. The mock archiver answers a history
query the other way round: it synthesizes a plausible-looking series at read
time, for questions nobody recorded the answer to. Configured together they
produce an agent whose past is fiction and whose present is not, with nothing
connecting the two — so the fiction can never be caught by disagreeing with the
machine it claims to describe.

Refused at three independent moments, because a deployment can acquire the
pairing at any of them: ``osprey build`` writes the config, ``osprey deploy``
stands the services up, and the MCP server reads whatever ``config.yml`` it is
pointed at — including one hand-edited long after the build. Each site raises in
its own vocabulary and names its own fix. What they share, and what lives here,
is the question they ask and the reason they ask it.

**The question is asked twice, because the two kinds of config are read by
different readers, and a guard must resolve a key exactly as the reader it
guards resolves it — otherwise the divergence *is* the bypass.**

- A build profile's ``config:`` block reaches the rendered project through the
  emitter, which honors the dotted spelling (``archiver.type:``, the canonical
  one) *and* a nested mapping — both land on the same rendered leaf. Both are
  therefore live, and :func:`pairing_in_profile` fails closed when they
  disagree: whichever one wins, the profile has stated the archive twice and is
  free to be wrong once.
- A rendered ``config.yml`` is read by :class:`~osprey.utils.config.ConfigBuilder`
  and by ``MCPServerConfig``, and both walk *nested sections only* — a top-level
  ``archiver.type:`` line there sets nothing at all. So
  :func:`pairing_in_rendered_config` resolves nested-only. A flat line is not
  evidence of an archiver; it is an inert line, and one this module names in its
  message rather than silently reading as "unset", because someone who typed it
  deserves to be told why it did nothing.

Both readers fall back to the mock when their key is absent or blank (see the
factory's ``… is not set; defaulting to …`` warnings), so *unset counts as mock*
at every site. That is the fallback the rule is really about: the common way
into the pairing is not naming the mock, it is naming nothing.

Deliberately *not* a refusal of every mock archiver: a mock control system paired
with the mock archiver is the honest storeless deployment, and it is the app
template's default. Nothing is claimed to be real there, so nothing lies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import (
    MOCK_ARCHIVER,
    VIRTUAL_ACCELERATOR,
    resolve_archiver_type,
    resolve_control_system_type,
)

#: Why the pairing is refused, in one sentence pair the three sites share so the
#: explanation cannot drift between them. Each site supplies its own fix.
VA_MOCK_ARCHIVER_WHY = (
    "a virtual accelerator serves a machine whose channels move for modelled "
    "reasons, while the mock archiver synthesizes history at read time. Paired, "
    "the agent reports a past that never happened and nothing can catch it — the "
    "one failure a simulated facility exists to make visible rather than to have."
)

_CONTROL_SYSTEM_TYPE = "control_system.type"
_ARCHIVER_TYPE = "archiver.type"


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
    archiver = _spellings(config, _ARCHIVER_TYPE, nested_only=False)

    # Each candidate is resolved through the factory's own resolver, as the
    # one-key section that spelling would render into. When two spellings both
    # exist, either may be the one that lands, so either being the mock is
    # enough to refuse.
    is_va = any(
        resolve_control_system_type({"type": value}) == VIRTUAL_ACCELERATOR
        for value in control_system
    )
    is_mock = not archiver or any(
        resolve_archiver_type({"type": value}) == MOCK_ARCHIVER for value in archiver
    )

    return ArchiverPairing(
        is_invented_history=is_va and is_mock,
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
    sections = config if isinstance(config, dict) else {}
    control_system_type = resolve_control_system_type(sections.get("control_system"))
    archiver_type = resolve_archiver_type(sections.get("archiver"))

    return ArchiverPairing(
        is_invented_history=(
            control_system_type == VIRTUAL_ACCELERATOR and archiver_type == MOCK_ARCHIVER
        ),
        archiver_phrase=_rendered_phrase(
            _spellings(config, _ARCHIVER_TYPE, nested_only=True),
            _flat_value(config, _ARCHIVER_TYPE),
        ),
    )


def _spellings(config: Any, dotted: str, *, nested_only: bool) -> list[Any]:
    """The values this config sets for *dotted*, in every spelling that is live.

    Returned exactly as written, never normalized: the resolvers in
    :mod:`osprey.connectors.types` are what turn a value into a decision, and
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
    section, _, leaf = dotted.partition(".")
    subtree = config.get(section) if isinstance(config, dict) else None
    if not isinstance(subtree, dict) or leaf not in subtree:
        return _ABSENT
    return subtree[leaf]


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
