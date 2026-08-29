"""What ``virtual_accelerator.live_standin:`` makes the rendered config say.

A stand-in is a *second* virtual accelerator, stood up beside the sandbox one
and dressed as the facility's machine, so an operator can rehearse the warnings,
the acknowledgments and the write refusals against something that cannot move a
magnet. It is a **third control target**, ``standin``, and not a mode of the
first: it gets a connector block of its own,
``control_system.connector.live_standin``, which
:func:`osprey_connectors.types.resolve_target` hands to the EPICS connector when
a session asks for ``standin``.

That block is the whole of this module's output. ``live`` means the machine the
facility authored under ``epics:``, on a deployment running a stand-in exactly
as on one that is not, and the build never writes a key there. A facility
already pointed at its own control system can therefore stand a rehearsal up
beside it, and "going to the real machine" is the switch a session already has
(``control_target_set live``) rather than three profile edits and a rebuild.

Two leaves per gateway lane plus a probe channel is everything a target needs to
be dialled, and each is a fact about the deployment rather than a preference:

- **where the stand-in is.** Both lanes point at loopback on the stand-in's
  port, over the CA name server, because that is the one host↔container Channel
  Access configuration confirmed to work across container runtimes. The port is
  written out, unlike the ``virtual_accelerator`` block's gateway rows: the VA
  connector default-fills those from ``services.virtual_accelerator.port`` and
  ``EPICSConnector`` has no such fill, so an unwritten port would be the EPICS
  default rather than the stand-in's — a target dialling somewhere nothing is
  listening.
- **what proves it reachable.** The probe channel is *derived* from the sandbox
  VA rather than invented, because the stand-in is the same soft IOC serving the
  same machine model — so the channel that proves one proves the other, and the
  target switch has something to prove ``standin`` with. A deployment whose VA
  block names no probe channel gets none here either, which is the honest state
  rather than a gap: a target with no probe channel is never switched to.

Nothing else. Write posture, limits checking and the operator acknowledgment are
the profile's to state, on a stand-in deployment exactly as on any other: they
describe how the *deployment* is run, not where one of its targets lives, and a
build that derived them would be answering a question the profile had already
answered. The one thing the build does own is the stand-in's own block, which is
why a profile spelling any of those leaves itself is refused
(:func:`live_standin_duplicate_key_errors`) rather than overridden.
"""

from __future__ import annotations

from typing import Any

from osprey.deployment.reach import dotted_get
from osprey_connectors.standin import LOOPBACK_HOSTNAME

from .build_profile_archiver import _expand_dotted
from .build_profile_schema import VAConfig

#: Rendered subtree the ``standin`` target is configured from — the stand-in's
#: own connector block, keyed by the connector type
#: :data:`osprey_connectors.types.LIVE_STANDIN` that
#: :func:`osprey_connectors.types.resolve_target` answers ``standin`` with.
_STANDIN_PREFIX = "control_system.connector.live_standin"

#: Rendered subtree of the sandbox virtual accelerator — read, never written:
#: it is where the probe channel below is copied *from*.
_VA_PREFIX = "control_system.connector.virtual_accelerator"

#: Where the sandbox VA states the channel that proves it reachable.
VA_PROBE_CHANNEL_KEY = f"{_VA_PREFIX}.probe_channel"

#: Where the stand-in restates it, so ``standin`` is switchable at all.
PROBE_CHANNEL_KEY = f"{_STANDIN_PREFIX}.probe_channel"

#: Every rendered key the stand-in derives — the emitted set and the refusal
#: set, one list so they cannot drift apart.
#:
#: Matched LEAF by leaf, never by prefix. A prefix over the stand-in's block
#: would refuse a persona's own
#: ``control_system.connector.live_standin.writes_enabled``, which says
#: something the build has no opinion about: *where* the stand-in is and *what
#: proves it reachable* are these keys; whether a given login may write to it is
#: the persona's to state, exactly as it is for the ``epics`` block beside it.
LIVE_STANDIN_DERIVED_KEYS: tuple[str, ...] = (
    f"{_STANDIN_PREFIX}.gateways.read_only.address",
    f"{_STANDIN_PREFIX}.gateways.read_only.port",
    f"{_STANDIN_PREFIX}.gateways.read_only.use_name_server",
    f"{_STANDIN_PREFIX}.gateways.write_access.address",
    f"{_STANDIN_PREFIX}.gateways.write_access.port",
    f"{_STANDIN_PREFIX}.gateways.write_access.use_name_server",
    PROBE_CHANNEL_KEY,
)


def live_standin_config_overrides(
    virtual_accelerator: VAConfig | None,
    config: Any,
    rendered_config: Any,
) -> dict[str, Any]:
    """The ``config:`` entries a stand-in contributes to the rendered project.

    Applied on the ordinary config-override path beside the ``deploy`` and
    ``va_archiver`` blocks, rather than by the VA injector, for the reason the
    archiver's keys are: an *attached* render — every web-terminal persona —
    scaffolds no services and never reaches an injector, yet its session can be
    pointed at the same ``standin`` target as the deployment it attaches to and
    must be told the same address.

    Args:
        virtual_accelerator: The parsed ``virtual_accelerator:`` block, or
            ``None``.
        config: The profile's own ``config:`` mapping — consulted only for the
            VA probe channel, which it may restate under any legal spelling.
        rendered_config: The project's freshly rendered ``config.yml``, as a
            nested mapping. The fallback source for that same probe channel:
            with the profile silent, what the template rendered is what the VA
            block will say.

    Returns:
        Dotted config keys to apply after the profile's own, empty when the
        profile asks for no stand-in. Every key is under
        ``control_system.connector.live_standin``: the facility's own ``epics``
        block is never among them.
    """
    if virtual_accelerator is None or virtual_accelerator.live_standin is None:
        return {}
    port = virtual_accelerator.live_standin
    overrides: dict[str, Any] = {}
    for lane in ("read_only", "write_access"):
        overrides[f"{_STANDIN_PREFIX}.gateways.{lane}.address"] = LOOPBACK_HOSTNAME
        overrides[f"{_STANDIN_PREFIX}.gateways.{lane}.port"] = port
        overrides[f"{_STANDIN_PREFIX}.gateways.{lane}.use_name_server"] = True
    probe_channel = _va_probe_channel(config, rendered_config)
    if probe_channel is not None:
        overrides[PROBE_CHANNEL_KEY] = probe_channel
    return overrides


def live_standin_duplicate_key_errors(
    virtual_accelerator: VAConfig | None, config: Any
) -> list[str]:
    """Refuse a profile that states the stand-in's address in both homes.

    The build derives these keys from ``live_standin``, so a ``config:`` entry
    saying the same thing is not an override — it is a second home for one
    fact, free to disagree with the first, and the derived value is the one that
    wins. Disagreement is the dangerous shape here: an address left in
    ``config:`` reads to anyone opening the profile as the endpoint the
    ``standin`` target dials, while every session on it is somewhere else.

    Scoped to the stand-in's own block. The facility's ``epics`` block is the
    profile's to spell however it likes, stand-in or no stand-in — that is the
    point of the stand-in being a target of its own rather than a rewrite of
    ``live``.

    Checked spelling-independently (the dotted key, a mapping under any dotted
    prefix, a fully nested ``control_system:`` subtree all reach the same
    rendered leaf), and leaf by leaf rather than by prefix — see
    :data:`LIVE_STANDIN_DERIVED_KEYS`.

    Args:
        virtual_accelerator: The parsed ``virtual_accelerator:`` block, or
            ``None``.
        config: The profile's own ``config:`` mapping.

    Returns:
        One error per derived key the profile also reaches, in the order the
        keys are derived; empty when there is no stand-in or no overlap.
    """
    if virtual_accelerator is None or virtual_accelerator.live_standin is None:
        return []
    addressed = _expand_dotted(config)
    port = virtual_accelerator.live_standin
    return [
        f"the profile's `config:` block reaches `{key}` while "
        f"`virtual_accelerator.live_standin: {port}` is set — one fact, two "
        f"homes, free to disagree. The stand-in owns that key: the build points "
        f"`{_STANDIN_PREFIX}` at the second virtual accelerator on "
        f"{LOOPBACK_HOSTNAME}:{port}, so the derived value wins and yours would "
        f"sit in the profile looking like it was in force. To address a machine "
        f"yourself, address the one your facility runs: "
        f"`control_system.connector.epics` is the `live` target and this build "
        f"never touches it."
        for key in LIVE_STANDIN_DERIVED_KEYS
        if _addresses(addressed, key)
    ]


def _va_probe_channel(config: Any, rendered_config: Any) -> Any:
    """The channel the VA block proves, as the finished render will state it.

    The profile's own ``config:`` first, because that overlay is applied in the
    same pass as these overrides and is what the rendered VA block will end up
    saying; the render's current value second, which is the template's. Both
    are read rather than one, so the two blocks cannot end up proving
    different channels on a deployment that named its own.
    """
    spelled = dotted_get(_expand_dotted(config), VA_PROBE_CHANNEL_KEY)
    if spelled is not None:
        return spelled
    return dotted_get(
        rendered_config if isinstance(rendered_config, dict) else None, VA_PROBE_CHANNEL_KEY
    )


def _addresses(tree: Any, dotted_key: str) -> bool:
    """Whether the expanded ``config:`` tree reaches *dotted_key*'s leaf.

    Presence, not truth: a key set to ``None`` or ``false`` is still a second
    home for the fact, and refusing it is the point. Which is why only the
    PARENT is walked with :func:`dotted_get` — reading the leaf through it too
    would answer ``None`` for a key that is present and empty, and let exactly
    the disagreeing entry this refuses through.
    """
    parent_key, _, leaf = dotted_key.rpartition(".")
    parent = dotted_get(tree, parent_key) if parent_key else tree
    return isinstance(parent, dict) and leaf in parent
