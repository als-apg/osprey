"""What ``virtual_accelerator.live_standin:`` makes the rendered config say.

A stand-in is a *second* virtual accelerator, stood up beside the sandbox one
and wired in as the deployment's ``live`` target, so an operator can rehearse
the whole go-live ritual — the switch, the acknowledgment, the warnings, the
write refusals — against something that cannot move a magnet. Standing the
service up is the injector's half of the feature; this module is the other
half: the keys the *shipped* ``epics:`` block needs before dialing "live"
reaches that stand-in rather than the facility's real gateway.

Two facts about the deployment come out of the one profile key, and neither is
a preference a second key should be able to contradict:

- **where the live target is.** Both ``epics`` gateways are pointed at
  loopback on the stand-in's port, over the CA name server, because that is
  the one host↔container Channel Access configuration confirmed to work across
  container runtimes. The port is written out here, unlike the
  ``virtual_accelerator`` block's gateways: the VA connector default-fills
  those from ``services.virtual_accelerator.port``, and ``EPICSConnector`` has
  no such fill — an unwritten port on the ``epics`` block is the EPICS
  default, not the stand-in's, so leaving it out would send "live" somewhere
  nothing is listening;
- **how it is run.** Limits checking on, unlisted channels refused. A
  rehearsal that runs looser than the machine teaches the wrong ritual: the
  first refused write an operator ever met would then be on real hardware.

The ``epics`` probe channel is *derived* rather than invented, because the
stand-in is the same soft-IOC serving the same machine model as the sandbox
VA — so the channel that proves one reachable proves the other, and the target
switch has something to prove ``live`` with. A deployment whose VA block names
no probe channel gets no ``epics`` probe channel either, which is the honest
state rather than a gap: a target with no probe channel is never switched to.

Going live is a documented three steps and deliberately not a knob: delete
``virtual_accelerator.live_standin``, point the ``epics`` gateways at your
facility, and replace the operator acknowledgment. That is why a
profile spelling any of these keys in its own ``config:`` block while a
stand-in is asked for is refused (:func:`live_standin_duplicate_key_errors`)
rather than overridden — the derived value wins at render time, and the
facility's real gateway address would sit in the profile looking like it was
in force.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ruamel.yaml import CommentedMap
from ruamel.yaml.error import CommentMark
from ruamel.yaml.tokens import CommentToken

from osprey.deployment.reach import dotted_get

# The framework's round-trip reader/writer, imported rather than reconstructed.
# This module re-opens a file `config_writer` has just written, and a second
# YAML instance under its own indent/width settings would reflow lines it never
# meant to touch — so the settings are not copied here, they are the same ones.
from osprey.utils.config_writer import _load, _save
from osprey_connectors.standin import LOOPBACK_HOSTNAME

from .build_profile_archiver import _expand_dotted
from .build_profile_schema import VAConfig

#: Rendered subtree of the connector the stand-in takes over. The ``epics``
#: block is the deployment's ``live`` target — with a stand-in built, ``live``
#: still resolves to it (``control_system.type`` stays ``virtual_accelerator``
#: and ``epics`` is the one non-simulated block), so this is where "live" is
#: told which machine it means.
_EPICS_PREFIX = "control_system.connector.epics"

#: Rendered subtree of the sandbox virtual accelerator — read, never written:
#: it is where the probe channel below is copied *from*.
_VA_PREFIX = "control_system.connector.virtual_accelerator"

#: Rendered subtree holding the write-safety posture the rehearsal runs under.
_LIMITS_PREFIX = "control_system.limits_checking"

#: Where the sandbox VA states the channel that proves it reachable.
VA_PROBE_CHANNEL_KEY = f"{_VA_PREFIX}.probe_channel"

#: Where the stand-in restates it, so ``live`` is switchable at all.
PROBE_CHANNEL_KEY = f"{_EPICS_PREFIX}.probe_channel"

#: The permissiveness the stand-in flips off, and whose rendered line carries
#: the template comment this module retruths.
ALLOW_UNLISTED_CHANNELS_KEY = f"{_LIMITS_PREFIX}.allow_unlisted_channels"

#: Every rendered key the stand-in derives — the emitted set and the refusal
#: set, one list so they cannot drift apart.
#:
#: Matched LEAF by leaf, never by prefix. A prefix over
#: ``control_system.connector.epics`` would refuse the shipped read-only
#: persona's own ``control_system.connector.epics.writes_enabled``, which says
#: something the stand-in has no opinion about: *where* the live target is and
#: *how strictly* it is run are these keys; whether this login may write to it
#: is the persona's to state.
LIVE_STANDIN_DERIVED_KEYS: tuple[str, ...] = (
    f"{_EPICS_PREFIX}.gateways.read_only.address",
    f"{_EPICS_PREFIX}.gateways.read_only.port",
    f"{_EPICS_PREFIX}.gateways.read_only.use_name_server",
    f"{_EPICS_PREFIX}.gateways.write_access.address",
    f"{_EPICS_PREFIX}.gateways.write_access.port",
    f"{_EPICS_PREFIX}.gateways.write_access.use_name_server",
    PROBE_CHANNEL_KEY,
    f"{_LIMITS_PREFIX}.enabled",
    ALLOW_UNLISTED_CHANNELS_KEY,
)

#: What the strict-limits line says once the stand-in has flipped it. The
#: templates ship the permissive value with an inline comment explaining it as
#: a convenience; the same comment above a ``false`` would be a rendered file
#: contradicting itself, so it is replaced rather than left to be read.
STRICT_LIMITS_COMMENT = "# strict: the stand-in rehearses the live machine's posture"


def live_standin_config_overrides(
    virtual_accelerator: VAConfig | None,
    config: Any,
    rendered_config: Any,
) -> dict[str, Any]:
    """The ``config:`` entries a stand-in contributes to the rendered project.

    Applied on the ordinary config-override path beside the ``deploy`` and
    ``va_archiver`` blocks, rather than by the VA injector, for the reason the
    archiver's keys are: an *attached* render — every web-terminal persona —
    scaffolds no services and never reaches an injector, yet its session dials
    the same ``live`` target as the deployment it attaches to and must be told
    the same address.

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
        profile asks for no stand-in.
    """
    if virtual_accelerator is None or virtual_accelerator.live_standin is None:
        return {}
    port = virtual_accelerator.live_standin
    overrides: dict[str, Any] = {}
    for lane in ("read_only", "write_access"):
        overrides[f"{_EPICS_PREFIX}.gateways.{lane}.address"] = LOOPBACK_HOSTNAME
        overrides[f"{_EPICS_PREFIX}.gateways.{lane}.port"] = port
        overrides[f"{_EPICS_PREFIX}.gateways.{lane}.use_name_server"] = True
    probe_channel = _va_probe_channel(config, rendered_config)
    if probe_channel is not None:
        overrides[PROBE_CHANNEL_KEY] = probe_channel
    overrides[f"{_LIMITS_PREFIX}.enabled"] = True
    overrides[ALLOW_UNLISTED_CHANNELS_KEY] = False
    return overrides


def live_standin_duplicate_key_errors(
    virtual_accelerator: VAConfig | None, config: Any
) -> list[str]:
    """Refuse a profile that states the live target's address in both homes.

    The build derives these keys from ``live_standin``, so a ``config:`` entry
    saying the same thing is not an override — it is a second home for one
    fact, free to disagree with the first, and the derived value is the one
    that wins. Disagreement is the dangerous shape here: a real gateway
    hostname left in ``config:`` reads to anyone opening the profile as the
    machine this deployment talks to, while every session is on the stand-in.

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
        f"the `epics` block at the second virtual accelerator on "
        f"{LOOPBACK_HOSTNAME}:{port} and runs it under the live machine's "
        f"limits, so the derived value wins and yours would sit in the profile "
        f"looking like it was in force. Going live is three steps: delete "
        f"`virtual_accelerator.live_standin`, point "
        f"`control_system.connector.epics.gateways` at your facility, and replace "
        f"`control_system.target_switch.live_gateway_acknowledged` with your own "
        f"live gateway's hostname."
        for key in LIVE_STANDIN_DERIVED_KEYS
        if _addresses(addressed, key)
    ]


def rewrite_strict_limits_comment(config_path: Path) -> bool:
    """Retruth the comment beside the strict-limits key the stand-in flipped.

    Both config templates ship ``allow_unlisted_channels: true``, and the
    Control Assistant one explains it on the same line as a tutorial
    convenience. The stand-in's override turns that key ``false`` while leaving
    the comment behind, and a rendered file whose comment contradicts the value
    beside it is worse than no comment at all — it is the sentence an operator
    reads when deciding whether the deployment is safe.

    Fixed here rather than in the templates because the templates are right:
    without a stand-in the permissive default *is* the tutorial convenience
    they describe. Only the build knows the value moved, so only the build can
    say why.

    Args:
        config_path: The rendered ``config.yml``, after the overrides have been
            applied to it.

    Returns:
        Whether the line was found and its comment rewritten.
    """
    if not config_path.exists():
        return False
    data = _load(config_path)
    parts = ALLOW_UNLISTED_CHANNELS_KEY.split(".")
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    if not isinstance(node, CommentedMap) or parts[-1] not in node:
        return False
    _replace_eol_comment(node, parts[-1], STRICT_LIMITS_COMMENT)
    _save(config_path, data)
    return True


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


def _replace_eol_comment(mapping: CommentedMap, key: str, comment: str) -> None:
    """Replace the comment on *key*'s own line, keeping the block below it.

    ruamel parks a comment block that *follows* a section's last key in that
    key's end-of-line slot, so this one token can hold both the inline comment
    and the banner introducing the next section — and
    ``allow_unlisted_channels`` is the last key of ``limits_checking`` in both
    templates. Only the token's first line is the key's own; everything after
    it is the file's layout and is carried over untouched.
    """
    entry = mapping.ca.items.setdefault(key, [None, None, None, None])
    token = entry[2]
    if token is None:
        # Column 0 asks the emitter for the minimum single space before `#`,
        # which is what a line with no comment to align against wants.
        entry[2] = CommentToken(f"{comment}\n", CommentMark(0), None)
        return
    text = token.value or ""
    head, _, tail = text.partition("\n")
    if head.strip():
        token.value = f"{comment}\n{tail}"
    elif text.startswith("\n"):
        # No inline comment, only a trailing block: the leading newline that
        # separates them is already at the front of the block's own text.
        token.value = f"{comment}{text}"
    else:
        token.value = f"{comment}\n"
