"""Whether a deployment's live target is the stand-in, decided in one place.

``virtual_accelerator.live_standin: <port>`` stands a *second* virtual
accelerator up — its own physics state, its own deterministic readout
perturbation — and wires it in as the deployment's ``live`` target. Nothing
about it pretends to be a simulation: the ``epics`` gateways really do dial it,
``real_machine`` stays true, and every warning, approval prompt and write
refusal an operator meets is the one hardware shows. What changes is the name
on the label — ``LIVE MACHINE (stand-in)`` — so an operator rehearsing the full
ritual is never told they are somewhere they are not.

Three readers need that fact, and none of them may work it out privately: the
roster's label, the archiver recorder's enablement gate, and the build
derivation that overwrites the gateways in the first place. **A guard that
re-implements what the reader resolves can disagree, and the disagreement is a
bypass** — the same reason :mod:`osprey_connectors.types` owns the target
resolvers and :mod:`osprey_connectors.honesty` owns the archiver pairing. A
recorder that believes it is sampling a stand-in while the roster tells an
operator "LIVE MACHINE" would file model readings into a real machine's
archive, which is the one thing an archive must never contain.

The predicate is three conjuncts, all of which must hold:

1. **The deployment built a stand-in.** ``services.live_standin.port`` is set
   and names a port. The build writes that block only when a profile asked for
   a stand-in, so its presence is the deployment saying it stood a second
   virtual accelerator up.
2. **The live endpoint is that port.** The endpoint a session would actually
   dial equals it. A config whose gateways were later moved to the real
   machine's port has gone live, whatever a leftover ``services:`` block still
   says — the label follows the endpoint, never the leftover.
3. **The live endpoint is loopback.** ``127.0.0.1``, ``::1``, or the literal
   ``localhost``. A container on this host is reached over the loopback
   interface; anything routed off the host is a different machine.

Deliberately **no** ``deployed_services`` conjunct. Attached projects and
persona renders carry ``services: {}`` except for the keys their reach contract
projects, so a conjunct on the deployed-services list would resolve to "not a
stand-in" on exactly the renders a multi-user deployment hands its operators —
one word of label for a single-user session and a different word for the same
machine seen through a persona. The projected port is the whole evidence, and
it is enough.

An SSH tunnel is the case this predicate must *not* claim. Forwarding a real
gateway to ``localhost:5064`` satisfies loopback and nothing else: either the
deployment stood no stand-in up (no ``services.live_standin`` block at all), or
it stood one up on another port. Conjunct 1 or 2 fails, the label stays
``LIVE MACHINE``, and that is the truth — the operator is one hop from
hardware.

An empty or unparseable host fails toward ``LIVE MACHINE`` for the reason every
honesty predicate in this package fails closed: the expensive mistake is
telling someone that the machine in front of them is only a stand-in when it is
not.

The archiver recorder follows this predicate rather than
``control_system.type`` because **the archive belongs to the machine, and a
model has no past**. Where there is a stand-in, the machine whose present is
recorded and the machine whose history was seeded are the same one — so the
recorder samples the stand-in, and the sandbox virtual accelerator keeps no
archive of its own.

The config is read as nested sections only, because that is how the rendered
``config.yml`` is read by everything that acts on it. A top-level
``services.live_standin.port:`` line in that file configures nothing, and
reading one here would let an inert line rename a live machine.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from typing import Any

#: Where a rendered ``config.yml`` states the stand-in's Channel Access port,
#: dotted as the build's service injector projects it and as this module walks
#: it. One key, because one key is the whole evidence that a stand-in exists.
LIVE_STANDIN_PORT_KEY = "services.live_standin.port"

#: The only host name that is loopback without being an address. Compared
#: case-insensitively; every other spelling has to parse as an IP address, so a
#: resolvable name that merely *happens* to point at this host is not accepted.
LOOPBACK_HOSTNAME = "localhost"

__all__ = ["LIVE_STANDIN_PORT_KEY", "LOOPBACK_HOSTNAME", "live_standin_active", "live_standin_port"]


def live_standin_port(config: Mapping[str, Any]) -> int | None:
    """The Channel Access port the deployment's stand-in serves, if it built one.

    Args:
        config: The full config mapping, as loaded from ``config.yml`` — not the
            ``services:`` section, since the whole point of the dotted key is
            that one walk answers for every reader.

    Returns:
        The port, or ``None`` when the deployment named no stand-in. ``None``
        also covers a value that names no port — blank, a mapping, a bare
        ``true`` — because a key that cannot be dialled is not a deployment
        saying where its stand-in is.
    """
    return _coerce_port(_nested_value(config, LIVE_STANDIN_PORT_KEY))


def live_standin_active(
    config: Mapping[str, Any], *, endpoint_host: str, endpoint_port: int | None
) -> bool:
    """Whether *endpoint* is this deployment's stand-in rather than a real machine.

    All three conjuncts described in the module docstring must hold. Anything
    less — no stand-in built, a port that no longer matches, a host that is not
    loopback or cannot be read at all — answers ``False``, which is the honest
    default: the endpoint is treated as a real machine until the config proves
    otherwise.

    Args:
        config: The full config mapping, as loaded from ``config.yml``.
        endpoint_host: The host a session on the ``live`` target would dial.
        endpoint_port: The port it would dial, or ``None`` when the deployment
            has not resolved one — in which case there is no endpoint to match
            and no stand-in to claim.

    Returns:
        ``True`` only for an endpoint that is the deployment's own stand-in.
    """
    if endpoint_port is None:
        return False
    port = live_standin_port(config)
    if port is None or port != endpoint_port:
        return False
    return _is_loopback(endpoint_host)


def _is_loopback(host: str) -> bool:
    """Whether *host* names this machine's loopback interface."""
    candidate = host.strip() if isinstance(host, str) else ""
    if not candidate:
        return False
    if candidate.lower() == LOOPBACK_HOSTNAME:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # A name this module cannot read is a machine it cannot vouch for.
        return False


def _coerce_port(value: Any) -> int | None:
    """*value* as a port number, or ``None`` when it does not name one.

    ``bool`` is rejected ahead of ``int`` on purpose: ``live_standin: true``
    says a stand-in is wanted without saying where it is, and Python would
    otherwise read it as port 1.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _nested_value(config: Any, dotted: str) -> Any:
    """The value of *dotted* walked as nested sections, or ``None`` when absent."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node
