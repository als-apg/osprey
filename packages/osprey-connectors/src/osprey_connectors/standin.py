"""The deployment's stand-in, decided in one place.

``virtual_accelerator.live_standin: <port>`` stands a *second* virtual
accelerator up — its own physics state, its own deterministic readout
perturbation — and wires it in as the deployment's own ``standin`` control
target, dialled through its own ``control_system.connector.live_standin``
block. It is not a relabelled ``live``: ``live`` always means the facility's
authored ``epics`` block, and an operator who selects ``standin`` has already
said which machine they mean. Nothing about the stand-in pretends to be a
simulation — the gateways really do dial it, ``real_machine`` stays true, and
every warning, approval prompt and write refusal an operator meets is the one
hardware shows.

Two predicates live here. They read the same key and answer different
questions, and the difference is the whole reason both are spelled out:

* :func:`live_standin_active` asks about **an endpoint, right now** — is the
  stand-in container actually up where a ``standin`` session would dial?
* :func:`archive_belongs_to_standin` asks about **the deployment's shape** —
  whose past is in the store this deployment records into?

Neither may be re-derived by a caller. **A guard that re-implements what the
reader resolves can disagree, and the disagreement is a bypass** — the same
reason :mod:`osprey_connectors.types` owns the target resolvers and
:mod:`osprey_connectors.honesty` owns the archiver pairing.

``live_standin_active`` — the ``standin`` target's deployed-container gate
-------------------------------------------------------------------------

Since the stand-in is its own target, nothing here has to rename anything: the
target carries its own name, and an operator on ``standin`` is told
``standin``. What still has to be established is that the endpoint that target
resolves to really is *this deployment's* stand-in container, and not some
other listener that happens to sit on this host. That is the question this
predicate answers, and its three conjuncts must all hold:

1. **The deployment built a stand-in.** ``services.live_standin.port`` is set
   and names a port. The build writes that block only when a profile asked for
   a stand-in, so its presence is the deployment saying it stood a second
   virtual accelerator up.
2. **The selected endpoint is that port.** The endpoint a session would
   actually dial equals it. A config whose gateways were later moved to the
   real machine's port is dialling a real machine, whatever a leftover
   ``services:`` block still says — the verdict follows the endpoint, never the
   leftover.
3. **The selected endpoint is loopback.** ``127.0.0.1``, ``::1``, or the
   literal ``localhost``. A container on this host is reached over the loopback
   interface; anything routed off the host is a different machine.

Deliberately **no** ``deployed_services`` conjunct. Attached projects and
persona renders carry ``services: {}`` except for the keys their reach contract
projects, so a conjunct on the deployed-services list would resolve to "not the
stand-in" on exactly the renders a multi-user deployment hands its operators —
the target offered to a single-user session and withheld from the same machine
seen through a persona. The projected port is the whole evidence, and it is
enough.

An SSH tunnel is the case this predicate must *not* claim. Forwarding a real
gateway to ``localhost:5064`` satisfies loopback and nothing else: either the
deployment stood no stand-in up (no ``services.live_standin`` block at all), or
it stood one up on another port. Conjunct 1 or 2 fails, the endpoint is treated
as a real machine, and that is the truth — the operator is one hop from
hardware.

An empty or unparseable host answers ``False`` for the reason every honesty
predicate in this package fails closed: the expensive mistake is telling
someone that the machine in front of them is only a stand-in when it is not.

``archive_belongs_to_standin`` — whose history the store holds
-------------------------------------------------------------

**The archive belongs to the machine it records, and a model has no past.** A
deployment that records its own store — one that runs an ``archiver_recorder``
service — beside a stand-in records the stand-in: the machine whose present is
sampled and the machine whose history was seeded are the same one, and the
sandbox virtual accelerator keeps no archive of its own. Two conjuncts, and no
endpoint anywhere in them:

1. **The deployment built a stand-in**, read through the same
   :func:`live_standin_port` accessor as above, so both predicates decide it
   from one reading of one key.
2. **The deployment runs the recorder** — ``archiver_recorder`` is in
   ``deployed_services``, or the render carries the recorder's own
   ``services.archiver_recorder`` block.

Two spellings for one fact, because a render has two ways of stating it. A
deploying render lists the service and the build's injector writes the block
beside the listing; an attached render — a web-terminal persona — lists
nothing at all, and is instead TOLD the block by the build, projected from its
host's render the way today's stand-in port is (the ``archiver_recorder`` Reach
Contract in :mod:`osprey.deployment.reach`). The persona reads the host's
store, so it must read the host's answer: the archive a multi-user session
queries is the same archive, and a gate that vanished behind a persona would
describe one machine two ways.

The second conjunct is the one ``live_standin_active`` must not have, and the
asymmetry is not an oversight. That predicate is about an endpoint some render
would dial, and a render that deploys nothing still dials. This one is about
which machine's history a store holds, and the recorder is what makes the store
the deployment's own — a fact about the host, which is why it is the host's
render, not the reader's, that answers it in both spellings.

Nor is there a loopback conjunct, because there is no endpoint in the question
— where a caller happens to be dialling does not change whose past is already
in the store.

While this holds, the ``live`` target is gated at runtime: a deployment cannot
record the stand-in and simultaneously offer sessions the facility's real
machine, because a real machine's readings spliced onto a stand-in's
synthesized past is the one thing an archive must never contain. The recorder's
compose entry, the recorder's own enablement gate and the archive seed
transform bind to this same predicate.

The config is read as nested sections only, because that is how the rendered
``config.yml`` is read by everything that acts on it. A top-level
``services.live_standin.port:`` line in that file configures nothing, and
reading one here would let an inert line vouch for a machine.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from typing import Any

#: Where a rendered ``config.yml`` states the stand-in's Channel Access port,
#: dotted as the build's service injector projects it and as this module walks
#: it. One key, because one key is the whole evidence that a stand-in exists.
LIVE_STANDIN_PORT_KEY = "services.live_standin.port"

#: The top-level list of services a rendered ``config.yml`` deploys. Empty or
#: absent on an attached render, which deploys none.
DEPLOYED_SERVICES_KEY = "deployed_services"

#: The ``deployed_services`` entry naming the host-side writer that samples this
#: deployment's machine into its archive.
ARCHIVER_RECORDER_SERVICE = "archiver_recorder"

#: Where a rendered ``config.yml`` carries the recorder's own block. Written by
#: the build's service injector on a deploying render (as
#: ``{path: ./services/archiver_recorder}``) exactly when the recorder is
#: deployed, and projected into an attached render from its host's — so a
#: non-empty block is the host saying it records, whichever render is reading.
ARCHIVER_RECORDER_BLOCK_KEY = "services.archiver_recorder"

#: The only host name that is loopback without being an address. Compared
#: case-insensitively; every other spelling has to parse as an IP address, so a
#: resolvable name that merely *happens* to point at this host is not accepted.
LOOPBACK_HOSTNAME = "localhost"

__all__ = [
    "ARCHIVER_RECORDER_BLOCK_KEY",
    "ARCHIVER_RECORDER_SERVICE",
    "DEPLOYED_SERVICES_KEY",
    "LIVE_STANDIN_PORT_KEY",
    "LOOPBACK_HOSTNAME",
    "archive_belongs_to_standin",
    "live_standin_active",
    "live_standin_port",
]


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

    The ``standin`` target's deployed-container gate: all three conjuncts
    described in the module docstring must hold. Anything less — no stand-in
    built, a port that no longer matches, a host that is not loopback or cannot
    be read at all — answers ``False``, which is the honest default: the
    endpoint is treated as a real machine until the config proves otherwise.

    Args:
        config: The full config mapping, as loaded from ``config.yml``.
        endpoint_host: The host a session on the ``standin`` target would dial.
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


def archive_belongs_to_standin(config: Mapping[str, Any]) -> bool:
    """Whether this deployment's archive is the stand-in's history.

    **The archive belongs to the machine it records; a deployment that records
    its own store records the stand-in.** True exactly when the deployment
    built a stand-in (:func:`live_standin_port` names a port) *and* runs the
    recorder that writes the store — ``archiver_recorder`` in
    ``deployed_services``, or the recorder's own
    :data:`ARCHIVER_RECORDER_BLOCK_KEY` block in the render.

    Either spelling, because a render states the fact one of two ways and both
    are the host's. A deploying render lists the service, and the build's
    injector writes the block beside the listing. An attached render — every
    web-terminal persona — lists nothing, and is told the block by the build,
    projected from its host's render like the stand-in's port (the
    ``archiver_recorder`` Reach Contract in :mod:`osprey.deployment.reach`).
    Reading only ``deployed_services`` would drop this gate in exactly the
    renders a multi-user deployment hands its operators, while the persona
    queries the very store the fact is about.

    Unlike :func:`live_standin_active` this asks nothing about an endpoint —
    not which one, not whether it is loopback. Where a caller dials does not
    change whose past a store already holds.

    Args:
        config: The full config mapping, as loaded from ``config.yml``.

    Returns:
        ``True`` only for a render whose deployment records its own stand-in.
        A render carrying neither spelling answers ``False``: no recorder is
        named, so nothing says this store holds a stand-in's history.
    """
    if live_standin_port(config) is None:
        return False
    return _records_its_own_store(config)


def _records_its_own_store(config: Mapping[str, Any]) -> bool:
    """Whether *config* says the deployment runs the archive recorder.

    The listing on a deploying render, the projected block on an attached one.
    A block that is not a non-empty mapping says nothing: an empty or null
    stanza declares no service, and a non-mapping value is not the block the
    injector writes.
    """
    if ARCHIVER_RECORDER_SERVICE in _deployed_services(config):
        return True
    block = _nested_value(config, ARCHIVER_RECORDER_BLOCK_KEY)
    return isinstance(block, Mapping) and bool(block)


def _deployed_services(config: Mapping[str, Any]) -> frozenset[str]:
    """The service names *config* deploys; empty for a render that deploys none.

    A value that is not a list is not a services list, and answering that way
    also keeps a bare string out: ``"archiver_recorder" in "archiver_recorder"``
    is true of a substring, and a substring is not a deployed service.
    """
    deployed = _nested_value(config, DEPLOYED_SERVICES_KEY)
    if not isinstance(deployed, list):
        return frozenset()
    return frozenset(str(service) for service in deployed)


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
