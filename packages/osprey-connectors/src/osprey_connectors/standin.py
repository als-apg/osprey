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

One predicate lives here. :func:`live_standin_active` asks about **an
endpoint, right now** — is the stand-in container actually up where a
``standin`` session would dial? It may not be re-derived by a caller. **A
guard that re-implements what the reader resolves can disagree, and the
disagreement is a bypass** — the same reason :mod:`osprey_connectors.types`
owns the target resolvers and :mod:`osprey_connectors.honesty` owns the
archiver pairing.

Nothing here is about the archive. The stand-in has no store of its own: it
rehearses the facility's real machine and is shown that machine's past, while
the deployment's own store is the virtual accelerator's — see
:mod:`osprey_connectors.archive_targets`, which decides whose history an
``archiver:`` block holds, and reads no stand-in key to do so.

:func:`archive_belongs_to_standin`, :data:`ARCHIVER_RECORDER_SERVICE` and
:data:`DEPLOYED_SERVICES_KEY` are the predicate that attribution replaces,
kept while the framework releases that import them are supported and removed
in a later connectors release. Nothing new reads them.

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
tier renders carry ``services: {}`` except for the keys their reach contract
projects, so a conjunct on the deployed-services list would resolve to "not the
stand-in" on exactly the renders a multi-user deployment hands its operators —
the target offered to a single-user session and withheld from the same machine
seen through a tier render. The projected port is the whole evidence, and it is
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

The config is read as nested sections only, because that is how the rendered
``config.yml`` is read by everything that acts on it. A top-level
``services.live_standin.port:`` line in that file configures nothing, and
reading one here would let an inert line vouch for a machine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey_connectors._config_read import LOOPBACK_HOSTNAME, coerce_port, is_loopback, nested_value

#: Where a rendered ``config.yml`` states the stand-in's Channel Access port,
#: dotted as the build's service injector projects it and as this module walks
#: it. One key, because one key is the whole evidence that a stand-in exists.
LIVE_STANDIN_PORT_KEY = "services.live_standin.port"

#: The top-level list of services a rendered ``config.yml`` deploys. Empty or
#: absent on an attached render, which deploys none. Read only by
#: :func:`archive_belongs_to_standin`; goes with it.
DEPLOYED_SERVICES_KEY = "deployed_services"

#: The ``deployed_services`` entry naming the host-side writer that samples this
#: deployment's machine into its archive. Goes with :func:`archive_belongs_to_standin`.
ARCHIVER_RECORDER_SERVICE = "archiver_recorder"

#: Where a rendered ``config.yml`` carries the recorder's own block, written by
#: the build's service injector on a deploying render and projected into an
#: attached render from its host's. Goes with :func:`archive_belongs_to_standin`.
_ARCHIVER_RECORDER_BLOCK_KEY = "services.archiver_recorder"

__all__ = [
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
    return coerce_port(nested_value(config, LIVE_STANDIN_PORT_KEY))


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
    return is_loopback(endpoint_host)


def archive_belongs_to_standin(config: Mapping[str, Any]) -> bool:
    """Whether the deployment built a stand-in and runs the archive recorder.

    The predicate :mod:`osprey_connectors.archive_targets` replaces: under
    attribution the deployment's store is the virtual accelerator's and no
    store belongs to the stand-in, so nothing in this package reads this. It
    is kept, unchanged, for the framework releases that still import it — the
    ``live`` switch gate, the recorder's enablement gate and the reach contract
    — and is removed in a later connectors release once none does.

    True exactly when :func:`live_standin_port` names a port *and* the render
    says the deployment runs the recorder: ``archiver_recorder`` in
    ``deployed_services``, or a non-empty recorder block under
    :data:`_ARCHIVER_RECORDER_BLOCK_KEY` (the spelling an attached render is
    told). No endpoint is consulted.

    Args:
        config: The full config mapping, as loaded from ``config.yml``.

    Returns:
        ``True`` only for a render whose deployment records beside its stand-in.
    """
    if live_standin_port(config) is None:
        return False
    deployed = nested_value(config, DEPLOYED_SERVICES_KEY)
    if isinstance(deployed, list) and ARCHIVER_RECORDER_SERVICE in {str(s) for s in deployed}:
        return True
    block = nested_value(config, _ARCHIVER_RECORDER_BLOCK_KEY)
    return isinstance(block, Mapping) and bool(block)
