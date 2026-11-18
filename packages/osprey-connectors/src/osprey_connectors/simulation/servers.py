"""The simulated IOCs: which IOC serves each channel of a simulator view.

A served physics model is one IOC, ``sim-<model>``; it serves the channels
the model owns, the model's status address and its IOC health addresses.
Every channel no served physics model owns is served by one more IOC,
:data:`TEXTURE_SERVER`.

A scenario's ``stopped_servers`` names IOCs by this rule. While the scenario is
active, every channel of a named IOC gives no answer: the one exception to
the simulator's rule that a read always answers
(:mod:`~osprey_connectors.simulation.channel_faults`). The in-process
connector raises :class:`~osprey_connectors.errors.ChannelUnreachableError`
for a read of such a channel, fails a write to it with nothing sent, and
tells its subscriptions the channel disconnected (and connected again on
restart); the served virtual accelerator withholds the channel, so a
client's read times out and its write is never sent. Either way, a channel
answers again on restart with its current value, never one held from the stop.

The module imports only the standard library, so the build's validator, the
composite and the directory share one rule.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

__all__ = ["TEXTURE_SERVER", "channel_server", "server_addresses", "model_server", "served_servers"]

#: The prefix of every simulated IOC name.
_SERVER_PREFIX = "sim-"

#: The IOC of every channel no served physics model owns.
TEXTURE_SERVER = f"{_SERVER_PREFIX}texture"


def model_server(model: str) -> str:
    """The IOC of the served physics model ``model``: ``sim-<model>``."""
    return f"{_SERVER_PREFIX}{model}"


def channel_server(owner: str | None, physics: Collection[str]) -> str:
    """The IOC serving a channel owned by ``owner``, given the served physics models."""
    return model_server(owner) if owner is not None and owner in physics else TEXTURE_SERVER


def served_servers(physics: Iterable[str]) -> frozenset[str]:
    """Every IOC name a view serving the physics models ``physics`` serves."""
    return frozenset({TEXTURE_SERVER, *(model_server(name) for name in physics)})


def server_addresses(
    physics: Iterable[str],
    owners: Mapping[str, str | None],
    status: Mapping[str, str],
    ioc_health: Mapping[str, Mapping[str, str]],
) -> dict[str, frozenset[str]]:
    """Every address of a view, by the IOC that serves it.

    Args:
        physics: The served physics models.
        owners: Each channel's owner, by address.
        status: Each served physics model's status address, by model.
        ioc_health: Each model's IOC health addresses by signal, by model.
    """
    served = set(physics)
    by_server: dict[str, set[str]] = {name: set() for name in served_servers(served)}
    for address, owner in owners.items():
        by_server[channel_server(owner, served)].add(address)
    for model, address in status.items():
        by_server.setdefault(model_server(model), set()).add(address)
    for model, signals in ioc_health.items():
        by_server.setdefault(model_server(model), set()).update(signals.values())
    return {name: frozenset(addresses) for name, addresses in sorted(by_server.items())}
