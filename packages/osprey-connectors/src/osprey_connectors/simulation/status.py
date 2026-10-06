"""A served physics model's status, read from a connector.

:func:`model_status` answers from the composite a connector serves in process.
:func:`read_model_status` reads a model's status channel through any
connector, the way a target served by a separate process reports it. The
module imports nothing beyond the standard library, so a status lookup loads
no model: the composite is reached through the connector when asked.
"""

from __future__ import annotations

from typing import Any

__all__ = ["model_status", "read_model_status"]

_COMPOSITE_ATTRIBUTE = "_composite"


def model_status(connector: Any, model: str) -> str:
    """A served physics model's status: ``ok`` or its engine's error text.

    Args:
        connector: A connected control system connector serving a composite
            in process.
        model: The name of a served physics model.

    Returns:
        ``ok`` while the model's engine serves it; otherwise the engine's
        error text, capped by the composite.

    Raises:
        ValueError: ``model`` is not a served physics model, the message naming
            the served ones; or the connector serves no composite in process.
    """
    composite = getattr(connector, _COMPOSITE_ATTRIBUTE, None)
    if composite is None:
        raise ValueError(f"{type(connector).__name__} serves no simulator in process")
    status: str = composite.status(model)
    return status


async def read_model_status(connector: Any, address: str) -> str:
    """A served physics model's status, read from its status channel.

    Args:
        connector: A connected control system connector.
        address: The model's status address, ``<code>:SIM:<model>:STATUS``.

    Returns:
        The text the channel holds: ``ok`` or the model's error text.
    """
    from osprey_connectors.simulation import decode_char_waveform

    reading = await connector.read_channel(address)
    return decode_char_waveform(reading.value)
