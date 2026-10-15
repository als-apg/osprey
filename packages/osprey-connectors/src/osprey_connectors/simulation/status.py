"""A served physics model's status, read from a connector.

A model is in one of three states, and its status text tells them apart:
``ok``; one of its engine's listed condition texts while the model is built,
serving, and reporting that condition; any other text, the engine's error
text, when no model is built.

:func:`model_status` and :func:`model_built` answer from the composite a
connector serves in process.
:func:`read_model_status` reads a model's status channel through any
connector, the way a target served by a separate process reports it. The
module imports nothing beyond the standard library, so a status lookup loads
no model: the composite is reached through the connector when asked.
"""

from __future__ import annotations

from typing import Any

__all__ = ["model_built", "model_status", "read_model_status"]

_COMPOSITE_ATTRIBUTE = "_composite"


def _composite(connector: Any) -> Any:
    """The composite ``connector`` serves in process.

    Raises:
        ValueError: The connector serves no composite in process.
    """
    composite = getattr(connector, _COMPOSITE_ATTRIBUTE, None)
    if composite is None:
        raise ValueError(f"{type(connector).__name__} serves no simulator in process")
    return composite


def model_status(connector: Any, model: str) -> str:
    """A served physics model's status: ``ok``, a condition text, or an error text.

    Args:
        connector: A connected control system connector serving a composite
            in process.
        model: The name of a served physics model.

    Returns:
        ``ok`` while the model's engine serves it and reports no condition;
        the condition text its engine reports while it does; otherwise the
        engine's error text. Each text is capped by the composite.

    Raises:
        ValueError: ``model`` is not a served physics model, the message naming
            the served ones; or the connector serves no composite in process.
    """
    status: str = _composite(connector).status(model)
    return status


def model_built(connector: Any, model: str) -> bool:
    """Whether a served physics model is built: serving, under a condition or not.

    Args:
        connector: A connected control system connector serving a composite
            in process.
        model: The name of a served physics model.

    Returns:
        ``False`` for a model whose engine failed it; ``True`` otherwise.

    Raises:
        ValueError: ``model`` is not a served physics model, the message naming
            the served ones; or the connector serves no composite in process.
    """
    built: bool = _composite(connector).built(model)
    return built


async def read_model_status(connector: Any, address: str) -> str:
    """A served physics model's status, read from its status channel.

    Args:
        connector: A connected control system connector.
        address: The model's status address, ``<code>:SIM:<model>:STATUS``.

    Returns:
        The text the channel holds: ``ok``, the condition text the model's
        engine reports, or the model's error text.
    """
    from osprey_connectors.simulation import decode_char_waveform

    reading = await connector.read_channel(address)
    return decode_char_waveform(reading.value)
