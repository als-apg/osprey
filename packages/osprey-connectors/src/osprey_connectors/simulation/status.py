"""A served physics model's status, read from a connector.

:func:`model_status` answers from the composite a connector serves in process.
:func:`read_model_status` reads a model's status channel through any
connector, the way a target served by a separate process reports it.
:data:`STATUS_OK` and :data:`STATUS_NO_DECK` are the two texts that are not
an error; :func:`is_error` is the one test of that. The module imports nothing beyond
the standard library, so a status lookup loads no model: the composite is
reached through the connector when asked.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "STATUS_NO_DECK",
    "STATUS_OK",
    "is_error",
    "model_status",
    "read_model_status",
]

#: A physics model's status while its engine serves it.
STATUS_OK = "ok"

#: The status of a model that names no deck and wires no channel. Nothing is
#: built for it, by design: the facility describes the machine, and every
#: channel of it is the texture's.
STATUS_NO_DECK = "no deck"

_COMPOSITE_ATTRIBUTE = "_composite"


@runtime_checkable
class _StatusSource(Protocol):
    """What a status lookup reads: the composite a connector serves in process."""

    def status(self, model: str) -> str: ...


def is_error(status: str) -> bool:
    """Whether a model's status text is an error.

    Args:
        status: A model's status, as :func:`model_status` or
            :func:`read_model_status` returns it.

    Returns:
        False for :data:`STATUS_OK` and :data:`STATUS_NO_DECK`, a model built
        on purpose from nothing; True for any other text, which is the error
        the model failed with.
    """
    return status not in (STATUS_OK, STATUS_NO_DECK)


def model_status(connector: object, model: str) -> str:
    """A served physics model's status: ``ok`` or its engine's error text.

    Args:
        connector: A connected control system connector serving a composite
            in process.
        model: The name of a served physics model.

    Returns:
        ``ok`` while the model's engine serves it, ``no deck`` for a model
        built on purpose from nothing; otherwise the engine's error text,
        capped by the composite.

    Raises:
        ValueError: ``model`` is not a served physics model, the message naming
            the served ones; or the connector serves no composite in process.
    """
    composite: object = getattr(connector, _COMPOSITE_ATTRIBUTE, None)
    if not isinstance(composite, _StatusSource):
        raise ValueError(f"{type(connector).__name__} serves no simulator in process")
    return composite.status(model)


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
