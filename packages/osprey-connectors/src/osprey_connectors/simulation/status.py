"""A served physics model's status, read from a connector.

:func:`model_status` answers from the composite a connector serves in process.
:func:`read_model_status` reads a model's status channel through any
connector, the way a target served by a separate process reports it.
:func:`model_state` sorts a status text into one of three states: the model
serves (:data:`MODEL_OK`), the model names no deck and wires nothing, so it is
not built on purpose (:data:`MODEL_NO_DECK`), or the model failed
(:data:`MODEL_FAILED`, its text the error). The module imports nothing beyond
the standard library, so a status lookup loads no model: the composite is
reached through the connector when asked.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MODEL_FAILED",
    "MODEL_NO_DECK",
    "MODEL_OK",
    "STATUS_NO_DECK",
    "STATUS_OK",
    "model_state",
    "model_status",
    "read_model_status",
]

#: A physics model's status while its engine serves it.
STATUS_OK = "ok"

#: The status of a model that names no deck and wires no channel. Nothing is
#: built for it, by design: the facility describes the machine, and every
#: channel of it is the texture's.
STATUS_NO_DECK = "no deck"

#: The states :func:`model_state` sorts a status text into.
MODEL_OK = "ok"
MODEL_NO_DECK = "no_deck"
MODEL_FAILED = "failed"

_COMPOSITE_ATTRIBUTE = "_composite"


def model_status(connector: Any, model: str) -> str:
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
    composite = getattr(connector, _COMPOSITE_ATTRIBUTE, None)
    if composite is None:
        raise ValueError(f"{type(connector).__name__} serves no simulator in process")
    status: str = composite.status(model)
    return status


def model_state(status: str) -> str:
    """The state a model's status text reports.

    Args:
        status: A model's status, as :func:`model_status` or
            :func:`read_model_status` returns it.

    Returns:
        :data:`MODEL_OK` for :data:`STATUS_OK`, :data:`MODEL_NO_DECK` for
        :data:`STATUS_NO_DECK`, and :data:`MODEL_FAILED` for any other text,
        which is the error the model failed with.
    """
    if status == STATUS_OK:
        return MODEL_OK
    if status == STATUS_NO_DECK:
        return MODEL_NO_DECK
    return MODEL_FAILED


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
