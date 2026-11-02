"""Every served physics model's status on one control target.

:func:`target_model_statuses` builds the connector a control target selects and
reads each model's status through it: from the composite a simulator served in
process holds, or from each model's status channel on a simulator served by a
process of its own. ``osprey sim status`` and the ``simulator`` health category
read it the same way, so the two can never disagree about a model.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["target_model_statuses"]


async def target_model_statuses(
    section: dict, target: str | None, models: list[str], addresses: Mapping[str, str] | None
) -> dict[str, str]:
    """Each model's status, read through the connector built for ``target``.

    The simulator served in process answers from the composite it holds; any
    other connector reads each model's status channel, ``addresses[model]``.

    Args:
        section: The ``control_system:`` config section.
        target: The control target, or ``None`` for the deployment's own.
        models: The served physics models to report.
        addresses: Each model's status address, or ``None`` for a target
            served in process.

    Returns:
        Each model's status text, by model name.
    """
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors
    from osprey_connectors.simulation import model_status, read_model_status

    register_builtin_connectors()
    connector = await ConnectorFactory.create_control_system_connector(
        section, control_target=target
    )
    try:
        if addresses is None:
            return {model: model_status(connector, model) for model in models}
        return {model: await read_model_status(connector, addresses[model]) for model in models}
    finally:
        await connector.disconnect()
