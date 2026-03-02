"""Registry JSON export.

Serialises the current :class:`RegistryConfig` (connectors, metadata) to
JSON for consumption by external tools and plan editors.

Extracted from :mod:`osprey.registry.manager` (RF-010).
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from osprey.utils.logger import get_logger

from .base import RegistryConfig

logger = get_logger(name="registry.export", color="sky_blue2")


def export_registry_to_json(
    config: RegistryConfig,
    registries: dict[str, dict[str, Any]],
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Export registry metadata for external tools and plan editors.

    :param config: Current registry configuration.
    :param registries: Live registries dict (unused today but reserved).
    :param output_dir: Directory for saving JSON files; *None* = return only.
    :return: Complete registry metadata dict.
    """
    export_data = {
        "connectors": _export_connectors(config),
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "registry_version": "1.0",
            "total_connectors": len(config.connectors),
        },
    }

    if output_dir:
        _save_export_data(export_data, output_dir)

    return export_data


def _export_connectors(config: RegistryConfig) -> list[dict[str, Any]]:
    """Transform connector registrations into a serialisable list."""
    connectors = []
    for conn_reg in config.connectors:
        connector_data = {
            "name": conn_reg.name,
            "connector_type": conn_reg.connector_type,
            "description": conn_reg.description,
            "module_path": conn_reg.module_path,
            "class_name": conn_reg.class_name,
        }
        connectors.append(connector_data)
    return connectors


def _save_export_data(export_data: dict[str, Any], output_dir: str) -> None:
    """Write *export_data* to ``registry_export.json`` inside *output_dir*."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        export_file = Path(output_dir) / "registry_export.json"
        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Registry export saved to: {export_file}")
        logger.info(
            f"Export contains: {export_data['metadata']['total_connectors']} connectors"
        )

    except Exception as e:
        logger.error(f"Failed to save export data: {e}")
        raise
