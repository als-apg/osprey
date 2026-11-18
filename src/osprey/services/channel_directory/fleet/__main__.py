"""Run the server fleet service.

``python -m osprey.services.channel_directory.fleet --config <config.yml> --state-dir <dir>
[--host 127.0.0.1] [--port <port>]``

Loads the rendered config, starts one refresher per refreshed target and serves
their snapshots (:mod:`osprey.services.channel_directory.fleet.app`). The port defaults
to ``services.server_fleet.port``, else the layout's ``server_fleet`` slot, the port
:func:`osprey.server_fleet_connection.fleet_url` resolves. A config without a
``channel_directory.fleet`` sub-block, or with a malformed one, exits 2 naming the key.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.port_layout import default_port, resolve_port_base

#: The rendered config's path when ``--config`` is not given.
CONFIG_ENV_VAR = "OSPREY_CONFIG"
#: The state directory when ``--state-dir`` is not given.
STATE_DIR_ENV_VAR = "SERVER_FLEET_STATE_DIR"


def service_port(config: Mapping[str, Any]) -> int:
    """``services.server_fleet.port``, else the layout's ``server_fleet`` slot."""
    services = config.get("services")
    block = services.get("server_fleet") if isinstance(services, Mapping) else None
    port = block.get("port") if isinstance(block, Mapping) else None
    if port:
        return int(port)
    return default_port("server_fleet", base=resolve_port_base(config))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m osprey.services.channel_directory.fleet",
        description="Supervise the server fleet's refreshers and serve their snapshots.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get(CONFIG_ENV_VAR),
        help=f"the rendered config.yml (default: ${CONFIG_ENV_VAR})",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=os.environ.get(STATE_DIR_ENV_VAR),
        help=f"where the refreshers write their files (default: ${STATE_DIR_ENV_VAR})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="the address to bind")
    parser.add_argument("--port", type=int, default=None, help="the port to bind")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Load the config, build the app and serve it until stopped."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.config:
        parser.error(f"--config is required when ${CONFIG_ENV_VAR} is not set")
    if not args.state_dir:
        parser.error(f"--state-dir is required when ${STATE_DIR_ENV_VAR} is not set")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    from osprey.services.channel_directory.fleet.app import create_app
    from osprey.services.channel_directory.resolve import SectionError
    from osprey_connectors.config import load_config

    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    try:
        app = create_app(config, config_path=config_path, state_dir=Path(args.state_dir))
    except SectionError as exc:
        parser.exit(2, f"ioc fleet: {exc}\n")

    import uvicorn

    port = args.port if args.port is not None else service_port(config)
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
