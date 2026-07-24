"""Entry point for ``python -m osprey.mcp_server.channel_finder_middle_layer``."""

from osprey.mcp_server.channel_finder_common import run_cf_main


def main() -> None:
    run_cf_main("osprey.mcp_server.channel_finder_middle_layer.server")


if __name__ == "__main__":
    main()
