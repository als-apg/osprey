"""Entry point for ``python -m osprey.mcp_server.channel_finder_in_context``."""

from osprey.mcp_server.channel_finder_common import run_cf_main


def main() -> None:
    run_cf_main("osprey.mcp_server.channel_finder_in_context.server")


if __name__ == "__main__":
    main()
