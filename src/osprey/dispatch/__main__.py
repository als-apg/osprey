"""Entry point for ``python -m osprey.dispatch``."""

from osprey.mcp_server.startup import run_mcp_server


def main() -> None:
    run_mcp_server("osprey.dispatch.server")


if __name__ == "__main__":
    main()
