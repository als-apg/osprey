"""Entry point for ``python -m osprey.mcp_server.textbooks``.

Subcommands:
    serve  — Start the MCP server (default).
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m osprey.mcp_server.textbooks",
        description="Textbooks MCP Server — accelerator physics textbook expert",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Start the Textbooks MCP server")

    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        from osprey.mcp_server.startup import run_mcp_server

        run_mcp_server("osprey.mcp_server.textbooks.server")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
