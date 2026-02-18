"""Entry point for ``python -m osprey.mcp_server.matlab``.

Subcommands:
    index  — Build the SQLite FTS5 index from matlab_dependencies.json.
    serve  — Start the MCP server (default).
"""

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m osprey.mcp_server.matlab",
        description="MATLAB MML MCP Server — search Middle Layer functions",
    )
    sub = parser.add_subparsers(dest="command")

    # --- index subcommand ---
    idx = sub.add_parser("index", help="Build SQLite FTS5 index from matlab_dependencies.json")
    idx.add_argument(
        "--data-file",
        required=True,
        help="Path to matlab_dependencies.json",
    )
    idx.add_argument(
        "--db-path",
        default=None,
        help="Output SQLite database path (default: ~/.matlab-mml/mml.db)",
    )
    idx.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows per INSERT batch (default: 500)",
    )

    # --- serve subcommand ---
    sub.add_parser("serve", help="Start the MATLAB MML MCP server")

    args = parser.parse_args()

    if args.command == "index":
        # Set up logging for CLI use
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        from osprey.mcp_server.matlab.indexer import build_index

        db_path = build_index(
            data_file=args.data_file,
            db_path=args.db_path,
            batch_size=args.batch_size,
        )
        print(f"Index built: {db_path}")

    elif args.command == "serve" or args.command is None:
        from osprey.mcp_server.common import run_mcp_server

        run_mcp_server("osprey.mcp_server.matlab.server")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
