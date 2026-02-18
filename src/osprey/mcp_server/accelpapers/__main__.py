"""Entry point for ``python -m osprey.mcp_server.accelpapers``.

Subcommands:
    index  — Build the SQLite FTS5 index from INSPIRE JSON files.
    serve  — Start the MCP server (default).
"""

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m osprey.mcp_server.accelpapers",
        description="AccelPapers MCP Server — search INSPIRE accelerator physics papers",
    )
    sub = parser.add_subparsers(dest="command")

    # --- index subcommand ---
    idx = sub.add_parser("index", help="Build SQLite FTS5 index from INSPIRE JSON files")
    idx.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing INSPIRE JSON files (with subdirectories)",
    )
    idx.add_argument(
        "--db-path",
        default=None,
        help="Output SQLite database path (default: ~/.accelpapers/papers.db)",
    )
    idx.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows per INSERT batch (default: 500)",
    )

    # --- serve subcommand ---
    sub.add_parser("serve", help="Start the AccelPapers MCP server")

    args = parser.parse_args()

    if args.command == "index":
        # Set up logging for CLI use
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        from osprey.mcp_server.accelpapers.indexer import build_index

        db_path = build_index(
            data_dir=args.data_dir,
            db_path=args.db_path,
            batch_size=args.batch_size,
        )
        print(f"Index built: {db_path}")

    elif args.command == "serve" or args.command is None:
        from osprey.mcp_server.common import run_mcp_server

        run_mcp_server("osprey.mcp_server.accelpapers.server")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
