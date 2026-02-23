"""Entry point for ``python -m osprey.mcp_server.accelpapers``.

Subcommands:
    index  — Build the Typesense collection from INSPIRE JSON files.
    status — Show Typesense collection info.
    serve  — Start the MCP server (default).
"""

import argparse
import logging
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m osprey.mcp_server.accelpapers",
        description="AccelPapers MCP Server — search INSPIRE accelerator physics papers",
    )
    sub = parser.add_subparsers(dest="command")

    # --- index subcommand ---
    idx = sub.add_parser("index", help="Build Typesense collection from INSPIRE JSON files")
    idx.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing INSPIRE JSON files (with subdirectories)",
    )
    idx.add_argument(
        "--typesense-host",
        default=None,
        help="Typesense server host (default: localhost, or ACCELPAPERS_TYPESENSE_HOST env)",
    )
    idx.add_argument(
        "--typesense-port",
        default=None,
        help="Typesense server port (default: 8108, or ACCELPAPERS_TYPESENSE_PORT env)",
    )
    idx.add_argument(
        "--api-key",
        default=None,
        help="Typesense API key (default: accelpapers-dev, or ACCELPAPERS_TYPESENSE_API_KEY env)",
    )
    idx.add_argument(
        "--collection",
        default=None,
        help="Typesense collection name (default: papers, or ACCELPAPERS_COLLECTION env)",
    )
    idx.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Documents per import batch (default: 200)",
    )

    # --- status subcommand ---
    sub.add_parser("status", help="Show Typesense collection info")

    # --- serve subcommand ---
    sub.add_parser("serve", help="Start the AccelPapers MCP server")

    args = parser.parse_args()

    if args.command == "index":
        # Set up logging for CLI use
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        # Set env vars from CLI args (takes precedence over existing env)
        if args.typesense_host:
            os.environ["ACCELPAPERS_TYPESENSE_HOST"] = args.typesense_host
        if args.typesense_port:
            os.environ["ACCELPAPERS_TYPESENSE_PORT"] = args.typesense_port
        if args.api_key:
            os.environ["ACCELPAPERS_TYPESENSE_API_KEY"] = args.api_key
        if args.collection:
            os.environ["ACCELPAPERS_COLLECTION"] = args.collection

        from osprey.mcp_server.accelpapers.indexer import build_index

        collection_name = build_index(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
        )
        print(f"Index built: collection '{collection_name}'")

    elif args.command == "status":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        from osprey.mcp_server.accelpapers.db import get_client, get_collection_name

        client = get_client()
        collection_name = get_collection_name()

        try:
            info = client.collections[collection_name].retrieve()
        except Exception as exc:
            print(f"Error: Could not connect to Typesense or collection not found: {exc}")
            sys.exit(1)

        print(f"Collection: {info['name']}")
        print(f"Documents:  {info.get('num_documents', 'unknown')}")
        print(f"Fields:     {len(info.get('fields', []))}")
        print("Field names:")
        for field in info.get("fields", []):
            facet = " (facet)" if field.get("facet") else ""
            optional = " (optional)" if field.get("optional") else ""
            print(f"  - {field['name']}: {field['type']}{facet}{optional}")

    elif args.command == "serve" or args.command is None:
        from osprey.mcp_server.common import run_mcp_server

        run_mcp_server("osprey.mcp_server.accelpapers.server")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
