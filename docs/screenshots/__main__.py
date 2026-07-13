"""CLI entry point: ``python -m docs.screenshots [list] [--only N] [--stack] [--agentic]``.

Default (no command, no flags) captures only the ``standalone_interface`` static
recipes — zero container, CI-safe locally. ``--stack`` opts into the
tutorial-stack recipes (needs a container runtime + free port 5432); ``--agentic``
opts into the live web-terminal hero (needs a live Claude session). ``list``
prints the registry without capturing anything.
"""

from __future__ import annotations

import argparse
import sys

from docs.screenshots.recipes import (
    REGISTRY,
    is_enabled,
    select_recipes,
    validate_registry,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m docs.screenshots", description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "list"],
        help="'run' (default) captures the selected recipes; 'list' prints the registry.",
    )
    parser.add_argument(
        "--only", metavar="NAME", default=None, help="Capture only the named recipe."
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="Include tutorial_stack recipes (containers + port 5432).",
    )
    parser.add_argument(
        "--agentic", action="store_true", help="Include agentic recipes (live Claude session)."
    )
    return parser


def _print_registry() -> None:
    if not REGISTRY:
        print("(registry is empty)")
        return
    width = max(len(s.name) for s in REGISTRY)
    for shot in REGISTRY:
        outs = ", ".join(f"{o}.png" for o in shot.output_names())
        print(
            f"{shot.name:<{width}}  {shot.environment:<20} {shot.kind:<8} themes={','.join(shot.themes)}"
        )
        print(f"{'':<{width}}    → {outs}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # A malformed registry is a hard error for every command — surface it clearly.
    try:
        validate_registry()
    except ValueError as exc:
        print(f"Invalid screenshot registry: {exc}", file=sys.stderr)
        return 2

    if args.command == "list":
        _print_registry()
        return 0

    selected = select_recipes(stack=args.stack, agentic=args.agentic, only=args.only)

    if not selected:
        if args.only is not None:
            known = next((s for s in REGISTRY if s.name == args.only), None)
            if known is None:
                print(
                    f"No recipe named {args.only!r}. Run 'list' to see the registry.",
                    file=sys.stderr,
                )
            elif not is_enabled(known, stack=args.stack, agentic=args.agentic):
                flag = "--agentic" if known.kind == "agentic" else "--stack"
                print(f"Recipe {args.only!r} needs {flag}. Re-run with it added.", file=sys.stderr)
            return 1
        print(
            "No recipes selected. (Default runs standalone recipes; add --stack/--agentic.)",
            file=sys.stderr,
        )
        return 1

    # Import the runner lazily: 'list' and selection must work even where the
    # heavy capture dependencies (Playwright/chromium) are unavailable.
    from docs.screenshots import capture

    capture.run(selected, stack=args.stack, agentic=args.agentic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
