"""Build orchestrator: load -> validate -> emit for the design-token generator.

Ties together the rest of the generator package into the one pipeline the
project actually runs:

    python -m osprey.interfaces.design_system.generator.build [--check]

Without ``--check``, it loads the real ``tokens/`` source tree, validates
it (aborting with every error printed if invalid), renders all three
generated artifacts, and writes them under ``static/``. With ``--check``,
it does the same load-and-validate-and-render, but never writes — instead
it diffs each rendered artifact against what's already on disk and exits
non-zero with a unified diff for anything that drifted. This is the
freshness gate a later task wires into CI.

The load/validate/render step is exposed as :func:`build_artifacts` with
explicit ``tokens_dir``/output paths (no hardcoded real paths baked into
its signature), so tests can run the whole pipeline hermetically against a
temporary tree instead of the real one.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from osprey.interfaces.design_system.generator.emit_css import emit_css
from osprey.interfaces.design_system.generator.emit_js import (
    render_theme_boot_js,
    render_tokens_js,
)
from osprey.interfaces.design_system.generator.model import TokenModelError, load_token_tree
from osprey.interfaces.design_system.generator.validate import TokenValidationError, assert_valid

__all__ = [
    "DEFAULT_TOKENS_DIR",
    "DEFAULT_STATIC_DIR",
    "CSS_RELATIVE_PATH",
    "TOKENS_JS_RELATIVE_PATH",
    "THEME_BOOT_JS_RELATIVE_PATH",
    "BuildArtifact",
    "BuildError",
    "ArtifactDiff",
    "build_artifacts",
    "write_artifacts",
    "check_artifacts",
    "main",
]

#: The design_system package root (.../src/osprey/interfaces/design_system).
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Real token source directory, used by the CLI entrypoint's defaults.
DEFAULT_TOKENS_DIR = _PACKAGE_ROOT / "tokens"
#: Real generated-artifact output directory, used by the CLI entrypoint's defaults.
DEFAULT_STATIC_DIR = _PACKAGE_ROOT / "static"

#: Output locations, relative to a static directory root.
CSS_RELATIVE_PATH = Path("css/tokens.css")
TOKENS_JS_RELATIVE_PATH = Path("js/tokens.js")
THEME_BOOT_JS_RELATIVE_PATH = Path("js/theme-boot.js")


class BuildError(Exception):
    """Raised when the load-validate-render pipeline cannot produce artifacts.

    The message already contains every underlying failure (token-source
    parse errors, or every :class:`~.validate.ValidationError` — never
    just the first), ready to print as-is.
    """


@dataclass(frozen=True)
class BuildArtifact:
    """One generated file: where it goes and what it contains.

    Attributes:
        relative_path: Location relative to a static directory root (see
            :data:`CSS_RELATIVE_PATH` and friends).
        content: The complete, ready-to-write file content.
    """

    relative_path: Path
    content: str


@dataclass(frozen=True)
class ArtifactDiff:
    """One artifact's on-disk-vs-regenerated mismatch, for ``--check`` mode.

    Attributes:
        relative_path: Which artifact drifted.
        unified_diff: A human-readable unified diff, on-disk vs regenerated
            (an on-disk file that doesn't exist at all is treated as empty).
    """

    relative_path: Path
    unified_diff: str


def build_artifacts(tokens_dir: Path) -> list[BuildArtifact]:
    """Load, validate, and render every generated artifact for ``tokens_dir``.

    Args:
        tokens_dir: The ``tokens/`` source directory to build from.

    Returns:
        The three build artifacts — ``tokens.css``, ``tokens.js``,
        ``theme-boot.js``, in that stable order.

    Raises:
        BuildError: If loading the token sources fails (wraps
            :class:`~.model.TokenModelError`), or if validation finds any
            failures (wraps :class:`~.validate.TokenValidationError`; the
            message lists every failure, per the "report ALL failures, not
            first-only" contract).
    """
    try:
        tree = load_token_tree(tokens_dir)
    except TokenModelError as exc:
        raise BuildError(f"failed to load token sources: {exc}") from exc

    try:
        assert_valid(tree)
    except TokenValidationError as exc:
        details = "\n".join(str(error) for error in exc.errors)
        raise BuildError(f"{len(exc.errors)} token validation error(s):\n{details}") from exc

    return [
        BuildArtifact(CSS_RELATIVE_PATH, emit_css(tree)),
        BuildArtifact(TOKENS_JS_RELATIVE_PATH, render_tokens_js(tree)),
        BuildArtifact(THEME_BOOT_JS_RELATIVE_PATH, render_theme_boot_js(tree)),
    ]


def write_artifacts(artifacts: Sequence[BuildArtifact], static_dir: Path) -> None:
    """Write every artifact under ``static_dir``, creating directories as needed.

    Args:
        artifacts: The artifacts to write, e.g. from :func:`build_artifacts`.
        static_dir: The static directory root each artifact's
            ``relative_path`` is resolved against.
    """
    for artifact in artifacts:
        target = static_dir / artifact.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact.content, encoding="utf-8")


def check_artifacts(artifacts: Sequence[BuildArtifact], static_dir: Path) -> list[ArtifactDiff]:
    """Compare freshly rendered artifacts against what's on disk. Never writes.

    Args:
        artifacts: Freshly rendered artifacts, e.g. from
            :func:`build_artifacts`.
        static_dir: The static directory root each artifact's
            ``relative_path`` is resolved against.

    Returns:
        One :class:`ArtifactDiff` per artifact whose on-disk content
        differs from the regenerated content (a missing on-disk file
        counts as differing). Artifacts that already match are omitted;
        an empty list means everything is fresh.
    """
    diffs: list[ArtifactDiff] = []
    for artifact in artifacts:
        target = static_dir / artifact.relative_path
        on_disk = target.read_text(encoding="utf-8") if target.is_file() else ""
        if on_disk == artifact.content:
            continue
        unified_diff = "".join(
            difflib.unified_diff(
                on_disk.splitlines(keepends=True),
                artifact.content.splitlines(keepends=True),
                fromfile=f"{artifact.relative_path} (on disk)",
                tofile=f"{artifact.relative_path} (regenerated)",
            )
        )
        diffs.append(ArtifactDiff(artifact.relative_path, unified_diff))
    return diffs


def main(
    argv: Sequence[str] | None = None,
    *,
    tokens_dir: Path = DEFAULT_TOKENS_DIR,
    static_dir: Path = DEFAULT_STATIC_DIR,
) -> int:
    """CLI entrypoint: build, or ``--check``, the design-token artifacts.

    Args:
        argv: Command-line arguments (excluding argv[0]); defaults to
            ``sys.argv[1:]`` via :meth:`argparse.ArgumentParser.parse_args`.
        tokens_dir: Override for the token source directory. Defaults to
            the real ``tokens/`` tree; overridable for hermetic testing.
        static_dir: Override for the generated-artifact output directory.
            Defaults to the real ``static/`` tree; overridable for
            hermetic testing.

    Returns:
        Process exit code: ``0`` on success (or a clean ``--check``),
        ``1`` on a build/validation failure or (``--check`` only) drift.
    """
    parser = argparse.ArgumentParser(
        prog="python -m osprey.interfaces.design_system.generator.build",
        description="Build (or --check) the OSPREY design-system generated artifacts.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Regenerate in-memory and diff against static/ without writing; exit non-zero on drift."
        ),
    )
    args = parser.parse_args(argv)

    try:
        artifacts = build_artifacts(tokens_dir)
    except BuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.check:
        diffs = check_artifacts(artifacts, static_dir)
        if not diffs:
            print("tokens.css, tokens.js, and theme-boot.js are up to date.")
            return 0
        for diff in diffs:
            print(f"--- drift in {diff.relative_path} ---", file=sys.stderr)
            print(diff.unified_diff, file=sys.stderr)
        return 1

    write_artifacts(artifacts, static_dir)
    for artifact in artifacts:
        print(f"wrote {static_dir / artifact.relative_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
