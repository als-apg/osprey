"""Build-time index of the plan catalog a built project will expose.

`osprey build` renders a bundled skill that tells the agent which Bluesky plans
exist before it ever calls the bridge. That listing has to be derived at build
time, from ``config.yml`` alone: the bridge is not running yet, and the plan
files must not be executed to be described. So this module walks the same
directory layers the runtime loader would
(:func:`osprey.services.bluesky_bridge.plan_loader._resolve_plan_dir_layers`)
and reads each file's ``PLAN_METADATA`` with the loader-free ``ast`` extractor
in :mod:`osprey.services.bluesky_bridge.plan_metadata`.

Two properties this module is built around:

- **Fail-soft.** The bridge's own plan surfaces are fail-closed: a malformed
  plan is quarantined rather than defaulted. That posture is right at run time
  and wrong here — a bad plan file must never abort a build. So a missing plan
  directory yields an entry in :attr:`PlanIndex.unreadable_dirs`, a malformed
  ``PLAN_METADATA`` yields a warning and no row, and neither raises. The two
  postures compose: the bad plan is still skipped, the build still finishes.
- **Import-clean.** Importing this module must not pull in bluesky, ophyd, or
  tiled. ``plan_loader`` and ``plan_metadata`` are both documented as free of
  those imports; this module only reaches for the pieces of them that keep that
  boundary intact, and a test asserts the property in a clean subprocess.

Provenance is assigned here exactly the way the loader assigns it — from the
directory layer a file came from, never self-declared by a plan author — so the
build-time listing and the runtime catalog agree on trust tier.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.services.bluesky_bridge.plan_loader import (
    _SHIPPED_PLANS_DIR,
    _TRUST_ORDER,
    _iter_plan_files,
)
from osprey.services.bluesky_bridge.plan_metadata import (
    PlanMetadataError,
    extract_plan_metadata_from_source,
)
from osprey.services.bluesky_bridge.plan_types import Provenance

__all__ = [
    "DESCRIPTION_LIMIT",
    "MAX_ROWS",
    "PlanIndex",
    "PlanIndexRow",
    "build_plan_index",
]

# Longest description a rendered row may carry. The skill body is agent context,
# not documentation: one line per plan is enough to decide whether to ask the
# bridge for the full schema.
DESCRIPTION_LIMIT = 120

# Most rows a rendered index may carry. Beyond this the listing stops paying for
# itself as context; the caller renders the overflow count as a pointer to
# ``list_plans`` instead.
MAX_ROWS = 40

# A sentence ends at .!? only when whitespace or end-of-string follows, so
# "bp.grid_scan)." and "v1.2" do not read as sentence boundaries.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


@dataclass(frozen=True)
class PlanIndexRow:
    """One plan's build-time listing entry.

    Attributes:
        name: The plan's registered name, from its ``PLAN_METADATA``. This is
            the key the loader registers under and the key ``excluded_plans``
            matches against — not the file stem.
        description: The plan's description, already shaped for rendering (see
            :func:`_shape_description`); never longer than `DESCRIPTION_LIMIT`.
        writes: Whether the plan moves anything, as declared by its author.
        provenance: The trust tier of the directory layer the file came from.
    """

    name: str
    description: str
    writes: bool
    provenance: Provenance


@dataclass(frozen=True)
class PlanIndex:
    """The result of one build-time index pass.

    An instance always means the pass ran. A pass that resolved no plans at all
    returns an instance with empty `rows`, which a caller can tell apart from
    "the pass never ran" (no instance) — the distinction matters because the
    first should render an honest "no plans" line and the second should render
    nothing.

    Attributes:
        rows: Listing rows, sorted by ``(trust tier, name)`` and capped at
            `MAX_ROWS`.
        overflow: How many resolved plans did not fit in `rows`, so the caller
            can render an "N further plans" pointer to ``list_plans``.
        warnings: Human-readable notes about files that were skipped, each
            naming the offending file. Never fatal.
        unreadable_dirs: Configured plan directories that do not exist or could
            not be read. The other layers' rows are still returned.
    """

    rows: tuple[PlanIndexRow, ...] = ()
    overflow: int = 0
    warnings: tuple[str, ...] = ()
    unreadable_dirs: tuple[str, ...] = ()

    @property
    def has_unreadable_dirs(self) -> bool:
        """Whether any configured plan directory could not be read."""
        return bool(self.unreadable_dirs)


def build_plan_index(config: Mapping[str, Any], repo_root: Path) -> PlanIndex:
    """Index the plans a built project will expose, from ``config.yml`` alone.

    Scans, in ascending trust order, the shipped ``plans_core/`` directory, the
    directories named by ``bluesky.plan_dirs``, and the directory named by
    ``services.bluesky.plan_dir``; reads each file's ``PLAN_METADATA`` without
    importing it; drops the names listed in either ``excluded_plans`` spelling;
    and returns the survivors as sorted, length-capped rows.

    Never raises for a bad input: this runs inside ``osprey build``, where an
    unreadable plan directory or a malformed plan file is a warning, not a
    build failure. See `PlanIndex` for how each is reported.

    Args:
        config: The parsed ``config.yml`` mapping.
        repo_root: The built project's root. A relative plan directory resolves
            against this, matching the compose bind rule in
            ``templates/services/bluesky/docker-compose.yml.j2``; the build does
            not run from the repo root, so resolving against the CWD would
            silently index the wrong directory (or nothing).

    Returns:
        The `PlanIndex` for this configuration.
    """
    excluded = _resolve_excluded_plans(config)
    warnings: list[str] = []
    unreadable: list[str] = []
    by_name: dict[str, PlanIndexRow] = {}

    for directory, provenance, configured in _resolve_layers(config, repo_root):
        for path in _scan_layer(directory, configured, warnings, unreadable):
            try:
                meta = extract_plan_metadata_from_source(path)
            except PlanMetadataError as exc:
                # Skip the file whole — a row built from half-read metadata
                # would misdescribe a plan the operator can actually run.
                warnings.append(f"skipped plan file: {exc}")
                continue
            if meta.name in excluded:
                continue
            existing = by_name.get(meta.name)
            if (
                existing is not None
                and _TRUST_ORDER[existing.provenance] > _TRUST_ORDER[provenance]
            ):
                # Mirrors the loader: a lower-trust layer never reclaims a name
                # a higher-trust layer already owns.
                continue
            by_name[meta.name] = PlanIndexRow(
                name=meta.name,
                description=_shape_description(meta.description),
                writes=meta.writes,
                provenance=provenance,
            )

    ordered = sorted(by_name.values(), key=lambda row: (_TRUST_ORDER[row.provenance], row.name))
    return PlanIndex(
        rows=tuple(ordered[:MAX_ROWS]),
        overflow=max(0, len(ordered) - MAX_ROWS),
        warnings=tuple(warnings),
        unreadable_dirs=tuple(unreadable),
    )


def _resolve_layers(
    config: Mapping[str, Any], repo_root: Path
) -> list[tuple[Path, Provenance, bool]]:
    """Ordered ``(directory, provenance, configured)`` layers, ascending trust.

    Mirrors ``plan_loader._resolve_plan_dir_layers`` for the tiers that exist at
    build time. The runtime ``session`` tier has no build-time counterpart — it
    is agent-authored at run time, so nothing is there to index.

    ``configured`` marks a directory the operator asked for, and so one whose
    absence is worth reporting. The shipped directory is not configured: the
    loader documents an absent/empty ``plans_core/`` as normal, not an error.
    """
    bluesky = _submapping(config, "bluesky")
    service = _submapping(_submapping(config, "services"), "bluesky")

    layers: list[tuple[Path, Provenance, bool]] = [(_SHIPPED_PLANS_DIR, "shipped", False)]
    layers.extend(
        (_resolve_dir(value, repo_root), "preset", True)
        for value in _as_str_list(bluesky.get("plan_dirs"))
    )
    layers.extend(
        (_resolve_dir(value, repo_root), "facility", True)
        for value in _as_str_list(service.get("plan_dir"))
    )
    return layers


def _scan_layer(
    directory: Path, configured: bool, warnings: list[str], unreadable: list[str]
) -> list[Path]:
    """List one layer's plan files, recording rather than raising on failure."""
    try:
        if not directory.is_dir():
            if configured:
                unreadable.append(str(directory))
                warnings.append(f"plan directory not found: {directory}")
            return []
        return list(_iter_plan_files(directory))
    except OSError as exc:
        if configured:
            unreadable.append(str(directory))
        warnings.append(f"plan directory could not be read: {directory} ({exc})")
        return []


def _resolve_excluded_plans(config: Mapping[str, Any]) -> set[str]:
    """Plan names to drop, unioned across both ``excluded_plans`` spellings.

    ``bluesky.excluded_plans`` is authored as a list (or a bare string);
    ``services.bluesky.excluded_plans`` is written by the build injector as one
    ``os.pathsep``-joined string, because that is the form the compose template
    hands the bridge as ``BLUESKY_EXCLUDED_PLANS``. Splitting every token on
    ``os.pathsep`` reads both shapes: a plan name never contains the separator,
    so splitting an already-split list is a no-op.
    """
    excluded: set[str] = set()
    for source in (
        _submapping(config, "bluesky"),
        _submapping(_submapping(config, "services"), "bluesky"),
    ):
        for value in _as_str_list(source.get("excluded_plans")):
            excluded.update(token for token in value.split(os.pathsep) if token)
    return excluded


def _resolve_dir(value: str, repo_root: Path) -> Path:
    """Resolve a configured plan directory against the built project's root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _submapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return ``config[key]`` when it is a mapping, else an empty one.

    Config comes off disk, so any key may be absent or hold the wrong type;
    neither is worth failing a build over.
    """
    value = config.get(key)
    return value if isinstance(value, Mapping) else {}


def _as_str_list(value: Any) -> list[str]:
    """Coerce a config value to a list of non-empty strings.

    A bare string is a one-element list, mirroring ``plan_loader``'s own
    handling of ``bluesky.plan_dirs`` — an operator who writes one directory
    without a list gets what they meant, not a scan of its characters.
    """
    if value is None:
        return []
    if isinstance(value, str | Path):
        value = [value]
    elif not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if str(item)]


def _shape_description(description: str) -> str:
    """Shape a plan description into one short listing line.

    Keeps the first sentence, then hard-caps at `DESCRIPTION_LIMIT` characters
    (ellipsis included), whichever cuts sooner. Whitespace is collapsed first,
    because shipped plans write their descriptions as implicitly concatenated
    source lines.
    """
    collapsed = " ".join(description.split())
    match = _SENTENCE_END.search(collapsed)
    if match is not None:
        collapsed = collapsed[: match.end()]
    if len(collapsed) > DESCRIPTION_LIMIT:
        collapsed = collapsed[: DESCRIPTION_LIMIT - 1].rstrip() + "…"
    return collapsed
