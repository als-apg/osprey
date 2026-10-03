"""Benchmark channel helpers over the hierarchical template database.

Expands the hierarchical channel database template into flat channel lists,
generates human-readable descriptions and aliases, declares the paradigms each
benchmark tier publishes, and checks a query set's targeted channels against
the tier databases.
"""

from __future__ import annotations

import json
from pathlib import Path

from osprey.build.modes import VALID_CHANNEL_FINDER_MODES
from osprey.services.channel_finder import naming

# Root of the control_assistant preset's shipped data tree. It anchors both the
# canonical template DB and the tier subsets, so consumers derive their paths
# from here rather than walking ``parents[]`` off TEMPLATE_DB_PATH.
TEMPLATE_DATA_DIR = (
    Path(__file__).resolve().parents[3] / "templates" / "apps" / "control_assistant" / "data"
)

# Path to the canonical hierarchical template database — the full ~2900-channel
# structural superset. It is the tier-3 hierarchical view (tier 3 is unfiltered),
# and every tier subset / paradigm is generated from the same content.
TEMPLATE_DB_PATH = TEMPLATE_DATA_DIR / "channel_databases" / "tiers" / "tier3" / "hierarchical.json"


def load_template(source_path: Path | None = None) -> tuple[dict, list[dict]]:
    """Load a hierarchical template and expand to flat channels.

    Args:
        source_path: Path to hierarchical JSON. Defaults to the packaged demo
            template — callers that write into a deployment's own database
            directory name the source explicitly rather than relying on it.

    Returns:
        (tree_data, expanded_channels) tuple.
    """
    path = source_path or TEMPLATE_DB_PATH
    tree_data = json.loads(path.read_text(encoding="utf-8"))
    channels = expand_hierarchy(tree_data)
    return tree_data, channels


# ---------------------------------------------------------------------------
# Description-phrase and alias-token maps
# ---------------------------------------------------------------------------
#
# The vocabulary itself lives in :mod:`osprey.services.channel_finder.naming`
# (shared with the tier-DB generator). The names below are this module's
# long-standing public API, kept as thin views of the canonical maps.

RING_NAMES: dict[str, str] = naming.RING_PHRASES
FIELD_NAMES: dict[str, str] = naming.FIELD_PHRASES
SUBFIELD_NAMES: dict[str, str] = naming.SUBFIELD_PHRASES
FAMILY_NAMES: dict[str, str] = naming.FAMILY_PHRASES

ALIAS_RING_NAMES: dict[str, str] = naming.RING_TOKENS
ALIAS_FIELD_NAMES: dict[str, str] = naming.FIELD_TOKENS
ALIAS_SUBFIELD_NAMES: dict[str, str] = naming.SUBFIELD_TOKENS
ALIAS_FAMILY_NAMES: dict[str, str] = naming.FAMILY_TOKENS


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _expand_instances(expansion_def: dict) -> list[str]:
    """Expand an ``_expansion`` directive into a list of instance names.

    Supports two expansion types present in the template database:

    * **range** -- ``_pattern`` + ``_range`` (inclusive on both ends)
    * **list** -- ``_instances`` explicit list
    """
    expansion_type = expansion_def.get("_type")

    if expansion_type == "range":
        pattern = expansion_def.get("_pattern", "{}")
        start, end = expansion_def.get("_range", [1, 1])
        return [pattern.format(i) for i in range(start, end + 1)]

    if expansion_type == "list":
        return list(expansion_def.get("_instances", []))

    return []


def _is_metadata_key(key: str) -> bool:
    """Return True if *key* is a metadata key (starts with ``_``)."""
    return key.startswith("_")


def expand_hierarchy(tree_data: dict) -> list[dict]:
    """Expand a hierarchical channel tree into flat channel entries.

    Traverses the 6-level hierarchy
    (ring -> system -> family -> DEVICE -> field -> subfield)
    and expands all ``_expansion`` directives into concrete channel records.

    Args:
        tree_data: The full JSON object loaded from the hierarchical
            template database (must contain a ``"tree"`` key).

    Returns:
        Sorted list of dicts, each with keys:
        ``pv``, ``ring``, ``system``, ``family``, ``device``,
        ``field``, ``subfield``.
    """
    tree = tree_data.get("tree", tree_data)
    channels: list[dict] = []

    for ring_name, ring_node in tree.items():
        if _is_metadata_key(ring_name):
            continue

        for system_name, system_node in ring_node.items():
            if _is_metadata_key(system_name):
                continue

            for family_name, family_node in system_node.items():
                if _is_metadata_key(family_name):
                    continue

                # The DEVICE key holds _expansion + field/subfield siblings
                device_node = family_node.get("DEVICE", {})
                expansion = device_node.get("_expansion")
                if expansion is None:
                    continue

                device_names = _expand_instances(expansion)

                # Collect field -> [subfield, ...] from siblings of
                # _expansion inside the DEVICE node
                for field_name, field_node in device_node.items():
                    if _is_metadata_key(field_name):
                        continue
                    if not isinstance(field_node, dict):
                        continue

                    for subfield_name, subfield_node in field_node.items():
                        if _is_metadata_key(subfield_name):
                            continue
                        if not isinstance(subfield_node, dict):
                            continue

                        for device in device_names:
                            pv = ":".join(
                                [
                                    ring_name,
                                    system_name,
                                    family_name,
                                    device,
                                    field_name,
                                    subfield_name,
                                ]
                            )
                            channels.append(
                                {
                                    "pv": pv,
                                    "ring": ring_name,
                                    "system": system_name,
                                    "family": family_name,
                                    "device": device,
                                    "field": field_name,
                                    "subfield": subfield_name,
                                }
                            )

    channels.sort(key=lambda c: c["pv"])
    return channels


def generate_description(pv_parts: dict) -> str:
    """Generate a natural-language description for a PV.

    Args:
        pv_parts: Dict with keys ``ring``, ``system``, ``family``,
            ``device``, ``field``, ``subfield``.

    Returns:
        Human-readable description string, e.g.
        ``"Storage ring dipole bending magnet B01 current setpoint"``.
    """
    ring = RING_NAMES.get(pv_parts["ring"], pv_parts["ring"])
    family = FAMILY_NAMES.get(pv_parts["family"], pv_parts["family"])
    device = pv_parts["device"]
    field_desc = FIELD_NAMES.get(pv_parts["field"], pv_parts["field"])
    subfield_desc = SUBFIELD_NAMES.get(pv_parts["subfield"], pv_parts["subfield"])

    # Capitalise first word only
    desc = f"{ring} {family} {device} {field_desc} {subfield_desc}"
    return desc[0].upper() + desc[1:]


def generate_alias(pv_parts: dict) -> str:
    """Generate a short alias for a PV.

    Composes aliases as ``{AliasRing}_{AliasFamily}_{Device}_{AliasField}_{AliasSubfield}``,
    falling back to the raw name for any component without a mapping.

    Args:
        pv_parts: Dict with keys ``ring``, ``system``, ``family``,
            ``device``, ``field``, ``subfield``.

    Returns:
        Alias string, e.g. ``"StorageRing_Dipole_B05_Current_Setpoint"``.
    """
    ring = ALIAS_RING_NAMES.get(pv_parts["ring"], pv_parts["ring"])
    family = ALIAS_FAMILY_NAMES.get(pv_parts["family"], pv_parts["family"])
    device = pv_parts["device"]
    field_alias = ALIAS_FIELD_NAMES.get(pv_parts["field"], pv_parts["field"])
    subfield_alias = ALIAS_SUBFIELD_NAMES.get(pv_parts["subfield"], pv_parts["subfield"])

    return f"{ring}_{family}_{device}_{field_alias}_{subfield_alias}"


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------

# The paradigms tier 3 publishes: every registered paradigm except ``graph``.
#
# ``graph`` is excluded deliberately, and by subtraction rather than by a
# hand-written list so registering a file-backed paradigm joins tier 3 without a
# second edit here. A tier view is a JSON database file this module
# cross-checks query-by-query; the graph paradigm has no such file — its
# store is seeded from the corpus TTL, so there is nothing under
# ``tiers/tier3/`` to name. Graph's equivalent of this cross-check is the
# corpus-vs-database PV-set equality asserted by
# ``tests/services/facility_knowledge/test_demo_ttl_consistency.py``.
_TIER3_PARADIGMS: tuple[str, ...] = tuple(sorted(set(VALID_CHANNEL_FINDER_MODES) - {"graph"}))

# Paradigms published per tier. Tier 1 ships the flat ``in_context`` view only;
# tier 3 ships every tier view. Query validation checks each tier's targeted PVs
# against exactly the paradigms declared here, and callers iterating "all tiers"
# iterate these keys — there is no tier 2.
TIER_PARADIGMS: dict[int, tuple[str, ...]] = {
    1: ("in_context",),
    3: _TIER3_PARADIGMS,
}

# Filename each paradigm view is stored under within a tier directory.
_PARADIGM_FILENAMES: dict[str, str] = {name: f"{name}.json" for name in _TIER3_PARADIGMS}


def collect_middle_layer_pvs(data: dict) -> set[str]:
    """Recursively collect all PVs from ``ChannelNames`` arrays in a middle-layer DB."""
    pvs: set[str] = set()

    for key, value in data.items():
        if key == "ChannelNames" and isinstance(value, list):
            pvs.update(value)
        elif isinstance(value, dict):
            pvs.update(collect_middle_layer_pvs(value))

    return pvs


def _validate_tier(
    queries: list[dict],
    tier_num: int,
    tier_dir: Path,
) -> tuple[list[dict], list[str]]:
    """Validate queries against a single tier's databases.

    Args:
        queries: List of query dicts, each with an optional ``targeted_pv`` list.
        tier_num: Tier number (1 or 3) for reporting; selects the paradigms
            validated via :data:`TIER_PARADIGMS`.
        tier_dir: Path to the tier directory containing the tier's database files.

    Returns:
        Tuple of (missing_entries, missing_database_paths).
    """
    paradigms = TIER_PARADIGMS.get(tier_num, tuple(_PARADIGM_FILENAMES))
    db_files: list[tuple[str, str]] = [(name, _PARADIGM_FILENAMES[name]) for name in paradigms]
    missing: list[dict] = []
    missing_databases: list[str] = []

    format_pv_sets: dict[str, set[str]] = {}
    for fmt_name, filename in db_files:
        path = tier_dir / filename
        if not path.exists():
            missing_databases.append(str(path))
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        if fmt_name == "in_context":
            # Handle both old (list) and new (envelope) formats
            if isinstance(data, dict) and "channels" in data:
                entries = data["channels"]
            else:
                entries = data
            # Use 'address' (PV) when available, fall back to 'channel'
            format_pv_sets[fmt_name] = {entry.get("address", entry["channel"]) for entry in entries}
        elif fmt_name == "hierarchical":
            hier_channels = expand_hierarchy(data)
            format_pv_sets[fmt_name] = {ch["pv"] for ch in hier_channels}
        elif fmt_name == "middle_layer":
            format_pv_sets[fmt_name] = collect_middle_layer_pvs(data)

    for q_idx, query in enumerate(queries):
        for pv in query.get("targeted_pv", []):
            for fmt_name, pv_set in format_pv_sets.items():
                if pv not in pv_set:
                    missing.append(
                        {
                            "query_id": q_idx,
                            "pv": pv,
                            "tier": tier_num,
                            "format": fmt_name,
                        }
                    )

    return missing, missing_databases


def validate_queries(
    queries_path: Path | None = None,
    db_dir: Path | None = None,
    *,
    tier_queries: dict[int, Path] | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Validate that all targeted PVs in query sets exist in the tier databases.

    Supports two modes:

    **Per-tier mode** (new): Pass ``tier_queries`` mapping tier numbers to
    per-tier query files.  Each tier's queries are validated only against
    that tier's databases in ``output_dir / f"tier{tier_num}"/``.

    **Legacy mode** (backward-compatible): Pass ``queries_path`` and
    ``db_dir``.  A single query file is validated against each tier's
    databases under ``db_dir``, using the per-tier paradigms declared in
    :data:`TIER_PARADIGMS`.

    Args:
        queries_path: Path to a single benchmark_queries.json (legacy mode).
        db_dir: Directory containing per-tier subdirs (``tier1/``, ``tier3/``)
            for the tiers in :data:`TIER_PARADIGMS` (legacy mode).
        tier_queries: Mapping of ``{tier_num: query_file_path}`` (per-tier mode).
        output_dir: Base output directory containing tier subdirs (per-tier mode).

    Returns:
        Dict with keys: ``valid`` (bool), ``total_queries``, ``total_pvs``,
        ``missing`` (list of ``{query_id, pv, tier, format}``),
        ``missing_databases`` (list of paths).

    Raises:
        ValueError: If required arguments are missing for the chosen mode.
    """
    all_missing: list[dict] = []
    all_missing_dbs: list[str] = []
    all_pvs: set[str] = set()
    total_queries = 0

    if tier_queries is not None:
        # New per-tier mode
        if output_dir is None:
            raise ValueError("output_dir is required when using tier_queries")
        for tier_num, query_path in sorted(tier_queries.items()):
            queries = json.loads(query_path.read_text(encoding="utf-8"))
            total_queries += len(queries)
            for q in queries:
                all_pvs.update(q.get("targeted_pv", []))
            tier_dir = output_dir / f"tier{tier_num}"
            missing, missing_dbs = _validate_tier(queries, tier_num, tier_dir)
            all_missing.extend(missing)
            all_missing_dbs.extend(missing_dbs)
    elif queries_path is not None:
        # Legacy backward-compatible mode
        if db_dir is None:
            raise ValueError("db_dir is required when using queries_path")
        queries = json.loads(queries_path.read_text(encoding="utf-8"))
        total_queries = len(queries)
        for q in queries:
            all_pvs.update(q.get("targeted_pv", []))
        for tier_num in TIER_PARADIGMS:
            tier_dir = db_dir / f"tier{tier_num}"
            missing, missing_dbs = _validate_tier(queries, tier_num, tier_dir)
            all_missing.extend(missing)
            all_missing_dbs.extend(missing_dbs)
    else:
        raise ValueError("Either queries_path or tier_queries must be provided")

    return {
        "valid": len(all_missing) == 0 and len(all_missing_dbs) == 0,
        "total_queries": total_queries,
        "total_pvs": len(all_pvs),
        "missing": all_missing,
        "missing_databases": all_missing_dbs,
    }
