"""Cross-paradigm benchmark database generator.

Expands the hierarchical channel database template into flat channel lists,
generates human-readable descriptions, and filters by tier specifications
for the three benchmark scale tiers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from osprey.services.channel_finder import naming
from osprey.services.channel_finder.tools.generate_from_spec import TIER1_FILTER

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
        source_path: Path to hierarchical JSON. Defaults to built-in template.

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
#
# The three spec families with no pre-unification precedent (QFA/SHF/SHD)
# are excluded here even though the template database now contains them:
# benchmark aliases/descriptions for those families have always rendered the
# raw family token (e.g. ``StorageRing_QFA_01_Current_Setpoint``) via the
# ``.get`` fallback, and this module preserves that output byte-for-byte.
# Dropping the exclusion would align benchmark prose with the shipped tier-3
# ``in_context.json`` (``QuadFocusAchromat`` ...) but changes generated
# benchmark DB content -- a deliberate decision, not a refactor.
_UNMAPPED_SPEC_FAMILIES = frozenset({"QFA", "SHF", "SHD"})

RING_NAMES: dict[str, str] = naming.RING_PHRASES
FIELD_NAMES: dict[str, str] = naming.FIELD_PHRASES
SUBFIELD_NAMES: dict[str, str] = naming.SUBFIELD_PHRASES
FAMILY_NAMES: dict[str, str] = {
    fam: phrase
    for fam, phrase in naming.FAMILY_PHRASES.items()
    if fam not in _UNMAPPED_SPEC_FAMILIES
}

ALIAS_RING_NAMES: dict[str, str] = naming.RING_TOKENS
ALIAS_FIELD_NAMES: dict[str, str] = naming.FIELD_TOKENS
ALIAS_SUBFIELD_NAMES: dict[str, str] = naming.SUBFIELD_TOKENS
ALIAS_FAMILY_NAMES: dict[str, str] = {
    fam: token for fam, token in naming.FAMILY_TOKENS.items() if fam not in _UNMAPPED_SPEC_FAMILIES
}


# ---------------------------------------------------------------------------
# Tier specification dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierSpec:
    """Specification for a benchmark scale tier.

    Attributes:
        name: Human-readable tier name.
        rings: Set of ring abbreviations to include (e.g. {"SR"}).
        families: Set of family abbreviations to include (None = all).
        allowed_subfields: Set of subfield names to include (None = all).
        allowed_fields_by_family: Per-family field restrictions. Maps
            family name to frozenset of allowed field names.
            ``None`` means all fields are allowed for all families.
    """

    name: str
    rings: frozenset[str]
    families: frozenset[str] | None = None
    allowed_subfields: frozenset[str] | None = None
    allowed_fields_by_family: dict[str, frozenset[str]] | None = field(default=None, hash=False)


# Tier 1 is a declared subset of tier 3; its ring/family/field/subfield
# allow-lists come from the single canonical :data:`TIER1_FILTER` definition
# rather than being restated here. Expected channel counts are computed from
# the live filter at use sites, never pinned as literals.
TIER_1 = TierSpec(
    name="tier1",
    rings=TIER1_FILTER.rings,
    families=TIER1_FILTER.families,
    allowed_subfields=TIER1_FILTER.subfields,
    allowed_fields_by_family={
        family: frozenset({field_name})
        for family, field_name in TIER1_FILTER.field_by_family.items()
    },
)

TIER_3 = TierSpec(
    name="tier3",
    rings=frozenset({"SR", "BR", "BTS"}),
    families=None,  # all families
    allowed_subfields=None,  # all subfields
)


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


def filter_channels(
    channels: list[dict],
    tier_spec: TierSpec,
) -> list[dict]:
    """Filter expanded channels according to a tier specification.

    Args:
        channels: Full list of expanded channel dicts from
            :func:`expand_hierarchy`.
        tier_spec: The tier spec controlling which rings, families,
            and subfields are included.

    Returns:
        Filtered (and sorted) list of channel dicts.
    """
    filtered: list[dict] = []

    for ch in channels:
        if ch["ring"] not in tier_spec.rings:
            continue
        if tier_spec.families is not None and ch["family"] not in tier_spec.families:
            continue
        if (
            tier_spec.allowed_subfields is not None
            and ch["subfield"] not in tier_spec.allowed_subfields
        ):
            continue
        if tier_spec.allowed_fields_by_family is not None:
            allowed = tier_spec.allowed_fields_by_family.get(ch["family"])
            if allowed is not None and ch["field"] not in allowed:
                continue
        filtered.append(ch)

    filtered.sort(key=lambda c: c["pv"])
    return filtered


# ---------------------------------------------------------------------------
# Paradigm-specific format functions
# ---------------------------------------------------------------------------

# DataType / HWUnits lookup tables for middle-layer output
_DATATYPE_BY_FIELD: dict[str, str] = {
    "STATUS": "enum",
    "CONTROL": "enum",
}
_DEFAULT_DATATYPE = "double"

_HWUNITS_BY_FIELD: dict[str, str] = {
    "CURRENT": "A",
    "POSITION": "mm",
    "VOLTAGE": "V",
    "PRESSURE": "Pa",
    "POWER": "W",
    "FREQUENCY": "MHz",
    "TEMPERATURE": "C",
    "TUNER": "cm",
    "GOLDEN": "mm",
    "OFFSET": "mm",
    "DOSE_RATE": "mrem/hr",
    "STATUS": "",
    "CONTROL": "",
}
_DEFAULT_HWUNITS = ""


def format_in_context(channels: list[dict], tier_spec: TierSpec) -> dict:
    """Format channels for the in-context (flat) database paradigm.

    Filters *channels* by *tier_spec* and returns a dict with
    ``_metadata`` and ``channels`` keys.  Each channel entry has
    ``channel`` (alias), ``address`` (PV name), and ``description``.

    Args:
        channels: Full list of expanded channel dicts from
            :func:`expand_hierarchy`.
        tier_spec: Tier specification controlling which channels to include.

    Returns:
        Dict with ``_metadata`` and ``channels`` keys, loadable by
        :class:`~osprey.services.channel_finder.databases.flat.ChannelDatabase`.
    """
    filtered = filter_channels(channels, tier_spec)
    channel_entries = [
        {
            "channel": generate_alias(ch),
            "address": ch["pv"],
            "description": generate_description(ch),
        }
        for ch in filtered
    ]
    return {
        "_metadata": {
            "version": "1.0",
            "tier": tier_spec.name,
            "total_channels": len(channel_entries),
            "generated_by": "osprey-benchmark-generator",
        },
        "channels": channel_entries,
    }


def format_hierarchical(tree_data: dict, tier_spec: TierSpec) -> dict:
    """Prune the hierarchical tree to match a tier specification.

    Deep-copies *tree_data* and removes branches that fall outside the
    tier spec (rings, families, fields, subfields).  Metadata keys
    (``_description``, ``_expansion``, ``_comment``, ``hierarchy``) are
    preserved as-is so the result is loadable by
    :class:`~osprey.services.channel_finder.databases.hierarchical.HierarchicalChannelDatabase`.

    Args:
        tree_data: Full hierarchical JSON (must contain ``"tree"`` and
            ``"hierarchy"`` keys).
        tier_spec: Tier specification controlling which branches survive.

    Returns:
        Pruned copy of the hierarchical JSON.
    """
    import copy

    result = copy.deepcopy(tree_data)
    tree = result.get("tree", {})

    # --- 1. Remove rings not in tier_spec ----------------------------------
    rings_to_remove = [r for r in tree if not _is_metadata_key(r) and r not in tier_spec.rings]
    for r in rings_to_remove:
        del tree[r]

    # --- 2. Walk remaining rings and prune ---------------------------------
    for ring_name, ring_node in list(tree.items()):
        if _is_metadata_key(ring_name):
            continue

        for system_name, system_node in list(ring_node.items()):
            if _is_metadata_key(system_name):
                continue

            families_to_remove: list[str] = []
            for family_name, family_node in list(system_node.items()):
                if _is_metadata_key(family_name):
                    continue

                # --- family filter ---
                if tier_spec.families is not None and family_name not in tier_spec.families:
                    families_to_remove.append(family_name)
                    continue

                # --- prune within DEVICE node ---
                device_node = family_node.get("DEVICE")
                if device_node is None:
                    continue

                for field_name, field_node in list(device_node.items()):
                    if _is_metadata_key(field_name):
                        continue
                    if not isinstance(field_node, dict):
                        continue

                    # --- per-family field filter ---
                    if tier_spec.allowed_fields_by_family is not None:
                        allowed = tier_spec.allowed_fields_by_family.get(family_name)
                        if allowed is not None and field_name not in allowed:
                            del device_node[field_name]
                            continue

                    # --- subfield filter ---
                    if tier_spec.allowed_subfields is not None:
                        for sf_name in list(field_node.keys()):
                            if _is_metadata_key(sf_name):
                                continue
                            if sf_name not in tier_spec.allowed_subfields:
                                del field_node[sf_name]

                        # If all non-meta subfields removed, drop field
                        remaining = [k for k in field_node if not _is_metadata_key(k)]
                        if not remaining:
                            del device_node[field_name]

                # Check if DEVICE node has any non-meta field left
                remaining_fields = [
                    k
                    for k in device_node
                    if not _is_metadata_key(k) and isinstance(device_node[k], dict)
                ]
                if not remaining_fields:
                    families_to_remove.append(family_name)

            for f in families_to_remove:
                if f in system_node:
                    del system_node[f]

            # Remove system node if it has no family children left
            remaining_families = [k for k in system_node if not _is_metadata_key(k)]
            if not remaining_families:
                del ring_node[system_name]

    return result


def format_middle_layer(channels: list[dict], tier_spec: TierSpec) -> dict:
    """Format channels for the middle-layer database paradigm.

    Filters *channels* by *tier_spec* and groups them into the MML-style
    functional hierarchy expected by
    :class:`~osprey.services.channel_finder.databases.middle_layer.MiddleLayerDatabase`.

    The output tree has four semantic levels:
    ``ring -> family -> field -> subfield``, with ``ChannelNames``,
    ``DataType``, and ``HWUnits`` at the leaf nodes.  Each family also
    receives a ``_setup`` block containing ``CommonNames``, ``DeviceList``,
    and ``ElementList`` for sector/device filtering.

    Args:
        channels: Full list of expanded channel dicts from
            :func:`expand_hierarchy`.
        tier_spec: Tier specification controlling which channels to include.

    Returns:
        Nested dict loadable as a middle-layer JSON database.
    """
    filtered = filter_channels(channels, tier_spec)

    result: dict = {}

    # Track unique devices per (ring, family) in insertion order
    family_devices: dict[tuple[str, str], list[str]] = {}

    for ch in filtered:
        ring = ch["ring"]
        family = ch["family"]
        device = ch["device"]
        field = ch["field"]
        subfield = ch["subfield"]
        pv = ch["pv"]

        # --- ring level ---
        if ring not in result:
            result[ring] = {
                "_description": RING_NAMES.get(ring, ring),
            }
        ring_node = result[ring]

        # --- family level ---
        if family not in ring_node:
            ring_node[family] = {
                "_description": FAMILY_NAMES.get(family, family),
            }
        family_node = ring_node[family]

        # --- field level ---
        if field not in family_node:
            family_node[field] = {
                "_description": FIELD_NAMES.get(field, field),
            }
        field_node = family_node[field]

        # --- subfield level (leaf) ---
        if subfield not in field_node:
            data_type = _DATATYPE_BY_FIELD.get(field, _DEFAULT_DATATYPE)
            hw_units = _HWUNITS_BY_FIELD.get(field, _DEFAULT_HWUNITS)
            leaf: dict = {
                "_description": SUBFIELD_NAMES.get(subfield, subfield),
                "ChannelNames": [],
                "DataType": data_type,
                "HWUnits": hw_units,
            }
            field_node[subfield] = leaf

        field_node[subfield]["ChannelNames"].append(pv)

        # Track devices in insertion order
        key = (ring, family)
        if key not in family_devices:
            family_devices[key] = []
        if device not in family_devices[key]:
            family_devices[key].append(device)

    # Post-process: add _setup blocks for every family
    for (ring, family), devices in family_devices.items():
        num_devices = len(devices)
        devices_per_sector = max(1, -(-num_devices // 12))

        common_names = [f"{family} {dev}" for dev in devices]
        device_list = [
            [(i // devices_per_sector) + 1, (i % devices_per_sector) + 1]
            for i in range(num_devices)
        ]
        element_list = list(range(1, num_devices + 1))

        result[ring][family]["_setup"] = {
            "CommonNames": common_names,
            "DeviceList": device_list,
            "ElementList": element_list,
        }

    return result


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------

# Paradigms published per tier. Tier 1 ships the flat ``in_context`` view only;
# tier 3 ships all three cross-paradigm views. Query validation checks each
# tier's targeted PVs against exactly the paradigms declared here, and callers
# iterating "all tiers" iterate these keys (tier 2 is retired).
TIER_PARADIGMS: dict[int, tuple[str, ...]] = {
    1: ("in_context",),
    3: ("in_context", "hierarchical", "middle_layer"),
}

# Filename each paradigm view is stored under within a tier directory.
_PARADIGM_FILENAMES: dict[str, str] = {
    "in_context": "in_context.json",
    "hierarchical": "hierarchical.json",
    "middle_layer": "middle_layer.json",
}


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
