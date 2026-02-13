"""
Simple Channel Database Builder

Builds a templated channel database from simple CSV format.

**CSV Format:**
address,description,family_name,instances,sub_channel

- Rows with family_name: grouped into templates
- Rows without family_name: standalone channels
"""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_csv(csv_path: Path) -> list[dict]:
    """Load CSV, skipping comment lines."""
    channels = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            address = row.get("address", "").strip()
            # Skip comments and empty rows
            if not address or address.startswith("#"):
                continue

            # Clean values
            cleaned = {k: v.strip() if v and v.strip() else None for k, v in row.items()}
            channels.append(cleaned)

    return channels


def group_by_family(channels: list[dict]) -> tuple:
    """Group channels by family_name."""
    families = defaultdict(list)
    standalone = []

    for ch in channels:
        family = ch.get("family_name")
        if family:
            families[family].append(ch)
        else:
            standalone.append(ch)

    return dict(families), standalone


def find_common_description(descriptions: list[str]) -> str:
    """Find the common part of all descriptions.

    Uses the longest common substring approach.
    """
    if not descriptions:
        return ""

    if len(descriptions) == 1:
        return descriptions[0]

    # Start with first description
    common = descriptions[0]

    # Find longest common substring with all others
    for desc in descriptions[1:]:
        # Find common parts (simple approach: find longest common prefix of words)
        common_words = []
        common_split = common.lower().split()
        desc_split = desc.lower().split()

        # Find common starting words
        for i, word in enumerate(common_split):
            if i < len(desc_split) and desc_split[i] == word:
                common_words.append(word)
            else:
                break

        if common_words:
            # Take the common part from the original (to preserve capitalization)
            common = " ".join(common.split()[: len(common_words)])
        else:
            # No common prefix, return generic
            return ""

    # Clean up the result
    common = common.strip()

    # Remove trailing punctuation (dash, comma, etc.)
    common = common.rstrip(" -,;:")

    # If it's too short or doesn't make sense, return empty
    if len(common.split()) < 3:
        return ""

    return common


def create_template(family_name: str, channels: list[dict]) -> dict:
    """Create a template from a family group."""
    first = channels[0]

    # Get instance count
    instances = int(first.get("instances", 1))

    # Get sub-channels from all rows in this family
    sub_channels = []
    channel_descriptions = {}
    all_descriptions = []

    for ch in channels:
        sub_ch = ch.get("sub_channel")
        desc = ch.get("description", "")
        if sub_ch and sub_ch not in sub_channels:
            sub_channels.append(sub_ch)
            channel_descriptions[sub_ch] = desc
            all_descriptions.append(desc)

    # Find common description from all channel descriptions
    base_description = find_common_description(all_descriptions)
    if not base_description:
        # Fallback to generic description
        base_description = f"{family_name} device family"

    # Strip the common prefix from sub-channel descriptions to avoid redundancy
    if base_description and base_description != f"{family_name} device family":
        cleaned_channel_descriptions = {}
        for sub_ch, desc in channel_descriptions.items():
            if desc.startswith(base_description):
                unique_part = desc[len(base_description) :].lstrip(" -:,;")
                if unique_part:
                    unique_part = unique_part[0].lower() + unique_part[1:]
                cleaned_channel_descriptions[sub_ch] = unique_part
            else:
                cleaned_channel_descriptions[sub_ch] = desc
        channel_descriptions = cleaned_channel_descriptions

    # Simple address pattern
    pattern = f"{family_name}" + "{instance:02d}{suffix}"

    # Build template
    template = {
        "template": True,
        "base_name": family_name,
        "instances": [1, instances],
        "sub_channels": sub_channels,
        "description": base_description,
        "address_pattern": pattern,
        "channel_descriptions": channel_descriptions,
    }

    return template


def build_database(
    csv_path: Path,
    output_path: Path,
    use_llm: bool = False,
    config_path: Path | None = None,
) -> dict:
    """Build channel database from CSV.

    Args:
        csv_path: Path to input CSV file.
        output_path: Path for output JSON database.
        use_llm: Whether to use LLM for name generation.
        config_path: Optional path to config file for LLM settings.

    Returns:
        The built database dict.
    """
    print("=" * 80)
    print("Channel Database Builder")
    print("=" * 80)
    print(f"\nInput CSV: {csv_path}")

    # Load CSV
    channels = load_csv(csv_path)
    print(f"Loaded {len(channels)} channels (excluding comments)")

    # Group by family
    families, standalone = group_by_family(channels)
    print("\nFound:")
    print(f"  - {len(families)} device families")
    print(f"  - {len(standalone)} standalone channels")

    # Create templates
    templates = []
    for family_name, family_channels in families.items():
        try:
            template = create_template(family_name, family_channels)
            templates.append(template)
            print(f"  \u2713 {family_name}: {len(family_channels)} channels \u2192 template")
        except Exception as e:
            print(f"  \u2717 {family_name}: ERROR - {e}")
            standalone.extend(family_channels)

    # Build database with metadata
    db = {
        "_metadata": {
            "generated_from": (
                str(csv_path.relative_to(Path.cwd()))
                if csv_path.is_relative_to(Path.cwd())
                else str(csv_path)
            ),
            "generation_date": datetime.now().strftime("%Y-%m-%d"),
            "generator": "osprey channel-finder build-database",
            "llm_naming": {"enabled": use_llm, "model": None, "purpose": None},
            "description": "Template-based channel database with automatic common description extraction",
        },
        "channels": [],
    }

    # Add standalone channels with optional LLM naming
    if standalone:
        if use_llm:
            print(
                f"\n\U0001f916 Generating descriptive names for "
                f"{len(standalone)} standalone channels using LLM..."
            )
            try:
                from osprey.services.channel_finder.tools.llm_channel_namer import (
                    create_namer_from_config,
                )

                namer = create_namer_from_config(config_path)
                print(f"  Using: {namer.model_id}")
                print(f"  Batch size: {namer.batch_size}")

                # Update metadata with LLM info
                db["_metadata"]["llm_naming"]["model"] = namer.model_id
                db["_metadata"]["llm_naming"]["purpose"] = (
                    f"Generated descriptive PascalCase names for {len(standalone)} standalone channels"
                )

                # Prepare channels for naming
                channels_to_name = [
                    {
                        "short_name": ch.get("address", ""),
                        "description": ch.get("description", ""),
                    }
                    for ch in standalone
                ]

                # Generate names
                generated_names = namer.generate_names(channels_to_name)
                print(f"  \u2713 Generated {len(generated_names)} names")

            except Exception as e:
                print(f"  \u26a0\ufe0f  LLM naming failed: {e}")
                print("  Using addresses as channel names")
                generated_names = [ch.get("address", "") for ch in standalone]
        else:
            print("\n\U0001f4dd Using addresses as channel names for standalone channels")
            generated_names = [ch.get("address", "") for ch in standalone]

        # Add standalone channels
        for i, ch in enumerate(standalone):
            db["channels"].append(
                {
                    "template": False,
                    "channel": (
                        generated_names[i] if i < len(generated_names) else ch.get("address", "")
                    ),
                    "address": ch.get("address", ""),
                    "description": ch.get("description", ""),
                }
            )

    # Add templates
    db["channels"].extend(templates)

    # Update metadata with final stats
    db["_metadata"]["stats"] = {
        "template_entries": len(templates),
        "standalone_entries": len(standalone),
        "total_entries": len(db["channels"]),
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(db, f, indent=2)

    print("\n\u2705 Database created successfully!")
    print(f"  \U0001f4cb Templates: {len(templates)}")
    print(f"  \U0001f4c4 Standalone: {len(standalone)}")
    print(f"  \U0001f4ca Total entries: {len(db['channels'])}")
    print(f"  \U0001f4be Output: {output_path}")
    print("=" * 80)

    return db
