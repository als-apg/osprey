"""
Simple Channel Database Builder

Builds a templated channel database from simple CSV format.

**CSV Format:**
address,description,family_name,instances,sub_channel

- Rows with family_name: grouped into templates
- Rows without family_name: standalone channels

The ``address`` column of a family row **is** that family's address pattern
whenever it holds a placeholder: ``{instance}`` (or ``{instance:02d}``) for the
device number and ``{sub_channel}`` for the row's sub-channel. A family whose
rows carry a literal address instead gets a pattern synthesised from its name,
``<family>{instance:02d}{suffix}``. ``instances`` is a count (``8`` means 1-8)
or an explicit range (``4-11``).
"""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from osprey.services.channel_finder.core.exceptions import (
    AddressPatternError,
    TemplateBuildError,
)


def load_csv(csv_path: Path, delimiter: str = ",") -> list[dict]:
    channels = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
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
    """Find the common prefix across all descriptions for a device family."""
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


#: Placeholders the template expander fills in when it formats an address
#: pattern (see ``databases/template.py``). A pattern naming anything else
#: would fail at expansion time, one address at a time, so it is refused here.
ADDRESS_PATTERN_FIELDS = ("base", "instance", "suffix", "axis")


def parse_instance_range(raw: object) -> list[int]:
    """Return the ``[start, end]`` instance range a family's ``instances`` cell names.

    ``8`` is the common case and means 1 through 8. ``4-11`` is the other one:
    a machine whose device numbering does not start at 1, which a bare count
    cannot express.

    Args:
        raw: The cell's value.

    Returns:
        ``[start, end]``, inclusive.

    Raises:
        ValueError: When the cell is not a count or a range, or the range runs
            backwards.
    """
    text = str(raw if raw is not None else 1).strip()
    start_text, dash, end_text = text.partition("-")
    if dash:
        start, end = int(start_text), int(end_text)
    else:
        start, end = 1, int(text)
    if start > end:
        raise ValueError(f"instance range {text!r} ends before it starts")
    return [start, end]


def family_address_pattern(family_name: str, channels: list[dict]) -> str | None:
    """The address pattern the family's own rows carry, or ``None``.

    An address column holding a placeholder already says how the family's
    addresses are built --- which levels there are, where the instance sits,
    what separates them --- and synthesising a second pattern from the family
    name silently throws that away. ``{sub_channel}`` is the CSV's name for the
    expander's ``{suffix}``, so it is translated rather than refused.

    Args:
        family_name: The family the rows belong to, for the error message.
        channels: The family's rows.

    Returns:
        The pattern, or ``None`` when no row carries a placeholder --- in which
        case the caller synthesises one from the family name as before.

    Raises:
        AddressPatternError: When the family's rows do not agree on one address.
    """
    addresses = {
        (channel.get("address") or "").replace("{sub_channel}", "{suffix}") for channel in channels
    }
    if not any("{" in address for address in addresses):
        return None
    if len(addresses) > 1:
        raise AddressPatternError(
            f"the rows of family {family_name!r} give different addresses "
            f"({', '.join(sorted(addresses))}); one family expands from one pattern, "
            "so put the varying part in {sub_channel}"
        )
    return addresses.pop()


def check_address_pattern(family_name: str, pattern: str, channels: list[dict]) -> None:
    """Refuse an address pattern the expander could not fill in.

    Formats *pattern* for the first instance with each row's own sub-channel,
    which is the same call the expander makes; a placeholder the expander does
    not know about fails here, once, instead of at expansion time.

    Args:
        family_name: The family the pattern belongs to.
        pattern: The pattern to check.
        channels: The family's rows.

    Raises:
        AddressPatternError: When the pattern names a placeholder the expander
            has no value for, or is not a valid format string.
    """
    for channel in channels:
        try:
            pattern.format(
                base=family_name,
                instance=1,
                suffix=channel.get("sub_channel") or "",
                axis="",
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise AddressPatternError(
                f"family {family_name!r} has address pattern {pattern!r}, which "
                f"cannot be filled in ({exc}); a pattern may use only "
                + ", ".join(f"{{{field}}}" for field in ADDRESS_PATTERN_FIELDS)
            ) from exc


def create_template(family_name: str, channels: list[dict]) -> dict:
    """Create a template from a family group.

    Raises:
        AddressPatternError: When the family's rows disagree about their
            address, or name a placeholder the expander cannot fill in.
        ValueError: When the ``instances`` cell is neither a count nor a range.
    """
    first = channels[0]

    # Instance range: a bare count, or an explicit start-end.
    instances = parse_instance_range(first.get("instances", 1))

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

    # The CSV's own address column is the pattern when it holds a placeholder;
    # only a family that gives none gets one synthesised from its name.
    pattern = family_address_pattern(family_name, channels)
    if pattern is None:
        pattern = f"{family_name}" + "{instance:02d}{suffix}"
    check_address_pattern(family_name, pattern, channels)

    # Build template
    template = {
        "template": True,
        "base_name": family_name,
        "instances": instances,
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
    delimiter: str = ",",
) -> dict:
    """Build channel database from CSV.

    Args:
        csv_path: Path to input CSV file.
        output_path: Path for output JSON database.
        use_llm: Whether to use LLM for name generation.
        config_path: Optional path to config file for LLM settings.
        delimiter: CSV field delimiter (default: ',').

    Returns:
        The built database dict.

    Raises:
        AddressPatternError: When a family's address pattern cannot be
            expanded.
        TemplateBuildError: When a family cannot be turned into a template for
            any other reason. Either way the build stops with the family named
            rather than writing a database that is missing it.
    """
    print("=" * 80)
    print("Channel Database Builder")
    print("=" * 80)
    print(f"\nInput CSV: {csv_path}")

    # Load CSV
    channels = load_csv(csv_path, delimiter=delimiter)
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
        except AddressPatternError:
            # Already names the family, and says which pattern it could not
            # fill in; re-wrapping would only bury that.
            raise
        except Exception as exc:
            # Not a family to skip: a family that reaches the database as plain
            # rows, or not at all, answers navigation queries about it with
            # nothing while the run still reports success. The input is wrong;
            # say which family and stop.
            raise TemplateBuildError(
                f"family {family_name!r} cannot be built into a template ({exc}); "
                "fix the family's rows in the input"
            ) from exc
        else:
            templates.append(template)
            print(f"  \u2713 {family_name}: {len(family_channels)} channels \u2192 template")

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
