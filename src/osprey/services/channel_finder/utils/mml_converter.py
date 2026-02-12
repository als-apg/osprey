"""
Utility for converting MATLAB Middle Layer (MML) exports to Channel Finder format.

This module provides tools to convert MATLAB Accelerator Object (AO) data exports
into the JSON format required by the middle_layer pipeline. It preserves all
metadata from the MML exports while organizing the data for the channel finder.

Usage:
    # From Python MML export files
    from your_facility.data.MML_ao_SR import MML_ao_SR
    from your_facility.data.MML_ao_BR import MML_ao_BR

    converter = MMLConverter()
    converter.add_system("SR", MML_ao_SR)
    converter.add_system("BR", MML_ao_BR)
    converter.save_json("data/channel_databases/my_facility.json")

    # Or from a single dictionary
    ao_data = {
        "SR": MML_ao_SR,
        "BR": MML_ao_BR,
    }
    MMLConverter.convert_and_save(ao_data, "my_facility.json")
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MMLConverter:
    """
    Convert MATLAB Middle Layer (MML) accelerator object data to Channel Finder format.

    The MML format uses a functional hierarchy:
    - System level (e.g., SR, BR, BTS) - added externally when combining
    - Family level (e.g., BPM, HCM, VCM) - device families
    - Field level (e.g., Monitor, Setpoint, X, Y) - data fields
    - Each field contains ChannelNames plus extensive metadata

    The converter preserves all metadata fields from the MML export, including:
    - DataType, Mode, Units (HW/Physics)
    - MemberOf, Range, Tolerance
    - HW2PhysicsParams, Physics2HWParams
    - And all other custom fields
    """

    def __init__(self) -> None:
        """Initialize converter with empty data dict."""
        self.data = {}

    def add_system(self, system_name: str, mml_data: dict[str, Any]) -> None:
        """
        Add a system's MML data to the converter.

        Args:
            system_name: System identifier (e.g., "SR", "BR", "LN")
            mml_data: MML dictionary for this system (Family → Field → ChannelNames + metadata)

        Example:
            from my_facility.data.MML_ao_SR import MML_ao_SR
            converter.add_system("SR", MML_ao_SR)
        """
        # Validate structure
        if not isinstance(mml_data, dict):
            raise ValueError(f"MML data for system '{system_name}' must be a dictionary")

        # Store directly - MML data already has the correct structure
        # System → Family → Field → {ChannelNames, metadata...}
        self.data[system_name] = mml_data

        # Count channels for logging
        num_channels = self._count_channels(mml_data)
        num_families = len([k for k in mml_data.keys() if isinstance(mml_data[k], dict)])

        logger.info(
            f"Added system '{system_name}': {num_families} families, {num_channels} channels"
        )

    def _count_channels(self, mml_data: dict) -> int:
        """Count total channels in MML data structure."""
        count = 0
        for _family, family_data in mml_data.items():
            if not isinstance(family_data, dict):
                continue
            for _field, field_data in family_data.items():
                if isinstance(field_data, dict) and "ChannelNames" in field_data:
                    channel_names = field_data["ChannelNames"]
                    # Count non-empty channel names (filter out whitespace padding)
                    count += len([ch for ch in channel_names if ch.strip()])
                elif isinstance(field_data, dict):
                    # Recurse for nested fields
                    count += self._count_nested_channels(field_data)
        return count

    def _count_nested_channels(self, data: dict) -> int:
        """Recursively count channels in nested structure."""
        count = 0
        for key, value in data.items():
            if key.lower() in ["setup", "pyat"]:
                continue
            if isinstance(value, dict):
                if "ChannelNames" in value:
                    channel_names = value["ChannelNames"]
                    count += len([ch for ch in channel_names if ch.strip()])
                else:
                    count += self._count_nested_channels(value)
        return count

    def save_json(self, output_path: str, indent: int = 2) -> None:
        """
        Save the converted data to JSON file.

        Args:
            output_path: Path to output JSON file
            indent: JSON indentation (default 2 spaces)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(self.data, f, indent=indent)

        total_channels = sum(self._count_channels(sys_data) for sys_data in self.data.values())
        logger.info(
            f"Saved {len(self.data)} systems ({total_channels} total channels) to {output_path}"
        )

    @classmethod
    def convert_and_save(cls, ao_data: dict[str, dict], output_path: str, indent: int = 2) -> None:
        """
        Convert and save MML data in one step.

        Args:
            ao_data: Dictionary mapping system names to MML data
                    e.g., {"SR": MML_ao_SR, "BR": MML_ao_BR}
            output_path: Path to output JSON file
            indent: JSON indentation

        Example:
            from my_facility.data.MML_ao_SR import MML_ao_SR
            from my_facility.data.MML_ao_BR import MML_ao_BR

            ao_data = {
                "SR": MML_ao_SR,
                "BR": MML_ao_BR,
            }

            MMLConverter.convert_and_save(
                ao_data,
                "data/channel_databases/my_facility.json"
            )
        """
        converter = cls()

        for system_name, mml_data in ao_data.items():
            converter.add_system(system_name, mml_data)

        converter.save_json(output_path, indent)

    @classmethod
    def convert_from_python_file(
        cls, module_path: str, variable_name: str, output_path: str
    ) -> None:
        """
        Convert directly from a Python file containing MML data.

        Args:
            module_path: Path to Python file (e.g., "data/MML_ao_250413_SR.py")
            variable_name: Variable name in file (e.g., "MML_ao_SR")
            output_path: Path to output JSON file

        Example:
            MMLConverter.convert_from_python_file(
                "data/MML_ao_250413_SR.py",
                "MML_ao_SR",
                "sr_channels.json"
            )
        """
        import importlib.util

        # Load module from file
        spec = importlib.util.spec_from_file_location("mml_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get MML data
        mml_data = getattr(module, variable_name)

        # Extract system name from variable name (e.g., "MML_ao_SR" → "SR")
        system_name = variable_name.split("_")[-1]

        # Convert and save
        cls.convert_and_save({system_name: mml_data}, output_path)


# === Command-line interface ===


def main() -> None:
    """
    Command-line interface for MML conversion.

    Usage:
        python -m channel_finder.utils.mml_converter \
            --input data/MML_ao_250413_SR.py:MML_ao_SR \
            --input data/MML_ao_250413_BR.py:MML_ao_BR \
            --output my_facility.json
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert MATLAB Middle Layer exports to Channel Finder JSON format"
    )
    parser.add_argument(
        "--input",
        "-i",
        action="append",
        required=True,
        help="Input Python file and variable (format: path/to/file.py:VariableName)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load all input files
    converter = MMLConverter()

    for input_spec in args.input:
        if ":" not in input_spec:
            logger.error(f"Invalid input format: {input_spec}")
            logger.error("Expected format: path/to/file.py:VariableName")
            return 1

        file_path, var_name = input_spec.split(":", 1)

        # Extract system name from variable
        system_name = var_name.split("_")[-1]

        # Load Python file
        import importlib.util

        spec = importlib.util.spec_from_file_location(f"mml_{system_name}", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get MML data
        mml_data = getattr(module, var_name)

        # Add to converter
        converter.add_system(system_name, mml_data)

    # Save combined data
    converter.save_json(args.output, indent=args.indent)

    logger.info("✓ Conversion complete!")
    return 0


if __name__ == "__main__":
    exit(main())
