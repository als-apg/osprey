"""
Middle Layer Database Implementation

Implements MATLAB Middle Layer (MML) style organization for channel databases.
This approach organizes channels by functional hierarchy (System → Family → Field)
rather than by naming patterns, mirroring production accelerator control systems.

The database structure follows:
- Systems: Top-level accelerator systems (SR, BR, BTS, etc.)
- Families: Device families within systems (BPM, HCM, VCM, etc.)
- Fields: Functional fields within families (Monitor, Setpoint, etc.)
- Subfields: Optional nested fields for complex data structures
- ChannelNames: Actual EPICS PV addresses

Example structure:
{
  "SR": {
    "BPM": {
      "Monitor": {
        "ChannelNames": ["SR01C:BPM1:X", "SR01C:BPM1:Y", ...]
      },
      "Setpoint": {
        "X": {
          "ChannelNames": ["SR01C:BPM1:XSet", ...]
        },
        "Y": {
          "ChannelNames": ["SR01C:BPM1:YSet", ...]
        }
      },
      "setup": {
        "CommonNames": ["BPM 1", "BPM 2", ...],
        "DeviceList": [[1, 1], [1, 2], ...]  # [sector, device] pairs
      }
    }
  }
}
"""

import json

from ..core.base_database import BaseDatabase


class MiddleLayerDatabase(BaseDatabase):
    """
    Database for middle-layer (MML) style channel organization.

    Supports functional hierarchy with optional device/sector filtering.
    Designed for React agent-style exploration using query tools.
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize middle layer database.

        Args:
            db_path: Path to middle layer database JSON file
        """
        super().__init__(db_path)

    def load_database(self) -> None:
        """Load middle layer database from JSON file."""
        with open(self.db_path) as f:
            self.data = json.load(f)

        # Build flat channel map for validation and lookup
        self.channel_map = self._build_channel_map()

    def _build_channel_map(self) -> dict[str, dict]:
        """
        Flatten MML hierarchy into channel map for O(1) validation.

        Returns:
            Dict mapping channel names to metadata
        """
        channels = {}

        for system, families in self.data.items():
            if not isinstance(families, dict):
                continue

            for family, fields in families.items():
                if not isinstance(fields, dict):
                    continue

                # Process each field in the family
                self._extract_channels_from_field(channels, system, family, fields, path=[])

        return channels

    def _extract_channels_from_field(
        self,
        channels: dict[str, dict],
        system: str,
        family: str,
        field_data: dict,
        path: list[str],
    ) -> None:
        """
        Recursively extract channels from nested field structure.

        Handles the MATLAB Middle Layer export format which includes:
        - ChannelNames: List of PV addresses (may contain whitespace padding)
        - Metadata fields: DataType, Mode, Units, HW2PhysicsParams, etc.
        - Special fields: setup (device metadata), pyAT (accelerator physics)

        Args:
            channels: Accumulator dict for channel map
            system: Current system name
            family: Current family name
            field_data: Current field/subfield data
            path: Current path through field hierarchy (e.g., ["Monitor"] or ["Setpoint", "X"])
        """
        for key, value in field_data.items():
            # Skip setup and special metadata fields
            # Note: 'pyAT' is skipped because it has ATIndex instead of ChannelNames
            if key.lower() in ["setup", "pyat"]:
                continue

            if not isinstance(value, dict):
                continue

            # Check if this is a terminal field with ChannelNames
            if "ChannelNames" in value:
                channel_names = value["ChannelNames"]
                # Normalize string to list (MML exports may use string for single channels)
                if isinstance(channel_names, str):
                    channel_names = [channel_names]
                field_path = path + [key]

                # Extract metadata if present (preserve from MML exports)
                metadata = {}
                for meta_key in [
                    "DataType",
                    "Mode",
                    "Units",
                    "HWUnits",
                    "PhysicsUnits",
                    "MemberOf",
                    "Range",
                    "Tolerance",
                    "Description",
                ]:
                    if meta_key in value:
                        metadata[meta_key] = value[meta_key]

                # Store each channel with its metadata
                for channel_name in channel_names:
                    # Strip whitespace from channel names (MML exports have padding)
                    clean_name = channel_name.strip()
                    if clean_name:  # Only add non-empty names
                        channels[clean_name] = {
                            "channel": clean_name,
                            "address": clean_name,
                            "system": system,
                            "family": family,
                            "field": field_path[0] if field_path else "",
                            "subfield": field_path[1:] if len(field_path) > 1 else None,
                            "description": f"{system}:{family}:{':'.join(field_path)}",
                            **metadata,  # Include MML metadata if present
                        }
            else:
                # Recurse into nested structure (handles subfields)
                self._extract_channels_from_field(channels, system, family, value, path + [key])

    def get_channel(self, channel_name: str) -> dict | None:
        """
        Get channel information by name.

        Args:
            channel_name: Channel name to lookup

        Returns:
            Channel dict or None if not found
        """
        return self.channel_map.get(channel_name.strip())

    def get_all_channels(self) -> list[dict]:
        """
        Get all channels in the database.

        Returns:
            List of channel dictionaries
        """
        return list(self.channel_map.values())

    def validate_channel(self, channel_name: str) -> bool:
        """
        Check if a channel exists.

        Args:
            channel_name: Channel name to validate

        Returns:
            True if channel exists, False otherwise
        """
        return channel_name.strip() in self.channel_map

    def get_statistics(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dict with statistics (total_channels, systems, families, etc.)
        """
        # Count systems
        systems = [s for s in self.data.keys() if isinstance(self.data[s], dict)]

        # Count families across all systems
        all_families = set()
        for system in systems:
            if isinstance(self.data[system], dict):
                families = [
                    f for f in self.data[system].keys() if isinstance(self.data[system][f], dict)
                ]
                all_families.update(families)

        return {
            "total_channels": len(self.channel_map),
            "format": "middle_layer",
            "systems": len(systems),
            "families": len(all_families),
        }

    # === Tool support methods for React agent ===

    def list_systems(self) -> list[dict[str, str]]:
        """
        Get list of all system names with descriptions.

        Returns:
            List of dicts with 'name' and 'description' keys.
            Description is empty string if not provided in database.
        """
        systems = []
        for s in self.data.keys():
            if isinstance(self.data[s], dict):
                systems.append({"name": s, "description": self.data[s].get("_description", "")})
        return systems

    def list_families(self, system: str) -> list[dict[str, str]]:
        """
        Get list of families in a system with descriptions.

        Args:
            system: System name

        Returns:
            List of dicts with 'name' and 'description' keys.
            Description is empty string if not provided in database.

        Raises:
            ValueError: If system not found
        """
        if system not in self.data or not isinstance(self.data[system], dict):
            available = [s["name"] for s in self.list_systems()]
            raise ValueError(f"System '{system}' not found. Available systems: {available}")

        families = []
        for f in self.data[system].keys():
            if f.startswith("_"):  # Skip metadata keys
                continue
            if isinstance(self.data[system][f], dict):
                families.append(
                    {"name": f, "description": self.data[system][f].get("_description", "")}
                )
        return families

    def inspect_fields(
        self, system: str, family: str, field: str | None = None
    ) -> dict[str, dict[str, str]]:
        """
        Inspect field structure with types and descriptions.

        Args:
            system: System name
            family: Family name
            field: Optional field name to inspect (if None, shows top-level fields)

        Returns:
            Dict mapping field names to dicts with 'type' and 'description' keys.
            Description is empty string if not provided in database.

        Raises:
            ValueError: If system/family/field not found
        """
        # Validate system
        if system not in self.data:
            raise ValueError(f"System '{system}' not found")

        # Validate family
        if family not in self.data[system]:
            families = [f["name"] for f in self.list_families(system)]
            raise ValueError(
                f"Family '{family}' not found in system '{system}'. Available families: {families}"
            )

        family_data = self.data[system][family]

        # If specific field requested, inspect that field
        if field:
            if field not in family_data:
                available = [
                    k for k in family_data.keys() if not k.startswith("_") and k.lower() != "setup"
                ]
                raise ValueError(
                    f"Field '{field}' not found in '{system}:{family}'. "
                    f"Available fields: {available}"
                )

            field_data = family_data[field]
            result = {}

            if isinstance(field_data, dict):
                for key, value in field_data.items():
                    if key.startswith("_"):  # Skip metadata
                        continue
                    if isinstance(value, dict):
                        # Check if it's a subfield with ChannelNames or nested structure
                        if "ChannelNames" in value:
                            result[key] = {
                                "type": "ChannelNames",
                                "description": value.get("_description", ""),
                            }
                        else:
                            result[key] = {
                                "type": "dict (subfield)",
                                "description": value.get("_description", ""),
                            }
                    else:
                        result[key] = {"type": type(value).__name__, "description": ""}

            return result

        # Otherwise show top-level fields
        result = {}
        for key, value in family_data.items():
            if key.startswith("_"):  # Skip metadata
                continue
            if key.lower() == "setup":
                result[key] = {
                    "type": "metadata",
                    "description": "Device setup information (CommonNames, DeviceList)",
                }
            elif isinstance(value, dict):
                if "ChannelNames" in value:
                    result[key] = {
                        "type": "ChannelNames",
                        "description": value.get("_description", ""),
                    }
                else:
                    result[key] = {
                        "type": "dict (has subfields)",
                        "description": value.get("_description", ""),
                    }
            else:
                result[key] = {"type": type(value).__name__, "description": ""}

        return result

    def list_channel_names(
        self,
        system: str,
        family: str,
        field: str,
        subfield: str | None = None,
        sectors: list[int] | None = None,
        devices: list[int] | None = None,
    ) -> list[str]:
        """
        Get channel names for a specific field/subfield with optional filtering.

        Args:
            system: System name
            family: Family name
            field: Field name
            subfield: Optional subfield name
            sectors: Optional list of sector numbers to filter by
            devices: Optional list of device numbers to filter by

        Returns:
            List of channel names

        Raises:
            ValueError: If path not found or invalid
        """
        # Navigate to field
        if system not in self.data:
            raise ValueError(f"System '{system}' not found")

        if family not in self.data[system]:
            raise ValueError(f"Family '{family}' not found in system '{system}'")

        family_data = self.data[system][family]

        if field not in family_data:
            raise ValueError(f"Field '{field}' not found in '{system}:{family}'")

        field_data = family_data[field]

        # If subfield specified, navigate deeper
        if subfield:
            if not isinstance(field_data, dict) or subfield not in field_data:
                raise ValueError(f"Subfield '{subfield}' not found in '{system}:{family}:{field}'")
            field_data = field_data[subfield]

        # Get channel names
        if "ChannelNames" not in field_data:
            raise ValueError(
                f"No ChannelNames found at '{system}:{family}:{field}"
                + (f":{subfield}" if subfield else "")
                + "'"
            )

        channel_names = field_data["ChannelNames"]
        # Normalize string to list (MML exports may use string for single channels)
        if isinstance(channel_names, str):
            channel_names = [channel_names]

        # Apply filtering if requested
        if sectors or devices:
            channel_names = self._filter_by_device_sectors(
                system, family, channel_names, sectors, devices
            )

        # Strip whitespace and filter empty strings
        return [name.strip() for name in channel_names if name.strip()]

    def _filter_by_device_sectors(
        self,
        system: str,
        family: str,
        channel_names: list[str],
        sectors: list[int] | None,
        devices: list[int] | None,
    ) -> list[str]:
        """
        Filter channel names by device and sector numbers.

        Args:
            system: System name
            family: Family name
            channel_names: Full list of channel names
            sectors: Optional list of sectors to include
            devices: Optional list of devices to include

        Returns:
            Filtered list of channel names

        Raises:
            ValueError: If DeviceList not available or filtering fails
        """
        # Get DeviceList from setup
        family_data = self.data[system][family]
        setup = family_data.get("setup", {})
        device_list = setup.get("DeviceList")

        if not device_list:
            raise ValueError(
                f"Cannot filter by sectors/devices for '{system}:{family}' - "
                f"DeviceList not defined in database"
            )

        if len(channel_names) != len(device_list):
            raise ValueError(
                f"ChannelNames length ({len(channel_names)}) does not match "
                f"DeviceList length ({len(device_list)}) for '{system}:{family}'"
            )

        # Build filtered list
        filtered = []
        sectors_set = set(sectors) if sectors else None
        devices_set = set(devices) if devices else None

        for i, (channel_name, device_entry) in enumerate(
            zip(channel_names, device_list, strict=True)
        ):
            if not isinstance(device_entry, list) or len(device_entry) != 2:
                raise ValueError(f"Invalid DeviceList entry at index {i}: {device_entry}")

            sector, device = device_entry

            # Check filters
            sector_match = sectors_set is None or sector in sectors_set
            device_match = devices_set is None or device in devices_set

            if sector_match and device_match:
                filtered.append(channel_name)

        if not filtered:
            raise ValueError(
                f"No channels match filter criteria. "
                f"Requested sectors: {sectors}, devices: {devices}"
            )

        return filtered

    def get_common_names(self, system: str, family: str) -> list[str] | None:
        """
        Get common/friendly names for devices in a family.

        Args:
            system: System name
            family: Family name

        Returns:
            List of common names or None if not available
        """
        if system not in self.data or family not in self.data[system]:
            return None

        family_data = self.data[system][family]
        setup = family_data.get("setup", {})
        return setup.get("CommonNames")
