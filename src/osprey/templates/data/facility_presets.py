"""EPICS facility configuration presets.

Provides gateway configurations for common facilities to simplify
project setup during osprey init.

Based on contributions from PR #37 by Marty Smith.
"""

# Facility gateway configurations
FACILITY_PRESETS = {
    "aps": {
        "name": "APS (Argonne National Laboratory)",
        "description": "Advanced Photon Source",
        "gateways": {
            "read_only": {
                "address": "pvgatemain1.aps4.anl.gov",
                "port": 5064,
                "use_name_server": False,
            },
            "write_access": {
                "address": "pvgatemain1.aps4.anl.gov",
                "port": 5064,
                "use_name_server": False,
            },
        },
    },
    "als": {
        "name": "ALS (Lawrence Berkeley National Laboratory)",
        "description": "Advanced Light Source",
        "gateways": {
            "read_only": {
                "address": "cagw-alsdmz.als.lbl.gov",
                "port": 5064,
                "use_name_server": False,
            },
            "write_access": {
                "address": "cagw-alsdmz.als.lbl.gov",
                "port": 5084,
                "use_name_server": False,
            },
        },
    },
    "simulation": {
        "name": "Local Simulation",
        "description": "Simulated soft IOC (caproto) - run 'osprey generate soft-ioc' first",
        "gateways": {
            "read_only": {
                "address": "localhost",
                "port": 5064,
                "use_name_server": False,
            },
            "write_access": {
                "address": "localhost",
                "port": 5064,
                "use_name_server": False,
            },
        },
    },
}


def get_facility_config(facility: str) -> dict | None:
    """Get gateway configuration for a facility.

    Args:
        facility: Facility identifier ('aps', 'als', etc.)

    Returns:
        Configuration dict suitable for config.yml template variables,
        or None if facility not found.

    Example:
        >>> config = get_facility_config('aps')
        >>> config['gateways']['read_only']['address']
        'pvgatemain1.aps4.anl.gov'
    """
    return FACILITY_PRESETS.get(facility)


def list_facilities() -> list[str]:
    """List available facility presets.

    Returns:
        List of facility identifiers.
    """
    return list(FACILITY_PRESETS.keys())


def get_facility_choices() -> list[tuple[str, str]]:
    """Get facility choices for CLI selection.

    Returns:
        List of (display_name, facility_id) tuples.
    """
    choices = [
        (f"{config['name']} - {config['description']}", facility)
        for facility, config in FACILITY_PRESETS.items()
    ]
    return choices
