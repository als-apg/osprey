#!/usr/bin/env python3
"""
EPICS Gateway Configuration - Opt-In Facility-Agnostic Utility

This module provides EPICS gateway configuration based on facility selection.
Applications explicitly call setup functions to configure EPICS environment variables.

Supported Facilities:
    - APS (Argonne National Laboratory / Advanced Photon Source)
      * Gateway when outside aps4.anl.gov network
      * Local broadcast when on aps4.anl.gov network
    - ALS (Lawrence Berkeley National Laboratory / Advanced Light Source)
      * Always uses gateway: cagw-alsdmz.als.lbl.gov
    - Custom facilities can be easily added

Usage:
    # In your agent's registry.py or __init__.py
    from osprey.utils.epics_gateway_config import setup_aps_epics
    
    # Configure EPICS for APS facility
    setup_aps_epics()
    
    # Then use EPICSManager
    from osprey.utils.epics_utils import EPICSManager
    epics_mgr = EPICSManager()

Adding New Facilities:
    Add an entry to FACILITY_CONFIGS below:
    
    FACILITY_CONFIGS = {
        'your_facility': {
            'internal_domain': 'your.facility.domain',
            'gateway': {
                'address': 'gateway.your.facility.domain',
                'port': 5064,
            },
            'use_gateway_when': 'always',  # or 'external' or 'never'
        }
    }

Environment Variables Set:
    - EPICS_CA_ADDR_LIST: Gateway address (if gateway needed)
    - EPICS_CA_SERVER_PORT: Gateway port
    - EPICS_CA_AUTO_ADDR_LIST: YES (broadcast) or NO (gateway)
    - EPICS_CA_MAX_ARRAY_BYTES: Maximum array size for EPICS
"""

import os
import socket
from typing import Dict, Optional

from osprey.utils.logger import get_logger

# Get logger for this module
logger = get_logger("epics_gateway")

# Facility-specific EPICS configuration
FACILITY_CONFIGS = {
    'aps': {
        'internal_domain': 'aps4.anl.gov',
        'gateway': {
            'address': 'pvgatemain1.aps4.anl.gov',
            'port': 5064,
        },
        'use_gateway_when': 'external',  # Gateway only when outside aps4 network
    },
    'als': {
        'internal_domain': 'als.lbl.gov',
        'gateway': {
            'address': 'cagw-alsdmz.als.lbl.gov',
            'port': 5064,
        },
        'use_gateway_when': 'always',  # Always use gateway
    },
    # Add more facilities here as needed
}


def detect_facility() -> Optional[str]:
    """
    Detect facility based on hostname domain.
    
    Returns:
        Facility name if detected, None otherwise
    """
    try:
        hostname = socket.getfqdn()
        logger.debug(f"Detected hostname: {hostname}")
        
        for facility, config in FACILITY_CONFIGS.items():
            if hostname.endswith(config['internal_domain']):
                logger.info(f"Auto-detected facility: {facility}")
                return facility
        
        logger.debug("No facility auto-detected from hostname")
        return None
        
    except Exception as e:
        logger.warning(f"Error detecting facility: {e}")
        return None


def is_internal_network(facility: str) -> bool:
    """
    Check if running on facility's internal network.
    
    Args:
        facility: Facility name ('aps', 'als', etc.)
        
    Returns:
        True if on internal network, False otherwise
    """
    if facility not in FACILITY_CONFIGS:
        return False
    
    try:
        hostname = socket.getfqdn()
        internal_domain = FACILITY_CONFIGS[facility]['internal_domain']
        is_internal = hostname.endswith(internal_domain)
        
        logger.debug(f"Hostname: {hostname}")
        logger.debug(f"Internal domain: {internal_domain}")
        logger.debug(f"Is internal: {is_internal}")
        
        return is_internal
        
    except Exception as e:
        logger.warning(f"Error checking internal network: {e}")
        return False


def setup_facility_epics(facility: Optional[str] = None, apply_env: bool = True) -> Dict[str, str]:
    """
    Configure EPICS environment variables based on facility.
    
    Args:
        facility: Facility name ('aps', 'als', etc.). Auto-detected if None.
        apply_env: If True, set environment variables. If False, just return config.
    
    Returns:
        Dictionary of EPICS environment variables
        
    Example:
        >>> # Auto-detect facility
        >>> env_vars = setup_facility_epics()
        
        >>> # Specify facility
        >>> env_vars = setup_facility_epics(facility='aps')
        
        >>> # Get config without applying
        >>> env_vars = setup_facility_epics(facility='aps', apply_env=False)
    """
    # Auto-detect facility if not specified
    if facility is None:
        facility = detect_facility()
        if facility:
            logger.info(f"Auto-detected facility: {facility}")
        else:
            logger.info("No facility detected - skipping EPICS configuration")
            return {}
    
    if facility not in FACILITY_CONFIGS:
        logger.warning(f"Unknown facility: {facility}")
        logger.info(f"Supported facilities: {list(FACILITY_CONFIGS.keys())}")
        return {}
    
    config = FACILITY_CONFIGS[facility]
    env_vars = {}
    
    # Determine if we should use gateway
    use_gateway_when = config.get('use_gateway_when', 'always')
    on_internal = is_internal_network(facility)
    
    use_gateway = (
        (use_gateway_when == 'always') or
        (use_gateway_when == 'external' and not on_internal) or
        (use_gateway_when == 'internal' and on_internal)
    )
    
    if use_gateway:
        # Use gateway configuration
        gateway = config['gateway']
        env_vars['EPICS_CA_ADDR_LIST'] = gateway['address']
        env_vars['EPICS_CA_SERVER_PORT'] = str(gateway['port'])
        env_vars['EPICS_CA_AUTO_ADDR_LIST'] = 'NO'
        logger.info(f"Using EPICS gateway: {gateway['address']}:{gateway['port']}")
    else:
        # Use local broadcast (internal network)
        env_vars['EPICS_CA_AUTO_ADDR_LIST'] = 'YES'
        logger.info("Using local EPICS broadcast (internal network)")
    
    # Common settings
    env_vars['EPICS_CA_MAX_ARRAY_BYTES'] = '16777216'
    
    # Apply to environment if requested
    if apply_env:
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.debug(f"Set {key}={value}")
        
        # Remove EPICS_CA_ADDR_LIST if not in env_vars (broadcast mode)
        if 'EPICS_CA_ADDR_LIST' not in env_vars and 'EPICS_CA_ADDR_LIST' in os.environ:
            del os.environ['EPICS_CA_ADDR_LIST']
            logger.debug("Removed EPICS_CA_ADDR_LIST (broadcast mode)")
        
        logger.info(f"EPICS configuration applied for facility: {facility.upper()}")
    
    return env_vars


def setup_aps_epics(apply_env: bool = True) -> Dict[str, str]:
    """
    Configure EPICS for APS facility.
    
    Automatically detects if running on aps4.anl.gov network:
    - On aps4 network: Uses local broadcast
    - Outside aps4: Uses gateway pvgatemain1.aps4.anl.gov:5064
    
    Args:
        apply_env: If True, set environment variables. If False, just return config.
    
    Returns:
        Dictionary of EPICS environment variables
        
    Example:
        >>> # In your agent's registry.py
        >>> from osprey.utils.epics_gateway_config import setup_aps_epics
        >>> setup_aps_epics()
    """
    return setup_facility_epics(facility='aps', apply_env=apply_env)


def setup_als_epics(apply_env: bool = True) -> Dict[str, str]:
    """
    Configure EPICS for ALS facility.
    
    Always uses gateway: cagw-alsdmz.als.lbl.gov:5064
    
    Args:
        apply_env: If True, set environment variables. If False, just return config.
    
    Returns:
        Dictionary of EPICS environment variables
        
    Example:
        >>> # In your agent's registry.py
        >>> from osprey.utils.epics_gateway_config import setup_als_epics
        >>> setup_als_epics()
    """
    return setup_facility_epics(facility='als', apply_env=apply_env)


def main():
    """Main function for command-line usage."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Configure EPICS gateway for supported facilities"
    )
    parser.add_argument(
        '--facility',
        choices=list(FACILITY_CONFIGS.keys()) + ['auto'],
        default='auto',
        help="Facility to configure (default: auto-detect)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show configuration without applying environment variables"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        facility = None if args.facility == 'auto' else args.facility
        env_vars = setup_facility_epics(
            facility=facility,
            apply_env=not args.dry_run
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EPICS Gateway Configuration Summary")
        print("="*60)
        
        if facility:
            print(f"Facility: {facility}")
        else:
            detected = detect_facility()
            print(f"Facility: {detected if detected else 'Not detected'}")
        
        print(f"Gateway Mode: {'Enabled' if 'EPICS_CA_ADDR_LIST' in env_vars else 'Disabled (Broadcast)'}")
        print()
        
        for key, value in env_vars.items():
            print(f"{key}: {value}")
        
        if not env_vars:
            print("No EPICS configuration applied")
        
        print("="*60)
        
        if args.dry_run:
            print("\nDRY RUN: Environment variables were not applied")
        
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()