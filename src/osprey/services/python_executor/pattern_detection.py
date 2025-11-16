"""
Pattern detection for control system operations in generated code.

This module provides config-based pattern detection for identifying control
system operations (reads/writes) in generated Python code. Patterns are defined
in the configuration file and can be easily extended for new control systems.

Related to Issue #18 - Control System Abstraction (Layer 1)
"""

import re
from typing import Optional

from osprey.utils.logger import get_logger

logger = get_logger("pattern_detection")


def detect_control_system_operations(
    code: str,
    patterns: dict[str, dict[str, list[str]]] | None = None,
    control_system_type: str | None = None
) -> dict[str, any]:
    """
    Detect control system operations using regex patterns from config.

    This function analyzes Python code to detect read and write operations
    to control systems (EPICS, LabVIEW, Tango, Mock, etc.) based on configurable
    regex patterns.

    Args:
        code: Python code string to analyze
        patterns: Pattern dictionary with structure:
                 {control_system_type: {operation_type: [regex_patterns]}}
                 If None, will attempt to load from config
        control_system_type: Control system type to check patterns for.
                            If None, will attempt to load from config

    Returns:
        Dict with operation detection results:
        {
            'has_writes': bool,
            'has_reads': bool,
            'control_system_type': str,
            'detected_patterns': {
                'writes': List[str],  # Matched patterns
                'reads': List[str]    # Matched patterns
            }
        }

    Examples:
        >>> code = "epics.caput('BEAM:CURRENT', 500.0)"
        >>> result = detect_control_system_operations(code)
        >>> result['has_writes']
        True

        >>> code = "value = epics.caget('BEAM:CURRENT')"
        >>> result = detect_control_system_operations(code)
        >>> result['has_reads']
        True
    """
    # Try to load from config if not provided
    if patterns is None or control_system_type is None:
        try:
            from osprey.utils.config import get_config_value

            if control_system_type is None:
                control_system_type = get_config_value('control_system.type', 'epics')

            if patterns is None:
                patterns = get_config_value('control_system.patterns', {})
        except Exception as e:
            logger.warning(f"Could not load patterns from config: {e}")
            patterns = patterns or {}
            control_system_type = control_system_type or 'epics'

    # Get patterns for the specified control system type
    cs_patterns = patterns.get(control_system_type, {})
    write_patterns = cs_patterns.get('write', [])
    read_patterns = cs_patterns.get('read', [])

    # Track which patterns matched
    detected_writes = []
    detected_reads = []

    # Check for writes
    for pattern in write_patterns:
        try:
            if re.search(pattern, code):
                detected_writes.append(pattern)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")

    # Check for reads
    for pattern in read_patterns:
        try:
            if re.search(pattern, code):
                detected_reads.append(pattern)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")

    has_writes = len(detected_writes) > 0
    has_reads = len(detected_reads) > 0

    result = {
        'has_writes': has_writes,
        'has_reads': has_reads,
        'control_system_type': control_system_type,
        'detected_patterns': {
            'writes': detected_writes,
            'reads': detected_reads
        }
    }

    if has_writes or has_reads:
        logger.debug(
            f"Detected control system operations: "
            f"writes={has_writes}, reads={has_reads}, "
            f"type={control_system_type}"
        )

    return result


def get_default_patterns() -> dict[str, dict[str, list[str]]]:
    """
    Get default pattern configurations for common control systems.

    Returns:
        Dictionary of default patterns for mock, EPICS, and future control systems
    """
    return {
        'mock': {
            'write': [
                r'\.caput\(',
                r'\.write_pv\(',
            ],
            'read': [
                r'\.caget\(',
                r'\.read_pv\(',
            ]
        },
        'epics': {
            'write': [
                r'epics\.caput\(',
                r'epics\.PV\([^)]*\)\.put\(',
                r'\.put\(',  # For pv.put()
            ],
            'read': [
                r'epics\.caget\(',
                r'epics\.PV\([^)]*\)\.get\(',
                r'\.get\(',  # For pv.get()
            ]
        },
        # Future patterns can be added here
        # 'tango': {
        #     'write': [
        #         r'DeviceProxy\([^)]*\)\.write_attribute\(',
        #         r'\.write_attribute\(',
        #     ],
        #     'read': [
        #         r'DeviceProxy\([^)]*\)\.read_attribute\(',
        #         r'\.read_attribute\(',
        #     ]
        # }
    }

