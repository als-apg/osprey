"""
Pattern detection for control system operations in generated code.

This module provides framework-standard pattern detection for identifying control
system operations (reads/writes) in generated Python code. The framework provides
unified patterns that work across all control systems. Users can optionally override
these patterns in their config.yml for custom workflows.

Key Design Principle:
    Patterns are control-system-agnostic because all modern code uses the
    osprey.runtime unified API (read_channel/write_channel). The control_system.type
    config only affects which connector is used at runtime, not which patterns
    are used for approval detection.

Related to Issue #18 - Control System Abstraction (Layer 1)
"""

import re

from osprey.utils.logger import get_logger

logger = get_logger("pattern_detection")


def get_framework_standard_patterns() -> dict[str, list[str]]:
    """
    Get framework-standard patterns for detecting control system operations.

    SECURITY PURPOSE: These patterns detect ALL attempts to interact with control systems,
    whether through the proper unified API or through direct library calls that would
    bypass the connector's safety features (limits checking, verification, approval).

    Pattern Categories:
    1. **Approved API**: osprey.runtime unified API with full safety features
    2. **Circumvention Detection**: Direct control system library calls that bypass safety
    3. **Future-Proofing**: Patterns for other control systems (Tango, LabVIEW, etc.)

    The framework provides these as sensible defaults. Users can override or extend
    them in config.yml under control_system.patterns if needed for custom workflows.

    Returns:
        Dictionary with 'write' and 'read' pattern lists

    Security Note:
        An LLM could try to circumvent the connector by directly importing control
        system libraries (epics, PyTango, etc.). These patterns ensure we catch
        such attempts in the approval workflow, regardless of which library is used.
    """
    return {
        'write': [
            # ============================================================
            # APPROVED: osprey.runtime unified API (has all safety features)
            # ============================================================
            r'\bwrite_channel\s*\(',      # write_channel('PV', value)
            r'\bwrite_channels\s*\(',     # write_channels({'PV1': val1, ...})

            # ============================================================
            # CIRCUMVENTION DETECTION: EPICS (PyEPICS library)
            # ============================================================
            r'\bcaput\s*\(',              # caput('PV', value) - standalone
            r'epics\.caput\(',            # epics.caput('PV', value)
            r'\.put\s*\(',                # pv.put(value) - PV object method
            r'\.set_value\s*\(',          # pv.set_value(value)
            r'PV\([^)]*\)\.put',          # PV('PV').put(value)
            r'epics\.PV\([^)]*\)\.put',   # epics.PV('PV').put(value)

            # ============================================================
            # CIRCUMVENTION DETECTION: Tango (PyTango library)
            # ============================================================
            r'DeviceProxy\([^)]*\)\.write_attribute\(',  # DeviceProxy(...).write_attribute(...)
            r'\.write_attribute\s*\(',                   # device.write_attribute(...)
            r'\.write_attribute_asynch\s*\(',            # device.write_attribute_asynch(...)
            r'tango\.DeviceProxy\([^)]*\)\.write',       # tango.DeviceProxy(...).write_attribute(...)

            # ============================================================
            # CIRCUMVENTION DETECTION: LabVIEW (potential patterns)
            # ============================================================
            # Note: LabVIEW Python integration varies by implementation
            # Add patterns as needed based on your LabVIEW integration
            r'labview\.set_control\(',                   # Example LabVIEW pattern
            r'\.SetControlValue\(',                      # Example LabVIEW ActiveX pattern

            # ============================================================
            # ADVANCED: Direct connector usage (internal/advanced use)
            # ============================================================
            r'connector\.write_channel\(',  # Direct connector access
        ],
        'read': [
            # ============================================================
            # APPROVED: osprey.runtime unified API (has all safety features)
            # ============================================================
            r'\bread_channel\s*\(',       # read_channel('PV')

            # ============================================================
            # CIRCUMVENTION DETECTION: EPICS (PyEPICS library)
            # ============================================================
            r'\bcaget\s*\(',              # caget('PV') - standalone
            r'epics\.caget\(',            # epics.caget('PV')
            r'\.get\s*\(',                # pv.get() - PV object method
            r'\.get_value\s*\(',          # pv.get_value()
            r'PV\([^)]*\)\.get',          # PV('PV').get()
            r'epics\.PV\([^)]*\)\.get',   # epics.PV('PV').get()

            # ============================================================
            # CIRCUMVENTION DETECTION: Tango (PyTango library)
            # ============================================================
            r'DeviceProxy\([^)]*\)\.read_attribute\(',   # DeviceProxy(...).read_attribute(...)
            r'\.read_attribute\s*\(',                    # device.read_attribute(...)
            r'\.read_attribute_asynch\s*\(',             # device.read_attribute_asynch(...)
            r'tango\.DeviceProxy\([^)]*\)\.read',        # tango.DeviceProxy(...).read_attribute(...)

            # ============================================================
            # CIRCUMVENTION DETECTION: LabVIEW (potential patterns)
            # ============================================================
            r'labview\.get_control\(',                   # Example LabVIEW pattern
            r'\.GetControlValue\(',                      # Example LabVIEW ActiveX pattern

            # ============================================================
            # ADVANCED: Direct connector usage (internal/advanced use)
            # ============================================================
            r'connector\.read_channel\(',   # Direct connector access
        ]
    }


def detect_control_system_operations(
    code: str,
    patterns: dict[str, list[str]] | None = None,
    control_system_type: str | None = None
) -> dict[str, any]:
    """
    Detect control system operations using framework-standard or custom patterns.

    This function analyzes Python code to detect read and write operations.
    By default, it uses framework-standard control-system-agnostic patterns.
    Users can optionally provide custom patterns via config or parameters.

    Args:
        code: Python code string to analyze
        patterns: Optional pattern dictionary with structure:
                 {'write': [patterns...], 'read': [patterns...]}
                 If None, will try to load from config, then fall back to framework standards
        control_system_type: Control system type (for logging/metadata only).
                            If None, will attempt to load from config
                            Note: This does NOT affect which patterns are used!

    Returns:
        Dict with operation detection results:
        {
            'has_writes': bool,
            'has_reads': bool,
            'control_system_type': str,  # For logging/metadata
            'detected_patterns': {
                'writes': List[str],  # Matched pattern regexes
                'reads': List[str]    # Matched pattern regexes
            }
        }

    Examples:
        Approved unified API (recommended):
        >>> code = "write_channel('BEAM:CURRENT', 500.0)"
        >>> result = detect_control_system_operations(code)
        >>> result['has_writes']
        True

        EPICS circumvention detected (security layer catches direct library calls):
        >>> code = "epics.caput('BEAM:CURRENT', 500.0)"
        >>> result = detect_control_system_operations(code)
        >>> result['has_writes']
        True

        >>> code = "value = read_channel('BEAM:CURRENT')"
        >>> result = detect_control_system_operations(code)
        >>> result['has_reads']
        True
    """
    # Load control_system_type for logging/metadata (doesn't affect patterns!)
    if control_system_type is None:
        try:
            from osprey.utils.config import get_config_value
            control_system_type = get_config_value('control_system.type', 'epics')
        except Exception:
            control_system_type = 'epics'  # Default for logging

    # Get patterns: config override → framework standard
    if patterns is None:
        try:
            from osprey.utils.config import get_config_value

            # Try to load custom patterns from config (optional override)
            custom_patterns = get_config_value('control_system.patterns', None)

            if custom_patterns is not None:
                # User provided custom patterns in config
                # Support both old nested format and new flat format for migration
                if isinstance(custom_patterns, dict):
                    # Check if it's the new flat format: {'write': [...], 'read': [...]}
                    if 'write' in custom_patterns or 'read' in custom_patterns:
                        patterns = custom_patterns
                        logger.info("Using custom patterns from config.yml")
                    # Check if it's the old nested format: {'epics': {'write': [...], ...}, ...}
                    elif control_system_type in custom_patterns:
                        patterns = custom_patterns[control_system_type]
                        logger.warning(
                            f"DEPRECATED: config.yml uses old nested pattern format "
                            f"(control_system.patterns.{control_system_type}). "
                            f"Please migrate to flat format (control_system.patterns.write/read). "
                            f"See documentation for details."
                        )
                    else:
                        # Unknown format, use framework standard
                        logger.warning(
                            f"Unrecognized pattern format in config.yml. "
                            f"Using framework standard patterns."
                        )
                        patterns = get_framework_standard_patterns()
                else:
                    patterns = get_framework_standard_patterns()
            else:
                # No custom patterns in config, use framework standard
                patterns = get_framework_standard_patterns()

        except Exception as e:
            logger.debug(f"Could not load patterns from config: {e}. Using framework standard patterns.")
            patterns = get_framework_standard_patterns()

    # Extract write and read patterns
    write_patterns = patterns.get('write', [])
    read_patterns = patterns.get('read', [])

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
    DEPRECATED: Get old nested pattern format.

    This function is maintained for backward compatibility with existing tests
    and code that still uses the old nested pattern format.

    New code should use get_framework_standard_patterns() instead.

    Returns:
        Dictionary in old nested format for backward compatibility

    .. deprecated::
        Use :func:`get_framework_standard_patterns` instead.
        This function will be removed in a future version.
    """
    logger.warning(
        "get_default_patterns() is deprecated. "
        "Use get_framework_standard_patterns() instead."
    )

    # Return framework standard patterns in old nested format for backward compatibility
    framework_patterns = get_framework_standard_patterns()

    return {
        'mock': framework_patterns,
        'epics': framework_patterns,
        # All control systems use the same patterns now (control-system-agnostic)
    }

