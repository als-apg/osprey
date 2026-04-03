"""Tests for pattern detection module."""

from unittest.mock import patch

from osprey.services.python_executor.analysis.pattern_detection import (
    detect_control_system_operations,
    get_framework_standard_patterns,
)


class TestPatternDetection:
    """Test pattern detection for control system operations."""

    def test_epics_write_detection(self):
        """Test detection of EPICS write operations."""
        code = "epics.caput('BEAM:CURRENT', 500.0)"
        result = detect_control_system_operations(code)

        assert result["has_writes"] is True
        assert result["has_reads"] is False
        assert len(result["detected_patterns"]["writes"]) > 0

    def test_epics_read_detection(self):
        """Test detection of EPICS read operations."""
        code = "value = epics.caget('BEAM:CURRENT')"
        result = detect_control_system_operations(code)

        assert result["has_writes"] is False
        assert result["has_reads"] is True
        assert len(result["detected_patterns"]["reads"]) > 0

    def test_epics_pv_write_detection(self):
        """Test detection of EPICS PV.put() operations."""
        code = """
pv = epics.PV('BEAM:CURRENT')
pv.put(500.0)
"""
        result = detect_control_system_operations(code)
        assert result["has_writes"] is True

    def test_epics_pv_read_detection(self):
        """Test detection of EPICS PV.get() operations."""
        code = """
pv = epics.PV('BEAM:CURRENT')
value = pv.get()
"""
        result = detect_control_system_operations(code)
        assert result["has_reads"] is True

    def test_unified_api_write_detection(self):
        """Test detection of unified API write operations."""
        code = "write_channel('BEAM:CURRENT', 500.0)"
        result = detect_control_system_operations(code)

        assert result["has_writes"] is True

    def test_unified_api_read_detection(self):
        """Test detection of unified API read operations."""
        code = "value = read_channel('BEAM:CURRENT')"
        result = detect_control_system_operations(code)

        assert result["has_reads"] is True

    def test_no_operations_detected(self):
        """Test code with no control system operations."""
        code = """
import numpy as np
data = np.array([1, 2, 3])
print(data.mean())
"""
        result = detect_control_system_operations(code)

        assert result["has_writes"] is False
        assert result["has_reads"] is False
        assert len(result["detected_patterns"]["writes"]) == 0
        assert len(result["detected_patterns"]["reads"]) == 0

    def test_mixed_operations_detection(self):
        """Test detection of both read and write operations."""
        code = """
current = epics.caget('BEAM:CURRENT')
if current < 400:
    epics.caput('ALARM:STATUS', 1)
"""
        result = detect_control_system_operations(code)

        assert result["has_writes"] is True
        assert result["has_reads"] is True

    def test_framework_patterns_structure(self):
        """Test that framework patterns have expected structure."""
        patterns = get_framework_standard_patterns()

        # New structure: flat dictionary with 'write' and 'read' keys
        assert "write" in patterns
        assert "read" in patterns
        assert isinstance(patterns["write"], list)
        assert isinstance(patterns["read"], list)
        assert len(patterns["write"]) > 0
        assert len(patterns["read"]) > 0

    def test_control_system_agnostic_patterns(self):
        """Test that patterns work regardless of control_system_type."""
        code = "write_channel('BEAM:CURRENT', 500.0)"

        # Should work the same regardless of control_system_type
        result_epics = detect_control_system_operations(code, control_system_type="epics")
        result_mock = detect_control_system_operations(code, control_system_type="mock")

        assert result_epics["has_writes"] == result_mock["has_writes"]
        assert result_epics["has_writes"] is True

    def test_custom_patterns_merged_with_framework(self):
        """Custom patterns from config extend framework patterns by default.

        When a facility adds custom write patterns via config (mode=extend),
        the framework's standard patterns must still be present.
        """
        code_custom = "my_custom_cs_lib.write('DEVICE', 42)"
        code_epics = "epics.caput('BEAM:CURRENT', 500.0)"

        custom_patterns = {
            "write": [r"my_custom_cs_lib\.write\("],
            "read": [],
        }

        def mock_config(key, default=None):
            if key == "control_system.patterns":
                return custom_patterns
            if key == "control_system.type":
                return "custom"
            return default

        with (
            patch(
                "osprey.services.python_executor.analysis.pattern_detection.get_config_value",
                side_effect=mock_config,
                create=True,
            ),
            patch(
                "osprey.utils.config.get_config_value",
                side_effect=mock_config,
            ),
        ):
            # Custom pattern detected
            result_custom = detect_control_system_operations(
                code_custom, control_system_type="custom"
            )
            assert result_custom["has_writes"] is True

            # Framework EPICS pattern still present after merge
            result_epics = detect_control_system_operations(
                code_epics, control_system_type="custom"
            )
            assert result_epics["has_writes"] is True

    def test_passed_patterns_merged_with_framework(self):
        """Patterns passed via the `patterns` parameter merge by default.

        The approval hook passes config-driven patterns directly. In extend
        mode (default), these are appended to framework standard patterns.
        """
        custom = {
            "write": [r"my_facility_write\("],
            "read": [r"my_facility_read\("],
        }

        # Custom pattern detected
        result = detect_control_system_operations(
            "my_facility_write('CH1', 10)", patterns=custom, control_system_type="custom"
        )
        assert result["has_writes"] is True

        # Framework pattern still present
        result2 = detect_control_system_operations(
            "epics.caput('PV', 1.0)", patterns=custom, control_system_type="custom"
        )
        assert result2["has_writes"] is True

        # Verify merged count is at least framework + custom unique
        result3 = detect_control_system_operations(
            "my_facility_write('CH1', 10)\nepics.caput('PV', 1.0)",
            patterns=custom,
            control_system_type="custom",
        )
        assert len(result3["detected_patterns"]["writes"]) >= 2

    def test_override_mode_replaces_framework_patterns(self):
        """With pattern_mode='override', custom patterns replace framework entirely.

        A facility that sets mode: override gets full control — only their
        patterns are used, framework standards are not included.
        """
        custom = {
            "write": [r"my_facility_write\("],
            "read": [],
        }

        # Custom pattern detected in override mode
        result = detect_control_system_operations(
            "my_facility_write('CH1', 10)",
            patterns=custom,
            control_system_type="custom",
            pattern_mode="override",
        )
        assert result["has_writes"] is True

        # Framework EPICS pattern NOT present in override mode
        result2 = detect_control_system_operations(
            "epics.caput('PV', 1.0)",
            patterns=custom,
            control_system_type="custom",
            pattern_mode="override",
        )
        assert result2["has_writes"] is False

    def test_override_mode_from_config(self):
        """Config with mode: override replaces framework patterns via config path."""
        custom_patterns = {
            "mode": "override",
            "write": [r"my_facility_write\("],
            "read": [],
        }

        def mock_config(key, default=None):
            if key == "control_system.patterns":
                return custom_patterns
            if key == "control_system.type":
                return "custom"
            return default

        with (
            patch(
                "osprey.services.python_executor.analysis.pattern_detection.get_config_value",
                side_effect=mock_config,
                create=True,
            ),
            patch(
                "osprey.utils.config.get_config_value",
                side_effect=mock_config,
            ),
        ):
            # Custom pattern detected
            result = detect_control_system_operations(
                "my_facility_write('CH1', 10)", control_system_type="custom"
            )
            assert result["has_writes"] is True

            # Framework pattern NOT present (override mode)
            result2 = detect_control_system_operations(
                "epics.caput('PV', 1.0)", control_system_type="custom"
            )
            assert result2["has_writes"] is False
