"""Tests for pattern detection module."""

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

        assert result['has_writes'] is True
        assert result['has_reads'] is False
        assert len(result['detected_patterns']['writes']) > 0

    def test_epics_read_detection(self):
        """Test detection of EPICS read operations."""
        code = "value = epics.caget('BEAM:CURRENT')"
        result = detect_control_system_operations(code)

        assert result['has_writes'] is False
        assert result['has_reads'] is True
        assert len(result['detected_patterns']['reads']) > 0

    def test_epics_pv_write_detection(self):
        """Test detection of EPICS PV.put() operations."""
        code = """
pv = epics.PV('BEAM:CURRENT')
pv.put(500.0)
"""
        result = detect_control_system_operations(code)
        assert result['has_writes'] is True

    def test_epics_pv_read_detection(self):
        """Test detection of EPICS PV.get() operations."""
        code = """
pv = epics.PV('BEAM:CURRENT')
value = pv.get()
"""
        result = detect_control_system_operations(code)
        assert result['has_reads'] is True

    def test_unified_api_write_detection(self):
        """Test detection of unified API write operations."""
        code = "write_channel('BEAM:CURRENT', 500.0)"
        result = detect_control_system_operations(code)

        assert result['has_writes'] is True

    def test_unified_api_read_detection(self):
        """Test detection of unified API read operations."""
        code = "value = read_channel('BEAM:CURRENT')"
        result = detect_control_system_operations(code)

        assert result['has_reads'] is True

    def test_no_operations_detected(self):
        """Test code with no control system operations."""
        code = """
import numpy as np
data = np.array([1, 2, 3])
print(data.mean())
"""
        result = detect_control_system_operations(code)

        assert result['has_writes'] is False
        assert result['has_reads'] is False
        assert len(result['detected_patterns']['writes']) == 0
        assert len(result['detected_patterns']['reads']) == 0

    def test_mixed_operations_detection(self):
        """Test detection of both read and write operations."""
        code = """
current = epics.caget('BEAM:CURRENT')
if current < 400:
    epics.caput('ALARM:STATUS', 1)
"""
        result = detect_control_system_operations(code)

        assert result['has_writes'] is True
        assert result['has_reads'] is True

    def test_framework_patterns_structure(self):
        """Test that framework patterns have expected structure."""
        patterns = get_framework_standard_patterns()

        # New structure: flat dictionary with 'write' and 'read' keys
        assert 'write' in patterns
        assert 'read' in patterns
        assert isinstance(patterns['write'], list)
        assert isinstance(patterns['read'], list)
        assert len(patterns['write']) > 0
        assert len(patterns['read']) > 0

    def test_control_system_agnostic_patterns(self):
        """Test that patterns work regardless of control_system_type."""
        code = "write_channel('BEAM:CURRENT', 500.0)"

        # Should work the same regardless of control_system_type
        result_epics = detect_control_system_operations(code, control_system_type='epics')
        result_mock = detect_control_system_operations(code, control_system_type='mock')

        assert result_epics['has_writes'] == result_mock['has_writes']
        assert result_epics['has_writes'] is True

