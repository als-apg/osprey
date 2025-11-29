"""Tests for pattern detection module."""

from osprey.services.python_executor.analysis.pattern_detection import (
    detect_control_system_operations,
    get_default_patterns,
)


class TestPatternDetection:
    """Test pattern detection for control system operations."""

    def test_epics_write_detection(self):
        """Test detection of EPICS write operations."""
        patterns = get_default_patterns()

        code = "epics.caput('BEAM:CURRENT', 500.0)"
        result = detect_control_system_operations(
            code, patterns=patterns, control_system_type='epics'
        )

        assert result['has_writes'] is True
        assert result['has_reads'] is False
        assert result['control_system_type'] == 'epics'
        assert len(result['detected_patterns']['writes']) > 0

    def test_epics_read_detection(self):
        """Test detection of EPICS read operations."""
        patterns = get_default_patterns()

        code = "value = epics.caget('BEAM:CURRENT')"
        result = detect_control_system_operations(
            code, patterns=patterns, control_system_type='epics'
        )

        assert result['has_writes'] is False
        assert result['has_reads'] is True
        assert result['control_system_type'] == 'epics'
        assert len(result['detected_patterns']['reads']) > 0

    def test_epics_pv_write_detection(self):
        """Test detection of EPICS PV.put() operations."""
        patterns = get_default_patterns()

        code = """
pv = epics.PV('BEAM:CURRENT')
pv.put(500.0)
"""
        result = detect_control_system_operations(
            code, patterns=patterns, control_system_type='epics'
        )

        assert result['has_writes'] is True

    def test_epics_pv_read_detection(self):
        """Test detection of EPICS PV.get() operations."""
        patterns = get_default_patterns()

        code = """
pv = epics.PV('BEAM:CURRENT')
value = pv.get()
"""
        result = detect_control_system_operations(
            code, patterns=patterns, control_system_type='epics'
        )

        assert result['has_reads'] is True

    def test_mock_write_detection(self):
        """Test detection of mock write operations."""
        patterns = get_default_patterns()

        code = "connector.caput('BEAM:CURRENT', 500.0)"
        result = detect_control_system_operations(
            code, patterns=patterns, control_system_type='mock'
        )

        assert result['has_writes'] is True
        assert result['control_system_type'] == 'mock'

    def test_mock_read_detection(self):
        """Test detection of mock read operations."""
        patterns = get_default_patterns()

        code = "value = connector.caget('BEAM:CURRENT')"
        result = detect_control_system_operations(
            code, patterns=patterns, control_system_type='mock'
        )

        assert result['has_reads'] is True
        assert result['control_system_type'] == 'mock'

    def test_no_operations_detected(self):
        """Test code with no control system operations."""
        patterns = get_default_patterns()

        code = """
import numpy as np
data = np.array([1, 2, 3])
print(data.mean())
"""
        result = detect_control_system_operations(
            code, patterns=patterns, control_system_type='epics'
        )

        assert result['has_writes'] is False
        assert result['has_reads'] is False
        assert len(result['detected_patterns']['writes']) == 0
        assert len(result['detected_patterns']['reads']) == 0

    def test_mixed_operations_detection(self):
        """Test detection of both read and write operations."""
        patterns = get_default_patterns()

        code = """
current = epics.caget('BEAM:CURRENT')
if current < 400:
    epics.caput('ALARM:STATUS', 1)
"""
        result = detect_control_system_operations(
            code, patterns=patterns, control_system_type='epics'
        )

        assert result['has_writes'] is True
        assert result['has_reads'] is True

    def test_default_patterns_structure(self):
        """Test that default patterns have expected structure."""
        patterns = get_default_patterns()

        assert 'mock' in patterns
        assert 'epics' in patterns

        for cs_type in ['mock', 'epics']:
            assert 'write' in patterns[cs_type]
            assert 'read' in patterns[cs_type]
            assert isinstance(patterns[cs_type]['write'], list)
            assert isinstance(patterns[cs_type]['read'], list)
            assert len(patterns[cs_type]['write']) > 0
            assert len(patterns[cs_type]['read']) > 0

