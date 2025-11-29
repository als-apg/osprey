"""Integration tests for pattern detection in domain analysis.

This module tests that the DefaultFrameworkDomainAnalyzer correctly uses
the pattern_detection module to detect control system operations from config.
"""

import pytest

from osprey.services.python_executor.analysis.policy_analyzer import (
    BasicAnalysisResult,
    DefaultFrameworkDomainAnalyzer,
)


class TestPatternDetectionIntegration:
    """Test that pattern detection is integrated correctly into domain analysis."""

    @pytest.mark.asyncio
    async def test_uses_pattern_detection_module(self):
        """Verify that domain analyzer uses pattern_detection module."""
        analyzer = DefaultFrameworkDomainAnalyzer(configurable={})

        epics_write_code = """
from epics import caget, caput
current = caget('BEAM:CURRENT')
caput('BEAM:CURRENT', current * 1.1)
results = {'status': 'success'}
"""

        basic_analysis = BasicAnalysisResult(
            syntax_valid=True,
            syntax_issues=[],
            security_issues=[],
            security_risk_level="low",
            import_issues=[],
            prohibited_imports=[],
            has_result_structure=True,
            code=epics_write_code,
            code_length=len(epics_write_code)
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # Should detect control system type
        assert 'control_system_type' in result.domain_data
        assert result.domain_data['control_system_type'] in ['epics', 'mock']

        # Should detect write operations
        assert any('_writes' in op for op in result.detected_operations), \
            f"Expected write operations in {result.detected_operations}"

        # Should store detected patterns
        assert 'detected_write_patterns' in result.domain_data

    @pytest.mark.asyncio
    async def test_detects_epics_writes(self):
        """Test detection of EPICS write operations."""
        analyzer = DefaultFrameworkDomainAnalyzer(configurable={})

        code = """
from epics import caput
caput('DEVICE:PV', 42.0)
results = {}
"""

        basic_analysis = BasicAnalysisResult(
            syntax_valid=True,
            syntax_issues=[],
            security_issues=[],
            security_risk_level="low",
            import_issues=[],
            prohibited_imports=[],
            has_result_structure=True,
            code=code,
            code_length=len(code)
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # Should detect write operation
        has_writes = any('_writes' in op for op in result.detected_operations)
        assert has_writes, f"Write operation not detected. Operations: {result.detected_operations}"

    @pytest.mark.asyncio
    async def test_detects_epics_reads(self):
        """Test detection of EPICS read operations."""
        analyzer = DefaultFrameworkDomainAnalyzer(configurable={})

        code = """
from epics import caget
value = caget('DEVICE:PV')
results = {'value': value}
"""

        basic_analysis = BasicAnalysisResult(
            syntax_valid=True,
            syntax_issues=[],
            security_issues=[],
            security_risk_level="low",
            import_issues=[],
            prohibited_imports=[],
            has_result_structure=True,
            code=code,
            code_length=len(code)
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # Should detect read operation
        has_reads = any('_reads' in op for op in result.detected_operations)
        assert has_reads, f"Read operation not detected. Operations: {result.detected_operations}"

    @pytest.mark.asyncio
    async def test_backward_compatibility_epics(self):
        """Test that EPICS operations also set legacy epics_writes/epics_reads flags."""
        analyzer = DefaultFrameworkDomainAnalyzer(configurable={})

        code = """
from epics import caget, caput
value = caget('PV:READ')
caput('PV:WRITE', value)
results = {}
"""

        basic_analysis = BasicAnalysisResult(
            syntax_valid=True,
            syntax_issues=[],
            security_issues=[],
            security_risk_level="low",
            import_issues=[],
            prohibited_imports=[],
            has_result_structure=True,
            code=code,
            code_length=len(code)
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # For EPICS control system, should also set legacy flags
        if result.domain_data.get('control_system_type') == 'epics':
            assert 'epics_writes' in result.detected_operations, \
                "Should set epics_writes for backward compatibility"
            assert 'epics_reads' in result.detected_operations, \
                "Should set epics_reads for backward compatibility"
            assert result.domain_data.get('epics_write_operations') is True
            assert result.domain_data.get('epics_read_operations') is True

    @pytest.mark.asyncio
    async def test_no_operations_detected(self):
        """Test code with no control system operations."""
        analyzer = DefaultFrameworkDomainAnalyzer(configurable={})

        code = """
import numpy as np
data = np.array([1, 2, 3])
mean = np.mean(data)
results = {'mean': mean}
"""

        basic_analysis = BasicAnalysisResult(
            syntax_valid=True,
            syntax_issues=[],
            security_issues=[],
            security_risk_level="low",
            import_issues=[],
            prohibited_imports=[],
            has_result_structure=True,
            code=code,
            code_length=len(code)
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # Should not detect any control system operations
        assert not any('_writes' in op for op in result.detected_operations)
        assert not any('_reads' in op for op in result.detected_operations)

    @pytest.mark.asyncio
    async def test_multiple_write_patterns(self):
        """Test detection of different write pattern styles."""
        analyzer = DefaultFrameworkDomainAnalyzer(configurable={})

        # Test different EPICS write patterns
        codes = [
            "caput('PV', 42)",
            "pv.put(42)",
            "from epics import caput\ncaput('PV', 42)",
        ]

        for code in codes:
            full_code = f"{code}\nresults = {{}}"
            basic_analysis = BasicAnalysisResult(
                syntax_valid=True,
                syntax_issues=[],
                security_issues=[],
                security_risk_level="low",
                import_issues=[],
                prohibited_imports=[],
                has_result_structure=True,
                code=full_code,
                code_length=len(full_code)
            )

            result = await analyzer.analyze_domain(basic_analysis)

            # Should detect write in all cases
            has_writes = any('_writes' in op for op in result.detected_operations)
            assert has_writes, f"Failed to detect write in: {code}"

    @pytest.mark.asyncio
    async def test_risk_categories_set(self):
        """Test that risk categories are set for control system writes."""
        analyzer = DefaultFrameworkDomainAnalyzer(configurable={})

        code = """
from epics import caput
caput('CRITICAL:PV', 100.0)
results = {}
"""

        basic_analysis = BasicAnalysisResult(
            syntax_valid=True,
            syntax_issues=[],
            security_issues=[],
            security_risk_level="low",
            import_issues=[],
            prohibited_imports=[],
            has_result_structure=True,
            code=code,
            code_length=len(code)
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # Should set risk category for writes
        assert len(result.risk_categories) > 0, \
            "Risk categories should be set for control system writes"
        assert 'control_system_write' in result.risk_categories or \
               'accelerator_control' in result.risk_categories


class TestPatternDetectionDirect:
    """Test the pattern_detection module directly."""

    def test_pattern_detection_module_exists(self):
        """Verify pattern_detection module can be imported."""
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
            get_default_patterns
        )
        assert detect_control_system_operations is not None
        assert get_default_patterns is not None

    def test_default_patterns_structure(self):
        """Test that default patterns have correct structure."""
        from osprey.services.python_executor.analysis.pattern_detection import (
            get_default_patterns
        )

        patterns = get_default_patterns()

        # Should have patterns for different control systems
        assert 'epics' in patterns
        assert 'mock' in patterns

        # Each should have write and read patterns
        for cs_type in ['epics', 'mock']:
            assert 'write' in patterns[cs_type]
            assert 'read' in patterns[cs_type]
            assert isinstance(patterns[cs_type]['write'], list)
            assert isinstance(patterns[cs_type]['read'], list)
            assert len(patterns[cs_type]['write']) > 0
            assert len(patterns[cs_type]['read']) > 0

    def test_detect_epics_operations(self):
        """Test direct pattern detection for EPICS."""
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
            get_default_patterns
        )

        patterns = get_default_patterns()

        # Test EPICS write
        write_code = "caput('PV', 42)"
        result = detect_control_system_operations(
            code=write_code,
            patterns=patterns,
            control_system_type='epics'
        )

        assert result['has_writes'] is True
        assert result['control_system_type'] == 'epics'
        assert len(result['detected_patterns']['writes']) > 0

        # Test EPICS read
        read_code = "value = caget('PV')"
        result = detect_control_system_operations(
            code=read_code,
            patterns=patterns,
            control_system_type='epics'
        )

        assert result['has_reads'] is True
        assert len(result['detected_patterns']['reads']) > 0

    def test_detect_mock_operations(self):
        """Test direct pattern detection for mock control system."""
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
            get_default_patterns
        )

        patterns = get_default_patterns()

        # Test mock write
        write_code = "cs.caput('PV', 42)"
        result = detect_control_system_operations(
            code=write_code,
            patterns=patterns,
            control_system_type='mock'
        )

        assert result['has_writes'] is True
        assert result['control_system_type'] == 'mock'

    def test_no_operations_detected_direct(self):
        """Test that non-control-system code doesn't trigger detection."""
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
            get_default_patterns
        )

        patterns = get_default_patterns()

        code = "import numpy as np\nresult = np.mean([1, 2, 3])"
        result = detect_control_system_operations(
            code=code,
            patterns=patterns,
            control_system_type='epics'
        )

        assert result['has_writes'] is False
        assert result['has_reads'] is False
        assert len(result['detected_patterns']['writes']) == 0
        assert len(result['detected_patterns']['reads']) == 0

