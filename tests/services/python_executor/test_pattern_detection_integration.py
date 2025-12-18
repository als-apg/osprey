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

        # Use new osprey.runtime API
        runtime_write_code = """
from osprey.runtime import read_channel, write_channel
current = read_channel('BEAM:CURRENT')
write_channel('BEAM:CURRENT', current * 1.1)
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
            code=runtime_write_code,
            code_length=len(runtime_write_code),
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # Should detect control system type
        assert "control_system_type" in result.domain_data
        assert result.domain_data["control_system_type"] in ["epics", "mock"]

        # Should detect write operations
        assert any("_writes" in op for op in result.detected_operations), (
            f"Expected write operations in {result.detected_operations}"
        )

        # Should store detected patterns
        assert "detected_write_patterns" in result.domain_data

    @pytest.mark.asyncio
    async def test_detects_epics_writes(self):
        """Test detection of control system write operations using osprey.runtime."""
        analyzer = DefaultFrameworkDomainAnalyzer(configurable={})

        # Use new osprey.runtime API
        code = """
from osprey.runtime import write_channel
write_channel('DEVICE:PV', 42.0)
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
            code_length=len(code),
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # Should detect write operation
        has_writes = any("_writes" in op for op in result.detected_operations)
        assert has_writes, f"Write operation not detected. Operations: {result.detected_operations}"

    @pytest.mark.asyncio
    async def test_detects_epics_reads(self):
        """Test detection of control system read operations using osprey.runtime."""
        analyzer = DefaultFrameworkDomainAnalyzer(configurable={})

        # Use new osprey.runtime API
        code = """
from osprey.runtime import read_channel
value = read_channel('DEVICE:PV')
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
            code_length=len(code),
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # Should detect read operation
        has_reads = any("_reads" in op for op in result.detected_operations)
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
            code_length=len(code),
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # For EPICS control system, should also set legacy flags
        if result.domain_data.get("control_system_type") == "epics":
            assert "epics_writes" in result.detected_operations, (
                "Should set epics_writes for backward compatibility"
            )
            assert "epics_reads" in result.detected_operations, (
                "Should set epics_reads for backward compatibility"
            )
            assert result.domain_data.get("epics_write_operations") is True
            assert result.domain_data.get("epics_read_operations") is True

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
            code_length=len(code),
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # Should not detect any control system operations
        assert not any("_writes" in op for op in result.detected_operations)
        assert not any("_reads" in op for op in result.detected_operations)

    @pytest.mark.asyncio
    async def test_multiple_write_patterns(self):
        """Test detection of different write pattern styles."""
        analyzer = DefaultFrameworkDomainAnalyzer(configurable={})

        # Test osprey.runtime write patterns (unified API)
        # The connector abstraction uses write_channel/read_channel, NOT caput/caget
        codes = [
            "from osprey.runtime import write_channel\nwrite_channel('PV', 42)",  # Standalone runtime utility
            "from osprey.runtime import write_channels\nwrite_channels({'PV': 42})",  # Bulk write utility
            "connector.write_channel('PV', 42)",  # Connector method (unified API)
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
                code_length=len(full_code),
            )

            result = await analyzer.analyze_domain(basic_analysis)

            # Should detect write in all cases
            has_writes = any("_writes" in op for op in result.detected_operations)
            assert has_writes, f"Failed to detect write in: {code}"

    @pytest.mark.asyncio
    async def test_risk_categories_set(self):
        """Test that risk categories are set for control system writes."""
        analyzer = DefaultFrameworkDomainAnalyzer(configurable={})

        # Use new osprey.runtime API
        code = """
from osprey.runtime import write_channel
write_channel('CRITICAL:PV', 100.0)
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
            code_length=len(code),
        )

        result = await analyzer.analyze_domain(basic_analysis)

        # Should set risk category for writes
        assert len(result.risk_categories) > 0, (
            "Risk categories should be set for control system writes"
        )
        assert (
            "control_system_write" in result.risk_categories
            or "accelerator_control" in result.risk_categories
        )


class TestPatternDetectionDirect:
    """Test the pattern_detection module directly."""

    def test_pattern_detection_module_exists(self):
        """Verify pattern_detection module can be imported."""
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
            get_framework_standard_patterns,
        )

        assert detect_control_system_operations is not None
        assert get_framework_standard_patterns is not None

    def test_framework_patterns_structure(self):
        """Test that framework patterns have correct structure."""
        from osprey.services.python_executor.analysis.pattern_detection import (
            get_framework_standard_patterns,
        )

        patterns = get_framework_standard_patterns()

        # Should have flat structure with write and read patterns
        assert "write" in patterns
        assert "read" in patterns
        assert isinstance(patterns["write"], list)
        assert isinstance(patterns["read"], list)
        assert len(patterns["write"]) > 0
        assert len(patterns["read"]) > 0

    def test_detect_epics_operations(self):
        """Test pattern detection for EPICS operations."""
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
        )

        # Test EPICS write (uses framework patterns automatically)
        write_code = "caput('PV', 42)"
        result = detect_control_system_operations(code=write_code)

        assert result["has_writes"] is True
        assert len(result["detected_patterns"]["writes"]) > 0

        # Test EPICS read
        read_code = "value = caget('PV')"
        result = detect_control_system_operations(code=read_code)

        assert result["has_reads"] is True
        assert len(result["detected_patterns"]["reads"]) > 0

    def test_detect_unified_api_operations(self):
        """Test pattern detection for unified API operations."""
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
        )

        # Test unified API write
        write_code = "write_channel('PV', 42)"
        result = detect_control_system_operations(code=write_code)

        assert result["has_writes"] is True

        # Test unified API read
        read_code = "value = read_channel('PV')"
        result = detect_control_system_operations(code=read_code)

        assert result["has_reads"] is True

    def test_no_operations_detected_direct(self):
        """Test that non-control-system code doesn't trigger detection."""
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
        )

        code = "import numpy as np\nresult = np.mean([1, 2, 3])"
        result = detect_control_system_operations(code=code)

        assert result["has_writes"] is False
        assert result["has_reads"] is False

    def test_control_system_agnostic_detection(self):
        """Test that patterns work regardless of control_system_type."""
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
        )

        # Patterns should work the same regardless of control_system_type
        code = "write_channel('PV', 42)"

        result_epics = detect_control_system_operations(code, control_system_type="epics")
        result_mock = detect_control_system_operations(code, control_system_type="mock")

        # Should detect writes regardless of control system type
        assert result_epics["has_writes"] is True
        assert result_mock["has_writes"] is True

        # Both should have detected write patterns
        assert len(result_epics["detected_patterns"]["writes"]) > 0
        assert len(result_mock["detected_patterns"]["writes"]) > 0
