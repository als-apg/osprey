"""
Execution Policy Analyzer - Configurable Decision Making

This module provides a clean extension point for applications to customize
both domain analysis and execution policy decisions based on code analysis results.

The framework handles basic safety checks (syntax, security) as hard-coded steps,
then delegates to configurable domain analyzers and policy analyzers for
application-specific decisions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple

from osprey.utils.logger import get_logger

from ..execution.control import ExecutionMode

logger = get_logger("python_analyzer")

@dataclass
class BasicAnalysisResult:
    """Results from framework's basic (non-configurable) analysis steps."""

    # Syntax analysis
    syntax_valid: bool
    syntax_issues: list[str]

    # Security analysis
    security_issues: list[str]
    security_risk_level: str  # "low", "medium", "high"

    # Import analysis
    import_issues: list[str]
    prohibited_imports: list[str]

    # Structure analysis
    has_result_structure: bool

    # Raw code for domain-specific analysis
    code: str
    code_length: int

    # Context information
    user_context: dict[str, Any] | None = None
    execution_context: dict[str, Any] | None = None


@dataclass
class DomainAnalysisResult:
    """Domain-specific analysis results that can be extended by applications."""

    # Base domain analysis that framework can populate
    detected_operations: list[str]
    risk_categories: list[str]

    # Extensible for domain-specific fields
    domain_data: dict[str, Any]

    def __post_init__(self):
        if self.domain_data is None:
            self.domain_data = {}


class ExecutionPolicyDecision(NamedTuple):
    """Decision from configurable execution policy analyzer."""
    execution_mode: ExecutionMode
    needs_approval: bool
    approval_reasoning: str
    additional_issues: list[str]
    recommendations: list[str]
    analysis_passed: bool
    additional_context: dict[str, Any] | None = None


# =============================================================================
# DOMAIN ANALYZER SYSTEM (Registry-Based)
# =============================================================================

class DomainAnalyzer(ABC):
    """
    Abstract base class for configurable domain analysis.

    Applications implement this to analyze code for domain-specific patterns,
    operations, and risks relevant to their use case.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return unique name for this domain analyzer"""
        pass

    @abstractmethod
    def get_priority(self) -> int:
        """Return priority (lower numbers = higher priority)"""
        pass

    @abstractmethod
    async def analyze_domain(
        self,
        basic_analysis: BasicAnalysisResult
    ) -> DomainAnalysisResult:
        """
        Perform domain-specific analysis of code.

        Args:
            basic_analysis: Results from framework's basic analysis

        Returns:
            Domain analysis results with detected operations, risks, etc.
        """
        pass


class DefaultFrameworkDomainAnalyzer(DomainAnalyzer):
    """
    Default domain analyzer that preserves existing framework EPICS analysis.

    This implements the current EPICS operation detection logic as the default,
    while allowing applications to override with custom domain analyzers.
    """

    def __init__(self, configurable: dict[str, Any]):
        self.configurable = configurable

    def get_name(self) -> str:
        return "default_osprey_domain"

    def get_priority(self) -> int:
        return 100  # Default priority

    async def analyze_domain(
        self,
        basic_analysis: BasicAnalysisResult
    ) -> DomainAnalysisResult:
        """Use config-based pattern detection for control system operations"""

        from .pattern_detection import detect_control_system_operations

        detected_operations = []
        risk_categories = []
        domain_data = {}

        code = basic_analysis.code

        # Use framework-standard pattern detection (control-system-agnostic)
        # Users can optionally override patterns via control_system.patterns in config.yml
        detection_result = detect_control_system_operations(
            code=code,
            patterns=None,  # Use framework standard (or config override if provided)
            control_system_type=None  # Load from config for logging/metadata
        )

        # Map detection results to domain analysis format
        control_system_type = detection_result['control_system_type']

        if detection_result['has_writes']:
            detected_operations.append(f"{control_system_type}_writes")
            risk_categories.append("control_system_write")
            domain_data[f"{control_system_type}_write_operations"] = True
            domain_data["detected_write_patterns"] = detection_result['detected_patterns']['writes']

        if detection_result['has_reads']:
            detected_operations.append(f"{control_system_type}_reads")
            domain_data[f"{control_system_type}_read_operations"] = True
            domain_data["detected_read_patterns"] = detection_result['detected_patterns']['reads']

        # Store control system type for downstream use
        domain_data["control_system_type"] = control_system_type

        # Backward compatibility: also set epics_* operations if control system is EPICS
        if control_system_type == "epics":
            if detection_result['has_writes']:
                detected_operations.append("epics_writes")
                risk_categories.append("accelerator_control")
                domain_data["epics_write_operations"] = True
            if detection_result['has_reads']:
                detected_operations.append("epics_reads")
                domain_data["epics_read_operations"] = True

        return DomainAnalysisResult(
            detected_operations=detected_operations,
            risk_categories=risk_categories,
            domain_data=domain_data
        )


class DomainAnalysisManager:
    """
    Manages domain analyzers and orchestrates domain analysis.
    Integrated with the registry system for pluggability.
    """

    def __init__(self, configurable: dict[str, Any]):
        self.configurable = configurable
        self._analyzers: list[DomainAnalyzer] = []
        self._initialized = False

    def initialize(self):
        """Initialize domain analyzers from registry"""
        if self._initialized:
            return

        try:
            from osprey.registry import get_registry
            registry = get_registry()

            # Get registered domain analyzers
            analyzers = registry.get_domain_analyzers()

            if not analyzers:
                logger.info("No custom domain analyzers registered, using default only")
                analyzers = [DefaultFrameworkDomainAnalyzer(self.configurable)]
            else:
                logger.info(f"Found {len(analyzers)} registered domain analyzers")
                # Always include default as fallback
                analyzers.append(DefaultFrameworkDomainAnalyzer(self.configurable))

            # Sort by priority (lower numbers first)
            self._analyzers = sorted(analyzers, key=lambda a: a.get_priority())
            self._initialized = True

            logger.info(f"Initialized domain analyzers: {[a.get_name() for a in self._analyzers]}")

        except Exception as e:
            logger.warning(f"Failed to load custom domain analyzers: {e}")
            logger.info("Using default framework domain analyzer only")
            self._analyzers = [DefaultFrameworkDomainAnalyzer(self.configurable)]
            self._initialized = True

    async def analyze_domain(
        self,
        basic_analysis: BasicAnalysisResult
    ) -> DomainAnalysisResult:
        """
        Perform domain analysis using registered analyzers in priority order.
        First analyzer (highest priority) is used.
        """
        self.initialize()

        for analyzer in self._analyzers:
            try:
                logger.debug(f"Using domain analyzer: {analyzer.get_name()}")
                result = await analyzer.analyze_domain(basic_analysis)
                logger.info(f"Domain analysis completed: {len(result.detected_operations)} operations detected")
                return result

            except Exception as e:
                logger.error(f"Domain analyzer {analyzer.get_name()} failed: {e}")
                continue

        # Fallback: safe default if all analyzers fail
        logger.error("All domain analyzers failed, using safe default")
        return DomainAnalysisResult(
            detected_operations=[],
            risk_categories=[],
            domain_data={"analyzer_failure": True}
        )


# =============================================================================
# EXECUTION POLICY ANALYZER SYSTEM (Registry-Based)
# =============================================================================

class ExecutionPolicyAnalyzer(ABC):
    """
    Abstract base class for configurable execution policy analysis.

    Applications implement this to customize execution mode selection
    and approval decisions based on their domain requirements.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return unique name for this policy analyzer"""
        pass

    @abstractmethod
    def get_priority(self) -> int:
        """Return priority (lower numbers = higher priority)"""
        pass

    @abstractmethod
    async def analyze_policy(
        self,
        basic_analysis: BasicAnalysisResult,
        domain_analysis: DomainAnalysisResult
    ) -> ExecutionPolicyDecision:
        """
        Analyze execution policy to determine mode and approval requirements.

        Args:
            basic_analysis: Results from framework's basic analysis
            domain_analysis: Domain-specific analysis results

        Returns:
            Policy decision on execution mode, approval, etc.
        """
        pass


class DefaultFrameworkPolicyAnalyzer(ExecutionPolicyAnalyzer):
    """
    Default policy analyzer that preserves existing framework behavior.

    This implements the current EPICS-based logic as the default,
    while allowing applications to override with custom analyzers.
    """

    def __init__(self, configurable: dict[str, Any]):
        self.configurable = configurable

    def get_name(self) -> str:
        return "default_osprey_policy"

    def get_priority(self) -> int:
        return 100  # Default priority

    async def analyze_policy(
        self,
        basic_analysis: BasicAnalysisResult,
        domain_analysis: DomainAnalysisResult
    ) -> ExecutionPolicyDecision:
        """Implement existing framework logic as default behavior"""

        # Block execution if basic analysis failed
        if not basic_analysis.syntax_valid:
            return ExecutionPolicyDecision(
                execution_mode=ExecutionMode.READ_ONLY,
                needs_approval=False,  # No approval needed for blocked execution
                approval_reasoning="Code blocked due to syntax errors",
                additional_issues=basic_analysis.syntax_issues,
                recommendations=["Fix syntax errors before execution"],
                analysis_passed=False
            )

        if basic_analysis.security_risk_level == "high":
            return ExecutionPolicyDecision(
                execution_mode=ExecutionMode.READ_ONLY,
                needs_approval=True,
                approval_reasoning="High security risk detected",
                additional_issues=basic_analysis.security_issues,
                recommendations=["Review security concerns before approval"],
                analysis_passed=False
            )

        # Get control system type from domain analysis
        control_system_type = domain_analysis.domain_data.get("control_system_type", "unknown")

        # Extract control system operations (generic, not EPICS-specific)
        has_control_writes = any(
            op.endswith("_writes") for op in domain_analysis.detected_operations
        )
        has_control_reads = any(
            op.endswith("_reads") for op in domain_analysis.detected_operations
        )

        # Backward compatibility: also check for legacy epics_* operations
        has_epics_writes = "epics_writes" in domain_analysis.detected_operations
        has_epics_reads = "epics_reads" in domain_analysis.detected_operations

        # Get execution control configuration
        try:
            from ..models import get_execution_control_config_from_configurable
            exec_control = get_execution_control_config_from_configurable(self.configurable)
        except Exception as e:
            logger.error(f"Failed to get execution control config: {e}")
            return ExecutionPolicyDecision(
                execution_mode=ExecutionMode.READ_ONLY,
                needs_approval=True,
                approval_reasoning=f"Configuration error: {e}",
                additional_issues=[str(e)],
                recommendations=["Fix configuration before execution"],
                analysis_passed=False
            )

        # Determine execution mode based on control system operations
        if has_control_writes:
            # Check if control system writes are enabled (with backward compatibility)
            writes_enabled = getattr(exec_control, 'control_system_writes_enabled',
                                    getattr(exec_control, 'epics_writes_enabled', False))

            if writes_enabled:
                execution_mode = ExecutionMode.WRITE_ACCESS
                additional_issues = [f"{control_system_type.upper()} writes detected - using write access mode"]
            else:
                return ExecutionPolicyDecision(
                    execution_mode=ExecutionMode.READ_ONLY,
                    needs_approval=False,
                    approval_reasoning=f"{control_system_type.upper()} writes detected but writes disabled in configuration",
                    additional_issues=[f"{control_system_type.upper()} writes blocked by configuration"],
                    recommendations=["Enable control system writes in configuration if needed"],
                    analysis_passed=False
                )
        else:
            execution_mode = ExecutionMode.READ_ONLY
            additional_issues = [f"No {control_system_type.upper()} writes detected - using read-only mode"]

        # Determine approval requirements using existing approval system
        # Use has_control_writes, but fall back to has_epics_writes for backward compatibility
        from osprey.approval.approval_manager import get_python_execution_evaluator
        approval_evaluator = get_python_execution_evaluator()
        approval_decision = approval_evaluator.evaluate(
            has_control_writes or has_epics_writes,
            has_control_reads or has_epics_reads
        )

        needs_approval = approval_decision.needs_approval
        approval_reasoning = approval_decision.reasoning

        # Generate recommendations
        recommendations = []
        if has_control_reads and not has_control_writes:
            recommendations.append(f"Read-only {control_system_type.upper()} operations detected - safe for execution")
        if has_control_writes:
            recommendations.append(f"{control_system_type.upper()} write operations require careful review")

        return ExecutionPolicyDecision(
            execution_mode=execution_mode,
            needs_approval=needs_approval,
            approval_reasoning=approval_reasoning,
            additional_issues=additional_issues,
            recommendations=recommendations,
            analysis_passed=True,
            additional_context={
                "has_control_writes": has_control_writes,
                "has_control_reads": has_control_reads,
                "control_system_type": control_system_type,
                "control_system_writes_enabled": getattr(exec_control, 'control_system_writes_enabled',
                                                         getattr(exec_control, 'epics_writes_enabled', False)),
                # Backward compatibility fields
                "has_epics_writes": has_epics_writes,
                "has_epics_reads": has_epics_reads,
                "epics_writes_enabled": getattr(exec_control, 'epics_writes_enabled', False)
            }
        )


class ExecutionPolicyManager:
    """
    Manages execution policy analyzers and orchestrates policy decisions.
    Integrated with the registry system for pluggability.
    """

    def __init__(self, configurable: dict[str, Any]):
        self.configurable = configurable
        self._analyzers: list[ExecutionPolicyAnalyzer] = []
        self._initialized = False

    def initialize(self):
        """Initialize policy analyzers from registry"""
        if self._initialized:
            return

        try:
            from osprey.registry import get_registry
            registry = get_registry()

            # Get registered execution policy analyzers
            analyzers = registry.get_execution_policy_analyzers()

            if not analyzers:
                logger.info("No custom execution policy analyzers registered, using default only")
                analyzers = [DefaultFrameworkPolicyAnalyzer(self.configurable)]
            else:
                logger.info(f"Found {len(analyzers)} registered execution policy analyzers")
                # Always include default as fallback
                analyzers.append(DefaultFrameworkPolicyAnalyzer(self.configurable))

            # Sort by priority (lower numbers first)
            self._analyzers = sorted(analyzers, key=lambda a: a.get_priority())
            self._initialized = True

            logger.info(f"Initialized execution policy analyzers: {[a.get_name() for a in self._analyzers]}")

        except Exception as e:
            logger.warning(f"Failed to load custom execution policy analyzers: {e}")
            logger.info("Using default framework policy analyzer only")
            self._analyzers = [DefaultFrameworkPolicyAnalyzer(self.configurable)]
            self._initialized = True

    async def analyze_policy(
        self,
        basic_analysis: BasicAnalysisResult,
        domain_analysis: DomainAnalysisResult
    ) -> ExecutionPolicyDecision:
        """
        Perform policy analysis using registered analyzers in priority order.
        First analyzer (highest priority) is used.
        """
        self.initialize()

        for analyzer in self._analyzers:
            logger.debug(f"Using execution policy analyzer: {analyzer.get_name()}")
            decision = await analyzer.analyze_policy(basic_analysis, domain_analysis)
            logger.info(f"Execution policy decision: {decision.execution_mode}, approval: {decision.needs_approval}")
            return decision
