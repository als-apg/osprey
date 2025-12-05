"""
Static Analysis Node - LangGraph Architecture

Analyzes generated Python code for potential issues before execution.
Enhanced with EPICS operation detection and execution mode selection.
Transformed for LangGraph integration with TypedDict state management.
"""

import ast
from typing import Any

from osprey.approval.approval_system import create_code_approval_interrupt
from osprey.utils.config import get_full_configuration
from osprey.utils.logger import get_logger

from ..exceptions import (
    CodeGenerationError,
    CodeSyntaxError,
    ContainerConfigurationError,
)
from ..models import (
    AnalysisResult,
    ExecutionError,
    PythonExecutionState,
    get_execution_mode_config_from_configurable,
    validate_result_structure,
)
from ..services import FileManager, NotebookManager
from .policy_analyzer import (
    BasicAnalysisResult,
    DomainAnalysisManager,
    ExecutionPolicyManager,
)

logger = get_logger("osprey")


class StaticCodeAnalyzer:
    """Clean code analyzer with proper exception handling"""

    def __init__(self, configurable):
        self.configurable = configurable

    async def analyze_code(self, code: str, context: Any) -> AnalysisResult:
        """Perform comprehensive static analysis using configurable execution policy analyzers"""

        try:
            # ========================================
            # BASIC ANALYSIS (Hard-coded Framework Safety Checks)
            # ========================================

            # 1. Syntax validation
            syntax_issues = self._check_syntax(code)
            syntax_valid = len(syntax_issues) == 0

            # Critical syntax errors should fail immediately
            if not syntax_valid:
                raise CodeSyntaxError(
                    "Code contains syntax errors",
                    syntax_issues=syntax_issues,
                    technical_details={
                        "code_preview": code[:200] + "..." if len(code) > 200 else code
                    },
                )

            # 2. Security analysis
            security_issues = self._check_security(code)
            security_risk_level = self._determine_security_risk_level(security_issues)

            # 3. Import validation
            import_issues = self._check_imports(code)
            prohibited_imports = self._get_prohibited_imports(import_issues)

            # 4. Result structure validation (static check - warns but doesn't fail)
            has_result_structure = validate_result_structure(code)
            if not has_result_structure:
                logger.warning(
                    "⚠️  Generated code does not appear to assign to 'results' variable. "
                    "This may cause issues if downstream code expects results. "
                    "Note: This is a static check - runtime validation will confirm."
                )

            # Create basic analysis result
            basic_analysis = BasicAnalysisResult(
                syntax_valid=syntax_valid,
                syntax_issues=syntax_issues,
                security_issues=security_issues,
                security_risk_level=security_risk_level,
                import_issues=import_issues,
                prohibited_imports=prohibited_imports,
                has_result_structure=has_result_structure,
                code=code,
                code_length=len(code),
                user_context=None,  # Could be populated from context if needed
                execution_context=None,
            )

            # ========================================
            # DOMAIN ANALYSIS (Framework-provided EPICS analysis)
            # ========================================

            domain_manager = DomainAnalysisManager(self.configurable)
            domain_analysis = await domain_manager.analyze_domain(basic_analysis)

            # ========================================
            # EXECUTION POLICY DECISION (Configurable)
            # ========================================

            policy_manager = ExecutionPolicyManager(self.configurable)
            policy_decision = await policy_manager.analyze_policy(basic_analysis, domain_analysis)

            # Combine all issues
            all_issues = []
            all_issues.extend(basic_analysis.syntax_issues)
            all_issues.extend(basic_analysis.security_issues)
            all_issues.extend(basic_analysis.import_issues)
            all_issues.extend(policy_decision.additional_issues)

            # Determine severity and pass/fail status
            critical_issues = [
                issue
                for issue in all_issues
                if any(word in issue.lower() for word in ["error", "invalid", "blocked"])
            ]
            passed = len(critical_issues) == 0 and policy_decision.analysis_passed
            severity = "error" if critical_issues else ("warning" if all_issues else "info")

            # Get execution mode configuration
            execution_mode_config = None
            try:
                mode_config = get_execution_mode_config_from_configurable(
                    self.configurable, policy_decision.execution_mode.value
                )
                execution_mode_config = mode_config.__dict__
            except Exception as e:
                logger.warning(f"Failed to load execution mode config: {e}")

            # Extract EPICS flags from domain analysis for backward compatibility
            has_epics_writes = "epics_writes" in domain_analysis.detected_operations
            has_epics_reads = "epics_reads" in domain_analysis.detected_operations

            analysis_result = AnalysisResult(
                passed=passed,
                issues=all_issues,
                recommendations=policy_decision.recommendations,
                severity=severity,
                has_epics_writes=has_epics_writes,
                has_epics_reads=has_epics_reads,
                recommended_execution_mode=policy_decision.execution_mode,
                needs_approval=policy_decision.needs_approval,
                approval_reasoning=policy_decision.approval_reasoning,
                execution_mode_config=execution_mode_config,
            )

            if passed:
                logger.info(
                    f"Static analysis passed: {len(all_issues)} non-critical issues, execution mode: {policy_decision.execution_mode.value}"
                )
            else:
                logger.warning(
                    f"Static analysis found critical issues: {len(critical_issues)} critical, {len(all_issues)} total"
                )

            return analysis_result

        except CodeSyntaxError:
            # Re-raise syntax errors
            raise
        except Exception as e:
            # Convert unexpected errors to configuration errors
            raise ContainerConfigurationError(
                f"Static analysis failed: {str(e)}", technical_details={"original_error": str(e)}
            ) from e

    def _check_syntax(self, code: str) -> list[str]:
        """Check Python syntax validity"""
        issues = []
        try:
            ast.parse(code)
            logger.debug("Syntax validation passed")
        except SyntaxError as e:
            issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
            logger.warning(f"Syntax error found: {e.msg}")
        except Exception as e:
            issues.append(f"Syntax parsing error: {str(e)}")
            logger.warning(f"Syntax parsing failed: {str(e)}")

        return issues

    def _check_security(self, code: str) -> list[str]:
        """Basic security checks for dangerous operations"""
        issues = []

        # Check for potentially dangerous operations
        dangerous_patterns = [
            ("exec(", "Use of exec() function"),
            ("eval(", "Use of eval() function"),
            ("__import__", "Dynamic import usage"),
            ("open(", "File operations - ensure proper handling"),
            ("subprocess", "Subprocess usage - potential security risk"),
            ("os.system", "System command execution"),
        ]

        for pattern, warning in dangerous_patterns:
            if pattern in code:
                if pattern in ["open(", "subprocess"]:
                    # These might be legitimate, just warn
                    issues.append(f"Warning: {warning}")
                else:
                    # These are more concerning
                    issues.append(f"Security risk: {warning}")

        return issues

    def _check_imports(self, code: str) -> list[str]:
        """Check for prohibited imports - returns list of issues"""
        issues = []

        prohibited_imports = ["subprocess", "os.system", "eval", "exec"]

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in prohibited_imports:
                            issues.append(f"Prohibited import detected: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in prohibited_imports:
                        issues.append(f"Prohibited import detected: {node.module}")

        except SyntaxError:
            # Syntax errors are handled elsewhere
            pass

        return issues

    def _determine_security_risk_level(self, security_issues: list[str]) -> str:
        """Determine security risk level based on security issues"""
        if not security_issues:
            return "low"

        high_risk_keywords = ["subprocess", "os.system", "eval", "exec", "shell"]
        for issue in security_issues:
            if any(keyword in issue.lower() for keyword in high_risk_keywords):
                return "high"

        return "medium"

    def _get_prohibited_imports(self, import_issues: list[str]) -> list[str]:
        """Extract list of prohibited imports from import issues"""
        prohibited = []
        for issue in import_issues:
            if "Prohibited import detected:" in issue:
                import_name = issue.split(": ", 1)[1]
                prohibited.append(import_name)
        return prohibited


def create_analyzer_node():
    """Create the static analysis node function."""

    async def analyzer_node(state: PythonExecutionState) -> dict[str, Any]:
        """Perform static analysis and package approval data to avoid double execution."""

        # Get logger with streaming support
        logger = get_logger("python_analyzer", state=state)
        logger.status("Analyzing Python code...")

        # Check if we have code to analyze
        generated_code = state.get("generated_code")
        if not generated_code:
            error = ExecutionError(
                error_type="generation",
                error_message="No code available for static analysis",
                attempt_number=state.get("generation_attempt", 0),
                stage="generation",
            )
            error_chain = state.get("error_chain", []) + [error]

            return {
                "analysis_failed": True,
                "error_chain": error_chain,
                "current_stage": "generation",
            }

        # Use existing analyzer logic - access request data via state.request
        # Get config from LangGraph configurable
        configurable = get_full_configuration()

        analyzer = StaticCodeAnalyzer(configurable)

        try:
            # Perform analysis using existing logic
            analysis_result = await analyzer.analyze_code(
                state["generated_code"],  # From service state
                state["request"],  # Original request context as dictionary
            )

            if not analysis_result.passed:
                # Analysis failed - need to regenerate
                error = ExecutionError(
                    error_type="analysis",
                    error_message="Static analysis failed",
                    failed_code=generated_code,
                    analysis_issues=analysis_result.issues,
                    attempt_number=state.get("generation_attempt", 0),
                    stage="analysis",
                )
                error_chain = state.get("error_chain", []) + [error]

                # Create attempt notebook for debugging static analysis failures
                error_message = f"Static analysis failed: {', '.join(analysis_result.issues)}"
                await _create_analysis_failure_attempt_notebook(
                    state, configurable, generated_code, error_message, analysis_result.issues
                )

                # Check retry limit here (not in conditional edge!)
                max_retries = state["request"].retries
                retry_limit_exceeded = len(error_chain) >= max_retries

                return {
                    "analysis_result": analysis_result,
                    "analysis_failed": True,
                    "error_chain": error_chain,
                    "current_stage": "generation",
                    # Mark as permanently failed if retry limit exceeded
                    "is_failed": retry_limit_exceeded,
                    "failure_reason": (
                        f"Code generation failed after {max_retries} attempts (analysis failures)"
                        if retry_limit_exceeded
                        else None
                    ),
                }
            # Analysis passed - check if approval needed
            requires_approval = analysis_result.needs_approval

            status_msg = "Code requires approval" if requires_approval else "Code analysis passed"
            logger.status(status_msg)

            if requires_approval:
                # Create pre-approval notebook for user review
                # This gives users a clickable Jupyter link to review code before approving
                execution_folder, notebook_path, notebook_link = (
                    await _create_pre_approval_notebook(
                        state, configurable, generated_code, analysis_result
                    )
                )

                # Create approval interrupt data with REAL notebook paths
                # Users now get clickable links instead of generic "code is available" message
                execution_mode_str = (
                    analysis_result.recommended_execution_mode.value
                    if hasattr(analysis_result.recommended_execution_mode, "value")
                    else str(analysis_result.recommended_execution_mode)
                )

                approval_interrupt_data = create_code_approval_interrupt(
                    code=state["generated_code"],
                    analysis_details=analysis_result.__dict__ if analysis_result else {},
                    execution_mode=execution_mode_str,
                    safety_concerns=analysis_result.issues + analysis_result.recommendations,
                    notebook_path=notebook_path,  # Real path for user review
                    notebook_link=notebook_link,  # Real clickable link
                    execution_request=state["request"],
                    expected_results=state["request"].expected_results,
                    execution_folder_path=execution_folder.folder_path,  # Real execution folder
                    step_objective=state["request"].task_objective,
                )

                return {
                    "analysis_result": analysis_result,
                    "analysis_failed": False,
                    "requires_approval": True,
                    "approval_interrupt_data": approval_interrupt_data,  # Use new LangGraph-native system
                    "execution_folder": execution_folder,  # Save folder to state for executor node
                    "current_stage": "approval",
                }
            else:
                return {
                    "analysis_result": analysis_result,
                    "analysis_failed": False,
                    "requires_approval": False,
                    "approval_interrupt_data": None,
                    "current_stage": "execution",
                }

        except (CodeSyntaxError, CodeGenerationError) as e:
            # Expected code quality issues - treat as analysis failure, not system error
            logger.warning(f"⚠️  Code analysis failed: {e}")

            error = ExecutionError(
                error_type="syntax",
                error_message=str(e),
                failed_code=generated_code,
                attempt_number=state.get("generation_attempt", 0),
                stage="analysis",
                analysis_issues=[str(e)],
            )
            error_chain = state.get("error_chain", []) + [error]

            # Create attempt notebook for debugging syntax errors
            await _create_syntax_error_attempt_notebook(state, configurable, generated_code, str(e))

            # Check retry limit here (not in conditional edge!)
            max_retries = state["request"].retries
            retry_limit_exceeded = len(error_chain) >= max_retries

            return {
                "analysis_failed": True,  # Retry with regeneration
                "error_chain": error_chain,
                "failure_reason": str(e),
                "current_stage": "generation",
                # Mark as permanently failed if retry limit exceeded
                "is_failed": retry_limit_exceeded,
            }

        except Exception as e:
            # Truly unexpected analyzer crashes are critical system errors
            # This should only happen due to framework bugs, not code quality issues
            logger.error(f"Critical system error: Analyzer crashed unexpectedly: {e}")
            logger.error("This indicates a framework bug, not a code quality issue")

            import traceback

            error = ExecutionError(
                error_type="system",
                error_message=f"Critical analyzer error: {str(e)}",
                failed_code=state.get("generated_code"),
                traceback=traceback.format_exc(),
                attempt_number=state.get("generation_attempt", 0),
                stage="analysis",
            )

            return {
                "analysis_failed": False,  # Don't retry - this is a system bug
                "is_failed": True,  # Mark as permanently failed
                "failure_reason": f"Critical analyzer error: {str(e)}",
                "error_chain": state.get("error_chain", []) + [error],
                "current_stage": "failed",
            }

    return analyzer_node


async def _create_analysis_failure_attempt_notebook(
    state: PythonExecutionState,
    configurable: dict[str, Any],
    code: str,
    error_message: str,
    issues: list[str],
) -> None:
    """Create attempt notebook for static analysis failures."""
    try:
        # Set up file and notebook managers
        file_manager = FileManager(configurable)
        notebook_manager = NotebookManager(configurable)

        # Ensure execution folder exists
        execution_folder = state.get("execution_folder")
        if not execution_folder:
            execution_folder = file_manager.create_execution_folder(
                state["request"].execution_folder_name
            )
            state["execution_folder"] = execution_folder

        # Save context to file if not already saved
        if execution_folder and not execution_folder.context_file_path:
            try:
                from osprey.context.context_manager import ContextManager
                from osprey.utils.config import get_config_value

                context_manager = ContextManager(state)

                # Add execution config snapshot for reproducibility
                execution_config = {}

                # Snapshot control system config
                control_system_config = get_config_value("control_system", {})
                if control_system_config:
                    execution_config["control_system"] = control_system_config

                # Snapshot Python executor config
                python_executor_config = get_config_value("python_executor", {})
                if python_executor_config:
                    execution_config["python_executor"] = python_executor_config

                # Add execution config to context
                context_manager.add_execution_config(execution_config)

                context_file_path = context_manager.save_context_to_file(
                    execution_folder.folder_path
                )
                execution_folder.context_file_path = context_file_path
            except Exception as e:
                logger.warning(f"Failed to save context: {e}")

        # Create detailed error context for the notebook
        error_context = f"""**Static Analysis Failed**

**Error:** {error_message}

**Issues Found:**
{chr(10).join(f'- {issue}' for issue in issues)}

**Debug Information:**
- Stage: Static analysis
- Generated Code Length: {len(code)} characters
- Analysis Result: Failed validation

**Note:** This notebook contains the code that failed static analysis. Review the issues above and regenerate the code accordingly."""

        # Create attempt notebook
        notebook_path = notebook_manager.create_attempt_notebook(
            context=execution_folder,
            code=code,
            stage="static_analysis_failed",
            error_context=error_context,
            silent=True,  # Don't log creation as we'll log it below
        )

        logger.info(f"📝 Created attempt notebook for static analysis failure: {notebook_path}")

    except Exception as e:
        logger.warning(f"Failed to create attempt notebook for static analysis failure: {e}")
        # Don't fail the entire analysis just because notebook creation failed


async def _create_syntax_error_attempt_notebook(
    state: PythonExecutionState, configurable: dict[str, Any], code: str, error_message: str
) -> None:
    """Create attempt notebook for syntax errors."""
    try:
        # Set up file and notebook managers
        file_manager = FileManager(configurable)
        notebook_manager = NotebookManager(configurable)

        # Ensure execution folder exists
        execution_folder = state.get("execution_folder")
        if not execution_folder:
            execution_folder = file_manager.create_execution_folder(
                state["request"].execution_folder_name
            )
            state["execution_folder"] = execution_folder

        # Save context to file if not already saved
        if execution_folder and not execution_folder.context_file_path:
            try:
                from osprey.context.context_manager import ContextManager
                from osprey.utils.config import get_config_value

                context_manager = ContextManager(state)

                # Add execution config snapshot for reproducibility
                execution_config = {}

                # Snapshot control system config
                control_system_config = get_config_value("control_system", {})
                if control_system_config:
                    execution_config["control_system"] = control_system_config

                # Snapshot Python executor config
                python_executor_config = get_config_value("python_executor", {})
                if python_executor_config:
                    execution_config["python_executor"] = python_executor_config

                # Add execution config to context
                context_manager.add_execution_config(execution_config)

                context_file_path = context_manager.save_context_to_file(
                    execution_folder.folder_path
                )
                execution_folder.context_file_path = context_file_path
            except Exception as e:
                logger.warning(f"Failed to save context: {e}")

        # Create detailed error context for the notebook
        error_context = f"""**Syntax Error Detected**

**Error:** {error_message}

**Debug Information:**
- Stage: Code syntax validation
- Generated Code Length: {len(code)} characters
- Error Type: Syntax validation failed

**Note:** This notebook contains the code that has syntax errors. The code below will not execute properly and needs to be corrected."""

        # Create attempt notebook
        notebook_path = notebook_manager.create_attempt_notebook(
            context=execution_folder,
            code=code,
            stage="syntax_error",
            error_context=error_context,
            silent=True,  # Don't log creation as we'll log it below
        )

        logger.info(f"📝 Created attempt notebook for syntax error: {notebook_path}")

    except Exception as e:
        logger.warning(f"Failed to create attempt notebook for syntax error: {e}")
        # Don't fail the entire analysis just because notebook creation failed


async def _create_pre_approval_notebook(
    state: PythonExecutionState,
    configurable: dict[str, Any],
    code: str,
    analysis_result: BasicAnalysisResult,
) -> tuple[Any, Any, str]:
    """Create pre-approval notebook for user review with clickable Jupyter link.

    This function creates the execution folder and a review notebook when code
    requires human approval. The notebook contains the generated code and analysis
    context, giving users a proper environment to review before approving.

    Returns:
        tuple: (execution_folder, notebook_path, notebook_link)
    """
    try:
        # Set up file and notebook managers
        file_manager = FileManager(configurable)
        notebook_manager = NotebookManager(configurable)

        # Ensure execution folder exists
        execution_folder = state.get("execution_folder")
        if not execution_folder:
            execution_folder = file_manager.create_execution_folder(
                state["request"].execution_folder_name
            )

        # Save context to file if not already saved
        # This ensures context.json exists before creating the notebook
        if not execution_folder.context_file_path:
            try:
                from osprey.context.context_manager import ContextManager
                from osprey.utils.config import get_config_value

                context_manager = ContextManager(state)

                # Add execution config snapshot for reproducibility
                execution_config = {}

                # Snapshot control system config
                control_system_config = get_config_value("control_system", {})
                if control_system_config:
                    execution_config["control_system"] = control_system_config

                # Snapshot Python executor config
                python_executor_config = get_config_value("python_executor", {})
                if python_executor_config:
                    execution_config["python_executor"] = python_executor_config

                # Add execution config to context
                context_manager.add_execution_config(execution_config)

                context_file_path = context_manager.save_context_to_file(
                    execution_folder.folder_path
                )
                # Update execution context with the saved context file path
                execution_folder.context_file_path = context_file_path
            except Exception as e:
                logger.warning(f"Failed to save context in pre-approval: {e}")
                # Don't fail the entire pre-approval process for context saving issues

        # Format execution mode for display
        execution_mode = (
            analysis_result.recommended_execution_mode.value
            if hasattr(analysis_result.recommended_execution_mode, "value")
            else str(analysis_result.recommended_execution_mode)
        )

        # Create approval context for the notebook
        approval_context = f"""**Code Analysis Passed - Approval Required**

**Execution Mode:** {execution_mode}

**Analysis Summary:**
- Safety Assessment: {'PASSED' if analysis_result.passed else 'FAILED'}
- Approval Required: {analysis_result.needs_approval}

**Safety Considerations:**
{chr(10).join(f'- {issue}' for issue in (analysis_result.issues + analysis_result.recommendations)) if (analysis_result.issues + analysis_result.recommendations) else '- No specific concerns identified'}

**Next Steps:**
Review the code below and approve/reject execution accordingly."""

        # Create pre-approval notebook for user review
        notebook_path = notebook_manager.create_attempt_notebook(
            context=execution_folder,
            code=code,
            stage="awaiting_approval",
            approval_context=approval_context,
            silent=True,  # Don't log creation as we'll log it below
        )

        # Generate Jupyter notebook link for user
        notebook_link = file_manager._create_jupyter_url(notebook_path)

        logger.info(f"📓 Created pre-approval notebook for user review: {notebook_path}")
        logger.info(f"🔗 Notebook link: {notebook_link}")

        return execution_folder, notebook_path, notebook_link

    except Exception as e:
        logger.warning(f"Failed to create pre-approval notebook: {e}")
        # Fall back to None values if notebook creation fails
        # This maintains backward compatibility but logs the failure
        execution_folder = state.get("execution_folder")
        return execution_folder, None, None
