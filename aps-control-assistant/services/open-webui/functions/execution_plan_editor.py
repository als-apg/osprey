"""
title: Execution Plan Editor
author: ALS Assistant Team
version: 1.0.0
required_open_webui_version: 0.5.1
icon_url: data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTkgNWgxdjE0SDlWNXptMy0yaC0xdjJoMVYzem0yIDBIOXYyaDVWM3ptMi0zSDlWMkg5djE0aDJWNmgxdjEwaDJWNmgxdjEwaDJWNmgxdjEwaDJWNmgxdjEwaDJWNmgxdjEwaDJWNiIgZmlsbD0iY3VycmVudENvbG9yIi8+CjxwYXRoIGQ9Ik0xMS41IDEyLjVsMyAzIDcuNS03LjUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgZmlsbD0ibm9uZSIvPgo8L3N2Zz4K
Description: Interactive execution plan editor for ALS Assistant Agent. Create, modify, and validate multi-step execution plans with visual dependency management and real-time validation.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Set up logger
logger = logging.getLogger(__name__)

# Import the config system for consistent configuration
try:
    # Add the framework source path to enable imports
    framework_src = Path("/app/src")
    if framework_src.exists():
        sys.path.insert(0, str(framework_src))

    from osprey.utils.config import get_agent_dir

    def get_execution_plans_path() -> Path:
        """Get the execution plans directory path using config system.

        This uses the exact same configuration system as the framework,
        ensuring complete consistency with the orchestrator.

        Returns:
            Path to the execution plans directory
        """
        execution_plans_dir = get_agent_dir("execution_plans_dir")
        return Path(execution_plans_dir)

    logger.info("Using config system for path resolution")

except ImportError as e:
    logger.warning(f"Could not import config system: {e}")
    logger.warning("Falling back to direct configuration loading")

    def get_execution_plans_path() -> Path:
        """Fallback path resolution when config is not available."""
        # Use hardcoded path that matches standard framework configuration
        return Path("/app/_agent_data/execution_plans")

# Registry data loading functionality - uses real registry data instead of dummy data
def load_registry_data(agent_data_dir: str = None) -> dict:
    """
    Load registry data from JSON files exported by the registry system.

    Args:
        agent_data_dir: Base directory for agent data (defaults to mounted path)

    Returns:
        Dictionary containing success flag, data/error info, and capabilities/context types/templates
    """
    try:
        # Use config system for consistent paths
        if agent_data_dir is None:
            try:
                # Use config to get the exact same path as the framework
                agent_data_dir = get_agent_dir("registry_exports_dir")
                # Get parent directory (agent_data_dir) since registry_exports is a subdirectory
                agent_data_dir = str(Path(agent_data_dir).parent)
            except Exception:
                # Fallback to standard path if config fails
                agent_data_dir = "/app/_agent_data"
                logger.warning(f"Using fallback path: {agent_data_dir}")

        registry_exports_dir = Path(agent_data_dir) / "registry_exports"

        # Load the complete registry export
        registry_export_file = registry_exports_dir / "registry_export.json"

        if registry_export_file.exists():
            with open(registry_export_file, encoding='utf-8') as f:
                registry_data = json.load(f)

            logger.info(f"Loaded registry data from: {registry_export_file}")
            logger.info(f"Registry data contains: {len(registry_data.get('capabilities', []))} capabilities, "
                       f"{len(registry_data.get('context_types', []))} context types, "
                       f"{len(registry_data.get('templates', []))} templates")

            return {
                "success": True,
                "capabilities": registry_data.get('capabilities', []),
                "context_types": registry_data.get('context_types', []),
                "templates": registry_data.get('templates', [])
            }
        else:
            error_msg = f"Registry export file not found: {registry_export_file}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "capabilities": [],
                "context_types": [],
                "templates": []
            }

    except Exception as e:
        error_msg = f"Failed to load registry data: {e}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "capabilities": [],
            "context_types": [],
            "templates": []
        }


class Action:
    class Valves(BaseModel):
        plans_data_path: str = Field(
            default=os.getenv("PLAN_EDITOR_DATA", "/app/data/plan_editor"),
            description="Directory for execution plan storage (must be mounted to host)"
        )
        max_plans_per_user: int = Field(
            default=100,
            description="Maximum number of saved plans per user"
        )

    class UserValves(BaseModel):
        show_validation_warnings: bool = Field(
            default=True,
            description="Show validation warnings in addition to errors"
        )
        auto_validate: bool = Field(
            default=True,
            description="Automatically validate plans as they are edited"
        )
        enable_templates: bool = Field(
            default=True,
            description="Enable template loading and saving"
        )
        enable_advanced_features: bool = Field(
            default=True,
            description="Enable advanced features like dependency visualization"
        )

    def __init__(self):
        self.valves = self.Valves()

    def extract_context_summary_from_messages(self, messages: list) -> dict[str, Any] | None:
        """Extract agent context summary from assistant messages."""

        try:
            logger.info(f"Extracting context from {len(messages)} messages")

            # Look through messages in reverse order (most recent first)
            for i, message in enumerate(reversed(messages)):
                try:
                    logger.debug(f"Checking message {i}: role={message.get('role')}, has_info={message.get('info') is not None}")

                    if message.get("role") == "assistant" and message.get("info"):
                        info_keys = list(message["info"].keys())
                        logger.debug(f"Message {i} info keys: {info_keys}")

                        # Check for context summary (check both old and new key names for compatibility)
                        if "als_assistant_agent_context" in message["info"]:
                            context_data = message["info"]["als_assistant_agent_context"]
                            logger.info(f"Found agent context with {context_data.get('total_context_items', 0)} items")
                            return context_data
                        elif "als_assistant_context_summary" in message["info"]:
                            context_data = message["info"]["als_assistant_context_summary"]
                            logger.info(f"Found agent context with {len(context_data.get('context_details', {}))} categories")
                            return context_data

                except Exception as e:
                    logger.error(f"Error processing message {i}: {e}")
                    continue

            logger.info("No agent context found in any message")
            return None

        except Exception as e:
            logger.error(f"Error extracting context from messages: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def extract_available_context_keys(self, context_summary: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract available context keys from agent context summary."""
        available_contexts = []

        try:
            # Handle both new and old data formats
            context_data = context_summary.get("context_data") or context_summary.get("context_details", {})

            if not context_data:
                return available_contexts

            # Process each context category
            for context_type, contexts_dict in context_data.items():
                # Process each context item in this category
                for context_key, context_info in contexts_dict.items():
                    context_type_name = context_info.get("type", "Unknown")

                    # Map context types to standard names
                    standard_type = self._map_context_type(context_type_name)

                    available_contexts.append({
                        "contextKey": context_key,
                        "contextType": standard_type,
                        "source": "agent_context",
                        "description": context_info.get("description", ""),
                        "category": context_type
                    })

            logger.info(f"Extracted {len(available_contexts)} context keys from agent context")
            return available_contexts

        except Exception as e:
            logger.error(f"Error extracting context keys: {e}")
            return available_contexts

    def _map_context_type(self, context_type_name: str) -> str:
        """Map agent context types to standard context type names."""
        type_mapping = {
            "PV Addresses": "PV_ADDRESSES",
            "Time Range": "TIME_RANGE",
            "PV Values": "PV_VALUES",
            "Archiver Data": "ARCHIVER_DATA",
            "Analysis Results": "ANALYSIS_RESULTS",
            "Visualization Results": "VISUALIZATION_RESULTS",
            "Operation Results": "OPERATION_RESULTS",
            "Memory Context": "MEMORY_CONTEXT",
            "Conversation Results": "CONVERSATION_RESULTS"
        }

        return type_mapping.get(context_type_name, context_type_name.upper().replace(" ", "_"))

    def _get_user_id(self, user_info: dict) -> str:
        """Get user ID from user info, using email prefix if available."""
        if not user_info:
            raise ValueError("User information not available")

        user_id = user_info.get("id")

        if user_info.get("email"):
            user_email = user_info.get("email")
            if "@" in user_email:
                user_id = user_email.split("@")[0]

        if not user_id:
            raise ValueError("User ID not available")

        return user_id

    def _get_user_plans_directory(self, user_id: str) -> Path:
        """Get the directory for user's saved plans."""
        # Sanitize user_id for filename
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
        return Path(self.valves.plans_data_path) / safe_user_id

    def _validate_plan(self, plan_steps: list[dict], available_context_keys: list[dict] = None) -> dict[str, Any]:
        """Enhanced validation of execution plan using registry data and agent context."""
        errors = []
        warnings = []

        if not plan_steps:
            return {
                "is_valid": False,
                "errors": ["Plan cannot be empty"],
                "warnings": []
            }

        # Load registry data for validation
        registry_data = load_registry_data()
        if not registry_data["success"]:
            errors.append(f"Cannot validate plan: {registry_data['error']}")
            return {
                "is_valid": False,
                "errors": errors,
                "warnings": warnings
            }

        valid_capabilities = {cap["name"] for cap in registry_data.get('capabilities', [])}
        valid_context_types = {ctx["type_name"] for ctx in registry_data.get('context_types', [])}

        # Track context keys and their types (including from agent context)
        context_key_types = {}

        # Add available context keys from agent context
        if available_context_keys:
            for ctx in available_context_keys:
                context_key_types[ctx["contextKey"]] = ctx["contextType"]

        for i, step in enumerate(plan_steps):
            step_id = f"Step {i+1} ({step.get('context_key', 'unknown')})"

            # Check required fields
            required_fields = ["context_key", "capability", "task_objective", "expected_output"]
            for field in required_fields:
                if not step.get(field):
                    errors.append(f"{step_id}: Missing required field '{field}'")

            # Check if capability exists
            capability = step.get("capability")
            if capability and capability not in valid_capabilities:
                errors.append(f"{step_id}: Unknown capability '{capability}'")

            # Check if expected output is valid
            expected_output = step.get("expected_output")
            if expected_output and expected_output not in valid_context_types:
                errors.append(f"{step_id}: Unknown context type '{expected_output}'")

            # Check if inputs reference valid context keys
            if step.get("inputs"):
                for input_item in step["inputs"]:
                    for context_type, context_key in input_item.items():
                        if context_key not in context_key_types:
                            errors.append(f"{step_id}: Input references unknown context key '{context_key}'")

            # Track context key from this step
            context_key = step.get("context_key")
            if context_key:
                if context_key in context_key_types:
                    # Check if it's from agent context or duplicate in plan
                    if any(ctx["contextKey"] == context_key for ctx in (available_context_keys or [])):
                        warnings.append(f"{step_id}: Context key '{context_key}' already exists in agent context")
                    else:
                        errors.append(f"{step_id}: Duplicate context key '{context_key}'")
                else:
                    context_key_types[context_key] = expected_output

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _save_plan(self, plan_steps: list[dict], user_id: str) -> dict[str, Any]:
        """Save execution plan to file."""
        try:
            user_plans_dir = self._get_user_plans_directory(user_id)
            user_plans_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"execution_plan_{timestamp}.json"
            file_path = user_plans_dir / filename

            plan_data = {
                "__metadata__": {
                    "version": "1.0",
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat(),
                    "serialization_type": "execution_plan"
                },
                "steps": plan_steps
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(plan_data, f, indent=2, ensure_ascii=False)

            return {
                "success": True,
                "filename": filename,
                "path": str(file_path)
            }
        except Exception as e:
            logger.error(f"Error saving execution plan: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def check_pending_plan(self, __event_emitter__=None, __event_call__=None) -> dict:
        """Check if there's a pending execution plan awaiting approval."""
        try:
            # Load registry data to check file paths
            registry_data = load_registry_data()
            if not registry_data["success"]:
                return {"has_pending": False, "error": "Registry data not available"}

            # Use framework configuration for consistent paths
            execution_plans_dir = get_execution_plans_path()
            pending_plans_dir = execution_plans_dir / "pending_plans"
            plan_file = pending_plans_dir / "pending_execution_plan.json"

            if plan_file.exists():
                with open(plan_file, encoding='utf-8') as f:
                    plan_data = json.load(f)
                return {
                    "has_pending": True,
                    "plan_data": plan_data,
                    "message": "Found pending execution plan for review"
                }
            else:
                return {"has_pending": False}

        except Exception as e:
            logger.error(f"Error checking for pending plan: {e}")
            return {"has_pending": False, "error": str(e)}

    async def save_modified_plan(self, plan_data: dict, __event_emitter__=None, __event_call__=None) -> dict:
        """Save modified execution plan for approval processing."""
        try:
            # Use framework configuration for consistent paths
            execution_plans_dir = get_execution_plans_path()
            pending_plans_dir = execution_plans_dir / "pending_plans"
            pending_plans_dir.mkdir(parents=True, exist_ok=True)

            modified_plan_file = pending_plans_dir / "modified_execution_plan.json"

            # Ensure plan_data has the correct format
            if 'steps' not in plan_data or '__metadata__' not in plan_data:
                # If it's just a steps array, wrap it properly
                if isinstance(plan_data, list):
                    plan_data = {
                        "__metadata__": {
                            "version": "1.0",
                            "modified_at": datetime.now().isoformat(),
                            "serialization_type": "modified_execution_plan"
                        },
                        "steps": plan_data
                    }
                else:
                    # Add metadata if missing
                    plan_data["__metadata__"] = {
                        "version": "1.0",
                        "modified_at": datetime.now().isoformat(),
                        "serialization_type": "modified_execution_plan"
                    }
            else:
                # Update existing metadata
                plan_data["__metadata__"]["modified_at"] = datetime.now().isoformat()

            with open(modified_plan_file, 'w', encoding='utf-8') as f:
                json.dump(plan_data, f, indent=2, ensure_ascii=False)

            return {"success": True, "message": f"Modified plan saved with {len(plan_data.get('steps', []))} steps"}

        except Exception as e:
            logger.error(f"Error saving modified plan: {e}")
            return {"success": False, "error": str(e)}

    async def create_plan_editor_interface(
        self,
        __event_emitter__=None,
        __event_call__=None,
        user_id: str = None,
        body: dict = None
    ) -> dict | None:
        """Create interactive execution plan editor using JavaScript."""

        # Check for pending plan first
        pending_check = await self.check_pending_plan(__event_emitter__, __event_call__)

        # Load real registry data
        registry_data = load_registry_data()

        # Set up registry data - use empty arrays if not available
        has_registry_data = registry_data["success"]

        # Extract agent context from messages
        agent_context_summary = None
        available_context_keys = []

        if body and body.get("messages"):
            agent_context_summary = self.extract_context_summary_from_messages(body.get("messages", []))
            if agent_context_summary:
                available_context_keys = self.extract_available_context_keys(agent_context_summary)
                logger.info(f"Found {len(available_context_keys)} context keys from agent context")

        # Prepare JSON data for the editor (empty arrays if no data)
        capabilities_json = json.dumps(registry_data.get('capabilities', [])).replace('"', '\\"')
        context_types_json = json.dumps(registry_data.get('context_types', [])).replace('"', '\\"')
        templates_json = json.dumps(registry_data.get('templates', [])).replace('"', '\\"')
        available_context_keys_json = json.dumps(available_context_keys).replace('"', '\\"')

        # Prepare pending plan data
        pending_plan_json = "null"
        editor_mode = "normal"
        if pending_check.get("has_pending"):
            pending_plan_json = json.dumps(pending_check["plan_data"]).replace('"', '\\"')
            editor_mode = "approval_review"

        # Prepare error message if registry data is missing
        error_message = ""
        if not has_registry_data:
            escaped_error = registry_data["error"].replace("'", "\\'")
            error_message = f'''
            // Show error alert immediately when editor loads
            setTimeout(function() {{
                alert(`⚠️ Registry Data Not Available

Error: {escaped_error}

🔧 Troubleshooting Steps:

1. Check Registry System
   • Ensure the ALS Assistant agent is running
   • Verify capabilities are properly registered
   • Check that registry initialization is working

2. Export Registry Data
   • Send a message to the ALS Assistant agent to trigger registry export
   • Registry data is exported every time the agent processes a message

3. Check File Paths
   • Verify files exist at: /app/_agent_data/registry_exports/
   • Expected files: registry_export.json, capabilities.json, context_types.json, templates.json

4. Container Mount Issues
   • If running in a container, ensure _agent_data directory is properly mounted
   • Check that the shared volume is accessible between the agent and Open WebUI
   • Verify file permissions allow read access

Please reload the page after fixing the registry data.`);
            }}, 500);
            '''

        try:

            # Emit status
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Loading execution plan editor...", "done": False},
                }
            )

            editor_js = f"""
            try {{
                {error_message}

                // ===== PROFESSIONAL SCIENTIFIC INTERFACE DESIGN CONFIGURATION =====

                // COLOR PALETTE - Professional Scientific Interface
                const colors = {{
                    primary: '#9eb0af',        // Muted blue-gray - primary actions, main borders
                    primaryHover: '#7a9291',   // Darker shade for hover states
                    secondary: '#95a4b8',      // Steel blue - secondary actions, accents
                    secondaryHover: '#7a8aa0', // Darker shade for hover states
                    success: '#9eb0af',        // Success states (use primary)
                    successHover: '#7a9291',   // Success hover states
                    danger: '#d66e6e',         // Muted red - warnings, errors, delete actions
                    dangerHover: '#c54545',    // Darker red for hover states
                    warning: '#95a4b8',        // Steel blue for warnings (use secondary)
                    warningHover: '#7a8aa0',   // Warning hover states
                    background: '#ffffff',     // Clean white - main backgrounds
                    panelBackground: '#f0d3d3', // Light pink - panel/card backgrounds
                    border: '#9eb0af',         // Primary color - main structural borders
                    borderLight: '#e0e5e5',    // Light gray - subtle dividers, field borders
                    text: '#000000',           // Pure black - primary text (high contrast)
                    textLight: '#4a4a4a',      // Dark gray - secondary text, labels
                    accent: '#95a4b8',         // Steel blue - highlights, badges
                    accentHover: '#7a8aa0'     // Accent hover states
                }};

                // LAYOUT DIMENSIONS - Main Interface Structure
                const layout = {{
                    // Main Container
                    containerMaxWidth: '1400px',      // Maximum editor width
                    containerWidth: '95%',            // Responsive width percentage
                    containerMaxHeight: '90vh',       // Maximum editor height
                    containerPadding: '32px',         // Main container internal padding
                    containerBorderWidth: '2px',      // Main container border thickness

                    // Header Section
                    headerBottomMargin: '32px',       // Space below header
                    headerBottomPadding: '20px',      // Header internal bottom padding
                    headerBorderWidth: '1px',         // Header divider line thickness

                    // Control Panel (buttons area)
                    controlPanelPadding: '20px',      // Control panel internal padding
                    controlPanelBottomMargin: '24px', // Space below control panel
                    controlPanelGap: '24px',          // Gap between left and right button groups
                    buttonSpacing: '12px',            // Space between individual buttons

                    // Main Content Area
                    contentGap: '28px',               // Gap between steps and capabilities panels
                    sectionHeaderBottomMargin: '16px', // Space below section headers
                    sectionHeaderBottomPadding: '8px', // Section header internal bottom padding

                    // Steps Container
                    stepsContainerMinHeight: '300px', // Minimum height for steps area
                    stepsContainerPadding: '20px',    // Steps container internal padding
                    stepCardBottomMargin: '16px',     // Space between step cards
                    stepCardPadding: '20px',          // Step card internal padding
                    stepCardHeaderBottomMargin: '16px', // Space below step card header
                    stepCardHeaderBottomPadding: '12px', // Step card header internal bottom padding
                    stepCardContentGap: '16px',       // Gap between step card content columns

                    // Capabilities Panel
                    capabilitiesPanelMaxHeight: '500px', // Maximum height for capabilities list
                    capabilitiesPanelPadding: '20px',    // Capabilities panel internal padding
                    capabilityCardBottomMargin: '12px',  // Space between capability cards
                    capabilityCardPadding: '16px',       // Capability card internal padding

                    // Validation Status
                    validationStatusPadding: '16px',     // Validation message internal padding
                    validationStatusBottomMargin: '20px', // Space below validation message
                    validationStatusBorderWidth: '1px'    // Validation message border thickness
                }};

                // COMPONENT SIZING - Individual Interface Elements
                const sizing = {{
                    // Buttons
                    buttonPadding: '12px 20px',       // Button internal padding (vertical horizontal)
                    buttonMinWidth: '120px',          // Minimum button width for consistency
                    buttonBorderWidth: '1px',         // Button border thickness
                    stepButtonPadding: '6px 12px',    // Smaller buttons in step cards

                    // Form Fields
                    fieldPadding: '8px',              // Input/select field internal padding
                    fieldBottomMargin: '6px',         // Space below field labels
                    fieldContentPadding: '8px',       // Content area padding in fields
                    fieldMinHeight: '36px',           // Minimum height for input fields

                    // Input Badges
                    badgePadding: '4px 8px',          // Input badge internal padding
                    badgeSpacing: '6px',              // Space between input badges
                    badgeBottomMargin: '4px',         // Bottom margin for input badges

                    // Modal Dialogs
                    modalWidth: '90%',                // Modal dialog width percentage
                    modalMaxWidth: '600px',           // Maximum modal dialog width
                    modalMaxHeight: '80vh',           // Maximum modal dialog height
                    modalPadding: '24px',             // Modal dialog internal padding
                    modalFieldBottomMargin: '16px',   // Space between modal form fields
                    modalButtonGap: '12px'            // Gap between modal buttons
                }};

                // TYPOGRAPHY - Text Styling and Spacing
                const typography = {{
                    // Main Headers
                    mainHeaderFontSize: '20px',       // Main editor title
                    mainHeaderLetterSpacing: '0.5px', // Main header letter spacing
                    sectionHeaderFontSize: '16px',    // Section headers (Steps, Capabilities)
                    sectionHeaderLetterSpacing: '0.5px', // Section header letter spacing

                    // Body Text
                    bodyFontSize: '14px',             // General body text
                    bodyLineHeight: '1.5',            // Body text line height
                    smallTextFontSize: '13px',        // Smaller text in cards
                    smallTextLineHeight: '1.5',       // Small text line height

                    // Labels and Buttons
                    labelFontSize: '12px',            // Field labels
                    labelLetterSpacing: '0.5px',      // Label letter spacing
                    buttonFontSize: '13px',           // Button text
                    buttonLetterSpacing: '0.5px',     // Button letter spacing
                    stepButtonFontSize: '11px',       // Smaller buttons in step cards
                    stepButtonLetterSpacing: '0.5px', // Step button letter spacing

                    // Capability Cards
                    capabilityTitleFontSize: '13px',  // Capability card titles
                    capabilityTitleLetterSpacing: '0.5px', // Capability title letter spacing
                    capabilityDescFontSize: '12px',   // Capability descriptions
                    capabilityDescLineHeight: '1.4',  // Capability description line height
                    capabilityMetaFontSize: '11px',   // Capability provides/requires text
                    capabilityMetaLineHeight: '1.4',  // Capability meta text line height

                    // Step Cards
                    stepTitleFontSize: '11px',        // Step number badge
                    stepTitleLetterSpacing: '0.5px',  // Step badge letter spacing
                    stepCapabilityFontSize: '14px',   // Step capability name
                    stepCapabilityLetterSpacing: '0.3px', // Step capability letter spacing

                    // Input Badges
                    badgeFontSize: '11px',            // Input badge text
                    badgeLetterSpacing: '0.3px',      // Input badge letter spacing

                    // Validation and Status
                    validationFontSize: '13px',       // Validation message text
                    validationLetterSpacing: '0.3px', // Validation message letter spacing

                    // Misc
                    subtitleFontSize: '13px',         // Subtitle text
                    noInputsFontSize: '11px',         // "No inputs configured" text
                    noInputsLetterSpacing: '0.3px'    // "No inputs" letter spacing
                }};

                // BORDER RADIUS - Corner Rounding (Professional = Minimal)
                const borderRadius = {{
                    container: '4px',                 // Main container corners
                    panel: '3px',                     // Panel and card corners
                    button: '3px',                    // Button corners
                    field: '3px',                     // Input field corners
                    badge: '3px'                      // Badge corners
                }};

                // SHADOWS - Subtle Depth Effects
                const shadows = {{
                    container: '0 8px 16px rgba(0, 0, 0, 0.1)',     // Main container shadow
                    card: '0 2px 4px rgba(0,0,0,0.05)',             // Card shadow
                    cardHover: '0 4px 8px rgba(0,0,0,0.1)',         // Card hover shadow
                    modal: '0 25px 50px -12px rgba(0, 0, 0, 0.25)'  // Modal dialog shadow
                }};

                // TRANSITIONS - Smooth Animations
                const transitions = {{
                    default: 'all 0.2s',             // Standard transition for most elements
                    fast: '0.15s',                    // Fast transitions
                    slow: '0.3s'                      // Slower transitions for complex changes
                }};

                // ===== ======================== =====
                // ===== END DESIGN CONFIGURATION =====

                // Remove any existing editor
                const existingEditor = document.getElementById('plan-editor-overlay');
                if (existingEditor) {{
                    existingEditor.remove();
                }}

                // Create overlay
                const overlay = document.createElement('div');
                overlay.id = 'plan-editor-overlay';
                overlay.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.8);
                    z-index: 10000;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                `;

                // Create editor container - professional scientific interface
                const editor = document.createElement('div');
                editor.style.cssText = `
                    background: ${{colors.background}};
                    border: ${{layout.containerBorderWidth}} solid ${{colors.border}};
                    border-radius: ${{borderRadius.container}};
                    padding: ${{layout.containerPadding}};
                    width: ${{layout.containerWidth}};
                    max-width: ${{layout.containerMaxWidth}};
                    max-height: ${{layout.containerMaxHeight}};
                    overflow-y: auto;
                    box-shadow: ${{shadows.container}};
                    color: ${{colors.text}};
                    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                    font-size: ${{typography.bodyFontSize}};
                    line-height: ${{typography.bodyLineHeight}};
                `;

                // Parse embedded data
                const capabilities = JSON.parse("{capabilities_json}");
                const contextTypes = JSON.parse("{context_types_json}");
                const templates = JSON.parse("{templates_json}");
                const availableContextKeys = JSON.parse("{available_context_keys_json}");
                const pendingPlan = JSON.parse("{pending_plan_json}");
                const editorMode = "{editor_mode}";

                let currentPlan = [];

                // Create step counter for unique IDs
                let stepCounter = 0;

                // Detect editor mode and setup accordingly
                let isReviewMode = (editorMode === "approval_review" && pendingPlan);
                let reviewModeData = null;

                if (isReviewMode) {{
                    reviewModeData = {{
                        originalTask: pendingPlan.__metadata__?.original_task || "Unknown task",
                        contextKey: pendingPlan.__metadata__?.context_key || "unknown",
                        createdAt: pendingPlan.__metadata__?.created_at || "Unknown time"
                    }};
                    currentPlan = pendingPlan.steps || [];
                    stepCounter = currentPlan.length;
                }}

                // Populate template dropdown
                function populateTemplateDropdown() {{
                    const templateSelect = document.getElementById('plan-template-select');
                    if (templateSelect) {{
                        templates.forEach((template, index) => {{
                            const option = document.createElement('option');
                            option.value = index;
                            option.textContent = `${{template.name}} - ${{template.description}}`;
                            templateSelect.appendChild(option);
                        }});
                    }}
                }}

                // Create main editor HTML based on mode
                function createEditorHTML() {{
                    let headerHTML = '';
                    let controlPanelHTML = '';

                    if (isReviewMode) {{
                        // Review mode header
                        headerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: ${{layout.headerBottomMargin}}; padding-bottom: ${{layout.headerBottomPadding}}; border-bottom: ${{layout.headerBorderWidth}} solid ${{colors.border}};">
                                <div>
                                    <h2 style="margin: 0; color: ${{colors.warning}}; font-size: ${{typography.mainHeaderFontSize}}; font-weight: 600; letter-spacing: ${{typography.mainHeaderLetterSpacing}}; text-transform: uppercase;">📋 PLAN REVIEW MODE</h2>
                                    <p style="margin: 8px 0 0 0; color: ${{colors.textLight}}; font-size: ${{typography.subtitleFontSize}}; font-weight: 400;">Review orchestrator-generated plan • Modify if needed • Return to chat to approve</p>
                                </div>
                                <button id="plan-close-btn" style="background: ${{colors.danger}}; color: white; border: ${{sizing.buttonBorderWidth}} solid ${{colors.danger}}; padding: ${{sizing.buttonPadding}}; border-radius: ${{borderRadius.button}}; cursor: pointer; font-weight: 500; font-size: ${{typography.buttonFontSize}}; text-transform: uppercase; letter-spacing: ${{typography.buttonLetterSpacing}}; transition: ${{transitions.default}}; min-width: 80px;">Close</button>
                            </div>
                        `;

                        // Review mode control panel
                        controlPanelHTML = `
                            <div style="background: ${{colors.warning}}; color: white; padding: 16px; border-radius: 6px; margin-bottom: 20px;">
                                <div style="font-weight: 600; margin-bottom: 8px;">Original Task:</div>
                                <div style="font-size: 14px; opacity: 0.9;">${{reviewModeData.originalTask}}</div>
                                <div style="font-size: 12px; margin-top: 8px; opacity: 0.7;">Created: ${{reviewModeData.createdAt}} • Context: ${{reviewModeData.contextKey}}</div>
                            </div>

                            <div style="display: flex; gap: 16px; margin-bottom: 24px; padding: 20px; background: ${{colors.panelBackground}}; border: 1px solid ${{colors.borderLight}}; border-radius: 3px; align-items: center; flex-wrap: nowrap;">
                                <button id="save-as-is-btn" style="background: ${{colors.success}}; color: white; border: 1px solid ${{colors.success}}; padding: 10px 16px; border-radius: 3px; cursor: pointer; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s; min-width: 120px;">💾 Use Plan As-Is</button>
                                <button id="save-modified-btn" style="background: ${{colors.primary}}; color: white; border: 1px solid ${{colors.primary}}; padding: 10px 16px; border-radius: 3px; cursor: pointer; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s; min-width: 140px;">✏️ Save Modifications</button>
                                <div style="margin-left: auto; color: ${{colors.textLight}}; font-size: 13px; text-align: right;">
                                    After saving, return to chat and respond with<br><strong>"yes"</strong> to approve the plan
                                </div>
                            </div>
                        `;
                    }} else {{
                        // Normal mode header
                        headerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: ${{layout.headerBottomMargin}}; padding-bottom: ${{layout.headerBottomPadding}}; border-bottom: ${{layout.headerBorderWidth}} solid ${{colors.border}};">
                                <div>
                                    <h2 style="margin: 0; color: ${{colors.text}}; font-size: ${{typography.mainHeaderFontSize}}; font-weight: 600; letter-spacing: ${{typography.mainHeaderLetterSpacing}}; text-transform: uppercase;">Execution Plan Editor</h2>
                                    <p style="margin: 8px 0 0 0; color: ${{colors.textLight}}; font-size: ${{typography.subtitleFontSize}}; font-weight: 400;">Advanced Light Source Expert Agent - Multi-Step Workflow Configuration</p>
                                </div>
                                <button id="plan-close-btn" style="background: ${{colors.danger}}; color: white; border: ${{sizing.buttonBorderWidth}} solid ${{colors.danger}}; padding: ${{sizing.buttonPadding}}; border-radius: ${{borderRadius.button}}; cursor: pointer; font-weight: 500; font-size: ${{typography.buttonFontSize}}; text-transform: uppercase; letter-spacing: ${{typography.buttonLetterSpacing}}; transition: ${{transitions.default}}; min-width: 80px;">Close</button>
                            </div>
                        `;

                        // Normal mode control panel
                        controlPanelHTML = `
                            <div style="display: flex; gap: 20px; margin-bottom: 24px; padding: 20px; background: ${{colors.panelBackground}}; border: 1px solid ${{colors.borderLight}}; border-radius: 3px; align-items: center; flex-wrap: nowrap;">
                                <button id="plan-add-step-btn" style="background: ${{colors.success}}; color: white; border: 1px solid ${{colors.success}}; padding: 10px 16px; border-radius: 3px; cursor: pointer; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s; min-width: 100px;">Add Step</button>
                                <select id="plan-template-select" style="background: ${{colors.accent}}; color: white; border: 1px solid ${{colors.accent}}; padding: 10px 16px; border-radius: 3px; cursor: pointer; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; min-width: 160px;">
                                    <option value="">Load Template</option>
                                </select>
                                <button id="plan-clear-btn" style="background: ${{colors.danger}}; color: white; border: 1px solid ${{colors.danger}}; padding: 10px 16px; border-radius: 3px; cursor: pointer; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s; min-width: 90px;">Clear</button>
                                <button id="plan-validate-btn" style="background: ${{colors.warning}}; color: white; border: 1px solid ${{colors.warning}}; padding: 10px 16px; border-radius: 3px; cursor: pointer; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s; min-width: 90px;">Validate</button>
                                <div style="margin-left: auto; display: flex; gap: 12px;">
                                    <button id="plan-save-btn" style="background: ${{colors.primary}}; color: white; border: 1px solid ${{colors.primary}}; padding: 10px 16px; border-radius: 3px; cursor: pointer; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s; min-width: 100px;">Save Plan</button>
                                    <button id="plan-cancel-btn" style="background: transparent; color: ${{colors.textLight}}; border: 1px solid ${{colors.borderLight}}; padding: 10px 16px; border-radius: 3px; cursor: pointer; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s; min-width: 80px;">Cancel</button>
                                </div>
                            </div>
                        `;
                    }}

                    return headerHTML + controlPanelHTML + `
                        <div id="validation-status" style="display: none; padding: 16px; border-radius: 3px; margin-bottom: 20px; font-size: 13px; font-weight: 500; border: 1px solid; letter-spacing: 0.3px;"></div>

                        <div style="display: flex; gap: 28px;">
                            <div style="flex: 2;">
                                <h3 style="margin: 0 0 16px 0; color: ${{colors.text}}; font-size: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid ${{colors.borderLight}}; padding-bottom: 8px;">Execution Steps</h3>
                                <div id="steps-container" style="border: 1px solid ${{colors.border}}; border-radius: 3px; min-height: 300px; padding: 20px; background: ${{colors.background}};"></div>
                            </div>
                            <div style="flex: 1;">
                                <h3 style="margin: 0 0 16px 0; color: ${{colors.text}}; font-size: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid ${{colors.borderLight}}; padding-bottom: 8px;">Available Context</h3>
                                <div id="context-panel" style="border: 1px solid ${{colors.border}}; border-radius: 3px; max-height: 200px; overflow-y: auto; padding: 20px; background: ${{colors.background}}; margin-bottom: 20px;"></div>

                                <h3 style="margin: 0 0 16px 0; color: ${{colors.text}}; font-size: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid ${{colors.borderLight}}; padding-bottom: 8px;">Available Capabilities</h3>
                                <div id="capabilities-panel" style="border: 1px solid ${{colors.border}}; border-radius: 3px; max-height: 300px; overflow-y: auto; padding: 20px; background: ${{colors.background}};"></div>
                            </div>
                        </div>
                    `;
                }}

                // Set editor HTML
                editor.innerHTML = createEditorHTML();

                // For review mode, we'll show instructions instead of trying to save directly
                function showSaveInstructions(action) {{
                    let message = '';
                    if (action === 'save_as_is') {{
                        message = `✅ Original Plan Ready for Approval!

Return to chat and respond with "yes" to approve the original execution plan.

No modifications were made - the original plan will be used as-is.`;
                    }} else {{
                        message = `✅ Plan modifications ready!

To save your modifications and proceed with approval:

1. Close this editor
2. Return to chat
3. Use the execution plan editor again (this will trigger a save of your modifications)
4. Then respond with "yes" to approve the modified plan

Note: Your modifications are preserved in this session until you reload the page.`;
                    }}

                    alert(message);
                }}

                // Review mode success notification
                function showReviewModeSuccess(message) {{
                    const notification = document.createElement('div');
                    notification.style.cssText = `
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        background: ${{colors.success}};
                        color: white;
                        padding: 16px 24px;
                        border-radius: 8px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                        z-index: 10002;
                        font-size: 14px;
                        font-weight: 500;
                        max-width: 400px;
                        word-wrap: break-word;
                        opacity: 0;
                        transform: translateX(100%);
                        transition: all 0.3s ease;
                    `;
                    notification.textContent = message;
                    document.body.appendChild(notification);

                    // Animate in
                    setTimeout(() => {{
                        notification.style.opacity = '1';
                        notification.style.transform = 'translateX(0)';
                    }}, 10);

                    // Auto-remove after 5 seconds
                    setTimeout(() => {{
                        notification.style.opacity = '0';
                        notification.style.transform = 'translateX(100%)';
                        setTimeout(() => {{
                            if (notification.parentNode) {{
                                notification.parentNode.removeChild(notification);
                            }}
                        }}, 300);
                    }}, 5000);
                }}

                // Setup review mode event handlers
                function setupReviewModeHandlers() {{
                    const saveAsIsBtn = document.getElementById('save-as-is-btn');
                    const saveModifiedBtn = document.getElementById('save-modified-btn');

                    if (saveAsIsBtn) {{
                        saveAsIsBtn.onclick = async () => {{
                            try {{
                                showSaveInstructions('save_as_is');
                                setTimeout(() => overlay.remove(), 500);
                            }} catch (error) {{
                                console.error('Error in save as-is:', error);
                                alert('Error preparing plan for approval. Please try again.');
                            }}
                        }};
                    }}

                    if (saveModifiedBtn) {{
                        saveModifiedBtn.onclick = async () => {{
                            try {{
                                // Store the current plan modifications in browser storage
                                const planData = {{
                                    "__metadata__": {{
                                        "version": "1.0",
                                        "modified_at": new Date().toISOString(),
                                        "context_key": reviewModeData.contextKey,
                                        "serialization_type": "modified_execution_plan"
                                    }},
                                    "steps": currentPlan
                                }};

                                // Store modifications in sessionStorage for persistence within this session
                                sessionStorage.setItem('pendingModifications', JSON.stringify(planData));

                                showSaveInstructions('save_modified');
                                setTimeout(() => overlay.remove(), 500);
                            }} catch (error) {{
                                console.error('Error preparing modified plan:', error);
                                alert(`Error preparing modifications: ${{error.message}}\\n\\nPlease try again.`);
                            }}
                        }};
                    }}
                }}

                // Render available context panel
                function renderAvailableContextPanel() {{
                    const panel = document.getElementById('context-panel');
                    let html = '';

                    if (availableContextKeys.length === 0) {{
                        html = `
                            <div style="text-align: center; padding: 20px; color: ${{colors.textLight}}; font-style: italic; font-size: 12px;">
                                No context data available from previous agent operations.
                            </div>
                        `;
                    }} else {{
                        html = `
                            <div style="margin-bottom: 12px; padding: 8px; background: ${{colors.success}}; color: white; border-radius: 3px; font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; text-align: center;">
                                📊 ${{availableContextKeys.length}} Available Context Keys
                            </div>
                        `;

                        availableContextKeys.forEach(ctx => {{
                            const categoryEmoji = getCategoryEmoji(ctx.category);
                            html += `
                                <div style="border: 1px solid ${{colors.borderLight}}; border-radius: 3px; padding: 12px; margin-bottom: 8px; background: ${{colors.background}}; font-size: 11px; border-left: 3px solid ${{colors.accent}};">
                                    <div style="font-weight: 600; color: ${{colors.text}}; margin-bottom: 4px; display: flex; align-items: center;">
                                        <span style="margin-right: 6px;">${{categoryEmoji}}</span>
                                        <span style="font-family: monospace; background: ${{colors.accent}}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px;">${{ctx.contextKey}}</span>
                                    </div>
                                    <div style="color: ${{colors.textLight}}; margin-bottom: 4px; font-size: 10px;">
                                        <strong>Type:</strong> ${{ctx.contextType}}
                                    </div>
                                    <div style="color: ${{colors.textLight}}; font-size: 10px;">
                                        <strong>Source:</strong> Agent Context
                                    </div>
                                </div>
                            `;
                        }});
                    }}

                    panel.innerHTML = html;
                }}

                // Get category emoji for context display
                function getCategoryEmoji(category) {{
                    const emojiMap = {{
                        "PV_ADDRESSES": "📍",
                        "TIME_RANGE": "⏰",
                        "PV_VALUES": "📊",
                        "ARCHIVER_DATA": "📈",
                        "ANALYSIS_RESULTS": "🔬",
                        "VISUALIZATION_RESULTS": "📊",
                        "OPERATION_RESULTS": "⚙️",
                        "MEMORY_CONTEXT": "🧠",
                        "CONVERSATION_RESULTS": "💬"
                    }};
                    return emojiMap[category] || "📁";
                }}

                // Render capabilities panel
                function renderCapabilitiesPanel() {{
                    const panel = document.getElementById('capabilities-panel');
                    let html = '';

                    capabilities.forEach(cap => {{
                        const providesText = cap.provides && cap.provides.length > 0 ? cap.provides.join(', ') : 'Any';
                        const requiresText = cap.requires && cap.requires.length > 0 ? cap.requires.join(', ') : 'None';

                        html += `
                            <div style="border: 1px solid ${{colors.borderLight}}; border-radius: 3px; padding: 16px; margin-bottom: 12px; background: ${{colors.panelBackground}}; cursor: pointer; transition: all 0.2s; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"
                                 onmouseover="this.style.borderColor='${{colors.primary}}'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.1)'"
                                 onmouseout="this.style.borderColor='${{colors.borderLight}}'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.05)'"
                                 onclick="addStepFromCapability('${{cap.name}}')">
                                <div style="font-weight: 600; color: ${{colors.text}}; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; font-size: 13px;">${{cap.name}}</div>
                                <div style="font-size: 12px; color: ${{colors.textLight}}; margin-bottom: 12px; line-height: 1.4;">${{cap.description}}</div>
                                <div style="font-size: 11px; color: ${{colors.textLight}}; line-height: 1.4;">
                                    <div style="margin-bottom: 4px;"><span style="color: ${{colors.success}}; font-weight: 500;" title="What data types this capability produces">PROVIDES:</span> ${{providesText}}</div>
                                    <div><span style="color: ${{colors.warning}}; font-weight: 500;" title="What data types this capability needs as input">REQUIRES:</span> ${{requiresText}}</div>
                                </div>
                            </div>
                        `;
                    }});

                    panel.innerHTML = html;
                }}

                // Add step from capability
                window.addStepFromCapability = function(capabilityName) {{
                    const capability = capabilities.find(cap => cap.name === capabilityName);
                    if (!capability) return;

                    const newStep = {{
                        context_key: `${{capability.name.replace(/_/g, '_')}}_${{stepCounter++}}`,
                        capability: capability.name,
                        task_objective: capability.description || `Execute ${{capability.name}} capability`,
                        expected_output: (capability.provides && capability.provides.length > 0) ? capability.provides[0] : 'UNKNOWN',
                        parameters: null,
                        inputs: []
                    }};

                    currentPlan.push(newStep);
                    renderSteps();
                    autoValidate();
                }};

                // Render steps
                function renderSteps() {{
                    const container = document.getElementById('steps-container');

                    if (currentPlan.length === 0) {{
                        container.innerHTML = `
                            <div style="text-align: center; padding: 40px; color: ${{colors.textLight}}; font-style: italic;">
                                No steps yet. Click "Add Step" or select a capability to get started.
                            </div>
                        `;
                        return;
                    }}

                    let html = '';
                    currentPlan.forEach((step, index) => {{
                        // Create inputs display
                        let inputsHtml = '';
                        if (step.inputs && step.inputs.length > 0) {{
                            inputsHtml = step.inputs.map(input => {{
                                const key = Object.keys(input)[0];
                                const value = input[key];
                                return `<span style="background: ${{colors.accent}}; color: white; padding: 4px 8px; border-radius: 3px; font-size: 11px; margin-right: 6px; margin-bottom: 4px; display: inline-block; font-weight: 500; text-transform: uppercase; letter-spacing: 0.3px; border: 1px solid ${{colors.accent}};">${{key}}: ${{value}}</span>`;
                            }}).join('');
                        }} else {{
                            inputsHtml = '<span style="color: ${{colors.textLight}}; font-style: italic; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px;">No inputs configured</span>';
                        }}

                        html += `
                            <div style="border: 1px solid ${{colors.borderLight}}; border-radius: 3px; padding: 20px; margin-bottom: 16px; background: ${{colors.panelBackground}}; position: relative; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid ${{colors.borderLight}};">
                                    <div style="flex: 1; display: flex; align-items: center;">
                                        <span style="background: ${{colors.primary}}; color: white; padding: 6px 12px; border-radius: 3px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">STEP ${{index + 1}}</span>
                                        <span style="margin-left: 12px; font-weight: 600; color: ${{colors.text}}; font-size: 14px; text-transform: uppercase; letter-spacing: 0.3px;">${{step.capability}}</span>
                                        <span style="margin-left: 12px; color: ${{colors.textLight}}; font-size: 12px; font-weight: normal; border: 1px solid ${{colors.borderLight}}; padding: 2px 6px; border-radius: 3px; cursor: help;" title="Context Key: Unique identifier for this step's output data">${{step.context_key}}</span>
                                    </div>
                                    <div>
                                        <button onclick="editStep(${{index}})" style="background: ${{colors.accent}}; color: white; border: 1px solid ${{colors.accent}}; padding: 6px 12px; border-radius: 3px; cursor: pointer; font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-right: 8px; transition: all 0.2s;">Edit</button>
                                        <button onclick="deleteStep(${{index}})" style="background: ${{colors.danger}}; color: white; border: 1px solid ${{colors.danger}}; padding: 6px 12px; border-radius: 3px; cursor: pointer; font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s;">Delete</button>
                                    </div>
                                </div>

                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                                    <div style="grid-column: 1 / -1;">
                                        <label style="display: flex; align-items: center; font-weight: 600; margin-bottom: 6px; color: ${{colors.text}}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                                            Task Objective:
                                            <span style="margin-left: 6px; cursor: help; color: ${{colors.accent}}; font-size: 10px; opacity: 0.7;" title="What this step will accomplish">ℹ</span>
                                        </label>
                                        <div style="font-size: 13px; color: ${{colors.text}}; line-height: 1.5; padding: 8px; background: ${{colors.background}}; border: 1px solid ${{colors.borderLight}}; border-radius: 3px;">${{step.task_objective}}</div>
                                    </div>
                                    <div>
                                        <label style="display: flex; align-items: center; font-weight: 600; margin-bottom: 6px; color: ${{colors.text}}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                                            Inputs:
                                            <span style="margin-left: 6px; cursor: help; color: ${{colors.accent}}; font-size: 10px; opacity: 0.7;" title="Data from previous steps that this step needs">ℹ</span>
                                        </label>
                                        <div style="font-size: 13px; line-height: 1.5; padding: 8px; background: ${{colors.background}}; border: 1px solid ${{colors.borderLight}}; border-radius: 3px; min-height: 36px;">${{inputsHtml}}</div>
                                    </div>
                                    <div>
                                        <label style="display: flex; align-items: center; font-weight: 600; margin-bottom: 6px; color: ${{colors.text}}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                                            Expected Output:
                                            <span style="margin-left: 6px; cursor: help; color: ${{colors.accent}}; font-size: 10px; opacity: 0.7;" title="Type of data this step will produce">ℹ</span>
                                        </label>
                                        <div style="font-size: 13px; color: ${{colors.text}}; padding: 8px; background: ${{colors.background}}; border: 1px solid ${{colors.borderLight}}; border-radius: 3px;">${{step.expected_output}}</div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }});

                    // Add "Add Step" button at the bottom
                    if (currentPlan.length > 0) {{
                        html += `
                            <div style="text-align: center; margin-top: 20px; padding: 20px;">
                                <button onclick="addNewStepAtBottom()" style="background: ${{colors.success}}; color: white; border: 1px solid ${{colors.success}}; padding: 12px 20px; border-radius: 50px; cursor: pointer; font-weight: 500; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s; display: inline-flex; align-items: center; gap: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
                                       onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.15)'"
                                       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'">
                                    <span style="font-size: 16px; font-weight: bold;">+</span>
                                    Add New Step
                                </button>
                            </div>
                        `;
                    }}

                    container.innerHTML = html;
                }}

                // Auto-update context key references in subsequent steps
                function updateContextKeyReferences(oldContextKey, newContextKey, changedStepIndex) {{
                    let updatedStepsCount = 0;
                    const updatedStepNumbers = [];

                    // Look through all steps after the changed step
                    for (let i = changedStepIndex + 1; i < currentPlan.length; i++) {{
                        const laterStep = currentPlan[i];
                        let stepWasUpdated = false;

                        // Check if this step has inputs that reference the old context key
                        if (laterStep.inputs && laterStep.inputs.length > 0) {{
                            laterStep.inputs.forEach((input, inputIndex) => {{
                                // Each input is an object like {{ "PV_ADDRESSES": "old_context_key" }}
                                Object.keys(input).forEach(contextType => {{
                                    if (input[contextType] === oldContextKey) {{
                                        input[contextType] = newContextKey;
                                        stepWasUpdated = true;
                                    }}
                                }});
                            }});
                        }}

                        if (stepWasUpdated) {{
                            updatedStepsCount++;
                            updatedStepNumbers.push(i + 1);
                        }}
                    }}

                    // Show notification if any references were updated
                    if (updatedStepsCount > 0) {{
                        const stepsList = updatedStepNumbers.join(', ');
                        const message = `✅ Auto-updated ${{updatedStepsCount}} reference${{updatedStepsCount > 1 ? 's' : ''}} to "${{oldContextKey}}" → "${{newContextKey}}" in step${{updatedStepsCount > 1 ? 's' : ''}} ${{stepsList}}`;

                        // Create temporary notification
                        const notification = document.createElement('div');
                        notification.style.cssText = `
                            position: fixed;
                            top: 20px;
                            right: 20px;
                            background: ${{colors.success}};
                            color: white;
                            padding: 12px 20px;
                            border-radius: 6px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                            z-index: 10002;
                            font-size: 13px;
                            font-weight: 500;
                            max-width: 400px;
                            word-wrap: break-word;
                            opacity: 0;
                            transform: translateX(100%);
                            transition: all 0.3s ease;
                        `;
                        notification.textContent = message;
                        document.body.appendChild(notification);

                        // Animate in
                        setTimeout(() => {{
                            notification.style.opacity = '1';
                            notification.style.transform = 'translateX(0)';
                        }}, 10);

                        // Auto-remove after 4 seconds
                        setTimeout(() => {{
                            notification.style.opacity = '0';
                            notification.style.transform = 'translateX(100%)';
                            setTimeout(() => {{
                                if (notification.parentNode) {{
                                    notification.parentNode.removeChild(notification);
                                }}
                            }}, 300);
                        }}, 4000);
                    }}
                }}

                // Add new step at bottom (same as top button)
                window.addNewStepAtBottom = function() {{
                    const firstCapability = capabilities[0];
                    const newStep = {{
                        context_key: `step_${{stepCounter++}}`,
                        capability: firstCapability?.name || 'unknown',
                        task_objective: firstCapability?.description || 'Define task objective',
                        expected_output: (firstCapability?.provides && firstCapability.provides.length > 0) ? firstCapability.provides[0] : 'UNKNOWN',
                        parameters: null,
                        inputs: []
                    }};

                    currentPlan.push(newStep);
                    renderSteps();
                    autoValidate();

                    // Scroll to the new step (last one)
                    setTimeout(() => {{
                        const stepsContainer = document.getElementById('steps-container');
                        const lastStep = stepsContainer.querySelector('div:last-child');
                        if (lastStep && lastStep.scrollIntoView) {{
                            lastStep.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                        }}
                    }}, 100);
                }};

                // Edit step
                window.editStep = function(index) {{
                    const step = currentPlan[index];

                    // Create edit modal
                    const editModal = document.createElement('div');
                    editModal.style.cssText = `
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(0, 0, 0, 0.8);
                        z-index: 10001;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    `;

                    const editForm = document.createElement('div');
                    editForm.style.cssText = `
                        background: white;
                        border-radius: 12px;
                        padding: 24px;
                        width: 90%;
                        max-width: 600px;
                        max-height: 80vh;
                        overflow-y: auto;
                        color: #333;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                    `;

                    // Generate capability options
                    const capabilityOptions = capabilities.map(cap =>
                        `<option value="${{cap.name}}" ${{cap.name === step.capability ? 'selected' : ''}}>${{cap.name}} - ${{cap.description}}</option>`
                    ).join('');

                    // Generate context type options for expected output
                    const contextTypeOptions = contextTypes.map(ctx =>
                        `<option value="${{ctx.type_name}}" ${{ctx.type_name === step.expected_output ? 'selected' : ''}}>${{ctx.type_name}} - ${{ctx.description}}</option>`
                    ).join('');

                    editForm.innerHTML = `
                        <h3 style="margin: 0 0 20px 0; color: ${{colors.text}};">Edit Step ${{index + 1}}</h3>

                        <div style="margin-bottom: 16px;">
                            <label style="display: flex; align-items: center; font-weight: 600; margin-bottom: 4px; color: ${{colors.text}};">
                                Context Key:
                                <span class="tooltip-icon" style="margin-left: 8px; cursor: help; color: ${{colors.accent}}; font-size: 12px; border: 1px solid ${{colors.accent}}; border-radius: 50%; width: 16px; height: 16px; display: inline-flex; align-items: center; justify-content: center; position: relative;" title="A unique identifier for this step (no spaces or special characters). This tags the step and its output data so other steps can reference it. Examples: 'pv_search', 'time_range', 'analysis_result'">?</span>
                            </label>
                            <input type="text" id="edit-context-key" value="${{step.context_key}}" style="width: 100%; padding: 8px; border: 1px solid ${{colors.border}}; border-radius: 4px; font-size: 14px; box-sizing: border-box;" placeholder="e.g., pv_search, time_range, analysis_result">
                            <div id="context-key-warning" style="margin-top: 4px; font-size: 12px; color: ${{colors.danger}}; display: none;"></div>
                        </div>

                        <div style="margin-bottom: 16px;">
                            <label style="display: flex; align-items: center; font-weight: 600; margin-bottom: 4px; color: ${{colors.text}};">
                                Capability:
                                <span class="tooltip-icon" style="margin-left: 8px; cursor: help; color: ${{colors.accent}}; font-size: 12px; border: 1px solid ${{colors.accent}}; border-radius: 50%; width: 16px; height: 16px; display: inline-flex; align-items: center; justify-content: center;" title="The specific capability/function that will execute this step. Each capability has different strengths - choose the one that best matches your task objective.">?</span>
                            </label>
                            <select id="edit-capability" style="width: 100%; padding: 8px; border: 1px solid ${{colors.border}}; border-radius: 4px; font-size: 14px; box-sizing: border-box;">
                                ${{capabilityOptions}}
                            </select>
                        </div>

                        <div style="margin-bottom: 16px;">
                            <label style="display: flex; align-items: center; font-weight: 600; margin-bottom: 4px; color: ${{colors.text}};">
                                Task Objective:
                                <span class="tooltip-icon" style="margin-left: 8px; cursor: help; color: ${{colors.accent}}; font-size: 12px; border: 1px solid ${{colors.accent}}; border-radius: 50%; width: 16px; height: 16px; display: inline-flex; align-items: center; justify-content: center;" title="A clear, descriptive explanation of what this step should accomplish. This is passed directly to the capability and may be processed by language models, so be specific and descriptive. Example: 'Find PV addresses for beam current monitors in storage ring'">?</span>
                            </label>
                            <textarea id="edit-task-objective" style="width: 100%; height: 80px; padding: 8px; border: 1px solid ${{colors.border}}; border-radius: 4px; font-size: 14px; resize: vertical; box-sizing: border-box;" placeholder="Describe exactly what this step should accomplish...">${{step.task_objective}}</textarea>
                        </div>

                        <div style="margin-bottom: 16px;">
                            <label style="display: flex; align-items: center; font-weight: 600; margin-bottom: 4px; color: ${{colors.text}};">
                                Expected Output:
                                <span class="tooltip-icon" style="margin-left: 8px; cursor: help; color: ${{colors.accent}}; font-size: 12px; border: 1px solid ${{colors.accent}}; border-radius: 50%; width: 16px; height: 16px; display: inline-flex; align-items: center; justify-content: center;" title="The type of data this step will produce. This determines how other steps can use this step's output. Usually automatically set based on the selected capability.">?</span>
                            </label>
                            <select id="edit-expected-output" style="width: 100%; padding: 8px; border: 1px solid ${{colors.border}}; border-radius: 4px; font-size: 14px; box-sizing: border-box;">
                                ${{contextTypeOptions}}
                            </select>
                        </div>

                        <div style="margin-bottom: 20px;">
                            <label style="display: flex; align-items: center; font-weight: 600; margin-bottom: 4px; color: ${{colors.text}};">
                                Inputs:
                                <span class="tooltip-icon" style="margin-left: 8px; cursor: help; color: ${{colors.accent}}; font-size: 12px; border: 1px solid ${{colors.accent}}; border-radius: 50%; width: 16px; height: 16px; display: inline-flex; align-items: center; justify-content: center;" title="Connect this step to outputs from previous steps. Select the context type you need, then choose which previous step provides that data. Steps execute in order, so you can only use outputs from steps above this one.">?</span>
                            </label>
                            <div id="inputs-container" style="border: 1px solid ${{colors.border}}; border-radius: 4px; padding: 12px; min-height: 60px; background: #fafafa;">
                                <div id="inputs-list"></div>
                                <button type="button" id="add-input-btn" style="background: ${{colors.success}}; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; margin-top: 8px;">+ Add Input</button>
                            </div>
                        </div>

                        <div style="display: flex; gap: 12px; justify-content: flex-end;">
                            <button id="edit-cancel-btn" style="background: ${{colors.secondary}}; color: white; border: none; padding: 10px 16px; border-radius: 6px; cursor: pointer; font-weight: 500;">Cancel</button>
                            <button id="edit-save-btn" style="background: ${{colors.primary}}; color: white; border: none; padding: 10px 16px; border-radius: 6px; cursor: pointer; font-weight: 500;">Save Changes</button>
                        </div>
                    `;

                    // Get available context types and keys from previous steps and agent context
                    function getAvailableContextData() {{
                        const availableContexts = [];

                        // Add agent context keys first
                        availableContextKeys.forEach(ctx => {{
                            availableContexts.push({{
                                contextKey: ctx.contextKey,
                                contextType: ctx.contextType,
                                stepNumber: null,
                                source: "agent_context",
                                description: ctx.description
                            }});
                        }});

                        // Look at all steps before the current one
                        for (let i = 0; i < index; i++) {{
                            const prevStep = currentPlan[i];
                            if (prevStep.context_key && prevStep.expected_output) {{
                                availableContexts.push({{
                                    contextKey: prevStep.context_key,
                                    contextType: prevStep.expected_output,
                                    stepNumber: i + 1,
                                    source: "plan_step",
                                    description: prevStep.task_objective
                                }});
                            }}
                        }}

                        return availableContexts;
                    }}

                    // Render current inputs
                    function renderEditInputs() {{
                        const inputsList = document.getElementById('inputs-list');
                        const availableContexts = getAvailableContextData();

                        // Get current capability requirements
                        const selectedCapability = document.getElementById('edit-capability').value;
                        const selectedCap = capabilities.find(cap => cap.name === selectedCapability);

                        let html = '';

                        if (step.inputs && step.inputs.length > 0) {{
                            step.inputs.forEach((input, inputIndex) => {{
                                const key = Object.keys(input)[0];
                                const value = input[key];

                                // Filter context types based on capability requirements
                                let validContextTypes = availableContexts.map(ctx => ctx.contextType);
                                if (selectedCap && selectedCap.requires && selectedCap.requires.length > 0) {{
                                    validContextTypes = availableContexts
                                        .filter(ctx => selectedCap.requires.includes(ctx.contextType))
                                        .map(ctx => ctx.contextType);
                                }}

                                // Remove duplicates
                                validContextTypes = [...new Set(validContextTypes)];

                                // Generate context type options (filtered by capability requirements)
                                const contextTypeOptions = validContextTypes.map(contextType =>
                                    `<option value="${{contextType}}" ${{contextType === key ? 'selected' : ''}}>${{contextType}}</option>`
                                ).join('');

                                // Generate context key options (from available previous steps and agent context that match requirements)
                                const validContextsForKey = availableContexts.filter(ctx => validContextTypes.includes(ctx.contextType));
                                const contextKeyOptions = validContextsForKey.map(ctx => {{
                                    const sourceLabel = ctx.source === "agent_context" ? "Agent Context" : `Step ${{ctx.stepNumber}}`;
                                    const sourceIcon = ctx.source === "agent_context" ? "🧠" : "📋";
                                    return `<option value="${{ctx.contextKey}}" ${{ctx.contextKey === value ? 'selected' : ''}}>${{sourceIcon}} ${{sourceLabel}}: ${{ctx.contextKey}} (${{ctx.contextType}})</option>`;
                                }}).join('');

                                html += `
                                    <div style="display: flex; gap: 8px; margin-bottom: 8px; align-items: center;">
                                        <select data-input-index="${{inputIndex}}" data-field="key" style="flex: 1; padding: 4px 8px; border: 1px solid ${{colors.border}}; border-radius: 3px; font-size: 12px;">
                                            <option value="">Select Context Type</option>
                                            ${{contextTypeOptions}}
                                        </select>
                                        <select data-input-index="${{inputIndex}}" data-field="value" style="flex: 1; padding: 4px 8px; border: 1px solid ${{colors.border}}; border-radius: 3px; font-size: 12px;">
                                            <option value="">Select Source Step</option>
                                            ${{contextKeyOptions}}
                                        </select>
                                        <button type="button" onclick="removeEditInput(${{inputIndex}})" style="background: ${{colors.danger}}; color: white; border: none; padding: 4px 6px; border-radius: 3px; cursor: pointer; font-size: 10px;">×</button>
                                    </div>
                                `;
                            }});
                        }} else {{
                            let helpText = 'No inputs configured';
                            if (selectedCap && selectedCap.requires && selectedCap.requires.length > 0) {{
                                helpText += `<br><br><small style="color: ${{colors.accent}};">This capability requires: ${{selectedCap.requires.join(', ')}}</small>`;
                            }}
                            html = `<div style="color: ${{colors.textLight}}; font-style: italic; font-size: 12px; text-align: center; padding: 20px;">${{helpText}}</div>`;
                        }}

                        inputsList.innerHTML = html;

                        // Add event listeners for input changes
                        inputsList.querySelectorAll('select').forEach(select => {{
                            select.onchange = function() {{
                                const inputIndex = parseInt(this.dataset.inputIndex);
                                const field = this.dataset.field;

                                if (!step.inputs) step.inputs = [];
                                if (!step.inputs[inputIndex]) step.inputs[inputIndex] = {{}};

                                if (field === 'key') {{
                                    const oldKey = Object.keys(step.inputs[inputIndex])[0];
                                    const value = step.inputs[inputIndex][oldKey];
                                    delete step.inputs[inputIndex][oldKey];
                                    step.inputs[inputIndex][this.value] = value;
                                }} else {{
                                    const key = Object.keys(step.inputs[inputIndex])[0];
                                    step.inputs[inputIndex][key] = this.value;
                                }}
                            }};
                        }});

                        // Add smart filtering: when context type changes, filter available source steps
                        inputsList.querySelectorAll('select[data-field="key"]').forEach(contextTypeSelect => {{
                            contextTypeSelect.onchange = function() {{
                                const inputIndex = parseInt(this.dataset.inputIndex);
                                const selectedContextType = this.value;
                                const sourceStepSelect = inputsList.querySelector(`select[data-input-index="${{inputIndex}}"][data-field="value"]`);

                                // Filter source steps to only show those that produce the selected context type
                                const filteredContexts = availableContexts.filter(ctx => ctx.contextType === selectedContextType);

                                let newOptions = '<option value="">Select Source Step</option>';
                                filteredContexts.forEach(ctx => {{
                                    const sourceLabel = ctx.source === "agent_context" ? "Agent Context" : `Step ${{ctx.stepNumber}}`;
                                    const sourceIcon = ctx.source === "agent_context" ? "🧠" : "📋";
                                    newOptions += `<option value="${{ctx.contextKey}}">${{sourceIcon}} ${{sourceLabel}}: ${{ctx.contextKey}} (${{ctx.contextType}})</option>`;
                                }});

                                sourceStepSelect.innerHTML = newOptions;

                                // Update the step data
                                if (!step.inputs) step.inputs = [];
                                if (!step.inputs[inputIndex]) step.inputs[inputIndex] = {{}};

                                const oldKey = Object.keys(step.inputs[inputIndex])[0];
                                const value = step.inputs[inputIndex][oldKey] || '';
                                delete step.inputs[inputIndex][oldKey];
                                step.inputs[inputIndex][selectedContextType] = value;

                                // Re-add the change listener for the source step select
                                sourceStepSelect.onchange = function() {{
                                    step.inputs[inputIndex][selectedContextType] = this.value;
                                }};
                            }};
                        }});
                    }}

                    window.removeEditInput = function(inputIndex) {{
                        step.inputs.splice(inputIndex, 1);
                        renderEditInputs();
                    }};

                    // Setup event handlers
                    editModal.appendChild(editForm);
                    document.body.appendChild(editModal);

                    renderEditInputs();

                    // Add real-time validation for context key
                    document.getElementById('edit-context-key').oninput = function() {{
                        const newContextKey = this.value.trim();
                        const warningDiv = document.getElementById('context-key-warning');
                        const currentContextKey = step.context_key;

                        // Check for duplicate context keys
                        const existingKeys = currentPlan
                            .filter((s, i) => i !== index) // Exclude current step
                            .map(s => s.context_key);

                        if (newContextKey && existingKeys.includes(newContextKey)) {{
                            warningDiv.textContent = `⚠️ Context key "${{newContextKey}}" already exists in another step`;
                            warningDiv.style.display = 'block';
                            this.style.borderColor = '${{colors.danger}}';
                        }} else if (newContextKey !== currentContextKey && newContextKey) {{
                            // Check if this context key is referenced by later steps
                            const referencingSteps = [];
                            for (let i = index + 1; i < currentPlan.length; i++) {{
                                const laterStep = currentPlan[i];
                                if (laterStep.inputs && laterStep.inputs.length > 0) {{
                                    const hasReference = laterStep.inputs.some(input =>
                                        Object.values(input).includes(currentContextKey)
                                    );
                                    if (hasReference) {{
                                        referencingSteps.push(i + 1);
                                    }}
                                }}
                            }}

                            if (referencingSteps.length > 0) {{
                                warningDiv.textContent = `ℹ️ This will auto-update references in step${{referencingSteps.length > 1 ? 's' : ''}} ${{referencingSteps.join(', ')}}`;
                                warningDiv.style.color = '${{colors.accent}}';
                                warningDiv.style.display = 'block';
                                this.style.borderColor = '${{colors.accent}}';
                            }} else {{
                                warningDiv.style.display = 'none';
                                this.style.borderColor = '${{colors.border}}';
                            }}
                        }} else {{
                            warningDiv.style.display = 'none';
                            this.style.borderColor = '${{colors.border}}';
                        }}
                    }};

                    document.getElementById('add-input-btn').onclick = function() {{
                        if (!step.inputs) step.inputs = [];
                        const availableContexts = getAvailableContextData();

                        if (availableContexts.length === 0) {{
                            alert('No inputs available from previous steps. Add some steps first!');
                            return;
                        }}

                        // Get the selected capability to check its requirements
                        const selectedCapability = document.getElementById('edit-capability').value;
                        const selectedCap = capabilities.find(cap => cap.name === selectedCapability);

                        // Filter available contexts based on capability requirements
                        let validContexts = availableContexts;
                        if (selectedCap && selectedCap.requires && selectedCap.requires.length > 0) {{
                            // Only show context types that this capability requires
                            validContexts = availableContexts.filter(ctx =>
                                selectedCap.requires.includes(ctx.contextType)
                            );
                        }}

                        if (validContexts.length === 0) {{
                            if (selectedCap && selectedCap.requires && selectedCap.requires.length > 0) {{
                                alert(`This capability requires: ${{selectedCap.requires.join(', ')}}. No previous steps provide these context types.`);
                            }} else {{
                                alert('No inputs available from previous steps.');
                            }}
                            return;
                        }}

                        // Add a new input with the first valid context type and key
                        const firstValidContext = validContexts[0];
                        step.inputs.push({{ [firstValidContext.contextType]: firstValidContext.contextKey }});
                        renderEditInputs();
                    }};

                    document.getElementById('edit-cancel-btn').onclick = function() {{
                        editModal.remove();
                    }};

                    document.getElementById('edit-save-btn').onclick = function() {{
                        // Get the old and new context keys to check for changes
                        const oldContextKey = step.context_key;
                        const newContextKey = document.getElementById('edit-context-key').value.trim();

                        // Check for duplicate context keys before saving
                        const existingKeys = currentPlan
                            .filter((s, i) => i !== index) // Exclude current step
                            .map(s => s.context_key);

                        if (newContextKey && existingKeys.includes(newContextKey)) {{
                            alert(`❌ Cannot save: Context key "${{newContextKey}}" already exists in another step. Please choose a unique context key.`);
                            return;
                        }}

                        if (!newContextKey) {{
                            alert('❌ Context key cannot be empty. Please provide a valid context key.');
                            return;
                        }}

                        // Update step with form values
                        step.context_key = newContextKey;
                        step.capability = document.getElementById('edit-capability').value;
                        step.task_objective = document.getElementById('edit-task-objective').value;
                        step.expected_output = document.getElementById('edit-expected-output').value;

                        // Auto-update references in subsequent steps if context key changed
                        if (oldContextKey !== newContextKey && oldContextKey && newContextKey) {{
                            updateContextKeyReferences(oldContextKey, newContextKey, index);
                        }}

                        editModal.remove();
                        renderSteps();
                        autoValidate();
                    }};

                    // Update expected output and input constraints when capability changes
                    document.getElementById('edit-capability').onchange = function() {{
                        const selectedCap = capabilities.find(cap => cap.name === this.value);
                        const expectedOutputSelect = document.getElementById('edit-expected-output');

                        // Update Expected Output dropdown to only show what this capability provides
                        if (selectedCap) {{
                            // Clear existing options
                            expectedOutputSelect.innerHTML = '';

                            if (selectedCap.provides && selectedCap.provides.length > 0) {{
                                // Add only the context types this capability can provide
                                selectedCap.provides.forEach(contextType => {{
                                    const contextInfo = contextTypes.find(ctx => ctx.type_name === contextType);
                                    const option = document.createElement('option');
                                    option.value = contextType;
                                    option.textContent = contextType + (contextInfo ? ' - ' + contextInfo.description : '');
                                    expectedOutputSelect.appendChild(option);
                                }});
                                // Select the first (or only) option
                                expectedOutputSelect.value = selectedCap.provides[0];
                            }} else {{
                                // No specific output - allow all context types
                                contextTypes.forEach(ctx => {{
                                    const option = document.createElement('option');
                                    option.value = ctx.type_name;
                                    option.textContent = ctx.type_name + ' - ' + ctx.description;
                                    expectedOutputSelect.appendChild(option);
                                }});
                            }}

                            // Clear existing inputs since capability changed
                            step.inputs = [];
                            renderEditInputs();
                        }}
                    }};

                    // Close on backdrop click
                    editModal.onclick = function(e) {{
                        if (e.target === editModal) {{
                            editModal.remove();
                        }}
                    }};
                }};

                // Delete step
                window.deleteStep = function(index) {{
                    if (confirm('Are you sure you want to delete this step?')) {{
                        currentPlan.splice(index, 1);
                        renderSteps();
                        autoValidate();
                    }}
                }};

                // Auto validation
                function autoValidate() {{
                    const statusDiv = document.getElementById('validation-status');

                    if (currentPlan.length === 0) {{
                        statusDiv.style.display = 'none';
                        return;
                    }}

                    // Enhanced validation with agent context
                    let hasErrors = false;
                    let hasWarnings = false;
                    const validCapabilities = capabilities.map(cap => cap.name);
                    const contextKeys = currentPlan.map(step => step.context_key);
                    const duplicateKeys = contextKeys.filter((key, index) => contextKeys.indexOf(key) !== index);

                    // Build available context keys (from agent context + previous steps)
                    const availableContextKeysMap = {{}};

                    // Add agent context keys
                    availableContextKeys.forEach(ctx => {{
                        availableContextKeysMap[ctx.contextKey] = ctx.contextType;
                    }});

                    // Add plan step context keys
                    currentPlan.forEach(step => {{
                        if (step.context_key) {{
                            availableContextKeysMap[step.context_key] = step.expected_output;
                        }}
                    }});

                    // Check for duplicate keys in plan
                    if (duplicateKeys.length > 0) {{
                        hasErrors = true;
                    }}

                    // Validate each step
                    currentPlan.forEach((step, index) => {{
                        // Check capability validity
                        if (!validCapabilities.includes(step.capability)) {{
                            hasErrors = true;
                        }}

                        // Check if context key conflicts with agent context
                        if (step.context_key && availableContextKeys.some(ctx => ctx.contextKey === step.context_key)) {{
                            hasWarnings = true;
                        }}

                        // Check if inputs reference valid context keys
                        if (step.inputs && step.inputs.length > 0) {{
                            step.inputs.forEach(input => {{
                                Object.values(input).forEach(contextKey => {{
                                    if (!availableContextKeysMap[contextKey]) {{
                                        hasErrors = true;
                                    }}
                                }});
                            }});
                        }}
                    }});

                    statusDiv.style.display = 'block';
                    if (hasErrors) {{
                        statusDiv.style.background = '#fef2f2';
                        statusDiv.style.border = `1px solid ${{colors.danger}}`;
                        statusDiv.style.color = colors.danger;
                        statusDiv.innerHTML = '<strong>❌ Plan has validation errors</strong>';
                    }} else if (hasWarnings) {{
                        statusDiv.style.background = '#fffbeb';
                        statusDiv.style.border = `1px solid ${{colors.warning}}`;
                        statusDiv.style.color = colors.warning;
                        statusDiv.innerHTML = `<strong>⚠️ Plan Ready with Warnings</strong> - ${{currentPlan.length}} steps configured`;
                    }} else {{
                        statusDiv.style.background = '#f0fdf4';
                        statusDiv.style.border = `1px solid ${{colors.success}}`;
                        statusDiv.style.color = colors.success;
                        statusDiv.innerHTML = `<strong>✅ Plan Ready</strong> - ${{currentPlan.length}} steps configured`;
                    }}
                }}

                // Load template from dropdown selection
                function loadTemplate(templateIndex) {{
                    if (templateIndex === '' || templateIndex === null) return;

                    const index = parseInt(templateIndex);
                    if (index >= 0 && index < templates.length) {{
                        if (currentPlan.length > 0) {{
                            if (!confirm('This will replace your current plan. Are you sure?')) {{
                                // Reset dropdown to default if user cancels
                                document.getElementById('plan-template-select').value = '';
                                return;
                            }}
                        }}

                        const template = templates[index];
                        currentPlan = template.steps.map(step => ({{ ...step }}));
                        stepCounter = currentPlan.length;

                        renderSteps();
                        autoValidate();

                        // Reset dropdown to default after loading
                        document.getElementById('plan-template-select').value = '';
                    }}
                }}

                // Clear the entire plan
                function clearPlan() {{
                    if (currentPlan.length === 0) {{
                        alert('Plan is already empty.');
                        return;
                    }}

                    if (confirm('This will delete all steps in your plan. Are you sure?')) {{
                        currentPlan = [];
                        stepCounter = 0;

                        renderSteps();
                        autoValidate();
                    }}
                }}

                // Setup event listeners
                function setupEventListeners() {{
                    if (isReviewMode) {{
                        // Setup review mode handlers
                        setupReviewModeHandlers();
                    }} else {{
                        // Setup normal mode handlers
                        setupNormalModeHandlers();
                    }}

                    // Common close handler
                    const closeBtn = document.getElementById('plan-close-btn');
                    if (closeBtn) {{
                        closeBtn.onclick = () => {{
                            overlay.remove();
                        }};
                    }}

                    // Close on overlay click
                    overlay.onclick = (e) => {{
                        if (e.target === overlay) {{
                            overlay.remove();
                        }}
                    }};
                }}

                // Setup normal mode event handlers
                function setupNormalModeHandlers() {{
                    // Main buttons
                    const addStepBtn = document.getElementById('plan-add-step-btn');
                    const templateSelect = document.getElementById('plan-template-select');
                    const clearBtn = document.getElementById('plan-clear-btn');
                    const validateBtn = document.getElementById('plan-validate-btn');
                    const saveBtn = document.getElementById('plan-save-btn');
                    const cancelBtn = document.getElementById('plan-cancel-btn');

                    if (addStepBtn) {{
                        addStepBtn.onclick = () => {{
                            const firstCapability = capabilities[0];
                            const newStep = {{
                                context_key: `step_${{stepCounter++}}`,
                                capability: firstCapability?.name || 'unknown',
                                task_objective: firstCapability?.description || 'Define task objective',
                                expected_output: (firstCapability?.provides && firstCapability.provides.length > 0) ? firstCapability.provides[0] : 'UNKNOWN',
                                parameters: null,
                                inputs: []
                            }};

                            currentPlan.push(newStep);
                            renderSteps();
                            autoValidate();
                        }};
                    }}

                    if (templateSelect) {{
                        templateSelect.onchange = function() {{
                            loadTemplate(this.value);
                        }};
                    }}

                    if (clearBtn) {{
                        clearBtn.onclick = clearPlan;
                    }}

                    if (validateBtn) {{
                        validateBtn.onclick = autoValidate;
                    }}

                    if (saveBtn) {{
                        saveBtn.onclick = () => {{
                            overlay.remove();
                            // Return save action with current plan data
                            return {{ action: 'save', plan: currentPlan }};
                        }};
                    }}

                    if (cancelBtn) {{
                        cancelBtn.onclick = () => {{
                            overlay.remove();
                            return {{ action: 'cancel' }};
                        }};
                    }}
                }}

                // Initialize
                overlay.appendChild(editor);
                document.body.appendChild(overlay);

                populateTemplateDropdown();
                renderAvailableContextPanel();
                renderCapabilitiesPanel();
                renderSteps();
                autoValidate();

                // Setup event listeners without waiting for them
                setupEventListeners();

                // Return immediately - the editor is now interactive
                return {{ action: 'editor_opened', mode: editorMode }};

            }} catch (error) {{
                return {{ action: 'error', message: 'Error creating plan editor: ' + error.message }};
            }}
            """

            # Execute JavaScript
            result = await __event_call__(
                {
                    "type": "execute",
                    "data": {"code": editor_js},
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error creating plan editor interface: {e}")
            return {"action": "error", "message": str(e)}

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> dict | None:
        """Main action handler for execution plan editor."""

        try:
            # Get user info
            user_id = self._get_user_id(__user__)
            logger.info(f"Execution Plan Editor - User: {user_id}")

            user_valves = __user__.get("valves")
            if not user_valves:
                user_valves = self.UserValves()

            # Emit initial status
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Initializing execution plan editor...", "done": False},
                }
            )

            # Create editor interface
            editor_result = await self.create_plan_editor_interface(
                __event_emitter__, __event_call__, user_id, body
            )

            if editor_result and editor_result.get("action") == "save":
                # Save the execution plan
                plan_data = editor_result.get("plan", [])

                if plan_data:
                    # Extract agent context for validation
                    agent_context_summary = None
                    available_context_keys = []

                    if body and body.get("messages"):
                        agent_context_summary = self.extract_context_summary_from_messages(body.get("messages", []))
                        if agent_context_summary:
                            available_context_keys = self.extract_available_context_keys(agent_context_summary)

                    # Validate the plan with agent context
                    validation_result = self._validate_plan(plan_data, available_context_keys)

                    if validation_result["is_valid"]:
                        # Save to file
                        save_result = self._save_plan(plan_data, user_id)

                        if save_result["success"]:
                            await __event_emitter__(
                                {
                                    "type": "message",
                                    "data": {"content": f"# ✅ Execution Plan Saved Successfully\\n\\n**Plan Details:**\\n- Steps: {len(plan_data)}\\n- Saved to: `{save_result['filename']}`\\n- Validation: Passed\\n\\n**Plan Summary:**\\n" + "\\n".join([f"{i+1}. {step.get('context_key', 'unknown')}: {step.get('capability', 'unknown')}" for i, step in enumerate(plan_data)])},
                                }
                            )
                        else:
                            await __event_emitter__(
                                {
                                    "type": "message",
                                    "data": {"content": f"# ❌ Error Saving Plan\\n\\n{save_result.get('error', 'Unknown error')}"},
                                }
                            )
                    else:
                        # Show validation errors
                        error_msg = "# ⚠️ Plan Validation Failed\\n\\n**Errors:**\\n" + "\\n".join([f"- {error}" for error in validation_result["errors"]])
                        if validation_result["warnings"]:
                            error_msg += "\\n\\n**Warnings:**\\n" + "\\n".join([f"- {warning}" for warning in validation_result["warnings"]])

                        await __event_emitter__(
                            {
                                "type": "message",
                                "data": {"content": error_msg},
                            }
                        )
                else:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": "# 📝 Empty Plan\\n\\nNo steps were defined in the execution plan."},
                        }
                    )

            elif editor_result and editor_result.get("action") == "save_modified":
                # Save modified execution plan for approval processing
                plan_data = editor_result.get("plan_data", {})

                if plan_data:
                    save_result = await self.save_modified_plan(plan_data, __event_emitter__, __event_call__)

                    if save_result["success"]:
                        await __event_emitter__(
                            {
                                "type": "message",
                                "data": {"content": f"# ✅ Modified Plan Saved Successfully\\n\\n{save_result['message']}\\n\\n**Next Steps:**\\nReturn to chat and respond with **'yes'** to approve the modified execution plan."},
                            }
                        )
                    else:
                        await __event_emitter__(
                            {
                                "type": "message",
                                "data": {"content": f"# ❌ Error Saving Modified Plan\\n\\n{save_result.get('error', 'Unknown error')}"},
                            }
                        )
                else:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": "# ❌ Error Saving Modified Plan\\n\\nNo plan data received."},
                        }
                    )

            elif editor_result and editor_result.get("action") == "save_as_is":
                # User wants to use the original plan as-is
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": "# ✅ Original Plan Ready for Approval\\n\\n**Next Steps:**\\nReturn to chat and respond with **'yes'** to approve the original execution plan."},
                    }
                )

            elif editor_result and editor_result.get("action") == "editor_opened":
                # Editor was successfully opened
                mode = editor_result.get("mode", "normal")
                if mode == "approval_review":
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": "# 📋 Execution Plan Review Mode\\n\\n**Review the pending execution plan above.**\\n\\n- **Use Plan As-Is**: Accept the original plan without changes\\n- **Save Modifications**: Edit the plan and save your changes\\n\\nAfter making your choice, return to chat and respond with **'yes'** to proceed with approval."},
                        }
                    )
                else:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": "# 📋 Execution Plan Editor\\n\\n**Create and configure your multi-step execution plan above.**\\n\\n- Add steps by clicking capabilities or using the **Add Step** button\\n- Configure inputs and outputs for each step\\n- Use **Save Plan** to save your configuration\\n\\nThe editor will validate your plan and show any issues before saving."},
                        }
                    )

            elif editor_result and editor_result.get("action") == "error":
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": f"# ❌ Editor Error\\n\\n{editor_result.get('message', 'Unknown error occurred')}"},
                    }
                )

            else:
                # Editor was opened and closed normally - no specific message needed
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": "# 📋 Execution Plan Editor\\n\\nEditor opened successfully. Use the interface above to create or review execution plans."},
                    }
                )

            # Final status
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Execution plan editor completed", "done": True},
                }
            )

        except Exception as e:
            logger.error(f"Execution Plan Editor error: {e}")
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": f"# ❌ Error\\n\\nExecution plan editor failed: {str(e)}"},
                }
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Execution plan editor failed", "done": True},
                }
            )
