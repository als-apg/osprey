"""Reference File Tools for State Identification Agent.

Provides tools for listing and reading reference documentation files
that inform machine state assessment decisions.

Supports two modes:
- mock_mode=True: Returns hardcoded mock data for testing
- mock_mode=False: Reads actual files from reference_path

The mock data provides realistic examples that help test the agent's
reasoning without requiring actual reference files to be set up.
"""

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from osprey.utils.logger import get_logger

logger = get_logger("xopt_optimizer")


# =============================================================================
# MOCK DATA FOR TESTING
# =============================================================================

MOCK_REFERENCE_FILES = {
    "machine_ready_criteria.md": """# Machine Ready Criteria

This document defines the criteria for determining if the machine is ready for optimization.

## Ready Conditions

The machine is considered **READY** for optimization when ALL of the following are true:

1. **Beam Current**: Above 10 mA (channel: `BEAM:CURRENT`)
2. **Vacuum Pressure**: Below 1e-8 Torr (channel: `VACUUM:PRESSURE`)
3. **Interlock Status**: No active interlocks (channel: `SAFETY:INTERLOCK`, value should be 0)
4. **Machine Mode**: Not in maintenance mode (channel: `MACHINE:MODE`, value should be 1 for "operational")

## Not Ready Conditions

The machine is **NOT_READY** when ANY of the following are true:

1. Beam current is zero or below 1 mA
2. Any safety interlock is active (SAFETY:INTERLOCK != 0)
3. Machine is in maintenance mode (MACHINE:MODE == 0)
4. Vacuum pressure exceeds safe limits

## Unknown State

Report **UNKNOWN** if:
- Unable to read critical channels
- Channel values are stale or unreliable
- Conflicting information from different sources
""",
    "optimization_channels.md": """# Optimization Channels Reference

This document lists the control system channels relevant to optimization operations.

## Primary Monitoring Channels

| Channel Name | Description | Units | Normal Range |
|--------------|-------------|-------|--------------|
| BEAM:CURRENT | Beam current | mA | 10-500 |
| VACUUM:PRESSURE | Vacuum level | Torr | < 1e-8 |
| SAFETY:INTERLOCK | Interlock status | - | 0 (clear) |
| MACHINE:MODE | Operating mode | - | 1 (operational) |

## How to Use

1. Read SAFETY:INTERLOCK first - if non-zero, machine is NOT_READY
2. Check MACHINE:MODE - must be 1 for operational
3. Verify BEAM:CURRENT is in acceptable range
4. Confirm VACUUM:PRESSURE is within limits

## Channel Naming Convention

All channels follow the pattern: `SYSTEM:SUBSYSTEM:PARAMETER`
- BEAM: Beam-related measurements
- VACUUM: Vacuum system readings
- SAFETY: Safety and interlock status
- MACHINE: Overall machine state
""",
    "safety_procedures.md": """# Safety Procedures for Optimization

## Pre-Optimization Checklist

Before starting any optimization run:

1. Verify no personnel in restricted areas
2. Confirm all interlocks are clear (SAFETY:INTERLOCK == 0)
3. Check beam current stability over last 5 minutes
4. Ensure vacuum levels are nominal

## Abort Conditions

Immediately abort optimization if:
- Any interlock activates
- Beam current drops below 5 mA
- Operator requests stop
- Unexpected machine behavior detected

## Contact Information

For questions about machine readiness, contact the control room operator.
""",
}


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================


def create_list_files_tool(reference_path: str | None = None, mock_mode: bool = False):
    """Create a tool for listing available reference files.

    Args:
        reference_path: Path to reference files directory (ignored in mock mode)
        mock_mode: If True, return mock file list

    Returns:
        LangChain tool function
    """

    @tool
    def list_reference_files() -> str:
        """List available reference documentation files.

        Returns a list of file names that can be read with the read_reference_file tool.
        These files contain important information about machine ready criteria,
        channel definitions, and safety procedures.

        Returns:
            Newline-separated list of available file names
        """
        if mock_mode:
            files = list(MOCK_REFERENCE_FILES.keys())
            logger.debug(f"[mock] Listing {len(files)} reference files")
            return "\n".join(files)

        if not reference_path:
            return "No reference files configured. Reference path not specified."

        path = Path(reference_path)
        if not path.exists():
            return f"Reference path does not exist: {reference_path}"

        # List markdown and text files
        files = []
        for ext in ["*.md", "*.txt", "*.yaml", "*.yml"]:
            files.extend(p.name for p in path.glob(ext))

        if not files:
            return f"No reference files found in {reference_path}"

        logger.debug(f"Found {len(files)} reference files in {reference_path}")
        return "\n".join(sorted(files))

    return list_reference_files


def create_read_file_tool(reference_path: str | None = None, mock_mode: bool = False):
    """Create a tool for reading reference file contents.

    Args:
        reference_path: Path to reference files directory (ignored in mock mode)
        mock_mode: If True, return mock file contents

    Returns:
        LangChain tool function
    """

    @tool
    def read_reference_file(filename: str) -> str:
        """Read the contents of a reference documentation file.

        Use list_reference_files first to see available files, then use this
        tool to read specific files that are relevant to assessing machine state.

        Args:
            filename: Name of the file to read (from list_reference_files output)

        Returns:
            Contents of the file, or error message if file not found
        """
        if mock_mode:
            if filename in MOCK_REFERENCE_FILES:
                logger.debug(f"[mock] Reading reference file: {filename}")
                return MOCK_REFERENCE_FILES[filename]
            else:
                available = ", ".join(MOCK_REFERENCE_FILES.keys())
                return f"File not found: {filename}. Available files: {available}"

        if not reference_path:
            return "Cannot read file - reference path not configured."

        file_path = Path(reference_path) / filename
        if not file_path.exists():
            return f"File not found: {filename}"

        # Security check - ensure file is within reference_path
        try:
            file_path.resolve().relative_to(Path(reference_path).resolve())
        except ValueError:
            return f"Access denied - file is outside reference directory: {filename}"

        try:
            content = file_path.read_text()
            logger.debug(f"Read reference file: {filename} ({len(content)} chars)")
            return content
        except Exception as e:
            return f"Error reading file {filename}: {e}"

    return read_reference_file


def create_reference_file_tools(
    reference_path: str | None = None,
    mock_mode: bool = False,
) -> list[Any]:
    """Create all reference file tools.

    Args:
        reference_path: Path to reference files directory
        mock_mode: If True, use mock data instead of real files

    Returns:
        List of LangChain tools [list_reference_files, read_reference_file]
    """
    return [
        create_list_files_tool(reference_path, mock_mode),
        create_read_file_tool(reference_path, mock_mode),
    ]
