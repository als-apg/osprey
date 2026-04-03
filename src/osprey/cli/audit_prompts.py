"""Prompt construction and models for the osprey audit command."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class AuditFinding(BaseModel):
    """A single finding from the audit."""

    category: str  # permissions, safety, lifecycle, mcp, overlay, config, deps
    severity: str  # info, warning, error
    title: str
    explanation: str
    file_path: str
    recommendation: str


class AuditReport(BaseModel):
    """Structured audit report from the AI reviewer."""

    summary: str
    overall_risk: str  # low, medium, high
    findings: list[AuditFinding]


AUDIT_SYSTEM_INSTRUCTIONS = """\
You are a senior security auditor for OSPREY, a framework that deploys AI agents \
to control safety-critical hardware (particle accelerators, fusion experiments, \
beamlines). Your job is to deeply analyze a build profile or built project and \
identify security, safety, and configuration risks.

## OSPREY Safety Model

OSPREY enforces a human-in-the-loop architecture for all hardware writes:

### Permission Model (FRAMEWORK_SERVERS)
- **controls server**: `channel_limits` is auto-allowed; `channel_write` requires \
explicit user approval via ask-permission.
- **python server**: `execute` requires explicit user approval. No tools are \
auto-allowed.
- **workspace server**: Read/visualization tools are auto-allowed; `setup_patch` \
requires approval.

### Safety Hook Chain for channel_write
Every `channel_write` call passes through three hooks in sequence:
1. **writes_check** — Validates the write operation is permitted
2. **limits** — Checks the value is within configured bounds
3. **approval** — Requires explicit human confirmation

### Known Dangerous Patterns
- Writes without the limits hook (bounds checking bypassed)
- Removed or empty deny entries in permissions
- Missing approval hooks on write operations
- Overlay files that replace safety-critical hooks or settings
- MCP server definitions that bypass the framework permission model
- Lifecycle scripts that run with elevated privileges or download external code
- Missing or overly permissive environment variable defaults

## Your Task

Analyze the provided files thoroughly. Use the Read, Glob, and Grep tools to \
examine file contents. Focus on:
1. **Permissions** — Are safety-critical operations properly gated?
2. **Safety hooks** — Is the write hook chain intact (writes_check → limits → approval)?
3. **MCP servers** — Do custom servers follow the framework permission model?
4. **Overlay files** — Do overlays replace safety-critical components?
5. **Lifecycle scripts** — Do build/deploy scripts introduce risks?
6. **Configuration** — Are config values safe and complete?
7. **Dependencies** — Are there concerning or unnecessary dependencies?

## Output Format

You MUST respond with ONLY a JSON object matching this schema (no markdown fences, \
no extra text):

{schema}
"""


def build_audit_prompt(target_type: str, target_path: Path, file_listing: str) -> str:
    """Build the full prompt for the audit agent.

    Args:
        target_type: Either "profile" or "project".
        target_path: Path to the target being audited.
        file_listing: Newline-separated list of files in the target.

    Returns:
        Complete prompt string with system instructions and context.
    """
    schema = json.dumps(AuditReport.model_json_schema(), indent=2)
    system = AUDIT_SYSTEM_INSTRUCTIONS.format(schema=schema)

    user_prompt = (
        f"Audit this OSPREY {target_type} at: {target_path}\n\n"
        f"Files present:\n{file_listing}\n\n"
        "Read the key files and produce your audit report as JSON."
    )

    return f"{system}\n\n{user_prompt}"
