"""Prompt construction and models for the osprey audit command."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class AuditFinding(BaseModel):
    """A single finding from the audit."""

    category: str  # permissions, safety, lifecycle, mcp, conventions, config, deps
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

### Execution Mode and Session Posture
`OSPREY_EXECUTION_MODE` carries the run's write posture, by value and never by \
presence: only the exact string `readonly` sandboxes. It reaches an agent two \
ways — the `execute` tool's own `execution_mode` argument for one run, and the \
Web Terminal's per-session sandbox toggle, which respawns that session's agent \
with `OSPREY_EXECUTION_MODE=readonly` so every process it launches inherits it. \
The posture is env-only and is never written to `config.yml`; it can only \
narrow what the deployment permits, never widen it.

Under a readonly posture each write route is refused by the layer that owns it: \
`channel_write` by the connector, a `readwrite` `execute` by the executor's \
gate, and the remaining writes-check-gated tools by the writes_check hook \
(best-effort — it is the first hook in the chain). Independently of the \
posture, executed Python may not write into the render zone (`build/`), the \
profile sources, or the audit ledger (`var/audit/`) in ANY mode; that runtime \
guard is defense in depth, and the enforcing boundary is the container's \
privilege split.

### Known Dangerous Patterns
- Writes without the limits hook (bounds checking bypassed)
- Removed or empty deny entries in permissions
- Missing approval hooks on write operations
- A write-capable MCP tool registered with no writes_check hook — for tools \
outside the controls and python servers, that hook is the only layer aware of \
the session posture at all
- Profile-supplied artifacts that shadow safety-critical hooks or settings — a \
file in a convention directory (`rules/`, `skills/`, `agents/`, `commands/`, \
`output-styles/`) that displaces a framework artifact of the same name, or a \
`project/` mirror entry aimed at a build-owned path
- MCP server definitions that bypass the framework permission model
- Lifecycle scripts that run with elevated privileges or download external code
- Missing or overly permissive environment variable defaults

## Your Task

Analyze the provided files thoroughly. Use the Read, Glob, and Grep tools to \
examine file contents. Focus on:
1. **Permissions** — Are safety-critical operations properly gated?
2. **Safety hooks** — Is the write hook chain intact (writes_check → limits → approval)?
3. **MCP servers** — Do custom servers follow the framework permission model?
4. **Convention directories** — A profile carries its artifacts in named \
directories at its root, each with one fixed destination: `rules/`, `skills/`, \
`agents/`, `commands/`, `output-styles/` land under `.claude/`; \
`web-terminal-context/<user>/` (plus its shared `base.md` baseline), \
`mcp_servers/<name>/` and `services/<name>/` land in their own trees; `project/` mirrors verbatim onto the project root. \
Does anything there shadow a safety-critical framework artifact? Does the \
`project/` mirror target a build-owned path — `config.yml`, `.mcp.json`, \
`.claude/settings.json`, `CLAUDE.md`, `.env`, `.env.example`, \
`.osprey-manifest.json`? Those each have their own channel and the mirror must \
not write them.
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
