# Security Policy

Osprey mediates access to control systems that can move real hardware. A defect in the
approval chain, the safety hooks, or a connector's write path is a safety issue as much as a
security one. Please report those privately.

## Reporting a vulnerability

**Do not open a public issue.** Use GitHub's private vulnerability reporting:

**[Report a vulnerability →](https://github.com/als-apg/osprey/security/advisories/new)**

The report stays private between you and the maintainers until an advisory is published.

Please include, as far as you are able:

- What the issue allows an attacker or an unattended agent to do.
- The affected version, and the connector and model provider in use.
- A minimal reproduction. A Mock-connector reproduction is preferred over one that requires
  live hardware.

Reports are handled on a best-effort basis by a small team. We will acknowledge your report
and keep you informed as we investigate; we do not commit to a fixed response deadline.

## Scope

In scope:

- Bypasses of the human-approval gate for hardware writes.
- Hooks that fail open — a guard that silently does nothing rather than blocking.
- Privilege escalation through MCP servers, the Python executor sandbox, or the runtime API.
- Credential or secret leakage through logs, artifacts, or agent transcripts.

Out of scope:

- Findings that require an operator to have already granted the agent write permission and
  approved the write. That is the system working as designed.
- Prompt injection that produces an *unapproved* action being *proposed*. Osprey's threat
  model assumes model output is untrusted; the approval gate is the control. A finding is
  in scope if injection causes an action to be **executed** without approval.
- Vulnerabilities in an upstream model provider, EPICS itself, or other third-party
  dependencies. Please report those to the relevant project, though we appreciate a
  heads-up if Osprey's usage makes the impact worse.

## Supported versions

Osprey follows CalVer (`vYYYY.M.P`). Security fixes land on `main` and ship in the next
release. Older releases are not backported.
