# Osprey Framework - Latest Release (v2026.6.1)

**Event dispatch & facility knowledge: turn external events into headless agent runs, serve subsystem knowledge to the agent on demand, and ship composable, data-driven simulation scenarios**

## Highlights

- **Event dispatch (opt-in).** A new `osprey.dispatch` server + dispatch worker turn external events (webhooks, cron) into headless agent runs, with a live dashboard, bearer-token-gated endpoints, per-trigger tool allowlists, and a server-side shell/web denylist. The `control-assistant` preset ships four control-system-free tutorial triggers; surfaced as an in-terminal **EVENTS** tab in `osprey web`.
- **Facility Knowledge (OKF).** A structured markdown bundle (`osprey_facility_knowledge` MCP server) for on-demand retrieval of subsystem descriptions, device details, and operational procedures; `facility.md` thins to identity-only and deep content is fetched by the agent on demand. New `facility-knowledge` subagent and `osprey knowledge` CLI.
- **Composable simulation scenarios.** Self-contained scenario bundles (telemetry + logbook narrative) under `data/simulation/scenarios/`, composed and applied via the new `osprey sim` CLI so the narrative the agent searches matches the telemetry it reads.
- **Build & config fixes.** `osprey build` no longer crashes when a profile removes agents and disables the MCP server they depended on (#266); the hello-world tutorial now launches via `osprey claude chat` so the configured provider is actually used (#261).

## Notable changes

- `claude-agent-sdk` upgraded to 0.2.101.
- **BREAKING:** the mock archiver no longer emits the built-in Sector-7 vacuum-burst and RF cavity-C1 thermal demo events from hard-coded source — that physics is now data-driven. Ship it as a scenario bundle (e.g. the `control-assistant` preset's `vacuum-burst` / `rf-thermal`). Without a `simulation_file`, the mock archiver synthesizes only generic per-PV waveforms.

## Installation

```bash
uv tool install --upgrade osprey-framework
```

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
