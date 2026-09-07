# Osprey Framework - Latest Release (v2026.9.0b1)

**First public beta.** A pre-release for evaluation: pip serves it only on
request (`pip install --pre osprey-framework`, or pin `==2026.9.0b1`), and a
plain install keeps resolving to the last stable release. With uv, add
`--prerelease=allow` so the paired `osprey-connectors` beta resolves too.
Interfaces may still move before the stable cut.

**Facility-agnostic core, the multi-user web workspace, and a hardened
write-safety chain.**

## Highlights

- **Facility-agnostic.** Core, shipped templates, and docs carry no named
  facility: what `osprey init` generates is yours, from the gateway URL to the
  deployment identity stamped into the web terminal.
- **Multi-user web workspace.** One deployment serves a roster of users behind
  a landing page, each with their own terminal and a capability tier —
  read-only, read-write, or admin — enforced by the same safety chain.
- **Write-safety chain, hardened.** Write verification, per-connector write
  posture, target-aware approval prompts, and a kill switch that fails closed.
- **Bluesky bridge and scan stack.** Plan authoring, queue UX, and scan
  orchestration against the run engine, with the safety chain in the loop.
- **Eight shipped agent skills.** The OSPREY plugin now carries the full
  maintainer and operator set — build-interview, panel, design-philosophy,
  contribute, pre-commit, release, housekeeping, doc-sync.
- **Pre-release channel.** The release pipeline understands beta/RC tags end
  to end; this release is the first to use it.

See `CHANGELOG.md` (section 2026.9.0b1) for the full list.
