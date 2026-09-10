# Osprey Framework - Latest Release (v2026.9.0b2)

**Second public beta.** A pre-release for evaluation: a plain install keeps
resolving to the last stable release, and a version pin alone fails under uv
because the paired `osprey-connectors` beta must resolve too. Install with

    uv tool install --prerelease allow "osprey-framework==2026.9.0b2"

(with pip, `pip install --pre osprey-framework` or the exact pin is enough).
Interfaces may still move before the stable cut.

**Web terminal with JupyterLab and Bluesky, per-target control switching, a
virtual accelerator with pluggable physics, and a guided installer.**

## Highlights

- A JupyterLab panel in the web terminal. Notebook kernels read and write the
  control system through `osprey.runtime`, under the same write gates as the
  agent's own Python.
- Bluesky plans run from the web terminal: up to two plan lanes, queue
  autostart, and a probe of the channels a plan declares before it runs.
- The simulator's physics is swappable; the shipped ring uses pyAT, and a
  surrogate or another tracking code can replace it while the channel names
  stay the same.
- One control target per deployment, switched from a header chip that labels
  the target (real machine, rehearsal, simulator, demo) and shows its write
  state. Write posture is set per target.
- Channel search answers from an index built over the facility corpus at build
  time, with an Explore view for browsing and an `ask_channels` tool for the
  agent.
- Event dispatch runs agents from triggers on machine events, with an Activity
  and Triggers panel, and bridges reach the deployment from Google Chat and
  Nextcloud Talk.
- First-session onboarding: Simple and Expert views onto one session, an
  optional tour, and a seeded example workspace.
- The header and status bars can be rearranged, feedback goes to the
  deployment owner from inside the terminal, and roster cards can be shared.
- `/osprey:install` builds a facility deployment interactively, confirming
  each step.
- Authentication, per-user roles, origin checks and an audit trail; see
  Security below.

See `CHANGELOG.md`: section 2026.9.0b2 for what changed since the first beta,
section 2026.9.0b1 for the beta line's breaking changes and the full list.
