CI's per-preset jobs are named for what they check rather than a tier number, which collided with the
agentic tests' difficulty tiers: `Preset Static Checks`, `Preset Build + MCP Boot — <preset>` and
`Preset Agentic Flow — <preset>`. Matrix jobs name their cell after an em dash, as the stack lanes
already did: `Test Python 3.12 on ubuntu-latest — rest-a`, `Agent Record E2E — <backend>`,
`Multi-User Deploy Lifecycle E2E — <runtime>`. release.yml's `REQUIRED_LANES` and `LANES` match the
new `Preset Agentic Flow` prefix, so `full_ci_on_main.py last-green agentic-per-preset` reports no
green until the first revalidation of main after this merges.
