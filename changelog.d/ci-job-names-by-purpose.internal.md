CI job names follow one rule, which `test_ci_workflow_wiring.py` checks. Each name is noun-first, in Title Case,
and says what the job checks, never when it runs. A matrix job is `<Class> — <cells>`, with every expression
after the one em dash, because GitHub shows a skipped matrix job's name template unexpanded. Descriptions
that used to sit in parentheses are now a comment above the job's `name:`.
The per-preset jobs lose the tier numbers that collided with the agentic tests' difficulty tiers: they are
now `Preset Static Checks`, `Preset Build + MCP Boot — <preset>` and `Preset Agentic Flow — <preset>`.
The connectors release gate in release-connectors.yml matches `Connectors Standalone Tests`,
`Connectors Compat` and `Framework at Connectors Floor Tests`. release.yml's `REQUIRED_LANES` and `LANES`
match `Preset Agentic Flow`, so `full_ci_on_main.py last-green agentic-per-preset` reports no green until
the first revalidation of main after this merges. The other spending-lane prefixes are unchanged.
