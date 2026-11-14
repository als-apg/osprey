The retired roster spellings, which `tier`, `default_tier` and `tiers/`
replaced, and the retired write-posture keys (`writes_enabled`,
`control_system.write_tools`, a facility write tool missing from its server's
`writes:` list) keep building with a warning through every 2026.11 release:
they are refused in the next stable OSPREY release, no longer from 2026.11.0.
A governed tool's `approval.tools.<tool>` policy is refused in the same
release.
