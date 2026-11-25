**Breaking change:** the orbit bump application is gone; OSPREY ships no
application beside its presets. A deployment that turned it on must delete
`apps/orbit-bump/` and its `apps:` entry: its copy no longer starts. The
Bluesky plan `orbit_bump_sweep` stays.
