pyAML is pinned to `accelerator-middle-layer-apg` 0.3.1.post2, and every image
that overlays local wheels now asserts that exactly one distribution owns the
`pyaml` package (`python -m osprey.deployment.pyaml_owner`).
