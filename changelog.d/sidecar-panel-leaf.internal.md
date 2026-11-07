The sidecar panel registry (`SidecarPanel`, `SIDECAR_PANELS`) lives in
`osprey.profiles.sidecar_panels`, a module that imports nothing, so readers
that cannot load the web stack share it; `osprey.profiles.web_panels`
re-exports both names.
