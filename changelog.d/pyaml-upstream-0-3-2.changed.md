OSPREY depends on upstream pyAML, `accelerator-middle-layer` 0.3.2, in place
of the `accelerator-middle-layer-apg` staging distribution. After upgrading,
rebuild rendered pyAML views with `osprey build`; a view written before fails
to load. An environment upgraded in place still holds the old distribution:
uninstall `accelerator-middle-layer-apg`, then reinstall
`accelerator-middle-layer`. In scripts, BPMs and the tune monitor are now
under `.diagnostic` (`sr.live.diagnostic.bpms`) and the tuning tools under
`.tool` (`sr.live.tool.trm`).
