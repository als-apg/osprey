pyAML is `accelerator-middle-layer-apg` 0.3.2.post1: upstream 0.3.2 plus the
first-turn trajectory tool (`pyaml.tuning_tools.trajectory`). After upgrading,
rebuild rendered pyAML views with `osprey build`; a view written before fails
to load. In scripts, BPMs and the tune monitor are now under `.diagnostic`
(`sr.live.diagnostic.bpms`) and the tuning tools under `.tool`
(`sr.live.tool.trm`).
