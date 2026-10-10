pyAML (`accelerator-middle-layer` 0.3.1) ships with OSPREY as the `pyaml-cs-osprey`
control-system backend: every pyAML read and write goes through the OSPREY
connector, so channel limits, write posture, control-target pinning and audit
apply to it. A pyAML write runs only inside a journaled guarded run; anywhere
else it is refused and nothing is written.
