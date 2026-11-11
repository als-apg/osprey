On a single-user deployment, a run started from the OPTIMIZER panel is now owned
by the deployment's operator (`deployment.host_identity`), so its writes are no
longer refused `owner_unknown` and its write request shows on the terminal's
chip. Run `osprey build` once to pick this up. A sidecar published off loopback
still names nobody, and the panel warns before Start.
