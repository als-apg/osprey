Control targets now carry an operator-facing display name on the web
terminal's control-target chip, minted server-side from what each target is:
"Real machine", "Rehearsal" for a deployed stand-in, "Simulator" for the
virtual accelerator, "Demo" on a simulated connector. A deployment renames
any of them per target via the new optional
`control_system.target_display_names` mapping.
