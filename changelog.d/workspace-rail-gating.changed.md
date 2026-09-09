`add_panel_to_rail`, `remove_panel_from_rail` and `register_panel` now ask for
approval first. They decide what an operator can launch at all, and they used to
run unannounced. The layout tools — `open_panel`, `close_panel` and
`arrange_workspace` — still run without a prompt. Set
`approval.tools.<tool>: skip` in `config.yml` on a deployment that would rather
not be asked — `skip` is the only policy that silences a prompt.

Six workspace tools that carried no policy either way are auto-allowed instead
of falling through: `artifact_pin` and the lattice dashboard's
`lattice_get_data`, `lattice_get_figure`, `lattice_get_settings`,
`lattice_update_settings` and `lattice_clear_baseline`. None of them reaches
hardware. Re-running `lattice_init` sets a fresh baseline, so
`lattice_clear_baseline` is undone by it; a settings change or a pin stands
until it is set back.

The three rail tools are refused outright in a headless read-only run, as
`lattice_clear_baseline` is.
