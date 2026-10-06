A virtual-accelerator build that cannot read `simulation/machine.json` or
`machine_state_channels.json` now only asks for the file to be repaired. It no
longer suggests removing it, which leaves a tree the build refuses as
incomplete. The warning for a tree whose every staged channel database is
unreadable drops the same suggestion.
