`osprey mml map --init` writes a `judgments:` block wherever a Middle Layer
export leaves a family's shape ambiguous: a field carrying more channel rows
than the family has devices, a device bound by no channel, or a PV shared
across devices. `PROFILE.md` lists each question with the devices and PVs it
concerns, and both `osprey mml map --check` and `osprey mml emit` refuse while
a slot is unanswered.
