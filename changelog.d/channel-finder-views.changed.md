**Breaking change:** `osprey build` writes each channel-finder index from the
facility description, and two things differ from the hand-written databases it
replaces.

- The hierarchical index's levels are the facility's place level words, then
  `class`, `device` and `leaf`. Every level is present for every channel, and
  the node `-` names an absent place, class or device. The `field` and
  `subfield` levels are dropped: their descriptions are kept in the groups'
  `signals` sentences and shown on leaves and middle_layer Fields. A query or
  a saved path that walks the old levels no longer resolves.
- The `channel-finder-standalone` preset serves 2912 channels where its
  hand-written database listed 1228; every one of the 1228 addresses is still
  served. The 1684 added are `SR:DIAG:BPM` 21 to 72, `SR:DIAG:CHROM` and
  `SR:DIAG:TUNE` `X` and `Y`, `SR:MAG:DIPOLE` 25 to 36, `SR:MAG:HCM` and
  `SR:MAG:VCM` 21 to 72, `SR:MAG:QD` and `SR:MAG:QF` 17 to 24, `SR:MAG:SD` and
  `SR:MAG:SF` 13 to 24, and the new families `SR:MAG:QFA`, `SR:MAG:SHD` and
  `SR:MAG:SHF` 1 to 24.

The three `channel_finder.pipelines.<pipeline>.database.type` keys are gone:
each pipeline reads the one index the build writes, so there is no loader to
choose.
