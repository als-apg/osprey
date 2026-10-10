A measurement file no longer takes `singular_values`: no measurement tool reads
it. The mml import no longer seeds it.

A measurement file no longer names `instruments.chromaticity`: pyAML measures
chromaticity from the tunes and an RF step, so `crm` uses `instruments.tune` and
`instruments.rf`.
