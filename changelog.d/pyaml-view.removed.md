A measurement file no longer takes `singular_values`: no measurement tool reads
it. The mml import no longer seeds it.

A measurement file no longer names `instruments.chromaticity`: pyAML measures
chromaticity from the tunes and an RF step, so `crm` uses `instruments.tune` and
`instruments.rf`.

The mml import no longer zeroes `PolynomA` and `PolynomB` on the elements its
corrector families own: an imported deck's correctors are served as the deck
states them.
