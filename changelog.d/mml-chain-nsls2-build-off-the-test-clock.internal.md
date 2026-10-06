`osprey mml verify` sweeps each corrector family in one pass that writes every
corrector back in the same solve as the next one's first arm, so a matrix of
`n` correctors costs `2n + 1` orbit solves instead of `3n`; the report is
unchanged bit for bit.
