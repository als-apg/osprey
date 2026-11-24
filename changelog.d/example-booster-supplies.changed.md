**Breaking change:** the example facility's booster bends and quadrupoles no
longer have a current and status per magnet. The 64 addresses
`BR:MAG:{DIPOLE,QF,QD}:NN:CURRENT:{SP,RB}` and `:STATUS:{READY,FAULT}` are gone; each
family is driven by one supply, read and set at
`BR:MAG:{DIPOLE,QF,QD}-RAMP:01:CURRENT:{SP,RB}`. The booster gains 16
horizontal and 16 vertical correctors (`BR:MAG:{HCM,VCM}:NN`, limited to ±6 A)
and 12 more BPMs.
