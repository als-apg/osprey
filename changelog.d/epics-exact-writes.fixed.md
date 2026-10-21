The EPICS connector no longer lets a write to an integer channel change the
value on its way out. It refuses (`VALIDATION_ERROR`, nothing sent) a fraction,
a NaN, a non-numeric string or an out-of-range number bound for a `longout`, a
short, a char, an enum or their arrays; before, a soft IOC stored 1.5 as 1 and
2**31 as -2147483648, and with confirmation off nothing said so. A whole float
bound for an integer field is sent as the integer it is. A confirmed text write
to a char array is `CONFIRMED` when the array holds the text, rather than a
mismatch against its bytes.
