The EPICS connector no longer lets a write to an integer channel change the
value on its way out. It refuses (`VALIDATION_ERROR`, nothing sent) a fraction,
a non-numeric string or an out-of-range number bound for a `longout`, a short,
a char or their arrays; before, a soft IOC stored 1.5 as 1 and 2**31 as
-2147483648, and with confirmation off nothing said so. Whole floats in an
integer array are now sent instead of raising. A confirmed text write to a char
array is now `CONFIRMED` rather than a mismatch against its bytes, and a write to
a channel that never connects says that nothing was sent.
