A channel with numeric limits now checks a string write as the number the EPICS
client parses it to (C rules: `"0x10"` is 16), instead of letting any string
`float()` rejects past the range check. A string the client would read
differently by channel type (`"010"`: 8 or 10), a non-numeral, `NaN`, `±inf`
and lists are refused. A `max_step` check whose current reading is not a finite
number now fails closed.
