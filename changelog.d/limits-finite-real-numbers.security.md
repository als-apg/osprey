A channel with numeric limits now refuses any value that is not a finite real
number — strings (even numeric-looking ones such as `"150"` or `"0x10"`),
`NaN`, `±inf` and lists — instead of letting it past the range check. A
`max_step` check whose current reading is not a finite number now fails
closed.
