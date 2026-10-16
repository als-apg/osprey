A readonly run now also refuses a raw Channel Access put made through the
`libca` handle that `epicscorelibs.ca.cadef` already holds, through any other
`ctypes` handle that resolves `ca_array_put`, `ca_array_put_callback` or
`ca_sg_array_put`, and through every spelling of the `ctypes` loader class.
Reads through aioca are unaffected.
