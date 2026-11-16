A connector-level limits refusal (the `max_step` check) reaches
`osprey.runtime.write_channel`, `ControlSession`, the optimizer and the pyAML
devices as `ChannelWriteBlockedError` carrying its violation, and the
`channel_write` tool still answers a single write with the `limits_violation`
envelope. The refusal's message is the one-line
`Write to '<address>' refused by the limits (<KIND>): <reason>` instead of the
multi-line limits banner. The control-system safety rule names
`ClientLibraryRefusedError`, not a `RAW_CLIENT_IMPORT` reason, for a refused
client import.
