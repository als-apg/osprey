A batch of writes is checked whole before any is sent (see the connectors
changelog). `channel_write` therefore writes nothing when the checks refuse one
of its operations (it wrote the others before), and its error names the refused
write. `write_channels()`, a guarded run's command batch and the optimizer
raise for the refused write, not for one its batch held back. A
`pyaml-cs-osprey` device list names as sent only the channels the runtime
reported sent.
