Facility seeds gain `pulse`, which gives a waveform channel a trace scaled by a
driver channel and switched by an enable channel, and `gate`, which holds a
channel at its `gated` value while any input reads its blocking label; the
build checks both and refuses a gate cycle. The Virtual Accelerator's archive
stores waveform traces in a sibling `<collection>_waveforms` collection once a
minute, the MongoDB archiver connector reads them raw as `list[float]`, and
`archiver_read` returns them unbinned within a 32,000-point budget per read.
