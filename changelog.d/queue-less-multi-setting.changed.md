Without a measurement queue, the agent no longer refuses a scan: it runs a
multi-setting measurement as separate `channel_write` calls or as one
`osprey.runtime` script the operator approves as a whole, whichever fits.
