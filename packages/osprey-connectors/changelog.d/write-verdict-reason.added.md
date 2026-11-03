`ChannelWriteResult.verdict_reason` and `ChannelWriteBlockedError.verdict_reason`
name the write verdict's reason for a `WRITES_DISABLED` refusal — `switch_off`,
`switching`, `auth_exhausted`, `auth_range` and the rest — across the
connector-host pipe as well, so a program can tell them apart without parsing
the message. The pipe's frame protocol version is 2: a version-1 peer, which
would drop the reason, is refused.
