`settle` raises `SettleReadError` naming the readback only when the readback
read failed with a `ChannelError` or answered no value; a lost connector
propagates as its own error.
