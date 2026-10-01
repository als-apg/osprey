A TANGO read or write made without a timeout of its own is now bounded by the
connector's `timeout`, as the device proxy beneath it already was; a write
that runs out is reported unconfirmed.
