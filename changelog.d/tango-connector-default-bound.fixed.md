A TANGO read or write made without a timeout of its own is now bounded by the
connector's `control_system.connector.tango.timeout_s`, as the device proxy
beneath it already was; a write that runs out is reported unconfirmed.
