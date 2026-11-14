**Breaking change:** a dotted connector or service `type` must name a subclass
of its base (`ControlSystemConnector`, `ArchiverConnector`, or one of the four
service bases); one that does not is refused with `ValueError` when the
connector is built, instead of running duck-typed or failing on first use. A
control-system class that is not a subclass never had its writes go through
the write door.
