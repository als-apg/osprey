Parameters that validate what they are given take `object` instead of `Any`, so
mypy checks the narrowing in their bodies; callers pass the same values as
before. The `ControlSystemConnector` methods a connector overrides keep `Any`.
The untyped client libraries (aioca, p4p, doocs4py, PyTango, aiokafka) are
reached through Protocols naming what the connectors call, and the JSON the
connectors write and read (grants, ledgers, frames, reports, service rows) is
typed with TypedDicts.
