Read-only connectors to the services beside a control system, one namespace and
contract each: `directory` (ChannelFinder, the simulator), `alarm_state` (Phoebus
alarm server over Kafka, a replay file), `alarm_log` (Phoebus alarm logger),
`put_log` (caPutLog); shared answers in `service_answers`, which source answers on
which target in `service_targets`.

`osprey_connectors.archive_targets.APPLIANCE_TARGETS` is public again: the directory
gives the facility's services the same targets the facility's archive serves.
