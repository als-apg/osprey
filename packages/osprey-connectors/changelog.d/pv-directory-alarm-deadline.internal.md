Alarm-state backends (Kafka, replay): `state_for` and `config_for` wait for
catch-up up to the deadline they are handed and only then answer `alarm feed
still loading`; a backend that already knows its answer answers at once.
