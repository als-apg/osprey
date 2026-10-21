A setpoint that an active scenario marks `stuck` now reads the value last
written to it, on every read and in `held()`, while its readback keeps showing
the model where it was. A write to it confirms on the virtual accelerator and
on the mock connector alike, with no dependence on when the next publishing
pass runs. Clearing the fault or a reset drops the written value along with
every other session write.
