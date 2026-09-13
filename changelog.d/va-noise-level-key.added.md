The Virtual Accelerator's noise level is now a configuration key,
`control_system.connector.virtual_accelerator.noise_level`, falling through to
`control_system.connector.mock.noise_level` when unset. It was reachable only
as the `VA_NOISE_LEVEL` container variable, which an operator editing the
deployment's configuration never saw; that variable still works and still wins
when a deployment exports it.
