`control_system.connector.virtual_accelerator.evolve: true` runs the
simulator's time-driven models in process, so a test of an application can
take shots with no container: state it in the `profile=` of
`osprey.apps.sdk.testing.synthetic_deployment`. A shot's outcome lives in the
connector that took the request.
