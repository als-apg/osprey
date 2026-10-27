The deployment-repo layout names, the `.env` writers, the executor's
execution-method vocabulary and `to_facility_iso` move from
`osprey_connectors` into the framework (`osprey.utils.workspace`,
`osprey.deployment.dotenv`, `osprey.utils.execution_method`,
`osprey.utils.facility`), and the audit layer, the executor and the hooks'
parity test take the read-only predicate from `osprey_connectors.posture_env`
instead of restating it.
