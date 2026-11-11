The uncalled legacy deploy helpers `deploy_down`, `deploy_restart` and
`rebuild_deployment` are removed from `osprey.deployment.container_lifecycle`;
`osprey down` and `osprey restart` already run `down_deployment` and
`restart_deployment`.
