The provider and model resolver for the agent's launch moves to
`osprey.agent_runner.provider_env`, inside the harness adapter, and importing
any `osprey.agent_runner` module no longer loads the agent SDK.
