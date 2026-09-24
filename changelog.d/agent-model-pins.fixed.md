`osprey build` refuses a `claude_code.agent_models` pin that names no agent,
and lists the agents there are. It also refuses a pin on a claimed or profile
agent whose own file runs a different model, instead of ignoring the pin.
`osprey status --agents` lists the agents the build ships, each with the model
its agent file names.
