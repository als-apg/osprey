An `optimizer` MCP server lets the agent read the optimizer service: its
environments, live readbacks, presets and bounds, its jobs and their configs,
stored results and saved models. It only reads and holds no launch token;
switch it on with `claude_code.servers.optimizer.enabled: true` in a profile
that deploys the optimizer.
