Behind a gateway (any provider reached through its own address, including
OSPREY's translation proxy) OSPREY now keeps Claude Code's tool search switched
off, so agents see all their tools from the first message, as before. Newer
Claude Code builds would otherwise hide most tools until the agent searches for
them, which dispatch runs and the translation proxy do not support yet. An agent
that talks to Anthropic directly is left as it is.
