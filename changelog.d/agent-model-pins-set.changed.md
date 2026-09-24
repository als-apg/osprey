`osprey set config.claude_code.agent_models.<agent>=<id>` makes the checks
`osprey set model=` makes: it refuses a bare `haiku`, `sonnet` or `opus` and
notes an id the provider does not list. It also refuses, before anything is
written, the pins `osprey build` refuses.
