The control-assistant preset runs every helper agent on the deployment's main
model. On anthropic and als-apg, logbook deep research moves from Opus 5.5 to
the main model. On cborg, the channel finder, the knowledge-graph agent and
deep research move to Haiku 4.5, the provider's default. On every other
provider those three agents now work instead of sending a model id the
provider does not serve. A deployment made from the preset earlier keeps the
three `claude_code.agent_models` lines in its own `profile.yml`, and
`osprey validate` now reports them as a difference from the preset: remove
them, or mark them `# DEVIATION:` to keep the pins.
