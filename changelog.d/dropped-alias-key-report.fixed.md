`osprey build` and `osprey status` each print one warning naming a
`claude_code.aliases` or provider `claude_code_aliases` key that is not haiku,
sonnet or opus, and so is ignored. Before, the build showed nothing and status
printed raw log lines.
