`osprey build` prints one warning when Claude Code's haiku, sonnet or opus
alias runs the main model because the provider serves no model of that family,
and one naming every configured model id the provider's served list does not
carry. `osprey status` says the alias substitution once instead of twice.
