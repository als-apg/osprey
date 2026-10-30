Each owner declares its checks: `scripts/check.py` collects the
`repo_checks.py` module beside the code that owns a check (the framework's is
`scripts/repo_checks.py`) instead of keeping every check, and the tables keyed
by check id, in one file. Each record carries its command, scope, tiers,
bound and hooks, so a branch adds a check without editing the command.
