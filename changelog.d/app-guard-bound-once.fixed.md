A login's render binds each facility guard once. A host's `claude_code.hooks.PreToolUse`
guard spelled as a mapping (`hook:` with a `matcher:`) was added again by the
guard floor, so the login's settings ran it twice.
