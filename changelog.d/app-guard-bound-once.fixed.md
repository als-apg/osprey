An app render binds each facility guard once. A host's `claude_code.hooks.PreToolUse`
guard spelled as a mapping (`hook:` with a `matcher:`) was carried into the app
and then added again by the guard floor, so the app's settings ran it twice.
