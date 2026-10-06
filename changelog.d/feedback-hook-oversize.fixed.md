The channel-finder feedback capture now records search answers too large for
Claude Code's tool-output limit. The hook reads the answer from the file Claude
Code saved it to, and follows only a file in the session's own `tool-results`
directory; before, such answers were dropped from the review queue silently.
