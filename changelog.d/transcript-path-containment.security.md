The web terminal's session diagnostics no longer read a JSON Lines file outside
the agent's transcript directory. A session or subagent id that is not a plain
file name in that directory, such as one with `../` or an absolute path, now
reads as a transcript that does not exist.
