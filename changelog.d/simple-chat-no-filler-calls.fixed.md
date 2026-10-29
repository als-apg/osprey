In the Simple view, a tool result no longer shows the notes that Claude Code
adds for the agent (`<system-reminder>…`). The agent still receives them. The
agent is also told to decide on a drawing before it writes its reply. Before,
it sometimes wrote the reply, and then made tool calls that did nothing (such
as a read of `/dev/null` or a subagent with nothing to do) so that it could
add the drawing in a second reply.
