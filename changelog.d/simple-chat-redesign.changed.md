The Simple view's chat is redesigned. Operator turns are soft bubbles, and
agent replies read as plain text. Thinking, tool calls and the agent's plan
render as inline blocks in the turn, and a turn shows a live status line with
its elapsed time. Chat text uses the interface's own font. Code, tool inputs
and numbers stay monospace. Math in agent replies (`$…$`, `$$…$$`, `\(…\)`,
`\[…\]`) is typeset with KaTeX. A subagent shows as one row in the turn, with
a live note of what it is doing. Its own tool calls and messages open under
it. A turn that hands work to a background agent stays open until the agent
reports, and Stop ends the background agent too.
