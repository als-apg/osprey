The pinned `claude-agent-sdk` moves from 0.2.136 to 0.2.165, whose bundled
Claude Code CLI is 2.1.294. That build knows current model prices, so a
per-run budget no longer ends a Haiku 5.5 run early on an overstated cost. It
no longer ships the task-list tools (`TaskCreate`, `TaskGet`, `TaskList`,
`TaskUpdate`), so dispatch jobs have no pass-through tools left; `TaskOutput`
and `TaskStop` stay denied.
