In the Simple view, the agent's task list sits in a bar above the message box.
The folded bar shows how many tasks are done, the task in hand or the next one,
and a progress bar. Click the bar to see the whole checklist. The bar reads the
`TaskCreate` and `TaskUpdate` calls that the current Claude Code makes, and also
`TodoWrite`. Before, the task card inside the turn read only `TodoWrite`, so it
never appeared. The task calls no longer show as rows in the turn. The list
stays for the whole conversation and comes back when the page reloads. When
every task is done, the list stays until the next prompt. The workflow rule now
tells the agent to mark each task in progress when it starts it, and completed
when it finishes it.
