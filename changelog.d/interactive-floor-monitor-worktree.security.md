Every built project's `permissions.deny` now also denies `Monitor`, which runs
shell commands in the background, and `EnterWorktree`, which creates a git
worktree on disk, in every session whatever the write posture. `osprey build`
refuses a profile that lifts either with `remove_deny` and gates it with
nothing else, as it does for `Bash` and `Edit`. `osprey up` refuses an open
deployment (`auth.method: none`) whose persona lifts `Monitor`, as it does for
`Bash`.
