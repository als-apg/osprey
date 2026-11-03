`ControlSession` judges a change of its target's write switch per write
instead of ending the process: only a move of the control record, or a switch
change that asks for the other gateway, exits 75. At
the `asked` position a write needs `confirmed=True` and is refused
`approval_missing` without it. `open(session_keys=...)` names the sessions a
process writes for; it is held while any of them is, and refuses every write
when the list cannot be read. A write's audit detail names its `position`,
`confirmed` and `sessions`.
