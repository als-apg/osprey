The controls server's target switch now holds a new connector-host child to the
same checks the connector-host pool applies. A child that reports another
connector type, or a write posture or readonly run other than the one the switch
derived, is refused and the previous target stays active; before, a child with
no gateways, or with a read-only gateway alone, passed whatever posture it came
up with. A target whose connector block still carries an unresolved `${VAR}`
placeholder is refused as `target_unresolvable`, including on a return to the
deployment baseline, instead of passing verification because the placeholder
matched itself. The switch's child now has a call deadline, so a call that names
no timeout ends; a child that then answers no ping is replaced on the same
target, while one that still answers is kept. A virtual-accelerator target whose
gateway port is unset is derived from the config file the child is handed, not
from whatever `CONFIG_FILE` the controls server's environment names, so the two
no longer disagree on the port.

The connector-host pool's refusal for a child that exits before answering its
init frame now names the child's real exit code; on Linux it could report 255
when the child was signalled after it had already exited.
