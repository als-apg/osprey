CANOPY screens can write through the deployment's control session. Each click
is judged by the execution verdict every write shares: tier, session hold,
record and switch, with no CANOPY setting. At `asked` the panel asks in place:
every run shows a confirm, and a write without it is refused
`approval_missing`; at `allowed` there is no dialog. The panel's writes belong
to the terminal's sessions: while a watchdog has flagged any of them, the panel
says "writes held" with the watchdog's reason and refuses `session_held`. A
switch or a hold reaches the open panel at its next heartbeat; a control target
move, or a switch moved to or from `off` where the machine has a separate write
gateway, replaces the data process, which lets admitted writes finish within
their bound first. The
panel's 34 s stop grace lets a terminal that is shutting down finish a write
already sent.
