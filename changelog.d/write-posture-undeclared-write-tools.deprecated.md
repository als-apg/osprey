A facility server with a `writes:` list that leaves out a tool it lists under
`permissions.ask`, or whose name looks destructive, now gets a build warning
naming the tool to add; the tool is gated as a write anyway. The warning
becomes a build refusal from OSPREY 2026.11.0.
