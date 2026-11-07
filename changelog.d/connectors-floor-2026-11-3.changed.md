The framework, `osprey-optimizer` and `pyaml-cs-osprey` require
`osprey-connectors>=2026.11.3`, the first release with programs' bounded write
authorization, which the web terminal's approvals, `osprey write-authorization`
and the optimizer's writes use. `pyaml-cs-osprey` states the framework's floor
because it imports the connectors too.
