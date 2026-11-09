A profile whose `config:` block pairs a scalar with a dotted key under it (say
`control_system: epics` beside `control_system.type: mock`) no longer refuses
readers that ask about an unrelated section such as `modules.web_terminals`;
the conflict is still reported to the reader whose section it decides.
