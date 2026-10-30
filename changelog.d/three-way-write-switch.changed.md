**Breaking change:** the write switch has three positions per login and
machine: `off`, `asked` (every machine write prompts a person) and `allowed`
(the write runs unprompted). Every stored `on` reads as `asked`. The tier key
gains a third value, `control_system.writes: never | asked | allowed`, the
furthest position a login may pick. Shipped stance: readonly, hello-world and
every app `never`; readwrite and the control-assistant root `asked`; admin
`allowed`.

The slider replaces `approval.tools.<tool>` for the governed machine-write
tools (`channel_write`, write-mode `execute`/`execute_file`, `pyaml_measure`,
`queue_add`, `queue_start`, hooked facility write tools). A profile spelling
one builds with a warning this release, is refused from OSPREY 2026.12.0, and
`osprey init --revalidate` removes the lines. Consequences: a `skip` policy now
prompts at `asked`; `execute: always` no longer prompts on read-only code;
`approval.enabled: false` no longer silences governed tools (it still does for
the rest).

Upgrade: install the connectors package first, then rebuild the lane and
dispatch containers and the project (an old container refuses every lane write
under an `asked` tier; an unrebuilt project reads it as `never`). Every
existing tier spelling `allowed` now offers the no-prompt position; spell
`asked` in a tier file where prompts are wanted. A project initialised from an
earlier preset gets a preset-drift NOTE on `control_system.writes`; spell
`asked`, or claim the line with a `# DEVIATION: <why>` comment, to silence it.
