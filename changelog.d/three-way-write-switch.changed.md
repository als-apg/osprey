**Breaking change:** the write switch has three positions per login and
machine: `off`, `asked` (every machine write prompts) and `allowed` (it runs
unprompted). The tier key `control_system.writes: never | asked | allowed` is
the furthest position a login may pick; the shipped readwrite tier is `asked`,
admin `allowed`.

- A stored `on` reads as `asked`. A tier spelling `allowed` now permits
  unprompted writes; spell `asked` where prompts are wanted.
- The switch, not `approval.tools.<tool>`, decides for the governed write tools
  (`channel_write`, write-mode `execute`/`execute_file`, `pyaml_measure`,
  `queue_add`, `queue_start`, hooked facility write tools); `skip` now prompts
  at `asked` and `approval.enabled: false` no longer silences them. Such a
  policy warns now and is refused from 2026.12.0; `osprey init --revalidate`
  removes it.
- Upgrade order: install the connectors package, rebuild the lane and dispatch
  containers, then rebuild the project. An old connectors container that meets
  an `asked` login-tier map refuses every login, not just the asked ones; an
  unrebuilt project reads `asked` as `never`.
