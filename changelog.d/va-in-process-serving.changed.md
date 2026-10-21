**Breaking change:** the `mock` control-system type is retired. The simulator is one
connector type, `virtual_accelerator`, served either from its container
(`control_system.connector.virtual_accelerator.serving: served`, the default) or inside
the process that asks (`serving: in_process`, no network and no containers). A profile,
`osprey set` or a connector build that states `control_system.type: mock`, or keeps a
`control_system.connector.mock` block, is refused with the fix: `osprey set
connector=virtual_accelerator config.control_system.connector.virtual_accelerator.serving=in_process`,
then `osprey build`.

- A deployment that states no control-system type is now the simulator in process, with
  baseline target `va`.
- A deployment serving the simulator in process that also configures a live or stand-in
  block can switch between them.
- The in-process simulator's writes journal lives under `<state>/inprocess/`; a journal
  left under `<state>/mock/` is ignored.
- The Bluesky bridge's browse-only remedy is `osprey set
  config.control_system.connector.virtual_accelerator.serving=served`.
