**Breaking change:** the channel directory is named in channel and server
words, which hold on every control system. Its config section, MCP server and
tools were named after EPICS PVs and IOCs. Change these when upgrading:

- the config section `pv_directory` is `channel_directory`, and its
  `display.ioc_health_suffixes` is `display.server_health_suffixes`;
- the MCP server `pv-directory` is `channel-directory`, so its switch is
  `claude_code.servers.channel-directory.enabled`;
- the tool `pv_record` is `channel_record` and takes `channels`, and the tool
  `ioc_record` is `server_record` and takes `server`;
- `alarm_state` takes `channels`, and `alarm_config`, `alarm_history` and
  `put_history` take `channel`;
- rows and pages carry `channel` and `server` where they carried `pv` and
  `ioc`, and the panel's routes are `#/channel/<name>`, `#/server/<name>` and
  `#/channels?server=<name>`.

A profile that still spells `pv_directory`, `display.ioc_health_suffixes` or
`claude_code.servers.pv-directory` fails the build with a message that names
the new key. The old tool names in a facility's own prompts, skills and
permission rules are not checked, so search for them.
