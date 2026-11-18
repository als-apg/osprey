**Breaking change:** the channel directory's names no longer use EPICS words.
The `pv_directory` config section is `channel_directory`, its
`display.ioc_health_suffixes` is `display.server_health_suffixes`, the MCP
server is `channel-directory` (switch:
`claude_code.servers.channel-directory.enabled`), and its tools `pv_record` and
`ioc_record` are `channel_record` and `server_record`, taking `channels` and
`server`; `alarm_state` takes `channels`, and `alarm_config`, `alarm_history`
and `put_history` take `channel`. Rows and pages carry `channel` and `server`
where they carried `pv` and `ioc`, and the panel's routes are `#/channel/<name>`,
`#/server/<name>` and `#/channels?server=<name>`. A profile that still spells an
old key fails the build, naming the new one.
