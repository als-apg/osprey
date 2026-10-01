**Breaking change:** `control_system.limits_checking.mode: exclusive | optional`
replaces `allow_unlisted_channels` (`false` is `exclusive`, `true` is
`optional`), deployment-wide and per connector type. Under `exclusive` only
channels in the limits file can be written; under `optional` channels in the
file are held to their limits and every other channel is written with no
limits. `osprey build` and `osprey validate` refuse any other `mode` and any
leaf a limits block does not define. The `control-assistant` and `hello-world`
presets ship `mode: optional`, the `control-assistant` preset writes no
per-type limits block, the `channel_limits` tool reports `mode` and `mode_key`,
and switching a session onto `live` or `standin` no longer depends on the
limits settings.
