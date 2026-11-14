`osprey chat` marks the rendered project it launches into as trusted in
Claude Code's state, as the host web terminal already does. Since Claude
Code 2.1.302 an untrusted render loses its `permissions.allow` list, so an
`osprey chat --print` run was refused reads of the project's artifacts, data
and uploads.
