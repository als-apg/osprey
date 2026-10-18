Every file OSPREY rewrites in place, the hooks' files included, is replaced
whole through a hidden `.<name>.*.tmp` file that is flushed to disk and removed
on failure. Files whose permissions came from the process umask now get fixed
ones whatever the umask: 0644, in-flight markers and approval stamps 0640, and
the agent's `.claude.json` 0600. Profiles add the `safety-file` helper to
their hook list. A container that sets a control identity mounts the helper
beside the identity module.
