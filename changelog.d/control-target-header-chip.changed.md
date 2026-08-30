The Web Terminal's per-session posture badge is replaced by a **control-target
chip** in the header: it names the machine the session writes to and the write
state on it, and opens on every configured control target with a
`writes | read-only` toggle and a `Switch` button per row. Write posture is now
held per target and changing it no longer restarts the session, so a narrowing
lands on the conversation already running.
