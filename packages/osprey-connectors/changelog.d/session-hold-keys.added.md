`session_hold_refusal()` and `execution_verdict()` take `session_keys=`, the
sessions a write belongs to, and refuse `session_held` while any one of them
has an open flag, with that session's sources; a process that serves several
sessions (a panel's data process) names them instead of the environment's
one key. Without the keyword nothing changes.
