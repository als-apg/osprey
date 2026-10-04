The web terminal's workspace file routes now answer `400 invalid_session_id` to
a `session_id` that is not a canonical session UUID, as the posture routes do,
instead of quietly serving the unscoped workspace tree.
