**Breaking change:** four more names nothing outside their own module used are
no longer public API: `posture_store.launch_pin_term`, and
`write_target.is_write_call`, `is_write_tool` and `short_tool_name`. Each is now
underscore-prefixed and out of `__all__`.
