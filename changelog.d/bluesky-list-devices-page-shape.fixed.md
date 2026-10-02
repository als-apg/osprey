`list_devices` answers a Bluesky bridge reply that is not a device page with a
`bluesky_bridge_error` refusal instead of failing with an unhandled `TypeError`.
