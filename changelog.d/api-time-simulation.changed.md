`osprey.simulation.apply.event_window` and `event_subwindows` take the anchor as
`anchor_s` and require `tz=`: they no longer fall back to the facility zone on
their own. The archive seed, the scenario rewrite and the session artifacts read
the activation anchor from the scenario state as an aware datetime and hand the
facility zone to the archive explicitly; the archive they write is unchanged.
