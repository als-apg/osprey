`resolve_tick_s` and `resolve_fast_tick_s` raise their documented `ValueError`
for a `simulation:` section that is not a mapping instead of an
`AttributeError`, and a simulator view channel whose `value_range` is not two
numbers is refused when the view is read instead of being kept with the wrong
length.
