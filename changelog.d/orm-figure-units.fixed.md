Run figures carry the units the facility declares for each channel, instead of a fixed "BPM units / A"; `get_run_data` states each column as `{name, unit}`, in the order a row's values follow.
The `orm` plan's largest-kick parameter is renamed `span` (was `span_a`): the kick is in the corrector's own unit.
