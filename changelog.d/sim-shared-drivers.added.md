Simulation scenarios can give channels a common cause: a `drivers` block declares
named slow signals (the `wander` texture, keyed by the driver's name and evaluated
on epoch time), `couple` adds `gain * driver(t)` to chosen channels before their
own noise — optionally with a `gain_wander` envelope so the coupling strength
drifts — and `noise` replaces a channel's noise sigmas while the scenario is
active. Live reads and synthesized history carry the same coupled term. The
control-assistant template ships `rf-thermal-live`, which couples cavity 01
temperature, reflected and forward power and tuner position for live
correlation plots.
