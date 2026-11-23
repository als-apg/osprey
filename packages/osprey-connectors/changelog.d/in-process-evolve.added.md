The in-process simulator runs its time-driven models when its connector block
states `evolve: true`: a request write takes a shot, reads, writes and the
tick advance the models, and the tick wakes at the composite's next deadline
to fire the subscriptions of what was emitted. The default is unchanged.
`Composite.now()` reads the composite's clock.
