EPICS CA subscriptions are served from one record per channel address,
however many subscribers it has (`X` and `ca://X` share one). An IOC restart
no longer delivers duplicate values, every subscription carries the channel's
limits, and `get_metadata` of a subscribed address answers from its control
data. `warm()` starts the CA context ahead of the first subscription.
