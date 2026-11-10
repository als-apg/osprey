An application container that drives the target now reads the machine each
login's session is on when the deployment's root tier is `never`. The
control-context tree is bound read-only to every such container instead of
only on an armed render, so `owner_target` no longer answers the baseline for
every login; writes are still refused on a `never` render.
