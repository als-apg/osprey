A build profile's `optimizer:` block (`environments_dir`, `port`)
deploys the optimizer API and its Redis job store. The API publishes at the
layout's `optimizer` slot on the deployment's bind address (loopback by
default), the job store publishes no port, `osprey up` mints the
`OPTIMIZER_LAUNCH_TOKEN` that arms the routes that start runs, write to the
machine or load a model, and the environment files are mounted read-only.
The login-tier map is staged into its config, so each run is judged at its
owner's rung.
