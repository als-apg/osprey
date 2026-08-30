`port_base` is now a top-level profile shorthand: `osprey init --set
port_base=42000` (or `port_base: 42000` in profile.yml) folds into
`config: deployment.port_base` and moves the whole deployment — gateway,
panels, services, stores — onto that thousand-port block. Handy for running a
second stack beside one on the default 10000 block.
