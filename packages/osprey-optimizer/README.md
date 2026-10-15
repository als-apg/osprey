# osprey-optimizer

The OSPREY optimizer service: Xopt-driven tuning runs against a control
system, with every write going through an OSPREY connector, so the write
switch, channel limits and `max_step` apply to each one. An optimization
environment is a YAML file of setpoint/readback channels and observables;
Python environments register on the `osprey.optimizer.environments`
entry-point group.

## Install

```
pip install osprey-optimizer             # models and environment files only
pip install "osprey-optimizer[service]"  # the engine and its HTTP service
```

The base install is what the OSPREY framework imports: the request, config
and status models, and the environment-file format. It does not load the
search stack. The `service` extra adds Xopt, PyTorch, BoTorch, Redis and
FastAPI, and is what the optimizer's container image installs. For a
CPU-only PyTorch, add `--extra-index-url https://download.pytorch.org/whl/cpu`.

The service app is `osprey_optimizer.api.main:app`.
