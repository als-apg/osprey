The Virtual Accelerator now answers a model RPC on the pvAccess channel
`model_rpc`: you can list the model's variables, read the ones it keeps off
the channel namespace, and compare what it holds against what the control
system serves. Those model-only variables — for the bundled demo lattice, the
per-BPM and per-magnet imperfections — are writable through the same RPC
when the container is started with `VA_MODEL_WRITE_TOKEN` set; a write that
presents no token, or the wrong one, is refused. The `virtual_accelerator`
instance publishes its pvAccess port alongside its Channel Access port so
clients outside the container can reach the RPC.
