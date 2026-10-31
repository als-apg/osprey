`python -m osprey.deployment.runtime_helper` prints the container runtime
`osprey up` would drive, and the virtual-accelerator developer scripts take
their runtime from it, set with `CONTAINER_RUNTIME`.
The virtual-accelerator build check reads the status vocabulary from the model
surface instead of restating it.
