`osprey build` now records the container runtime it used in
`build/config.yml` (`container_runtime: docker` or `podman`) instead of
copying `auto` into the render. `down`, `restart`, `reset` and the port
preflight act on that runtime and refuse if it is not running; they no longer
re-detect one per invocation, which on a host with both Docker and Podman
installed could let one slow `docker ps` send a `down` to Podman, where it
found nothing, exited 0 and reported the still-running Docker stack as
stopped. Auto-detection also warns when it skips an installed runtime, naming
the probe that failed.
