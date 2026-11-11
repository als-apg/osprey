The scaffold's and orbit bump's `build_connector` drop the configured
gateways inside a container only for the deployment's baseline target, whose
name server the container is given; any other target is reached through its
own gateways instead of silently reading the baseline's machine. A second
Channel Access target in one backend process is refused, since the process's
one Channel Access context would move the first target's reads and writes.
