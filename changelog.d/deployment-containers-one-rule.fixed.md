The `container` health check grades only this deployment's containers. A
container labelled for another OSPREY or compose project on the same host is
never reported as this deployment's, a service is matched by whole name
segments or its compose service name, and a service that runs several
containers reports each one that is not running.
