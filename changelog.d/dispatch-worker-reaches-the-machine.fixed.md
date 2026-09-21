In the `control-assistant` preset a dispatched job could not reach the control
system or the plan queue: its worker ran on the container network, where the
`localhost` addresses it was given name the container itself. The worker now
runs on the host network, like the web terminals. A worker kept on the
container network is given the Bluesky bridge's in-network address.
