Before a plan moves anything, the bluesky worker now probes every channel it
declares — setpoint, readback, and read addresses — on the lane's own
connector, retrying once on a failed probe. A plan naming an address that
never responds is refused before its first move, instead of aborting
mid-run with a partially applied setpoint.
