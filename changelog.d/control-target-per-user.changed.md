Control-target records, reports, switch requests and in-flight markers now
live per user under `var/agent_data/control_target/<identity>/`, bound from
the host in multi-user deployments. Pick the control target again after
upgrading: the old single-file record is not read.
