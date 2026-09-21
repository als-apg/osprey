Inside a deployed container, the hooks looked for agent state and wrote their
audit records under the build host's project path instead of the path the
deployment runs at. A dispatched job's control-system write was refused with
"write state unknown", and hook audit records landed outside the audit
directory. The hooks now anchor on the repo they run in.
