`osprey up` refuses to start a deployment whose compose file publishes the telemetry store
on a different host port than `services.openobserve.port`, and names both ports. The start
used to print one port and then provision the ingest account against the other, timing out
without naming the mismatch.
