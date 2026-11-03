`osprey.runtime.session.ControlSession` gives a long-lived process, such as a
panel backend, one binding to the deployment's control target. It reads the
control-context record once and builds one connector for the target it names,
reports an absent or unservable control system as a state rather than an
error, names its target by the display name every other surface uses, and
tells each subscription its channel's connection whenever it changes. The
process exits with a dedicated status so its supervisor starts a fresh one:
`EXIT_BINDING_MOVED` (75) when the record or the bound target's switch
position moves, `EXIT_TRANSPORT_STALE` (76) when the connector reports its
transport stale, and `EXIT_START_REFUSED` (78), never respawned, for an
environment refused at start.

`ControlSession.write()` writes through the bound connector in one fixed
order: an in-flight marker a target switch names when it refuses, the target
pin, the binding state, the execution verdict, the switch gate, an `attempt`
audit record, the checked write and one outcome record. Whether this login
may write is the execution verdict shared with every other write path, the
session's hold included; a refusal names the verdict's reason in the
verdict's phrase, and a hold on the session pauses its writes in place
without a restart. `writes` states the verdict's own terms (`allowed`,
`position`, `reason`, `kind`). A session write's audit records carry the
value written as `value_len=` and `value=` detail tokens, the one exception
to the envelope's rule; `value=` is a JSON string, so a value cannot forge a
token.
