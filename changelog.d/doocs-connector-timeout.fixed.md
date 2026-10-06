The DOOCS connector gives every ENS lookup, property read and property set a
bound, `control_system.connector.doocs.timeout_s` (default 5 seconds). A read
that runs out raises a timeout error, and a set that runs out is reported
unconfirmed, because the value may still arrive.
