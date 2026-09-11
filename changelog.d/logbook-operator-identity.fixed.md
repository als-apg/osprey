Logbook entries name the operator the audit ledger names. The session metadata
attached to an entry resolved the operator from the process account, which in a
per-user terminal container is a service account rather than the person at the
keyboard; it now climbs the same identity ladder every audit record uses, and
falls back to `unknown` rather than to no name at all.
