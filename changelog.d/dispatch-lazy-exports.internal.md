The `osprey.dispatch` package resolves its public names on first attribute
access, so importing one leaf of it — the trigger-configuration reader, say —
no longer pulls the HTTP worker client in behind it.
