Publishing an ARIEL logbook entry now asks for approval, as creating one always
did. `entry_publish` is the call that writes an entry through to the facility's
logbook, and it was gated nowhere — so a headless read-only query could publish
to the logbook with no prompt at all.
