The PV directory service serves its search, record, alarm and history answers
with the same keys and words as before, with three exceptions. A search whose
backend gave no answer serves `request: ""`, as a search that timed out
already did, instead of the request the backend sent. An alarm or put history
section with no answer no longer carries `truncated` and `retention_known`, as
a history that timed out already did not. An alarm severity word that is not
one of `OK`, `MINOR`, `MAJOR`, `INVALID` and `UNDEFINED` (which Phoebus never
sends) is served as `UNDEFINED`. A logged acknowledged alarm is still served as
`MAJOR_ACK`.
