Queue rows, history entries, stream frames and `GET /runs` records carry an
`owner`, present only when the item named somebody. The deployment mints that
name from the credential the request arrived with, in `X-Osprey-Owner`;
nothing else names an owner.
