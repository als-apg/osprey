A control target switch made in an agent session before its first channel
read no longer sits on `switching…` for 30 s and then reports
`request_expired`. A controls server that has not launched its connector yet
now brings it up on the new target when the record moves, so the switch is
reported like any other instead of waiting on a server that could never
answer.
