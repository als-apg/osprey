`archiver_read` no longer guesses at ambiguous dates. `start_time` and
`end_time` accept ISO-8601, relative expressions like `2h ago`, and `now`; a
dotted or slashed date such as `03.04.2026` — which reads as two dates a month
apart depending on where you are — is now a validation error naming the
accepted spellings, instead of being read month-first and stamped
facility-local.
