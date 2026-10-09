An ARIEL search or listing whose `end_date` is a bare date (`2025-10-06` or
`20251006`) now includes the entries written on that day; it used to stop at
midnight at its start. An end date with a time of day is still taken exactly
as written, and a number is still read as epoch seconds.
