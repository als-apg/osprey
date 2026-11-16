The PV directory reads the connectors' aware service instants and serves them
under the keys it always has (`time`, `last_contact`, `last_announced`, `at`),
as UTC ISO-8601 text to the millisecond ending in `Z`. A time the alarm logger
or the alarm feed recorded without milliseconds now reads `.000Z`, one recorded
with another offset reads in UTC, and one recorded with no zone reads `null`;
a put-log or directory stamp with no zone is still served as recorded. A
put-log stamp with no zone no longer counts as UTC: it stays in a `since`
window and sorts after the timed rows.
