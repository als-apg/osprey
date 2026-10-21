A write journal the in-process simulator refuses on replay is now logged under
the address the simulator refused, or under the journal file when the refusal
names no single channel, instead of under the journal's first address. The
in-process connector also reads its write journal only when the file changed,
so reads in a long session no longer re-parse the whole journal each time.
