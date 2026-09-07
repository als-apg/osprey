The archiver recorder refuses to start when the channel manifest lists an
address the archive cannot store as a field name — one containing `.`, one
starting with `$`, or one containing a NUL byte — and names the address. It
previously started and warned once per tick while archiving nothing for that
channel.
