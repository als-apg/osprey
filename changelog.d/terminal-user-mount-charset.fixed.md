A web terminal whose `OSPREY_TERMINAL_USER` falls outside `[A-Za-z0-9._-]` no
longer answers 500 on every panel request. That name is both the container's
URL mount and a header on each proxied panel hop, and neither carries a
character outside that class — nor does the nginx front door route one — so the
container now refuses to start and names the variable instead.
