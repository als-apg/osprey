The ARIEL sync daemon and the dispatch workers now reach the qmd search
sidecars by their compose service names on the compose network, rather than at
the host's published port, which from inside those containers is their own
loopback. The dispatch workers are also given the ARIEL store's address, as the
sync daemon already was, so their agent's ARIEL tools reach the store.
