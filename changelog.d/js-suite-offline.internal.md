The browser-side unit suite no longer opens sockets to the test environment's
own origin. An iframe's src and an unstubbed `fetch` are both answered in
process, so a refused connection can no longer surface after the last test and
fail a run in which everything passed.
