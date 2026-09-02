Redeploying over a running web-terminal stack no longer leaves nginx serving
the previous deploy's configuration — or, on Docker Desktop, a 404 in place of
the landing page and a permanently unhealthy nginx container. Each deploy
regenerates the rendered `nginx.conf` and `landing.html` from scratch, which
detached the running container's file bind mounts from the files on disk;
`osprey up` now recreates the nginx container (a sub-second, nginx-only
restart) instead of asking it to hot-reload configuration it could no longer
see. Roster changes on a live stack still hot-reload with zero downtime.
