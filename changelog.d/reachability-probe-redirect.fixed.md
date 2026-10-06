The post-deploy check that nginx is reachable from the host no longer follows
nginx's redirect. On a TLS deployment, or one whose origin is not the loopback
address, it no longer reports a healthy stack as unreachable.
