Container runtime detection no longer reports a running Docker or Podman as
absent when the host is busy: each detection probe may take up to 30 s.
