**Breaking change:** the web tier is named after the deployment's
`project_name`: the reverse proxy is `<project>-nginx`, the auth sidecar
`<project>-auth` (its locally built image `<project>-auth:local`) and a user's
terminal `<project>-web-<user>`. Lint, provisioning, `osprey users remove`,
orphan reconciliation and teardown all resolve the same name. The first
`osprey up` after upgrading replaces the old-named containers and leaves the
old `<prefix>-assistant-auth:local` image for you to remove. Anything that
addresses the containers by name (scripts, monitoring, `docker exec`) needs
the new names.
