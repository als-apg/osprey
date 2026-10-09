**Breaking change:** the web tier is named after the deployment's
`project_name`: the reverse proxy is `<project>-nginx`, the auth sidecar
`<project>-auth` (its locally built image `<project>-auth:local`) and a user's
terminal `<project>-web-<user>`. Lint, provisioning, `osprey users remove`,
orphan reconciliation and teardown all resolve the same name. A terminal with
no persona render of its own runs `<project>-assistant[-<persona>]:local` under
`/app/<project>-assistant`; Claude Code keys saved sessions by that directory,
so a deployment whose old `facility.prefix` spelled a different word no longer
lists the sessions saved under `/app/<prefix>-assistant` (the volume keeps
them). The first `osprey up` after upgrading replaces the old-named containers
and leaves the old `<prefix>-assistant-auth:local` image for you to remove.
Anything that addresses the containers by name (scripts, monitoring,
`docker exec`) needs the new names.
