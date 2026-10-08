**Breaking change:** the web-terminal containers are named after the
deployment's `project_name` instead of `facility.prefix`: `<project>-nginx`,
`<project>-auth` and `<project>-web-<user>`, and the locally built login image
is `<project>-auth:local`. A terminal with no persona render of its own runs an
image tagged `<project>-assistant[-<persona>]:local`, but keeps its
`/app/<prefix>-assistant` directory, so its saved sessions stay where they
were. The first `osprey up` after upgrading replaces the old-named containers
and leaves the old `<prefix>-assistant-auth:local` image for you to remove.
Anything that addresses the containers by name (scripts, monitoring,
`docker exec`) needs the new names.
