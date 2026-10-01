**Breaking change:** web-terminal provisioning, `osprey users remove`, orphan
reconciliation and teardown address containers and images by the project name
(`project_name`, else the project directory's name) instead of
`facility.prefix`: the no-persona project is `<project>-assistant` under
`/app/<project>-assistant`, the locally built auth sidecar image is
`<project>-assistant-auth:local`, and a user's container is
`<project>-web-<user>`.
