Generated projects and the dispatch-worker image now pin Claude Code CLI
`2.1.258` (was `2.1.146`), and the generated GitLab CI pipeline builds on
`docker:29` / `docker:29-dind`, deploys from `alpine:3.23`, and validates the
profile on `python:3.12-slim` — the project image's own Python — instead of
the end-of-life `docker:27`, `alpine:3.20` and `python:3.11-slim`.
