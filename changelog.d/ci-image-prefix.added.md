A profile's `deploy:` block takes a `ci_image_prefix:`, which the scaffolded
GitLab pipeline puts in front of the base images its own jobs run in. A runner
that cannot reach Docker Hub pulls them from a site mirror instead of needing
the generated file hand-edited.
