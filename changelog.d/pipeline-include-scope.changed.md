The scaffolded GitLab pipeline and the `ci-extra.yml` header no longer say that
a job in the facility's own include can override a scaffolded job of the same
name. The including file is merged last, so the scaffolded pipeline's keys win
on every key it sets; a facility job needs its own name to run as written.
