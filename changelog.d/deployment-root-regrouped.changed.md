**Breaking change:** the deployment root's sources are regrouped under
`agent/` and `ops/`, and a root still laid out the old way is refused. Sources
move:

- `rules/`, `skills/`, `agents/`, `commands/`, `output-styles/` and `hooks/` to
  `agent/<same>/`
- `web-terminal-context/` to `agent/context/`
- `watchdogs/` to `agent/watchdogs/`
- `data/code_review/` to `agent/code_review/`
- `mcp_servers/` to `agent/mcp_servers/`
- `services/`, `project/` and `scripts/` to `ops/<same>/`
- `osprey.service` to `ops/osprey.service`, and `ci-extra.yml` to
  `ops/ci-extra.yml`
- `data/.runtime/` to `var/runtime/`

Profile keys and list prefixes such as `rules/safety` or `skills/orbit-check`
name a row and do not change. The canopy payload and the two canopy wheels are
no longer staged at the repo root: `osprey build` writes them to
`build/canopy_payload/` and `build/*.whl`, and the migration deletes the
copies left at the root.

Every command that reads the deployment root now stops before reading anything
when a retired path is present. It prints the `mkdir -p`, `git mv` (or `mv`)
lines that move the sources, the edits to path values in the profile,
`.gitlab-ci.yml` and `.dockerignore`, and the host steps; when nothing listed
needs you first, the block's last line runs `osprey init --revalidate <repo>`.
In `.gitlab-ci.yml` a moved path is rewritten where it is spelled as a path:
with a directory, after `./`, `$CI_PROJECT_DIR/` or `${CI_PROJECT_DIR}/`, or as
an `include` or `exists:` value. A path present at both the old and the new
location is reported and left to you. To migrate, run
`osprey init --revalidate`, or paste the printed lines.

Upgrade OSPREY in every deployment first, then migrate each one. The refusal
and `--revalidate` come with the new OSPREY, and an older OSPREY does not read
the new layout. A deployment that mounts a machine repository follows the
machine-repository upgrade order described with the facility tree change.

Host steps, once the files have moved: run `osprey build`. If the generated
`osprey.service` moved, copy `ops/osprey.service` to
`~/.config/systemd/user/`, then `systemctl --user daemon-reload` and
`systemctl --user enable --now osprey.service`. If the generated boot hook
moved, edit the crontab with `crontab -e` and replace the `osprey-boot-hook`
lines with the ones printed.

Check by hand; the paths inside these files are not rewritten:
`ops/ci-extra.yml`, each `ops/services/<name>/Dockerfile`, the operator
scripts in `ops/scripts/`, and any shell command in the profile's
`lifecycle` `run` values. A `.gitlab-ci.yml` line that holds the bare name
`osprey.service` or `ci-extra.yml` anywhere else is listed with its line
number and never rewritten: in `systemctl --user enable osprey.service` it is
the unit's name, not the file.
