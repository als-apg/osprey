---
name: osprey-deploy-ops
description: >
  Operating an already-deployed OSPREY stack: triaging a service that is down,
  deciding when to re-emit a facility repo's deploy scaffolding, and resolving
  the secrets a container volume adopted at first start. Use when someone says
  "a service is down", "deploy status shows X", "the deploy failed", "the CI
  pipeline is out of date", "re-scaffold", "verify.sh", "openobserve won't
  authenticate", "the token doesn't work anymore", or is otherwise running a
  deployment rather than authoring one. To create or change the profile a
  deployment comes from, use `osprey-build-interview` instead.
allowed-tools: Read, Glob, Grep, Bash
---

# OSPREY Deploy Operations

This is a judgment runbook for the operate-time side of a deployment — what to
look at, in what order, and which verb is the right size for the problem. It
holds no command catalog on purpose: `osprey deploy --help` lists every verb and
`osprey deploy VERB --help` explains its options, and both stay correct across
releases in a way a copied list here would not.

## Orient first

A facility repo is a git repo holding `profile/` (the profile, and the
`profile/project/` mirror that every build copies into the project), a
`.gitlab-ci.yml` and `ci-extra.yml` at the root, and `build/<name>/` — the built
project the containers actually run from.

That split decides which flag a verb takes. Almost every verb acts on one built
project: run it from inside `build/<name>/`, or pass `--project`. The exception
is `scaffold`, which emits into the *repo* and takes `--repo`; run it from
anywhere inside the repo and it finds the root itself.

## A service is down

Work outward from the deployment's own account of itself.

1. **`osprey deploy status`** — read the lines *above* the table first. A stale
   render warning means the project was rendered from an older profile, or by an
   older OSPREY than the one installed now, so the running services are not the
   ones the profile currently describes. Re-run **`osprey build --force`** before
   diagnosing anything else, or you will debug a version of the stack nobody is
   trying to run. Note the exact command: what is stale is the *render*, and
   `osprey deploy rebuild` does not re-render anything.
2. **Absent or exited?** A service missing from the table entirely was probably
   never in `deployed_services` — that is a config question, not a runtime one.
   A service that is present and exited has something to say; go ask it.
3. **Logs, oldest first.** Containers retry, so the tail of a log is usually the
   twentieth symptom rather than the cause — reading tail-first is how an hour
   goes into a downstream timeout. Start at the first error in the failing
   service's log; if it names a dependency, follow it there and repeat.
4. **`scripts/verify.sh`** in the project root separates "the process is gone"
   from "the process is up but not answering". Run one group at a time by
   passing its id (the script's `PROBE_GROUPS` default is all of them). It
   always exits 0 — the output is the report, the exit code says nothing.

### What each verb costs you

Escalate one rung at a time — each rung throws away evidence the next diagnosis
would have wanted — and know which rung destroys data before you type it. There
are two kinds of volume in play: the **service state volumes** (openobserve's
data, ARIEL's Postgres, the dispatch workspace, the Bluesky catalog) and the
**per-user web-terminal volumes**. No verb touches both.

- **`restart`** preserves everything. Containers are restarted in place with the
  secrets they already hold, and nothing re-initializes.
- **`osprey build --force` then `deploy up`** is the re-render rung, and the
  right answer to a stale render. `up` creates volumes that are missing and
  destroys none.
- **`rebuild` and `clean` destroy this project's service state volumes** —
  `rebuild` because it *is* `clean` followed by `up`, and `clean` because it
  runs `down --volumes`. If you reach for `rebuild` expecting a restart with
  fresh containers, you also just deleted openobserve's data and ARIEL's
  database. Per-user web-terminal volumes survive both: they are declared only
  in `docker-compose.web.yml`, which `clean` never sees.
- **`nuke` is the mirror image, and its name invites the wrong guess.** It
  removes every user's web-terminal workspace, their volumes, and the roster's
  locally-built images — but its teardown deliberately carries no `--volumes`,
  so the service state volumes are left exactly where they were. Reaching for
  `nuke` to "start completely clean" does not give you a fresh openobserve.

`up` and `rebuild` both run the secret-continuity preflight described below;
`restart` deliberately does not.

## When to re-scaffold

`.gitlab-ci.yml` and `profile/project/scripts/verify.sh` are generated from the
profile's `deploy:` block, not maintained by hand. Re-emit them with
`osprey deploy scaffold` whenever that block changes — CI platform, registry,
deploy host coordinates, external projects — or whenever the set of services
changes what the health check should probe.

Re-running is safe and quiet. A file whose content already matches is left
untouched, version stamp included, so an OSPREY upgrade alone produces no diff.
If scaffold reports `Updated`, something genuinely changed: read the diff.

The refusal is the part that needs judgment. Each emitted file carries a
two-line provenance header (`# osprey-scaffold: deploy/gitlab-ci` or
`deploy/verify`, plus `# osprey-version:`). A file without that marker is
treated as hand-written and left alone. **Do not answer that with `--force`
first.** Find out what the hand-edit was for: facility-specific CI jobs belong
in `ci-extra.yml`, which the pipeline includes and the scaffolder never
rewrites, and deployment facts belong in the `deploy:` block, where the next
re-emission will carry them forward. `--force` is correct only once the content
being replaced is genuinely safe to lose.

A re-scaffolded `verify.sh` reaches a running deployment only at the next
`osprey build`, which is what copies the mirror into `<project>/scripts/`.

The web terminals' `.env.production` is generated too, and is likewise not a
file to hand-edit on the host. `osprey deploy render-env-production --project
<build dir> --output <path>` creates it at mode 0600 from its first byte on
disk, so the secrets never exist at a wider mode, and that is how the emitted
pipeline produces it just before `osprey deploy up`. Without `--output` it
prints to stdout — useful for inspecting what would be written, wrong on a
deploy host where stdout is a job log.

## Secrets a volume already owns

Some service secrets are adopted exactly once, by a docker volume, at the moment
the service first initializes — openobserve's root password and the ARIEL
database password among them. After that the volume is the authority, and a
deploy supplying a different value simply fails to authenticate against data it
can no longer open.

`osprey deploy up` preflights for this and warns when a service's volume already
exists but the profile `.env` does not hold the variable that opened it. Read
the warning precisely: nothing is broken yet, and no command you are about to
run locally will break it. The project `.env` holds the value, and re-rendering
with `osprey build --force` preserves it — these keys are runtime-written, so an
existing project value outranks both the profile and the fresh render, and
`--force` preserves `.env` wholesale besides.

The problem the warning names is that the only copy is in the wrong place. The
project `.env` is local state: a rebuild keeps it, but nothing carries it
anywhere. The profile is what travels. Deploy from a fresh checkout on the
deploy host — or after the project directory is recreated from nothing — and the
value falls back to the profile, which does not have it, so a new secret is
minted and the volume that survived rejects it.

Two ways out, and the choice is the operator's: copy the value from the project
`.env` into the profile `.env`, which keeps the data and is almost always what
you want, or remove the named volume and let the service re-initialize, which
does not.

Two limits worth knowing before you trust a quiet run. The preflight checks the
profile by variable *name* only; it cannot tell you whether the value matches
the one the volume adopted. And `osprey deploy restart` never runs it, by
design — restart reuses the secrets the running containers already hold and
re-initializes nothing — so a silent restart is not evidence the warning is
resolved. `rebuild` does run it, through the `up` it delegates to.

## Composition with other skills

- **`osprey-build-interview`** — authoring or changing the profile, `deploy:`
  block included. Anything ending in "the profile should say something
  different" belongs there, not here.
- **`osprey-contribute`** — landing a facility-repo change once you have one.
