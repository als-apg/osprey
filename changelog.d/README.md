# Changelog fragments

Changelog entries live here, one small file per change, instead of being written
straight into `CHANGELOG.md`. Two pull requests that both edit `CHANGELOG.md`
collide on the same lines; two pull requests that each add their own file here
never do.

## Adding one

Create a file named `changelog.d/<name>.<type>.md`:

- **`<name>`** — the issue number when there is one (`745.changed.md`,
  `745-gate.fixed.md`). A leading number *is* the issue number: it becomes a
  `(#745)` reference at the end of the entry, so never start a name with a date
  or any other number that is not an issue number — and do not repeat the
  number in the text; a trailing `(#745)` fails the check. With no issue, use a
  short slug (`gate.fixed.md`); no reference is added, and writing one yourself
  at the end of the text (`(#735, #737)`) is fine.
- **`<type>`** — one of:

  | Type | File | Use it for |
  | --- | --- | --- |
  | added | `745.added.md` | New capability |
  | changed | `745.changed.md` | Different behaviour of something that existed |
  | deprecated | `745.deprecated.md` | Still works, will go away |
  | removed | `745.removed.md` | Gone |
  | fixed | `745.fixed.md` | Bug fix |
  | security | `745.security.md` | Security fix |
  | internal | `745.internal.md` | Work users never see — satisfies the check, renders nothing |

The body is the changelog bullet's text *without* the leading `- `: one or two
sentences in present tense, written for someone using OSPREY, wrapped at about
78 columns. A bold opener (`**Breaking change:** …`) is fine. Do not start the
file with a list marker, a heading, or a code fence — sub-bullets and fenced
blocks further down are fine.

```
Web terminal writes from a proxied panel no longer fail with `HTTP 403`. The
proxy now drops the browser's `Origin` header on the hop to the panel's
sidecar, as it already did for the operator's credentials.
```

## The check

A pull request that touches `src/` or `packages/` needs a fragment. To check
before pushing:

```bash
uv run python scripts/changelog_fragments.py check --base origin/main
```

`./scripts/premerge_check.sh main` runs the same check, and the `lint` CI job
enforces it.

## Hands off CHANGELOG.md

The `[Unreleased]` section of `CHANGELOG.md` is written only by the release
fold, which turns every fragment here into a bullet and deletes the file. Adding
a bullet by hand fails the check. To correct an entry that is already in
`CHANGELOG.md`, open a pull request that changes nothing but that file.

This README is the only permanent file in this directory.
