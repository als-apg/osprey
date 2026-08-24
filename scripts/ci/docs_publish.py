#!/usr/bin/env python3
"""Decide, stage, and describe the multi-version documentation site.

The docs site root is the stable release. A push to `main` must therefore
never change it -- `main` publishes to `/latest/`, behind the theme's
development banner, and only the newest `vX.Y.Z` release tag is allowed to
rewrite the root. The workflow used to work this out in bash, where the
"newest" tag came from `git tag --sort=-version:refname | head -n1`. That
sorts in plain string order over *every* tag in the repository, so a
`checkpoint-*` backup tag, a connector release like `osprey-connectors-v0.1.0`,
or a two-component `v2026.7` could win and silently overwrite the site root
with a build of whatever tree it pointed at.

This module makes that impossible by defining a release tag in exactly one
place (`RELEASE_TAG_RE`) and sorting by the numeric triple rather than by
text. Everything downstream -- the deploy decision, the staging of the
`gh-pages` tree, the version switcher -- is derived from that single
definition rather than passed around the workflow as strings, so the two
halves of a deploy can never disagree about which build is the stable one.

Standard library only: it runs in CI jobs that never installed the project.

Usage:
    docs_publish.py plan --ref REF --event EVENT [--input-tag TAG] [--tags TAG ...]
    docs_publish.py stage --ref REF --event EVENT [--input-tag TAG] [--tags TAG ...]
        --build DIR --existing DIR --deployment DIR [--allow-empty-site]
    docs_publish.py versions --deployment DIR [--tags TAG ...]

`plan` writes the four `key=value` lines the workflow appends to
`$GITHUB_OUTPUT`; `stage` re-derives the same plan from the same inputs rather
than being handed a directory or a flag to trust, and `versions` re-reads the
same tag list to describe the staged tree.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Canonical published location of the site. Used for the version switcher's
# URLs and as the canonical base every build points at.
SITE_URL = "https://als-apg.github.io/osprey/"

# The one definition of a release tag. `release.yml` triggers on the broader
# `v*.*.*`, which also matches pre-releases such as `v1.0.0-rc1`; those build
# but must never be treated as the stable release.
RELEASE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

# Root entries a release's root refresh must not delete, beyond the `vX.Y.Z`
# directories `RELEASE_TAG_RE` matches. `latest` is the development build;
# `CNAME` is GitHub Pages' custom-domain file, which no Sphinx build produces
# and whose deletion takes the custom domain offline.
ROOT_KEEP = frozenset({"latest", "CNAME"})


class PlanError(ValueError):
    """Malformed publish input, such as a tag that is not `vX.Y.Z`.

    The CLI maps this to exit status 2 so a bad dispatch fails loudly
    instead of quietly deploying somewhere unintended.
    """


@dataclass(frozen=True)
class Plan:
    """What a single documentation build should publish, and where."""

    version: str
    """Version the build identifies as: `"2026.6.2"` for a release, else `"dev"`."""

    deploy_dir: str
    """Site subdirectory to write: `"v2026.6.2"`, `"latest"`, or `"pr-preview"`."""

    is_latest: bool
    """True only for the newest release tag -- the one build allowed to write the root."""

    deploy: bool
    """False for pull-request builds and any ref that must not reach the site."""


def release_tags(tags: Iterable[str]) -> list[str]:
    """Return the release tags among `tags`, newest first.

    Accepts anything iterable, including `git tag` output split on newlines:
    entries are stripped and blanks skipped. Anything that is not a full
    `vX.Y.Z` match is dropped, and the survivors are ordered by the numeric
    triple, so `v2026.10.0` correctly outranks `v2026.6.2` where a plain
    string sort would not.
    """
    parsed: list[tuple[tuple[int, int, int], str]] = []
    for raw in tags:
        tag = raw.strip()
        if not tag:
            continue
        match = RELEASE_TAG_RE.fullmatch(tag)
        if match is None:
            continue
        major, minor, patch = match.groups()
        parsed.append(((int(major), int(minor), int(patch)), tag))
    parsed.sort(key=lambda item: item[0], reverse=True)
    return [tag for _, tag in parsed]


def plan(ref: str, event: str, input_tag: str, tags: Iterable[str]) -> Plan:
    """Decide what the current build publishes, and where.

    The rules are applied in this order:

    1. An explicit `input_tag` -- the `workflow_dispatch` input -- wins over
       the ref it was dispatched from, so a release can be re-published from
       any branch. It must be a full `vX.Y.Z`; anything else raises
       `PlanError` rather than guessing at what was meant.
    2. A `refs/tags/vX.Y.Z` push publishes that release into its own
       directory, and rewrites the site root if it is the newest release.
    3. Any other tag ref -- `v2026.7`, `v1.0.0rc1`,
       `osprey-connectors-v0.1.0` -- builds but does not deploy. This
       workflow and `release.yml` share loose `v*`-style triggers, so such
       refs arrive here routinely and must not reach the site.
    4. `refs/heads/main` publishes `/latest/` and is **never** `is_latest` --
       for every possible tag list, including one where the tag `git
       describe` would name on main is the newest release, and including no
       tags at all. This is the invariant the whole module exists for: the
       bash it replaces set `IS_LATEST=true` on main, so every merge
       overwrote the site root -- the stable documentation users land on --
       with the development build. `event` being `push` or
       `workflow_dispatch` makes no difference here.
    5. Everything else -- pull requests, a dispatch from a topic branch such
       as `refs/heads/feature/x` -- builds for inspection only.

    `event` is accepted for readability at the call site and for parity with
    the workflow's `github.event_name`, and so later rules can consult it
    without changing the signature. The rules above deliberately consult only
    `ref` and `input_tag`, because main behaves identically whether it was
    pushed or dispatched.

    Deriving the whole plan here, once, is the point. The workflow used to
    compute `version`, `deploy_dir`, and an `is_latest` *string* in one bash
    step and pass them between later steps, where a single stale or mistyped
    value could aim the deploy at the root. Every step now consumes one
    `Plan`, so the halves of a deploy cannot disagree about which build is
    the stable one.
    """
    tag = input_tag.strip()
    if tag:
        if RELEASE_TAG_RE.fullmatch(tag) is None:
            raise PlanError(
                f"tag input {tag!r} is not a release tag; expected the form "
                "vX.Y.Z, for example v2026.6.2"
            )
    elif ref.startswith("refs/tags/"):
        candidate = ref[len("refs/tags/") :]
        if RELEASE_TAG_RE.fullmatch(candidate) is None:
            # A build of a pre-release or a component tag: useful to have
            # succeed, but it has no place on the published site.
            return Plan(version="dev", deploy_dir="pr-preview", is_latest=False, deploy=False)
        tag = candidate
    elif ref == "refs/heads/main":
        return Plan(version="dev", deploy_dir="latest", is_latest=False, deploy=True)
    else:
        return Plan(version="dev", deploy_dir="pr-preview", is_latest=False, deploy=False)

    known = release_tags(tags)
    return Plan(
        version=tag[1:],
        deploy_dir=tag,
        # `bool(known) and ...` rather than indexing: a repository with no
        # release tags yet must yield "not latest", not an IndexError that
        # fails the release build.
        is_latest=bool(known) and tag == known[0],
        deploy=True,
    )


def stage(
    build: Path,
    existing: Path | None,
    deployment: Path,
    p: Plan,
    allow_empty_site: bool = False,
) -> None:
    """Assemble the tree that will be pushed to `gh-pages`.

    The staged tree is the *whole* published site, not a delta: whatever ends
    up in `deployment` replaces `gh-pages` wholesale. Three consequences shape
    this function.

    It fails closed on an empty `existing`. The workflow clones the current
    site from `gh-pages` first; if anything between that clone and this call
    leaves the tree missing or empty -- a half-failed checkout, a wrong path,
    a renamed branch -- publishing from it would push a site containing only
    this one build and delete every other version. A genuine first deploy is
    rare and deliberate, so the workflow has to say `allow_empty_site` out
    loud, and only does so when the `gh-pages` branch does not exist yet.

    A root refresh *deletes* before it copies. Copying the build over the root
    would leave any page that was deleted or renamed on `main` sitting at the
    root forever, still linked from search engines and still claiming to be
    current documentation. Clearing the root first makes removals propagate.

    That clearing skips `latest/` and every `vX.Y.Z/` directory, because those
    are other builds' output, not stale copies of this one. Wiping them would
    take every archived release and the development build offline the moment a
    release was published -- and the release's own directory is itself a
    `vX.Y.Z` name, so the same rule keeps the build that was just written. It
    also skips `CNAME`, the file GitHub Pages reads to serve the site from a
    custom domain: Sphinx never emits one, so clearing it would drop the custom
    domain the moment a release was published.
    """
    if existing is None or not existing.is_dir() or not any(existing.iterdir()):
        if not allow_empty_site:
            raise PlanError(
                "gh-pages tree is empty; refusing to publish — pass "
                "allow_empty_site for a first deploy"
            )
    else:
        shutil.copytree(existing, deployment, dirs_exist_ok=True)

    deployment.mkdir(parents=True, exist_ok=True)

    versioned = deployment / p.deploy_dir
    # Replace rather than merge: a re-run of the same tag must not leave
    # pages behind that the current build no longer produces.
    _remove(versioned)
    shutil.copytree(build, versioned)

    if p.is_latest:
        for entry in sorted(deployment.iterdir()):
            if entry.name in ROOT_KEEP or RELEASE_TAG_RE.fullmatch(entry.name):
                continue
            _remove(entry)
        _copy_into(build, deployment)

    # GitHub Pages runs Jekyll unless told not to, and Jekyll drops every
    # directory beginning with an underscore -- which is all of Sphinx's
    # `_static/` and `_images/`.
    (deployment / ".nojekyll").touch()


def _remove(target: Path) -> None:
    """Delete `target`, whether it is a file, a symlink, or a whole tree."""
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)


def _copy_into(source: Path, destination: Path) -> None:
    """Copy the *contents* of `source` into an existing `destination`."""
    for entry in sorted(source.iterdir()):
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)


def versions(tags: Iterable[str], deployment: Path) -> list[dict[str, object]]:
    """Build the version switcher's `versions.json` payload.

    Each entry's `version` field is what the theme matches against the page it
    is rendering: `conf.py` sets `version_match` from `release`, which is
    `X.Y.Z` for a release build and the literal `dev` otherwise. The strings
    here must equal those exactly, or the switcher renders with nothing
    highlighted and no way to tell which version the reader is on.

    Exactly one entry carries `preferred`, as pydata-sphinx-theme requires; it
    is the newest release, which is also what the site root serves. Older
    releases appear only when their directory actually exists in `deployment`,
    so a tag cut before this workflow existed does not become a dead link.

    The development entry is emitted whenever `latest/` is present, whatever
    event triggered the build. The bash this replaces emitted it only on a
    push to `main`, so publishing a release rewrote `versions.json` without
    it: `/latest/` vanished from every switcher on the site until the next
    merge to `main` happened to put it back.
    """
    entries: list[dict[str, object]] = []
    rt = release_tags(tags)
    if rt:
        newest = rt[0]
        entries.append(
            {
                "name": f"{newest} (stable)",
                "version": newest[1:],
                "url": SITE_URL,
                "preferred": True,
            }
        )
        for tag in rt[1:]:
            if (deployment / tag).is_dir():
                entries.append({"name": tag, "version": tag[1:], "url": f"{SITE_URL}{tag}/"})
    if (deployment / "latest").is_dir():
        entries.append(
            {
                "name": "latest (development)",
                "version": "dev",
                "url": f"{SITE_URL}latest/",
            }
        )
    return entries


# --------------------------------------------------------------------------
# Command line interface
# --------------------------------------------------------------------------


def _git_tags() -> list[str]:
    """Read this repository's tags, newest-first ordering left to callers.

    Shelling out lives here, in the CLI layer, and nowhere else. `plan`,
    `stage`, and `versions` take the tag list as an argument and stay pure, so
    the release-selection rules -- the part that can overwrite the published
    site -- are tested against explicit tag lists rather than against whatever
    tags happen to exist in the checkout the tests run from.
    """
    completed = subprocess.run(["git", "tag"], capture_output=True, text=True, check=True)
    # `release_tags` strips entries and skips blanks, so the trailing newline
    # `git tag` always emits needs no special handling here.
    return completed.stdout.split("\n")


def _plan_parser() -> argparse.ArgumentParser:
    """Parser holding the inputs every subcommand derives its `Plan` from.

    Shared as a parent parser rather than duplicated: `stage` deliberately has
    no `--is-latest` or `--deploy-dir` option. Passing those between workflow
    steps is exactly how a deploy could be aimed at the site root by a stale
    string, so each subcommand recomputes the plan from the same inputs.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--ref", default="", help="the ref being built, e.g. refs/heads/main")
    parent.add_argument("--event", default="", help="the triggering event name")
    parent.add_argument(
        "--input-tag",
        default="",
        help="workflow_dispatch tag input; must be a full vX.Y.Z when given",
    )
    parent.add_argument(
        "--tags",
        action="append",
        metavar="TAG",
        help="repeatable; when omitted the repository's own tags are read",
    )
    return parent


def _resolve_tags(args: argparse.Namespace) -> list[str]:
    return args.tags if args.tags is not None else _git_tags()


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface; return the process exit status."""
    parent = _plan_parser()
    parser = argparse.ArgumentParser(
        prog="docs_publish.py",
        description="Decide, stage, and describe the multi-version documentation site.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "plan",
        parents=[parent],
        help="print version/deploy_dir/is_latest/deploy for $GITHUB_OUTPUT",
    )

    stage_parser = subparsers.add_parser(
        "stage", parents=[parent], help="assemble the tree to publish to gh-pages"
    )
    stage_parser.add_argument("--build", required=True, type=Path, help="the Sphinx output tree")
    stage_parser.add_argument(
        "--existing", required=True, type=Path, help="clone of the current gh-pages site"
    )
    stage_parser.add_argument(
        "--deployment", required=True, type=Path, help="tree to assemble and publish"
    )
    stage_parser.add_argument(
        "--allow-empty-site",
        action="store_true",
        help="permit publishing when the gh-pages clone is empty (first deploy only)",
    )

    versions_parser = subparsers.add_parser(
        "versions", parents=[parent], help="write the version switcher's versions.json"
    )
    versions_parser.add_argument(
        "--deployment", required=True, type=Path, help="the staged tree to describe"
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "plan":
            p = plan(args.ref, args.event, args.input_tag, _resolve_tags(args))
            # Exactly these four lines on stdout: the workflow redirects them
            # straight into $GITHUB_OUTPUT, where anything else is a parse
            # error or a smuggled output.
            print(f"version={p.version}")
            print(f"deploy_dir={p.deploy_dir}")
            print(f"is_latest={str(p.is_latest).lower()}")
            print(f"deploy={str(p.deploy).lower()}")
        elif args.command == "stage":
            p = plan(args.ref, args.event, args.input_tag, _resolve_tags(args))
            stage(
                args.build,
                args.existing,
                args.deployment,
                p,
                allow_empty_site=args.allow_empty_site,
            )
            print(
                f"staged {args.build} into {args.deployment / p.deploy_dir}"
                + (" and the site root" if p.is_latest else ""),
                file=sys.stderr,
            )
        else:
            entries = versions(_resolve_tags(args), args.deployment)
            payload = json.dumps(entries, indent=2)
            target = args.deployment / "_static" / "versions.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload + "\n")
            print(payload)
    except PlanError as error:
        # Nothing has reached stdout at this point, so a failed `plan` cannot
        # append a half-written value to $GITHUB_OUTPUT.
        print(f"docs_publish: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
