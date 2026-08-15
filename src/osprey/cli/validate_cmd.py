"""The ``osprey validate`` verb — check a deployment's source without building.

Validation is the cheap half of a build: resolve the ``extends:`` chain, check
the convention directories, the ``data:`` tree, service templates, lifecycle
steps and env vars, then lint the web stack against the config a build would
render. Nothing is written and nothing is started, which is what makes this the
verb a CI pipeline runs first and an operator runs after every profile edit.

Zero-argument is the normal spelling: the repo enclosing the working directory
is the thing being validated, found by the same walk every repo-scoped command
uses. The optional path argument stays for the one profile that is not a repo
root — a persona delta under ``personas/``, which is a profile in its own right
and has its own way to be wrong.

``osprey profile validate`` is the same verb spelled as part of the ``profile``
noun group, and both spellings are documented surface. Only the INTERFACE
differs — a required TARGET there, an optional one plus ``--repo`` here — so the
check itself lives once, in :func:`check_profile_file`, and both commands
resolve a profile file and hand it over. The two used to be copy-pasted bodies
and had already drifted into two failure headers, two next-step lines and two
wordings of the same refusal.
"""

from __future__ import annotations

from pathlib import Path

import click

from osprey.errors import BuildProfileError

from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option


def profile_file_at(target: Path) -> Path:
    """Return the profile file *target* names, directly or as its directory.

    Both spellings are accepted because both are things an operator has on
    hand: a directory (the repo root, or any directory holding a
    ``profile.yml``) and an explicit path to a profile file (how persona deltas
    are named, since a delta is a file beside its siblings, not a directory).

    Raises:
        click.UsageError: When *target* is a directory with no profile.yml.
    """
    if target.is_dir():
        candidate = target / PROFILE_FILENAME
        if not candidate.is_file():
            raise click.UsageError(
                f"No {PROFILE_FILENAME} in {target}. Pass the profile file directly, or "
                f"create a deployment repo with `osprey init {target} --preset <NAME>`."
            )
        return candidate.resolve()
    return target.resolve()


def check_profile_file(profile_file: Path) -> None:
    """Validate the profile at *profile_file* and print the verdict.

    The whole of what both spellings of the verb do once a profile file has been
    named: resolve the ``extends:`` chain, run the profile's own consistency
    check, lint the declared web stack against the config a build would render,
    and report.

    Args:
        profile_file: An existing profile file — a repo's ``profile.yml`` or a
            persona delta.

    Raises:
        click.UsageError: With every accumulated problem, so the caller exits 2.
    """
    from .build_profile import resolve_build_profile
    from .build_profile_deploy import deploy_aware_config_errors

    try:
        build_profile, profile_dir = resolve_build_profile(profile_file, None)
        # Named explicitly rather than left to resolution's internals: this
        # command exists to run exactly this check, so it must not become a
        # no-op if resolution ever stops validating on its own.
        build_profile.validate(profile_dir)
    except BuildProfileError as e:
        raise click.UsageError(str(e)) from e

    # The multi-user web stack the profile declares, judged on the config the
    # build would render — the `config:` block with the `deploy:` block's
    # contributions applied. Checked from the command rather than inside
    # `validate()`, which also runs during profile resolution — see
    # `lint_profile_config`.
    web_errors = deploy_aware_config_errors(build_profile.deploy, build_profile.config)
    if web_errors:
        # "Profile validation failed", not "Build profile ...": the success line
        # below says "Profile is valid", and `BuildProfile.validate` already owns
        # the "Build profile validation failed" header for its own errors.
        raise click.UsageError("Profile validation failed:\n  - " + "\n  - ".join(web_errors))

    click.echo(f"✓ Profile is valid: {profile_file}")
    click.echo(f"  Name: {build_profile.name}")
    # Named separately because "valid" says nothing about whether the deploy
    # coordinates were read at all: a profile that omits the block and one whose
    # block checked out otherwise print the same line.
    if build_profile.deploy is not None:
        deploy = build_profile.deploy
        click.echo(f"  Deploy: {deploy.ci} CI → {deploy.host.user}@{deploy.host.name}")
    click.echo("\nNext steps:")
    click.echo("  1. Render the deployment: osprey build")
    click.echo("  2. Re-run this command after editing the profile")


@click.command()
@click.argument("target", required=False, type=click.Path(exists=True, path_type=Path))
@repo_option
def validate(target: Path | None, repo: Path | None) -> None:
    """Check the deployment profile without building.

    With no argument, validates the deployment repo enclosing the working
    directory. TARGET names a different profile to check instead — a persona
    delta file under personas/, or a directory holding a profile.yml.

    Resolves extends: chains and runs the full consistency check — convention
    directories, the data: tree, service templates, lifecycle steps, env vars —
    then lints the declared web stack against the config a build would render.
    Every problem found is reported, not just the first.

    Exits 0 when the profile is valid, 2 with the accumulated errors when it is
    not, so a CI job can gate on it.

    Examples:

    \b
      $ osprey validate
      $ osprey validate personas/reader.yml
      $ osprey validate --repo ~/als-assistant
    """
    if target is not None and repo is not None:
        raise click.UsageError(
            "--repo and TARGET both name what to validate. Drop one: --repo picks "
            "the deployment repo, TARGET picks a specific profile file."
        )

    if target is not None:
        profile_file = profile_file_at(target)
    else:
        # The resolver guarantees the marker is a file at the root it returns,
        # so there is no second existence check to make here.
        profile_file = find_repo_root(repo) / PROFILE_FILENAME

    check_profile_file(profile_file)
