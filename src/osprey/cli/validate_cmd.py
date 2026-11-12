"""The ``osprey validate`` verb — check a deployment's source without building.

Validation is the cheap half of a build: resolve the ``extends:`` chain, check
the convention directories, the ``data:`` tree, service templates, lifecycle
steps and env vars, then lint the web stack against the config a build would
render. Nothing is written and nothing is started, which is what makes this the
verb a CI pipeline runs first and an operator runs after every profile edit.

Zero-argument is the normal spelling: the repo enclosing the working directory
is the thing being validated, found by the same walk every repo-scoped command
uses. The optional path argument stays for the one profile that is not a repo
root — a tier under ``tiers/`` or an app under ``apps/``, each a profile in its
own right with its own way to be wrong.

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

from osprey_connectors.errors import BuildProfileError

from .output import note, report, section
from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option
from .styles import Styles


def profile_file_at(target: Path) -> Path:
    """Return the profile file *target* names, directly or as its directory.

    Both spellings are accepted because both are things an operator has on
    hand: a directory (the repo root, or any directory holding a
    ``profile.yml``) and an explicit path to a profile file (how tiers and apps
    are named, since each is a file beside its siblings, not a directory).

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


def _reject_unknown_layer_keys(profile_file: Path) -> None:
    """Refuse a tier or app file spelling a key not legal in its kind.

    Checked on the file's own document, before it is merged over the root
    profile, so a deployment key in a tier is named against the tier's key set
    rather than accepted by the root set the merged document is parsed with.
    A root profile is left to the parser.
    """
    from .build_profile_document import _read_profile_document
    from .build_profile_load import layer_file_label, reject_unknown_kind_keys
    from .profile_root import resolve_profile_root

    kind = resolve_profile_root(profile_file)[1]
    if not kind.is_layer or not profile_file.is_file():
        return
    raw = _read_profile_document(profile_file)
    if isinstance(raw, dict):
        reject_unknown_kind_keys(raw, kind, layer_file_label(profile_file))


def check_profile_file(profile_file: Path) -> None:
    """Validate the profile at *profile_file* and print the verdict.

    The whole of what both spellings of the verb do once a profile file has been
    named: resolve the ``extends:`` chain, run the profile's own consistency
    check, lint the declared web stack against the config a build would render,
    say whether the preset a materialized profile came from has moved on, and
    report.

    The host-variant overlay is applied first, exactly as ``osprey build``
    applies it. Without that, this verb would judge a document no host builds:
    a repo whose ``.env.variant`` selects ``profiles/teststand.yml`` would be
    called valid on the strength of the tracked profile while the render the
    build produces is the merged one — and the operator would find out at build
    time, from the verb they ran validate to stay ahead of.

    Args:
        profile_file: An existing profile file — a repo's ``profile.yml`` or a
            tier or app file.

    Raises:
        click.UsageError: With every accumulated problem, so the caller exits 2.
        FacilityBuildError: When ``machine:`` names no usable machine root, the
            same stop ``osprey build`` raises.
    """
    from .build_profile import resolve_build_profile
    from .build_profile_canopy import profile_home_screen_errors, repo_home_screen_errors
    from .build_profile_deploy import (
        deploy_aware_config_errors,
        deploy_aware_config_warnings,
        limits_block_errors,
        profile_dispatch_tier,
    )
    from .build_profile_presets import bundled_layer_file_refusal
    from .build_profile_timezone import system_timezone_errors, system_timezone_reminders
    from .build_profile_updates import preset_update_note
    from .profile_root import resolve_profile_root
    from .variant_selection import VARIANT_DIRNAME, VariantSelection, resolve_variant_selection

    layer_refusal = bundled_layer_file_refusal(profile_file)
    if layer_refusal is not None:
        raise click.UsageError(layer_refusal)

    variant = VariantSelection(name=None, path=None)
    try:
        # Only a deployment has a host variant, and it overlays the deployment's
        # profile.yml: a tier or app named directly is judged over the root its
        # render merges over, so the setting is read off that root, never off
        # the `tiers/` or `apps/` directory.
        root_dir, kind = resolve_profile_root(profile_file)
        if kind.is_layer or profile_file.name == PROFILE_FILENAME:
            variant = resolve_variant_selection(root_dir)
        _reject_unknown_layer_keys(profile_file)
        overlays = (variant.path,) if variant.path is not None else ()
        build_profile, profile_dir = resolve_build_profile(profile_file, None, overlays)
        # Named explicitly rather than left to resolution's internals: this
        # command exists to run exactly this check, so it must not become a
        # no-op if resolution ever stops validating on its own.
        build_profile.validate(profile_dir)
    except BuildProfileError as e:
        raise click.UsageError(str(e)) from e

    # Where the deployment reads its facility, checked exactly as the build
    # checks it: a `machine:` naming an empty submodule or overlapping `data:`
    # stops here with the same facility stop the build would raise.
    build_profile.resolved_facility_roots(profile_dir)

    # The multi-user web stack the profile declares, judged on the config the
    # build would render — the `config:` block with the `deploy:` block's
    # contributions applied. Checked from the command rather than inside
    # `validate()`, which also runs during profile resolution — see
    # `lint_profile_config`.
    # The profile's own directory, not the working directory: the `tiers/` and
    # `apps/` files the web lint reads sit beside the profile. Passing it is what
    # makes `osprey validate --repo <path>`, and a run from any subdirectory,
    # read the same files a run from the repo root reads — without it the
    # render half of the lint silently sees nothing and reports the profile
    # valid.
    profile_root = profile_file.parent
    dispatch_tier = profile_dispatch_tier(build_profile)
    web_errors = deploy_aware_config_errors(
        build_profile.deploy,
        build_profile.config,
        profile_root=profile_root,
        project_name=build_profile.project_name,
        dispatch_tier=dispatch_tier,
        agent_record=build_profile.agent_record,
    )
    # A sibling call, not a line inside `deploy_aware_config_errors`: that
    # function judges the multi-user web stack against the deploy block, and
    # `osprey build` runs it over the RENDERED config as well. A per-type
    # `limits_checking` block is neither web-stack business nor a render fact —
    # it is a claim the `config:` block makes about a posture, checked once,
    # here and in the build's own profile-side pass. Folded into the web lint it
    # would be asked twice during a build and refuse the same profile twice.
    web_errors = [
        *web_errors,
        *limits_block_errors(build_profile.config),
        *system_timezone_errors(build_profile.config),
    ]
    # The CANOPY home screen, as `osprey build` judges it: a repo's profile
    # together with every tier and app file the build renders, a file named
    # directly on its own, against the library under the profile's data root.
    try:
        if profile_file.name == PROFILE_FILENAME:
            web_errors.extend(repo_home_screen_errors(build_profile, profile_root, overlays))
        else:
            web_errors.extend(profile_home_screen_errors(build_profile, profile_dir))
    except BuildProfileError as e:
        raise click.UsageError(str(e)) from e
    if web_errors:
        # "Profile validation failed", not "Build profile ...": the success line
        # below says "Profile is valid", and `BuildProfile.validate` already owns
        # the "Build profile validation failed" header for its own errors.
        raise click.UsageError("Profile validation failed:\n  - " + "\n  - ".join(web_errors))

    # Advisory findings are printed rather than raised: they name real exposures
    # (a privileged terminal with no login wall — `auth.method: token` or `none`) that are not
    # mistakes every deployment has made, so they must not fail a CI gate — but
    # a finding nobody prints is a finding nobody has.
    for warning in (
        *deploy_aware_config_warnings(
            build_profile.deploy,
            build_profile.config,
            profile_root=profile_root,
            project_name=build_profile.project_name,
            dispatch_tier=dispatch_tier,
            agent_record=build_profile.agent_record,
        ),
        # The same UTC reminder `osprey build` prints.
        *system_timezone_reminders(build_profile.config),
    ):
        note(f"⚠ {warning}")

    # Whether the presets this repo took have changed since it last took them:
    # one line, never a refusal. The profile is the operator's to edit, and
    # applying what a preset changed is `osprey init --revalidate`'s job. The
    # real checks above are what refuse.
    try:
        update_note = preset_update_note(profile_file)
    except BuildProfileError as e:
        raise click.UsageError(str(e)) from e
    if update_note:
        note(update_note)

    # Before the verdict, and worded as `osprey build` words it: the reader has
    # to know WHICH document was judged before being told it is fine.
    if variant.selected:
        note(
            f"host variant {variant.name} "
            f"({VARIANT_DIRNAME}/{variant.name}.yml over {PROFILE_FILENAME})"
        )

    report(f"✓ Profile is valid: {profile_file}", style=Styles.SUCCESS)

    rows: list[tuple[str, object]] = [("Name", build_profile.name)]
    # Named separately because "valid" says nothing about whether the deploy
    # coordinates were read at all: a profile that omits the block and one whose
    # block checked out otherwise print the same line.
    if build_profile.deploy is not None:
        deploy = build_profile.deploy
        rows.append(("Deploy", f"{deploy.ci} CI → {deploy.host.user}@{deploy.host.name}"))
    section("", rows)

    report("")
    report("Next steps:")
    note("1. Render the deployment: osprey build")
    note("2. Re-run this command after editing the profile")


@click.command()
@click.argument("target", required=False, type=click.Path(exists=True, path_type=Path))
@repo_option
def validate(target: Path | None, repo: Path | None) -> None:
    """Check the deployment profile without building.

    With no argument, validates the deployment repo enclosing the working
    directory. TARGET names a different profile to check instead — a tier
    under tiers/, an app under apps/, or a directory holding a profile.yml.

    Resolves extends: chains and runs the full consistency check — convention
    directories, the data: tree, service templates, lifecycle steps, env vars —
    then lints the declared web stack against the config a build would render.
    Every problem found is reported, not just the first.

    For a profile `osprey init` wrote, and each tier and app it copied, a note
    says when the bundled preset has changed since. Your own edits to the
    profile are never refused: it is yours to change.

    Judges what this host builds: when .env.variant selects an overlay under
    profiles/, that overlay is merged in first, and named in the output.

    Exits 0 when the profile is valid, 2 with the accumulated errors when it is
    not, so a CI job can gate on it.

    Examples:

    \b
      $ osprey validate
      $ osprey validate apps/logbook.yml
      $ osprey validate --repo ~/my-assistant
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
