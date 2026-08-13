"""The ``osprey set`` verb — the one sanctioned CLI write into ``profile.yml``.

Every other lifecycle verb reads the deployment repo's source zone; this one
writes it. The write lands in ``profile.yml`` and nowhere else: ``build/`` is
100% derived, so a CLI edit of the rendered ``build/config.yml`` would be
undone by the next build without ever having reached the file that describes
the deployment. That failure mode — a setting that works until someone
rebuilds — is what having exactly one writer prevents.

The machinery is the build's own write-back
(:func:`~osprey.cli.build_profile_resolve.write_back_cli_overrides`), promoted
rather than reimplemented: comment-preserving round-trip YAML, one atomic
replace of the whole document, ``--set`` key spellings unchanged, and the same
refusal of an ``extends:`` write that materialization makes. A second writer
with its own spelling rules would mean ``osprey set connector=epics`` and
``osprey init --set connector=epics`` landing differently in the same file.

Two shorthands from the retired ``osprey config set-*`` commands survive here
as key spellings: ``connector=`` (folded into ``config.control_system.type`` by
the shared layering step) and ``epics_gateway=`` (expanded below into the
facility's gateway addresses). Both write the literal dotted keys a reader
would otherwise type by hand, so nothing lands in the profile that only this
command can understand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option

#: CLI-only shorthand absorbed from ``osprey config set-epics-gateway
#: --facility NAME``. It never reaches the profile under this name: it is
#: expanded into the dotted gateway keys before anything is written, so the
#: file holds the same entries a hand-editor would write.
EPICS_GATEWAY_KEY = "epics_gateway"

#: Where the EPICS connector's gateway table lives in the rendered config, as
#: the ``config.``-prefixed key path ``--set`` addresses it by.
GATEWAY_KEY_PREFIX = "config.control_system.connector.epics.gateways"


def _expand_shorthands(pairs: tuple[str, ...]) -> tuple[str, ...]:
    """Expand CLI-only shorthands, passing every other pair through untouched.

    Runs before the shared parser sees anything, so a shorthand is indistin-
    guishable from the dotted keys it stands for by the time the write happens.
    """
    expanded: list[str] = []
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if separator and key.strip() == EPICS_GATEWAY_KEY:
            expanded.extend(_gateway_pairs(value.strip()))
        else:
            expanded.append(pair)
    return tuple(expanded)


def _unrecognized_top_level_keys(pairs: tuple[str, ...]) -> list[str]:
    """Top-level keys in *pairs* that the profile schema does not recognize.

    The write itself is happy to put any key into the file — it is a YAML edit,
    not a schema-aware one — but the profile schema is CLOSED, so an unknown
    top-level key makes the very next ``osprey build`` refuse the whole profile.
    Without this the operator learns that at the build, from a message about
    ``profile.yml`` rather than about the command that just edited it.

    A warning rather than a refusal: the value did reach the file, saying so is
    the honest report, and an operator writing a key ahead of the release that
    reads it should not be blocked by this command. The build still refuses.

    ``config.``-prefixed keys are excluded — they address the rendered config,
    whose key space is open, and the profile's ``config:`` block holds them as
    the dotted paths they are.
    """
    from .build_profile import _KNOWN_PROFILE_KEYS

    # Deduplicated through a dict, not a `set()`: the command in this module is
    # NAMED `set`, so at module scope that name is a click Command and calling
    # it re-enters the CLI instead of building a set.
    unknown: dict[str, None] = {}
    for pair in pairs:
        key, separator, _ = pair.partition("=")
        if not separator:
            continue
        head = key.strip().split(".", 1)[0]
        if head and head != "config" and head not in _KNOWN_PROFILE_KEYS:
            unknown[head] = None
    return sorted(unknown)


def _gateway_pairs(facility: str) -> list[str]:
    """The dotted gateway keys ``epics_gateway=<facility>`` stands for.

    Values are rendered as JSON, which is a YAML subset: the port stays an
    integer and ``use_name_server`` stays a boolean through the same
    ``yaml.safe_load`` the ``--set`` parser puts every value through.
    """
    from osprey.templates.data import get_facility_config, list_facilities

    preset = get_facility_config(facility)
    if preset is None:
        known = ", ".join(sorted(list_facilities()))
        raise click.UsageError(
            f"Unknown facility {facility!r} for {EPICS_GATEWAY_KEY}. Known facilities: "
            f"{known}.\n\nA gateway that list does not carry is set by its own keys:\n"
            f"  osprey set {GATEWAY_KEY_PREFIX}.read_only.address=gw.example.org "
            f"{GATEWAY_KEY_PREFIX}.read_only.port=5064"
        )

    gateways: dict[str, dict[str, Any]] = preset["gateways"]
    return [
        f"{GATEWAY_KEY_PREFIX}.{role}.{field}={json.dumps(value)}"
        for role, gateway in gateways.items()
        for field, value in gateway.items()
    ]


@click.command(name="set")
@click.argument("pairs", nargs=-1, metavar="KEY=VALUE...")
@repo_option
def set(pairs: tuple[str, ...], repo: Path | None) -> None:
    """Write settings into the deployment profile.

    Each KEY=VALUE is written into this repo's profile.yml in place, comments
    intact. That file is the source of truth, so this is the only command that
    edits configuration for you — the rendered build/config.yml is generated
    from it and is never hand-edited or CLI-edited. Run `osprey build` to carry
    a setting through to build/, then `osprey up` to deploy it.

    KEY is a top-level profile key (provider, model, tier, channel_finder_mode,
    connector) or a dotted path. Keys under `config.` address the rendered
    config: `config.control_system.type=epics` writes that literal dotted entry
    into the profile's config: block, replacing the value already there.

    VALUE is read as YAML — true/false become booleans, bare numbers become
    numbers, everything else is text.

    Two shorthands stand in for longer key paths: `connector=` writes
    config.control_system.type, and `epics_gateway=` writes a known facility's
    EPICS gateway addresses.

    Examples:

    \b
      $ osprey set model=sonnet
      $ osprey set connector=epics
      $ osprey set tier=1 channel_finder_mode=in_context
      $ osprey set config.facility.name='ALS Storage Ring'
      $ osprey set epics_gateway=als
      $ osprey set --repo ~/als-assistant config.control_system.writes_enabled=true
    """
    from osprey.deployment.staleness import check_drift
    from osprey.errors import BuildProfileError

    from .build_profile import write_back_cli_overrides

    if not pairs:
        raise click.UsageError(
            "Nothing to set. Name at least one KEY=VALUE, e.g. `osprey set model=sonnet`.\n\n"
            "To see what the profile currently says, run `osprey config`."
        )

    # Resolved before anything is parsed: standing outside a deployment repo is
    # the operator's first problem, and a message about a malformed pair would
    # send them looking in the wrong place.
    repo_root = find_repo_root(repo)
    profile_path = repo_root / PROFILE_FILENAME

    expanded = _expand_shorthands(pairs)

    try:
        # Every pair is merged into one layer before any of it is written, so a
        # command line with one bad pair in it changes nothing at all.
        written = write_back_cli_overrides(profile_path, set_pairs=expanded)
    except BuildProfileError as e:
        raise click.UsageError(str(e)) from e

    click.echo(f"✓ Wrote {len(written)} setting(s) into {profile_path}")
    for key in written:
        click.echo(f"    {key}")

    unrecognized = _unrecognized_top_level_keys(expanded)
    if unrecognized:
        click.echo(
            f"⚠ Not a profile key: {', '.join(unrecognized)} — written, but the "
            "profile schema is closed, so `osprey build` will refuse this profile "
            "until it is corrected or removed. To address the rendered config "
            "instead, prefix the key with `config.`."
        )

    # The build this edit just invalidated, named while the operator is still
    # here. `check_drift` never raises, so a repo with no build/ — the common
    # case right after `init` — reports that rather than failing the write.
    click.echo(check_drift(repo_root).status_line)
