"""Profile command group — inspect build profiles, and materialize one.

A build profile is the editable source a deployment owns: the ``profile.yml`` at
its repo root plus the data tree and convention directories beside it. This
group holds the read-only verbs that act on that source, kept separate from
``osprey build``, which consumes a profile and renders ``build/`` from it.

The write half lives elsewhere by design: ``osprey init`` creates the deployment
repo (through :func:`_materialize_profile_directory` here, which is the one
function in this module that writes anything), ``osprey init --revalidate``
brings an existing one up to date, and ``osprey set`` edits the profile. Every
verb in this group only reads.

``osprey validate`` is the top-level spelling of ``profile validate``, and the
check both run is :func:`osprey.cli.validate_cmd.check_profile_file` — one
implementation, because the only thing that differs between the spellings is
whether the target is required. ``presets`` is the other end of the same rule:
:func:`echo_preset_names` here is also what ``osprey init --list-presets``
prints.

Usage:
    osprey profile presets
    osprey profile validate ~/deployments/my-assistant
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import click

from osprey.profiles.providers import PROVIDERS_FILENAME
from osprey.utils.workspace import APPS_DIRNAME, LAYER_DIR_KINDS, TIERS_DIRNAME
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.logger import get_logger

from .output import report, section
from .profile_conventions import (
    CONTEXT_BASELINE_FILENAME,
    PER_USER_CONTEXT_DIRNAME,
    PROFILE_TRIGGERS_FILENAME,
    context_baseline_slot,
)
from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option

if TYPE_CHECKING:
    # Annotation only — the profile model is imported lazily inside the command
    # bodies to keep `osprey --help` off the build-profile import chain (the
    # lazy-import budget test in tests/cli/test_main.py pins this).
    from .build_profile_model import BuildProfile
    from .templates.manager import TemplateManager
    from .templates.preset_data import PresetData

logger = get_logger("profile")


@click.group()
def profile() -> None:
    """Author, validate, and inspect build profiles."""


def echo_preset_names() -> None:
    """Print every bundled preset name, one per line.

    Two surfaces ask this question — ``osprey profile presets`` below and
    ``osprey init --list-presets``, an eager flag that answers before anything
    else parses — and an operator comparing the two lists is entitled to the
    same answer from both. One writer, so there is nothing to drift.
    """
    from .build_profile import list_presets

    for name in list_presets():
        report(name)


@profile.command()
def presets() -> None:
    """List bundled preset names, one per line.

    Every name printed here is usable as 'osprey init --preset NAME'.
    """
    echo_preset_names()


@profile.command()
def artifacts() -> None:
    """List every artifact the six profile lists can name.

    One section per list key (hooks, rules, skills, agents, output_styles,
    web_panels), each entry with a one-line description. The same menu
    appears as commented entries in an emitted profile.yml.
    """
    from .build_profile_emit import artifact_menu_catalog

    for kind, entries in artifact_menu_catalog().items():
        section(kind, list(entries))


@profile.command()
@click.argument("target", type=click.Path(exists=True, path_type=Path))
def validate(target: Path) -> None:
    """Check a profile without building anything.

    TARGET is a profile directory (its profile.yml is used) or a path to a
    profile file. Resolves 'extends:' chains and runs the full consistency
    check — convention directories, the 'data:' tree, service templates,
    lifecycle steps, env vars — reporting every problem found, not just the
    first.

    Exits 0 when the profile is valid, 2 with the accumulated errors when it is
    not.

    Examples:

    \b
      $ osprey profile validate ~/deployments/my-assistant
      $ osprey profile validate ~/deployments/my-assistant/tiers/readonly.yml
    """
    # The check itself belongs to the verb, not to either spelling of it: this
    # command differs from `osprey validate` only in requiring its TARGET, so
    # that requirement is the whole of what lives here.
    from .validate_cmd import check_profile_file, profile_file_at

    check_profile_file(profile_file_at(target))


@profile.command(name="card")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the card as a single JSON document on stdout (machine-readable)",
)
@repo_option
def card(as_json: bool, repo: Path | None) -> None:
    """Show what this deployment is made of.

    The composition card `osprey init` prints, for a repo that already exists:
    who can sign in and with what rights, what the agent runs on, what machine
    it talks to, and what runs beside it. Everything is read off the repo's
    profile.yml and the tier files beside it — nothing is built, started, or
    probed.

    With --json, the same rows as one JSON array of {group, label, value}
    objects on stdout, and every warning on stderr.

    Examples:

    \b
      $ osprey profile card
      $ osprey profile card --repo ~/deployments/my-assistant
      $ osprey profile card --json
    """
    import json
    from contextlib import nullcontext

    from .build_profile import resolve_build_profile
    from .main import lifecycle_reporter
    from .output import machine_mode
    from .phase_reporter import current_reporter
    from .profile_card import Segment, _card_segment_lines, card_rows

    # `--json` gets `machine_mode` and no reporter; the plain card gets the
    # reporter and no `machine_mode`. The two are exclusive on purpose: the
    # reporter's live region mounts on a terminal's stdout and would re-wrap
    # the document a `--json` reader is parsing.
    with (
        machine_mode() if as_json else nullcontext(),
        nullcontext() if as_json else lifecycle_reporter(),
    ):
        repo_root = find_repo_root(repo)
        resolved, _profile_dir = resolve_build_profile(repo_root / PROFILE_FILENAME, None)
        deltas = read_layer_deltas(repo_root)

        # One derivation, one refusal. The card `init` prints is advisory and
        # swallows what it cannot derive, because the repo is already made by
        # then; here the card IS the output, so a shape this reader never met
        # is a failure in both modes rather than a card with a hole in it.
        rows: list[dict[str, str]] = []
        lines: list[list[Segment]] = []
        try:
            if as_json:
                rows = card_rows(resolved, deltas)
            else:
                lines = _card_segment_lines(resolved, deltas)
        except Exception as e:
            raise click.ClickException(f"Cannot derive the card: {e}") from e

        if as_json:
            click.echo(json.dumps(rows))
            return

        # Printed the way `init` prints it, so the two cards are one card.
        reporter = current_reporter()
        for line in lines:
            if line:
                reporter.echo_segments(line)
            else:
                reporter.echo("")


# Directory name the materialized data tree gets, and the value written to the
# profile's ``data:`` key. One constant so the copy target and the emitted key
# cannot drift apart.
_PROFILE_DATA_DIRNAME = "data"

# Convention directory holding one subdirectory of per-user web-terminal context
# per roster user. Named from the convention table rather than spelled again, so
# the slots seeded here are the ones the build copies from.
_CONTEXT_CONVENTION_DIRNAME = PER_USER_CONTEXT_DIRNAME

# The deployment's secret channel (FR-1). `.env.example` is the documented
# variable list, rendered from the SAME template every other render uses, so the
# two can never document different variables. `.env` beside it holds the values
# and IS the deployment's secret store — the file compose is handed as
# `--env-file` and every `${VAR}` expansion reads. Nothing derives a second copy
# of it, and a build only ever appends, which is what makes a secret survive a
# rebuild.
_PROFILE_ENV_FILENAME = ".env"
_PROFILE_ENV_EXAMPLE_FILENAME = ".env.example"
_ENV_EXAMPLE_TEMPLATE = "project/env.example.j2"

# Section header the shell-harvested keys are written under, distinct from the
# banner `osprey up` appends its minted service tokens beneath: the two have
# different origins, and a reader should be able to tell which values came from
# their own shell. Named after the verb that actually ran, because the banner is
# what tells a reader months later where a value in their `.env` came from — and
# an emitted artifact naming a command that does not exist is worse than no
# attribution at all (SC-8).
REPO_SEEDED_ENV_BANNER = "# ── Seeded by `osprey init` from your shell ──"

# Section header for values the profile itself declares under `env.defaults`.
# A third origin, and a third banner for the same reason the two above are
# distinct: these came from the preset's author, not from this operator's shell
# and not from a deploy mint, and the reader deciding whether a value is safe
# to change needs to know that.
PROFILE_DEFAULTS_ENV_BANNER = "# ── Declared by this profile (env.defaults) — edit freely ──"

#: Source-zone entries a repo-root materialization owns, and therefore the exact
#: set a re-materialization is allowed to replace. Everything else in a
#: deployment repo — ``.git``, ``.env``, ``var/``, ``build/``, ``ci-extra.yml``,
#: the CI files — belongs to the operator or to another command, so re-running
#: ``osprey init`` over an existing repo can never cost a secret, an agent's
#: memory, or a hand-written CI job.
MATERIALIZED_SOURCE_ENTRIES: tuple[str, ...] = (
    "profile.yml",
    PROVIDERS_FILENAME,
    _PROFILE_DATA_DIRNAME,
    *LAYER_DIR_KINDS,
    PROFILE_TRIGGERS_FILENAME,
    _CONTEXT_CONVENTION_DIRNAME,
    _PROFILE_ENV_EXAMPLE_FILENAME,
)

# Build exhaust that a source checkout may hold inside a bundle's data tree but
# a wheel install never does — hatch excludes it from the package (see the
# `exclude` in pyproject.toml). Copying it would make an emission from a
# checkout differ from an emission from a wheel, so the copy drops it. Paths are
# relative to the data root, matched segment-wise so an unrelated `results/`
# elsewhere in the tree is untouched.
_EXCLUDED_DATA_SUBTREES: tuple[tuple[str, ...], ...] = (("benchmarks", "results"),)


def _directory_derived_name(dirname: str) -> str:
    """The facility's display name, read off the directory the operator chose.

    ``my-facility`` becomes ``My Facility``. A guess, and a replaceable one —
    ``--set name=...`` overrides it, and the emitted ``profile.yml`` carries the
    result as an ordinary editable key.
    """
    return dirname.replace("-", " ").replace("_", " ").title()


def _data_copy_ignore(source_root: Path) -> Callable[[str, list[str]], set[str]]:
    """``shutil.copytree`` ignore callable dropping :data:`_EXCLUDED_DATA_SUBTREES`.

    Matching is by position in the tree, not by name: only the directory at
    exactly ``benchmarks/results`` under ``source_root`` is dropped, so a
    ``results/`` a bundle legitimately ships anywhere else comes across.
    """
    root = source_root.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        try:
            here = Path(directory).resolve().relative_to(root).parts
        except ValueError:
            # Outside the tree (a symlinked subdirectory): nothing to drop.
            return set()
        return {
            subtree[-1]
            for subtree in _EXCLUDED_DATA_SUBTREES
            if subtree[:-1] == here and subtree[-1] in names
        }

    return ignore


class _ProviderCatalogPlan(NamedTuple):
    """The ``providers.yml`` a materialization is about to write."""

    text: str
    """Complete file content, ready to write at the repo root."""

    entries: dict[str, Any]
    """Provider name → entry, parsed. What the ``.env`` seeding reads to learn
    which environment variable each provider's key lives in."""

    content_hash: str
    """:func:`~osprey.profiles.providers.compute_providers_hash` of that content,
    stamped into the emitted profile's ``provenance.providers_hash``."""

    carried: tuple[str, ...]
    """Entry names taken from the repo's existing catalog because the packaged
    one does not declare them — the operator's own gateways. Empty on a first
    init, and on a re-init over a catalog nobody has added to."""


def _previous_provider_catalog(target: Path) -> Path | None:
    """The catalog a re-materialization must carry the operator's entries out of.

    ``osprey init --force`` moves the whole source zone into
    :data:`~.repo_resolver.HELD_SOURCE_ZONE_DIRNAME` before this materializer
    runs, so by the time the question is asked the repo root normally no longer
    holds one and the held copy is the live answer. The repo root is checked
    first anyway, so the function answers the same question for a materialization
    that runs over a standing zone.
    """
    from .repo_resolver import HELD_SOURCE_ZONE_DIRNAME

    for candidate in (
        target / PROVIDERS_FILENAME,
        target / HELD_SOURCE_ZONE_DIRNAME / PROVIDERS_FILENAME,
    ):
        if candidate.is_file():
            return candidate
    return None


def _plan_provider_catalog(target: Path, *, keep_values: bool = False) -> _ProviderCatalogPlan:
    """Resolve the catalog to write into ``target`` from the one OSPREY ships.

    A first init copies the packaged catalog byte for byte. Over a repo that
    already has one there are two rules, chosen by the caller:

    * ``osprey init --force`` re-materializes the source zone, discarding
      edits, so it REFRESHES every packaged entry to the packaged text while
      carrying over every entry the packaged catalog does not declare,
      comments included. Those are the operator's own gateways, and nothing
      else in the repo remembers them.
    * ``osprey init --revalidate`` (``keep_values``) treats the file the way it
      treats ``profile.yml``: every value it states is kept, an edited packaged
      entry included, and only what it lacks is added — a packaged entry it
      does not have, and a packaged key an existing entry does not state.
      ``osprey validate`` reports each difference that remains as drift.

    Build-time resolution is the opposite rule on purpose
    (:func:`~osprey.profiles.providers.load_provider_catalog` replaces rather
    than merges): once the file exists it is the whole catalog, so deleting an
    entry there deletes it. This function is the one place the two documents
    meet, and it runs only when the operator asks for one: ``osprey init`` over
    an existing repo, and ``osprey init --revalidate``.

    Args:
        target: The repo root being materialized. Read only — the plan is
            returned, and the caller writes it once the directory exists.
        keep_values: Keep every value the existing file states and add only
            what it lacks, rather than refreshing the packaged entries.

    Returns:
        The file content, its parsed entries, its hash, and the names carried
        over (:class:`_ProviderCatalogPlan`).

    Raises:
        click.UsageError: The repo's existing catalog does not parse. Refusing
            is the only honest answer: the merge cannot tell an operator's
            entry from a packaged one in a file it cannot read, and silently
            replacing it would drop gateways.
    """

    from osprey.profiles.providers import (
        compute_providers_hash,
        load_provider_catalog,
        packaged_catalog_path,
    )

    packaged_path = packaged_catalog_path()
    packaged_text = packaged_path.read_text(encoding="utf-8")

    def _packaged_only() -> _ProviderCatalogPlan:
        return _ProviderCatalogPlan(
            packaged_text,
            dict(load_provider_catalog(None).entries),
            compute_providers_hash(packaged_path),
            (),
        )

    previous = _previous_provider_catalog(target)
    if previous is None:
        return _packaged_only()

    try:
        packaged_entries = load_provider_catalog(None).entries
        existing_entries = load_provider_catalog(previous.parent).entries
    except BuildProfileError as e:
        raise click.UsageError(
            f"Cannot re-materialize over the provider catalog already in this repo: {e} "
            f"The merge tells the entries OSPREY ships from yours, and it cannot "
            f"while that file does not parse — fix it, or move it aside."
        ) from e

    carried = tuple(str(name) for name in existing_entries if name not in packaged_entries)

    # Merged as TEXT, not through a YAML round-trip, either way: the packaged
    # half arrives byte-identical, and the repo's own lines keep their comments,
    # spacing and quoting — precisely what a round-trip loses, because ruamel
    # files a comment before a key under the key BEFORE it.
    if not keep_values:
        if not carried:
            return _packaged_only()
        blocks = _catalog_entry_blocks(previous.read_text(encoding="utf-8"))
        text = packaged_text if packaged_text.endswith("\n") else packaged_text + "\n"
        for name in carried:
            block = blocks.get(name)
            if block:
                text += block if block.endswith("\n") else block + "\n"
        return _staged_provider_plan(text, previous, packaged_entries, existing_entries, carried)

    text = previous.read_text(encoding="utf-8")
    packaged_raw = _catalog_mapping(packaged_text)
    existing_raw = _catalog_mapping(text)
    packaged_blocks = _catalog_entry_blocks(packaged_text)
    for name, entry in packaged_raw.items():
        present = existing_raw.get(name)
        if not isinstance(present, dict) or not isinstance(entry, dict):
            continue
        lacking = [key for key in entry if key not in present]
        if lacking:
            text = _add_entry_keys(text, name, _entry_key_lines(packaged_blocks[name], lacking))
    missing = [name for name in packaged_raw if name not in existing_raw]
    if missing:
        text = _add_entries(text, "".join(packaged_blocks[name] for name in missing))
    return _staged_provider_plan(text, previous, packaged_entries, existing_entries, carried)


def _staged_provider_plan(
    text: str,
    previous: Path,
    packaged_entries: Mapping[str, Any],
    existing_entries: Mapping[str, Any],
    carried: tuple[str, ...],
) -> _ProviderCatalogPlan:
    """The plan for writing *text*, validated and hashed as every reader of the file sees it."""
    import tempfile

    from osprey.profiles.providers import compute_providers_hash, load_provider_catalog

    # Staged rather than hashed in memory: the validation and the hash are the
    # ones every other reader of a catalog file gets, over the exact bytes this
    # repo is about to hold, so a merge that produced something unreadable
    # fails here rather than at the next build.
    with tempfile.TemporaryDirectory() as staging:
        staged = Path(staging) / PROVIDERS_FILENAME
        staged.write_text(text, encoding="utf-8")
        entries = dict(load_provider_catalog(Path(staging)).entries)
        digest = compute_providers_hash(staged)
    if set(entries) != set(packaged_entries) | set(existing_entries):
        raise click.UsageError(
            f"Could not merge the provider catalog at {previous} with the one OSPREY "
            f"ships: the result would not carry every entry. Nothing was changed. Move "
            f"that file somewhere you can read it, re-run this command to get a fresh "
            f"`{PROVIDERS_FILENAME}`, then paste your own entries back into it."
        )
    return _ProviderCatalogPlan(text, entries, digest, carried)


def _catalog_mapping(text: str) -> dict[str, Any]:
    """The ``providers:`` mapping of a catalog file, as written (no defaults applied)."""
    import yaml

    document = yaml.safe_load(text) or {}
    providers = document.get("providers") if isinstance(document, dict) else None
    return providers if isinstance(providers, dict) else {}


#: An entry's own key: four spaces, then a name.
_CATALOG_ENTRY_FIELD = re.compile(r"^ {4}(?![ #])(.+?):")


def _entry_key_lines(block: str, keys: list[str]) -> str:
    """The lines of *block* (one sliced entry) that state *keys*, each with its body."""
    wanted = set(keys)
    picked: list[str] = []
    taking = False
    for line in block.splitlines(keepends=True):
        field = _CATALOG_ENTRY_FIELD.match(line)
        if field:
            taking = field.group(1).strip().strip("'\"") in wanted
        elif _CATALOG_ENTRY_KEY.match(line) or line.lstrip().startswith("#"):
            taking = False
        if taking:
            picked.append(line)
    return "".join(picked)


def _catalog_spans(text: str) -> tuple[list[str], dict[str, int], int]:
    """The file's lines, the index after each entry's last stated line, and after the last entry's.

    A trailing comment run belongs to whatever follows it, so an addition goes
    above it.
    """
    lines = text.splitlines(keepends=True)
    ends: dict[str, int] = {}
    current: str | None = None
    inside = False
    last = len(lines)
    for index, line in enumerate(lines):
        if not inside:
            inside = bool(_CATALOG_ENTRIES_KEY.match(line.rstrip()))
            last = index + 1
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _CATALOG_ENTRY_KEY.match(line)
        if match:
            current = match.group(1).strip().strip("'\"")
        elif not line.startswith("  "):
            break
        if current is not None:
            ends[current] = index + 1
        last = index + 1
    return lines, ends, last


def _add_entry_keys(text: str, name: str, key_lines: str) -> str:
    """*text* with *key_lines* added at the end of entry *name*."""
    lines, ends, _last = _catalog_spans(text)
    at = ends[name]
    if at and not lines[at - 1].endswith("\n"):
        lines[at - 1] += "\n"
    lines[at:at] = key_lines.splitlines(keepends=True)
    return "".join(lines)


def _add_entries(text: str, blocks: str) -> str:
    """*text* with whole entry *blocks* added after its last entry."""
    lines, _ends, at = _catalog_spans(text)
    if at and not lines[at - 1].endswith("\n"):
        lines[at - 1] += "\n"
    lines[at:at] = blocks.splitlines(keepends=True)
    return "".join(lines)


#: The line the entries hang under. A trailing comment on it is ordinary YAML
#: and must not stop the slicer: matching only the bare key left every entry
#: unrecovered, which the caller reads as "the merge lost them" and refuses on.
_CATALOG_ENTRIES_KEY = re.compile(r"^providers:\s*(?:#.*)?$")

#: A ``providers:`` entry key: two spaces, then a name. Deeper lines are the
#: entry's body, column-0 lines end the block, and a comment run at this depth
#: belongs to the entry BELOW it — the convention the packaged catalog follows
#: and the one an operator copying an entry inherits.
_CATALOG_ENTRY_KEY = re.compile(r"^ {2}(?![ #])(.+?):\s*(?:#.*)?$")


def _catalog_entry_blocks(text: str) -> dict[str, str]:
    """Slice a catalog file into one verbatim text block per provider entry.

    Text rather than parsed data because what has to survive being carried into
    a refreshed catalog is exactly what a parse throws away: the operator's
    comments, their spacing, their quoting. The blocks are re-validated as YAML
    once they are joined (:func:`_plan_provider_catalog`), so a slice that came
    out malformed is caught before the file is written rather than trusted.

    Two shapes the slicer does not read, both of which end as a refusal rather
    than as a lost entry, because the caller compares the names it recovered
    against the names the loader parsed: a flow-style mapping
    (``providers: {cborg: {...}}``), where there are no per-entry lines to cut;
    and a YAML 1.1 boolean-like key (``yes:``, ``on:``), which the loader reads
    as ``True`` while this reads the characters. Both are rare, both leave the
    operator's file untouched, and the refusal says how to recover.

    Args:
        text: A whole ``providers.yml``.

    Returns:
        Entry name → its lines, leading comment run included. Entries the
        slicer cannot delimit are simply absent, which the caller notices as a
        missing name rather than as silent damage.
    """
    blocks: dict[str, str] = {}
    pending: list[str] = []
    body: list[str] = []
    current: str | None = None
    inside = False

    def close() -> None:
        if current is not None:
            blocks[current] = "".join(body)

    for line in text.splitlines(keepends=True):
        if not inside:
            inside = bool(_CATALOG_ENTRIES_KEY.match(line.rstrip()))
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            pending.append(line)
            continue
        match = _CATALOG_ENTRY_KEY.match(line)
        if match:
            close()
            current = match.group(1).strip().strip("'\"")
            body = [*pending, line]
            pending = []
        elif line.startswith("  ") and current is not None:
            body.extend(pending)
            body.append(line)
            pending = []
        else:
            # A column-0 key: the `providers:` mapping has ended.
            break
    close()
    return blocks


def _triggers_source(resolved: BuildProfile, preset_dir: Path) -> Path | None:
    """The trigger-config file ``resolved``'s ``dispatch:`` block names.

    ``None`` for a profile that declares no dispatch block — there is nothing to
    materialize and nothing to repoint.

    Resolution is the build's own
    (:func:`~osprey.cli.build_profile_presets.resolve_triggers_path`):
    profile-relative first, then the bundled triggers directory.
    ``resolve_build_profile`` has already rejected a value that resolves to
    neither, so a miss here is a packaging problem rather than something the
    caller could have got wrong.

    Raises:
        BuildProfileError: If neither candidate exists.
    """
    if resolved.dispatch is None:
        return None

    from .build_profile_presets import resolve_triggers_path

    source = resolve_triggers_path(preset_dir, resolved.dispatch.triggers)
    if source is not None:
        return source.path
    raise BuildProfileError(
        f"dispatch.triggers not found: {resolved.dispatch.triggers!r} — looked in "
        f"{preset_dir} and the bundled triggers directory."
    )


def _roster_user_names(config: Mapping[str, Any]) -> list[str]:
    """Web-terminal user names a profile's roster declares, in roster order.

    Empty for a profile with no web-terminal module and for one whose module is
    switched off — a layer, say, which attaches to a hosting project's
    web tier and stands up no roster of its own.

    Only the *route* to the subtree is this function's own: a profile spells its
    settings as a flat ``config:`` block, so the dotted keys have to be folded
    (:func:`~.build_profile_emit.effective_web_terminals`) before there is a
    subtree to read. Deriving names from it is
    :func:`~osprey.deployment.web_terminals.roster.roster_user_names`, the
    same call the build makes against the built project's ``config.yml`` — which
    is what makes the slots seeded here exactly the ones the build looks for.
    """
    from osprey.deployment.web_terminals.roster import roster_user_names

    from .build_profile_emit import effective_web_terminals

    return roster_user_names(effective_web_terminals(config))


def _stands_up_a_web_tier(config: Mapping[str, Any]) -> bool:
    """Whether the profile switches the web-terminal module on.

    An empty roster covers two cases the seeding has to tell apart: a profile
    with no web-terminal module at all, which has no context convention to
    seed, and one whose module is on but names no users yet, which does — its
    shared baseline is roster-independent, so it belongs in the profile whether
    or not anyone is on the roster. The subtree is read through the same fold
    :func:`_roster_user_names` uses, on the same "not switched on means no web
    tier" rule as
    :func:`~osprey.deployment.web_terminals.roster.roster_user_names`.
    """
    from .build_profile_emit import effective_web_terminals

    return bool(effective_web_terminals(config).get("enabled"))


def _config_node(path: tuple[str, ...], value: Any, wanted: tuple[str, ...]) -> Any:
    """What a single ``config:`` key sets at ``wanted``, or ``None``.

    A ``config:`` block addresses paths in whatever spelling its author chose:
    the dotted key itself (``claude_code.provider``), or an ancestor key
    (``claude_code``) carrying the path nested inside its value. Both are read
    here, so a profile cannot hide a provider selection behind a spelling.

    A key DEEPER than ``wanted`` addresses something inside the value rather
    than the value itself and returns ``None``.
    """
    if path == wanted:
        return value
    if wanted[: len(path)] != path:
        return None
    probe: Any = value
    for part in wanted[len(path) :]:
        probe = probe.get(part) if isinstance(probe, Mapping) else None
    return probe


def _providers_named_by(provider: Any) -> set[str]:
    """The provider name one profile layer selects.

    The top-level ``provider:`` field is the only place a layer names one.
    ``claude_code.provider`` and ``api.providers.*`` are rendered by the build
    from this field and ``providers.yml``, and a ``config:`` spelling of either
    is refused (:mod:`osprey.cli.derived_keys`), so there is no second spelling
    to union in.

    Args:
        provider: The layer's top-level ``provider:`` key. A layer that
            overrides it names its own; one that does not inherits the host's,
            which is already counted.

    Returns:
        The name the layer selects, or empty when it selects none.
    """
    if isinstance(provider, str) and provider.strip():
        return {provider.strip()}
    return set()


def _parsed_layer_deltas(
    render_texts: Mapping[str, str], *, require_mapping: bool = False
) -> dict[str, Mapping[str, Any]]:
    """Parse every emitted layer, naming the file a bad one came from.

    A delta gets no resolution round-trip of its own — it is meaningless without
    the host beside it — so this parse is the ONLY check that the line-level
    ``extends:`` surgery (:func:`~.build_profile_emit.emit_layer_delta_yaml`)
    left valid YAML behind. It therefore happens once, here, and its result
    serves every later reader: a second parse elsewhere would either raise a
    bare ``YAMLError`` first or silently disagree with this one.

    A delta that parses to something other than a mapping (an empty file, a
    stray scalar) is carried as an empty mapping: readers ask it for keys, and
    the emitted text is written out either way.

    Args:
        render_texts: ``{repo-relative layer file: text}``, as
            :func:`_layer_profile_texts` emits them. The result is keyed by
            each file's stem — the render name every reader asks for.
        require_mapping: Refuse a document that is not a mapping instead of
            carrying it as an empty one. Off for the emission path above, whose
            texts this module wrote and which is written out whatever it parses
            to; on for :func:`read_layer_deltas`, where the text came off disk
            and an empty delta would misreport a file the operator has to fix.

    Raises:
        BuildProfileError: If any delta is not valid YAML, or — under
            ``require_mapping`` — parses to something that is not a mapping.
    """
    return {
        _layer_key_name(key): _parse_layer_delta(
            text,
            f"Emitted layer {_layer_key_path(key)}",
            require_mapping=require_mapping,
        )
        for key, text in render_texts.items()
    }


def _layer_key_name(key: str) -> str:
    """The render name an emitted text's key stands for: ``tiers/admin.yml`` -> ``admin``.

    A bare key (no folder, no suffix) is already the name.
    """
    return Path(key).stem if key.endswith(".yml") else key


def _layer_key_path(key: str) -> str:
    """The repo-relative file an emitted text's key names, as a refusal shows it."""
    return key if key.endswith(".yml") else f"{TIERS_DIRNAME}/{key}.yml"


def _parse_layer_delta(text: str, shown: str, *, require_mapping: bool) -> Mapping[str, Any]:
    """Parse one layer, naming ``shown`` in any refusal.

    The single parse behind :func:`_parsed_layer_deltas` and
    :func:`read_layer_deltas`. The two differ only in what a refusal calls the
    document — the file the emission would have written, or the file the
    operator opens — so the caller supplies that and nothing else varies.

    Args:
        text: The delta's YAML text.
        shown: How a refusal names this document (``"Layer
            tiers/admin.yml"``).
        require_mapping: Refuse a document that is not a mapping instead of
            carrying it as an empty one.

    Raises:
        BuildProfileError: If ``text`` is not valid YAML, or — under
            ``require_mapping`` — parses to something that is not a mapping.
    """
    import yaml

    try:
        delta = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise BuildProfileError(f"{shown} is not valid YAML: {e}") from e
    if isinstance(delta, Mapping):
        return delta
    if require_mapping:
        raise BuildProfileError(f"{shown} is not a YAML mapping")
    return {}


def read_layer_deltas(repo_root: Path) -> dict[str, Mapping[str, Any]]:
    """Read the on-disk delta of every tier and app layer a repo carries.

    The read half of what :func:`_layer_profile_texts` emits, over the same
    listing the build renders one project per layer from
    (:func:`~osprey.cli.profile_root.layer_files`): the files directly under
    ``tiers/``, then ``apps/``, beside ``profile.yml``. A file there IS a layer
    the build deploys, so every one is read and nothing else is.

    Args:
        repo_root: The deployment repo whose layer directories are read. The
            layer set is the files on disk, not anything the profile names.

    Returns:
        ``{layer name: parsed delta}`` keyed by file stem (the render name every
        reader asks for), tiers before apps, sorted by name within each — the
        order the emission keys them in.

    Raises:
        click.ClickException: If a layer file cannot be read or is not a YAML
            mapping. A layer that is there and unreadable is something the
            operator has to fix, and a caller that carried on would state
            something wrong about that layer rather than admit it knows nothing.
    """
    from .profile_root import layer_files

    deltas: dict[str, Mapping[str, Any]] = {}
    for _kind, path in layer_files(repo_root):
        name = path.stem
        if name in deltas:
            # Two layers sharing a name render into one directory; the build
            # refuses that. The first one listed is the one a reader keeps.
            continue
        relative = path.relative_to(repo_root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise click.ClickException(f"Layer {relative} cannot be read: {e}") from e
        try:
            deltas[name] = _parse_layer_delta(text, f"Layer {relative}", require_mapping=True)
        except BuildProfileError as e:
            raise click.ClickException(str(e)) from e
    return deltas


def _referenced_providers(
    resolved: BuildProfile,
    layer_deltas: Mapping[str, Mapping[str, Any]],
    catalog_additions: Collection[str] = (),
) -> set[str]:
    """Every provider the materialized profile — host and renders — references.

    The layers are read too because they share this profile's ``.env``:
    a delta sits in ``tiers/`` and anchors its secrets at the profile root
    (:func:`~.profile_root.resolve_profile_root`), so a render that switches
    provider needs its key in the same file. A delta with no ``provider:`` of
    its own inherits the host's selection, which is already counted.

    An entry the operator added to this repo's ``providers.yml`` counts too,
    named or not. Writing a gateway into the catalog is the statement that this
    deployment reaches it — the same statement a ``config: api.providers.*``
    block used to make, before the catalog became a file of its own — and the
    entry names the variable its key lives in.

    Args:
        resolved: The resolved host profile.
        layer_deltas: The parsed deltas (:func:`_parsed_layer_deltas`).
            Parsed rather than raw text so the one parse that validates them
            is the one read here.
        catalog_additions: Entry names the repo's catalog carries and the
            packaged one does not (:attr:`_ProviderCatalogPlan.carried`).
    """
    names = _providers_named_by(resolved.provider)
    for delta in layer_deltas.values():
        names |= _providers_named_by(delta.get("provider"))
    return names | {str(name) for name in catalog_additions}


class _ShellProviderKeys(NamedTuple):
    """The exported provider variables, split by whether this profile references them.

    A referenced provider contributes every variable it cannot start without:
    its API key and, for a gateway that ships no default host, the endpoint
    beside it (:func:`_exported_provider_keys`). Only keys are ever reported as
    skipped — an endpoint is harvested for a referenced provider or not at all.
    """

    seeded: dict[str, str]
    """``{VAR: value}`` to write into the profile ``.env``: keys in registry
    order, then the endpoints of the providers that need one."""

    skipped: tuple[str, ...]
    """Key variables the shell exports for providers the profile never names."""


#: An ``api_key:`` that defers to the environment, e.g. ``${CBORG_API_KEY}`` or
#: ``${ARGO_API_KEY:-}``. A catalog entry spelling its key any other way — a
#: literal, or an empty placeholder — has no variable to seed.
_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}$")


def _catalog_key_variables(entries: Mapping[str, Any]) -> dict[str, str]:
    """Provider name → the environment variable its catalog entry's key names.

    The catalog is where an endpoint says which variable holds its credential,
    so it is what decides the variable to seed — for a gateway the operator
    added, which no code-level registry knows, and for a packaged provider an
    operator has re-pointed at a variable of their own. Entries with a literal
    key (``ollama``, ``ds4``) and entries with no ``api_key:`` at all contribute
    nothing, which is right: there is no secret to carry.

    Args:
        entries: The resolved catalog entries
            (:attr:`_ProviderCatalogPlan.entries`).
    """
    variables: dict[str, str] = {}
    for name, entry in entries.items():
        if not isinstance(entry, Mapping):  # pragma: no cover - loader refuses these
            continue
        match = _ENV_REFERENCE.match(str(entry.get("api_key", "")).strip())
        if match:
            variables[str(name)] = match.group(1)
    return variables


def _exported_provider_keys(
    providers: Collection[str], catalog_entries: Mapping[str, Any] | None = None
) -> _ShellProviderKeys:
    """Split the shell's provider variables against the providers ``providers`` names.

    ``os.environ`` is the ONLY source (FR-1). A ``.env`` that happens to sit in
    whatever directory ``osprey init`` was run from is ambient state the profile
    cannot reproduce, so nothing is harvested from it.

    This is the one place the shell may seed a key, and it seeds the repo's own
    ``.env`` — the file an operator can read, edit, and account for, and the one
    store compose and every ``${VAR}`` expansion read. There is no second copy
    to derive: a build only ever appends to that file
    (:func:`~osprey.deployment.dotenv.append_profile_env`), so a key seeded here
    survives every rebuild and a key that reaches a running deployment was
    written to this file first.

    Only the keys of providers the RESOLVED PROFILE references are seeded: a
    whole-keyring import copies secrets the profile has no use for into a file
    it then owns forever, which is more surface than the profile needs. The rest
    are reported by name rather than dropped silently (:func:`_skipped_keys_note`)
    — they were seen, and the operator decides whether the omission is right.

    The variable list starts from
    :func:`~.templates.scaffolding.provider_api_key_entries`, the same registry
    derivation the ``.env.example`` beside it is rendered from, so the file that
    holds the values and the file that documents them cannot name different
    variables for a provider OSPREY ships. The catalog is then laid over it: an
    entry's ``api_key: ${VAR}`` is that provider's variable, which is the only
    way a gateway the operator added has one at all, and the right answer when
    a packaged entry has been re-pointed. The two agree for every packaged
    provider, so nothing about a stock repo changes.

    A key is not always the whole of what a provider needs. One that fronts a
    gateway shipping no default host reads its endpoint from the environment
    too, and a ``.env`` holding the key alone is refused by the ``.env.users``
    gate over the endpoint — with the value that would have settled it exported
    in the same shell this harvest read. So the endpoint is taken beside the
    key, named by
    :func:`~osprey.deployment.web_terminals.env_production.required_provider_endpoint_var`
    rather than by a rule of this module's own: ``osprey up``'s seed
    (:func:`~.deploy_cmd.ensure_repo_env`) asks the same helper, and it never
    reaches a repo ``init`` has already written a ``.env`` for, so the two seeds
    have one chance between them to get the name right. A variable the shell
    does not export is left out rather than written empty — an empty assignment
    in ``.env`` is a value, and it would beat the fallback every later reader
    has for an unset one.

    Args:
        providers: Provider names the profile references
            (:func:`_referenced_providers`).
        catalog_entries: The resolved provider catalog
            (:attr:`_ProviderCatalogPlan.entries`). ``None`` falls back to the
            registry alone, for a caller with no catalog in hand.

    Returns:
        The exported variables split into ``seeded`` and ``skipped``. Both are
        empty when the caller exported none.
    """
    import os

    from osprey.deployment.web_terminals.env_production import required_provider_endpoint_var

    from .templates.scaffolding import provider_api_key_entries

    variables: dict[str, str] = {
        entry["provider"]: entry["var"] for entry in provider_api_key_entries()
    }
    variables.update(_catalog_key_variables(catalog_entries or {}))

    seeded: dict[str, str] = {}
    skipped: list[str] = []
    for provider, var in variables.items():
        value = os.environ.get(var)
        if not value:
            continue
        if provider in providers:
            seeded[var] = value
        elif var not in skipped:
            skipped.append(var)
    # The endpoints come after the keys, each resolved against the minimal
    # config the helper reads: the provider it is asked about, and the catalog
    # that may spell that provider's `base_url` as a variable of the operator's
    # own choosing.
    for provider in sorted(providers):
        endpoint_var = required_provider_endpoint_var(
            {
                "claude_code": {"provider": provider},
                "api": {"providers": dict(catalog_entries or {})},
            }
        )
        if not endpoint_var:
            continue
        endpoint = os.environ.get(endpoint_var)
        if endpoint:
            seeded.setdefault(endpoint_var, endpoint)

    # Two entries may name one variable — an operator's gateway sharing a
    # packaged provider's key, say. Seeding wins: the variable IS carried, so
    # reporting it as left out would be a false statement about the `.env`.
    return _ShellProviderKeys(seeded, tuple(var for var in skipped if var not in seeded))


def _skipped_keys_note(skipped: Collection[str]) -> str:
    """One line naming the exported keys this profile did not take.

    One wording, used by both the materializer's log and ``osprey init``'s
    summary: a skipped secret is a thing the operator has to be able to account
    for, and two spellings of the same fact read as two different facts.
    """
    return (
        f"Left out {', '.join(skipped)}. Your shell exports them, but this "
        f"assistant uses a different provider."
    )


def _write_secret_channel(
    target: Path,
    manager: TemplateManager,
    resolved: BuildProfile,
    profile_name: str,
    exported: Mapping[str, str],
    providers: Collection[str] = (),
) -> list[str]:
    """Write the profile's secret channel: ``.env.example``, ``.env``, ``.gitignore``.

    The profile owns its secrets (FR-1): the build derives a project's ``.env``
    from the one written here, so a value set once survives every rebuild.

    ``.env.example`` is always written, and comes from the project template
    (:data:`_ENV_EXAMPLE_TEMPLATE`) rather than from prose of its own — one
    template documents the variable set wherever it is rendered. ``.env`` is
    written ONLY when there is something to seed it with — shell-exported
    provider keys (``exported``) and/or the profile's own ``env.defaults``
    values: an empty secrets file reads as a configured one, and ``cp
    .env.example .env`` is the honest starting point when there is nothing to
    seed.

    ``env.defaults`` values are seeded as real starting values, under their own
    banner (:data:`PROFILE_DEFAULTS_ENV_BANNER`), because a preset declares one
    for exactly the deployments that should come up working without a hand
    edit — a demo login password, say. The append-only writer keeps every
    later authority intact: a value the operator has already set (or that
    ``osprey up`` minted) always wins over the declared default.

    Args:
        target: The profile directory, already created.
        manager: Template manager whose Jinja environment renders the example.
        resolved: The resolved profile, for the ``env:`` block the example
            documents.
        profile_name: Display name, for the example's title line.
        exported: Provider keys to seed (:attr:`_ShellProviderKeys.seeded`).
            Passed in rather than read here because the README rendered earlier
            says whether a ``.env`` was seeded, and the two must agree.

    Returns:
        Profile-relative names of the files written, for the caller's summary.
    """
    from osprey.deployment.dotenv import append_profile_env

    from .templates.scaffolding import (
        provider_api_key_entries,
        provider_base_url_entries,
        service_token_var_entries,
    )

    manager.render_template(
        _ENV_EXAMPLE_TEMPLATE,
        {
            "project_name": profile_name,
            "provider_api_keys": provider_api_key_entries(),
            # Which of those this profile actually uses, so the example puts
            # the one or two keys someone has to fill in above the rest rather
            # than in a list of eight they have to search. Empty from the
            # programmatic path, where the template lists them all as before.
            "active_provider_vars": [
                entry["var"]
                for entry in provider_api_key_entries()
                if entry["provider"] in providers
            ],
            # Endpoints no provider ships a default for: a deployment that
            # picks one of these has to name its own gateway or the launch is
            # refused, so the variable belongs in the file it is told to fill in.
            "provider_base_urls": provider_base_url_entries(),
            # Which of those this profile actually needs — the rest are
            # commented, the same way the unused API keys are.
            "active_provider_base_url_vars": [
                entry["var"]
                for entry in provider_base_url_entries()
                if entry["provider"] in providers
            ],
            "service_token_vars": service_token_var_entries(),
            # The profile's `env:` block, for the example's documentation —
            # the same two keys `osprey build` feeds this template. The
            # `env.defaults` VALUES are additionally seeded into `.env` below.
            "env_required": list(resolved.env.required or []),
            "env_defaults": dict(resolved.env.defaults or {}),
            # No project exists yet; the key is a commented hint either way.
            "project_root": "/path/to/your/project",
        },
        target / _PROFILE_ENV_EXAMPLE_FILENAME,
    )
    written = [_PROFILE_ENV_EXAMPLE_FILENAME]

    if exported:
        # Written through the same append-only, 0600, atomic path `osprey up`
        # uses for its write-back, so there is one writer discipline for the
        # profile `.env` rather than a second one that only new profiles get.
        append_profile_env(target / _PROFILE_ENV_FILENAME, exported, REPO_SEEDED_ENV_BANNER)
        written.append(_PROFILE_ENV_FILENAME)

    defaults = dict(resolved.env.defaults or {})
    if defaults:
        # Same writer, own banner: a declared default an operator has already
        # overridden (or that a mint got to first) is never rewritten.
        append_profile_env(target / _PROFILE_ENV_FILENAME, defaults, PROFILE_DEFAULTS_ENV_BANNER)
        if _PROFILE_ENV_FILENAME not in written:
            written.append(_PROFILE_ENV_FILENAME)

    return written


def _unreadable_access_problems(config: Mapping[str, Any]) -> list[str]:
    """Every roster entry whose ``access`` value the vocabulary cannot resolve.

    ``users[].access`` is authored as a principal set — the ``own``/``any``
    shorthands, or a list of ``self``, ``roster``, ``user:<id>`` and
    ``domain:<domain>`` members — and
    :func:`~osprey.deployment.web_terminals.roster.resolve_access_principals`
    is its single parser. That parser REFUSES an unreadable value rather than
    answering, because a value one surface reads as owner-only while another
    reads it as admitting somebody is the failure the whole key exists to
    avoid.

    So the refusal has to be caught at an altitude that can explain it. Every
    later reader in this materialization parses the same values strictly — the
    per-user context slots, the deploy summary printed at the end — and one of
    them would otherwise surface the parser's ``ValueError`` as a traceback.
    Reported here instead, before anything is written, in the deploy lint's own
    words: this returns the parser's message verbatim, exactly as
    ``web_terminals.invalid_user_access`` does, so both surfaces say the same
    sentence about the same entry. Every offending entry is named at once,
    because a roster is fixed in one edit.

    Args:
        config: The host profile's ``config:`` block, from which the roster is
            folded out.

    Returns:
        One message per unreadable entry, in roster order; empty when every
        authored ``access`` resolves.
    """
    from osprey.deployment.web_terminals.roster import (
        authored_people,
        resolve_access_principals,
    )

    from .build_profile_emit import effective_web_terminals

    problems: list[str] = []
    for entry in authored_people(effective_web_terminals(config)):
        # Bare-string entries carry no `access` at all, and a non-dict entry is
        # a different (separately reported) mistake.
        if not isinstance(entry, dict) or "access" not in entry:
            continue
        try:
            resolve_access_principals(entry)
        except ValueError as refusal:
            problems.append(str(refusal))
    return problems


def _privileged_tier_problems(
    config: Mapping[str, Any],
    privileges: Mapping[str, tuple[str, ...]],
    absolute_privileges: Mapping[str, tuple[str, ...]],
) -> list[str]:
    """Why this profile's roster must not deploy as written, from the layers it names.

    The build-time half of the deployment-editing guard, run where every tier
    and app the roster names has just been emitted and its privileges are
    therefore known exactly (see
    :func:`~osprey.deployment.web_terminals.roster.tier_privileges`). Two
    rules, both owned by that module so the lint belt behind this reports the
    same thing in the same words:

    * ``default_tier`` must not be a privileged tier — it is what every person
      who names no tier inherits;
    * no shared card may resolve to one. Shared is the whole of
      :func:`~osprey.deployment.web_terminals.roster.entry_is_shared` —
      ``access: any``, a ``[user:…]`` or ``[domain:…]`` list, any principal set
      naming somebody beyond the entry's own user, and every app card, which
      everyone on the roster opens.

    **The two rules read two different maps**, and handing one map to both
    would leave this surface exempt on a floorless host preset: every tier
    there sits at the baseline, so the relative map is empty for all of them
    and the shared-card rule would have nothing to report about a terminal
    holding everything. See
    :func:`~osprey.deployment.web_terminals.roster.privileges_beyond_baseline`
    for why the ``default_tier`` rule keeps the relative reading.

    Args:
        config: The host profile's ``config:`` block, from which the roster is
            folded out — and the baseline the relative map is measured
            against.
        privileges: Tier or app name → what it holds BEYOND that baseline. A
            name absent from this mapping contributes nothing.
        absolute_privileges: Tier or app name → what it holds outright. What
            the shared-card rule is judged on.

    Returns:
        One message per problem, ``default_tier`` first, then roster order.
    """
    from osprey.deployment.web_terminals.roster import (
        privileged_default_tier_problem,
        resolve_cards,
        shared_card_privileged_problems,
    )

    from .build_profile_emit import effective_web_terminals

    web_terminals = effective_web_terminals(config)
    problems: list[str] = []
    default_tier = web_terminals.get("default_tier")
    default_problem = privileged_default_tier_problem(
        default_tier,
        privileges.get(default_tier, ()) if isinstance(default_tier, str) else (),
    )
    if default_problem is not None:
        problems.append(default_problem)

    # Every card the roster runs — people, then the derived app cards
    # (`roster_entries`) — each resolved to the tier or app it renders from.
    # Lenient, exactly as the lint belt resolves it: an entry carrying an
    # `access` value the vocabulary does not recognise, or a role nothing
    # binds, is a different mistake, reported by the rule that owns it
    # (`_unreadable_access_problems` here, the deploy lint there), and raising
    # from this rule would replace that report with a worse one.
    resolved = resolve_cards(web_terminals, {}, strict=False)
    problems.extend(shared_card_privileged_problems(resolved, absolute_privileges))
    return problems


def _roster_layers(config: Mapping[str, Any], dispatch_tier: Any) -> tuple[list[str], list[str]]:
    """The tiers and apps this profile's roster names, as ``(tiers, apps)``, sorted.

    A tier is named by any card a person runs on (their ``tier:`` or role, or
    ``default_tier`` when they name neither), by ``default_tier`` itself — the
    tier the next person added inherits — by every role ``authorization``
    binds, since single sign-on may hand any of them out, and by
    ``dispatch.tier``. An app is named by ``shared:``. A profile that stands up
    no web terminals deploys no roster, so it names only ``dispatch.tier``: the
    dispatch workers run that tier with or without a terminal beside them.

    Unreadable roster values name nothing here; the rules that own them report
    them.

    The served set itself is
    :func:`~osprey.deployment.web_terminals.roster.served_render_names`; this
    splits it by kind. A ``shared:`` name the served set already counts as a
    tier is listed under both, so the caller's refusal that a card renders
    from one file reports it.
    """
    from osprey.deployment.web_terminals.render_naming import RENDER_KIND_APP
    from osprey.deployment.web_terminals.roster import served_render_names, shared_app_name

    from .build_profile_emit import effective_web_terminals

    web_terminals = effective_web_terminals(config)
    if not web_terminals.get("enabled"):
        web_terminals = {}
    served = served_render_names(web_terminals, dispatch_tier)
    tiers = {name for name, kind in served.items() if kind != RENDER_KIND_APP}
    apps = {name for name, kind in served.items() if kind == RENDER_KIND_APP}
    shared = web_terminals.get("shared")
    for item in shared if isinstance(shared, list) else []:
        name = shared_app_name(item)
        if name is not None and name in tiers:
            apps.add(name)
    return sorted(tiers), sorted(apps)


def _layer_profile_texts(
    resolved: BuildProfile,
    profile_name: str,
    render_path_prefix: str,
) -> dict[str, str]:
    """Emit the tier and app files this profile's roster names.

    Each is the bundled tier or app materialized as an explicit file
    (:func:`~.build_profile_emit.emit_layer_delta_yaml`): its whole content,
    comments and a provenance block, and no pointer into the package. Which
    ones is :func:`_roster_layers`' answer, so a fresh repo holds a file for
    every tier and app it deploys and nothing else.

    ``render_path_prefix`` is what the emitted headers prefix ``tiers/`` and
    ``apps/`` with when they name a file: the profile's directory name and a
    slash for a nested profile directory, and empty for a repo root, where the
    reader already stands where those paths are relative to.

    One thing beyond emission is decided here, because this is the only moment
    every layer's content is in hand: whether the roster hands deployment
    editing to a terminal that must not have it (see
    :func:`_privileged_tier_problems`). Refusing at materialization is what
    keeps the bad roster from reaching a repo at all.

    Args:
        resolved: The host profile, resolved.
        profile_name: The deployment's display name, which each tier is titled
            after.
        render_path_prefix: See above.

    Returns:
        ``{repo-relative file (tiers/<n>.yml, apps/<n>.yml): text}``, tiers
        first, each kind sorted.

    Raises:
        click.UsageError: With every unusable roster reference named at once —
            or, once they all emit, with every privileged-terminal problem
            named at once.
    """
    from osprey.deployment.web_terminals.roster import (
        privileges_beyond_baseline,
        tier_privileges,
    )

    from .build_profile_emit import emit_layer_delta_yaml
    from .build_profile_merge import render_document
    from .build_profile_presets import list_apps, list_tiers
    from .profile_root import ProfileKind

    dispatch_tier = resolved.dispatch.tier if resolved.dispatch is not None else None
    tiers, apps = _roster_layers(resolved.config, dispatch_tier)
    texts: dict[str, str] = {}
    # What each layer can edit, recorded BOTH ways, because the two rules in
    # `_privileged_tier_problems` read two different answers: relative to
    # the host for the inherited `default_tier`, absolute for a shared card.
    # See `privileges_beyond_baseline`.
    host_privileges = tier_privileges(resolved.config)
    privileges: dict[str, tuple[str, ...]] = {}
    absolute: dict[str, tuple[str, ...]] = {}
    problems: list[str] = [
        f"{name!r} is both a tier the roster names and an app in shared:, but a "
        f"card renders from one file — rename one of them"
        for name in sorted(set(tiers) & set(apps))
    ]
    for dirname, word, names, bundled in (
        (TIERS_DIRNAME, LAYER_DIR_KINDS[TIERS_DIRNAME], tiers, list_tiers()),
        (APPS_DIRNAME, LAYER_DIR_KINDS[APPS_DIRNAME], apps, list_apps()),
    ):
        for name in names:
            # `..` needs naming explicitly: `Path("..").name` is `".."`, so the
            # plain-name check alone would pass it and write a `tiers/..yml`.
            if name in (".", "..") or Path(name).name != name:
                problems.append(
                    f"{name!r}: a {word} name becomes a file name under {dirname}/, so "
                    "it must be a plain name — no path separators"
                )
                continue
            if name not in bundled:
                problems.append(
                    f"{name!r}: the roster names {word} {name!r}, but OSPREY ships no "
                    f"such {word} (bundled: {', '.join(bundled) or 'none'})"
                )
                continue
            relative = f"{dirname}/{name}.yml"
            try:
                text = emit_layer_delta_yaml(
                    preset_name=f"{dirname}/{name}",
                    profile_name=profile_name,
                    profile_filename=f"{render_path_prefix}{relative}",
                )
            except BuildProfileError as e:
                problems.append(f"{name!r}: {e}")
                continue
            texts[relative] = text
    if problems:
        raise click.UsageError(
            "Cannot materialize the tiers and apps this profile's web-terminal "
            "roster names:\n  - " + "\n  - ".join(problems)
        )
    # Each layer is judged on the document the build renders it from — an app
    # is composed, never merged like a tier — and the composer reads only the
    # host's `config`, `hooks` and `mcp_servers`, so `config` stands in for it.
    host_doc = {"config": resolved.config}
    for relative, text in texts.items():
        [(name, delta)] = _parsed_layer_deltas({relative: text}).items()
        kind = ProfileKind(LAYER_DIR_KINDS[relative.split("/", 1)[0]])
        try:
            rendered = render_document(kind, host_doc, dict(delta), relative)
        except BuildProfileError as e:
            problems.append(f"{name!r}: {e}")
            continue
        held = tier_privileges(rendered.get("config"))
        if held:
            absolute[name] = held
        lifted = privileges_beyond_baseline(held, host_privileges)
        if lifted:
            privileges[name] = lifted
    if problems:
        raise click.UsageError(
            "Cannot materialize the tiers and apps this profile's web-terminal "
            "roster names:\n  - " + "\n  - ".join(problems)
        )
    privilege_problems = _privileged_tier_problems(resolved.config, privileges, absolute)
    if privilege_problems:
        raise click.UsageError(
            "This profile's web-terminal roster would hand deployment editing to a "
            "terminal that must not have it:\n  - " + "\n  - ".join(privilege_problems)
        )
    return texts


def _cleanup(target: Path, seeded: tuple[str, ...] = ()) -> str:
    """Remove what a failed materialization wrote, and say what is left.

    Only the entries a materialization owns (:data:`MATERIALIZED_SOURCE_ENTRIES`)
    are removed: the target is a deployment repo root, which routinely holds an
    operator's own files — a ``.git``, an ``.env``, a clone's README — and a
    failed run must never cost one of those.

    Args:
        target: The repo root the failed materialization was writing into.
        seeded: Entries THIS run created that the table does not cover — the
            ``seed_dirs`` it actually copied, and a ``.env`` it wrote where
            there was none. Never one it found already there. They are not in
            :data:`MATERIALIZED_SOURCE_ENTRIES`, because a later ``--force``
            must leave an operator's own copy alone; but a run that just
            created one and then failed owns it, so a first init that fails
            leaves nothing behind.

    Returns:
        A sentence for the refusal it is appended to, naming anything that could
        not be removed so the operator knows a retry will refuse too.
    """
    import shutil

    owned = (*MATERIALIZED_SOURCE_ENTRIES, *seeded)
    for name in owned:
        entry = target / name
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    remaining = [name for name in owned if (target / name).exists()]
    if remaining:
        return (
            f"Partly-written files remain in {target}: {', '.join(remaining)} — "
            f"remove them before retrying."
        )
    return "Nothing was materialized."


def _context_baseline_source(manager: TemplateManager, data_bundle: str) -> Path:
    """The ``base.md`` text a new profile's context baseline starts from.

    Bundle first: an app template that ships its own
    ``web-terminal-context/base.md`` (the control assistant does — its text
    describes that family's renders) seeds the profile with it. Any other
    bundle falls back to the framework's generic baseline, the same file
    ``osprey build`` installs into every render — so the materialized slot
    starts byte-identical to what the build would have used anyway, and every
    edit to it from then on is visible in the operator's own repo.
    """
    bundle = context_baseline_slot(
        manager.template_root / "apps" / data_bundle / PER_USER_CONTEXT_DIRNAME
    )
    if bundle.is_file():
        return bundle
    return context_baseline_slot(manager.template_root / "claude_code" / PER_USER_CONTEXT_DIRNAME)


def _app_template_root(manager: TemplateManager, data_bundle: str) -> Path:
    """The packaged app-template directory a bundle name points at.

    The one place that turns a ``data_bundle`` into a path on disk, so every
    reader of an app template — the ``data/`` tree, and the whole-template
    copiers beside it — fails the same way on the same missing directory.
    Checked up front, before anything is written, so a packaging regression
    surfaces as an actionable error here rather than as a missing-file error
    mid-copy. It cannot be caused by anything the caller passed.

    Args:
        manager: The :class:`~.templates.manager.TemplateManager` locating the
            installed template root.
        data_bundle: App template to locate.

    Returns:
        The template directory, guaranteed to exist.

    Raises:
        BuildProfileError: If the template is absent from the installation.
    """
    from .templates.preset_data import app_template_root

    return app_template_root(Path(manager.template_root), data_bundle)


def _preset_data(manager: TemplateManager, preset_name: str) -> PresetData:
    """The packaged data *preset_name* materializes: app template + facility.

    Both names are preset-side facts read off the preset chain
    (:func:`~.build_profile_presets.preset_data_bundle`,
    :func:`~.build_profile_presets.preset_facility`), so a preset inheriting
    either through ``extends:`` composes the same tree its parent does.

    Args:
        manager: The :class:`~.templates.manager.TemplateManager` locating the
            installed template root.
        preset_name: Bundled preset, in either spelling.

    Returns:
        The composition :mod:`~.templates.preset_data` defines.

    Raises:
        BuildProfileError: If a named directory is absent, or the app template
            ships a facility of its own beside the one the preset names.
    """
    from .build_profile_presets import preset_data_bundle, preset_facility
    from .templates.preset_data import compose_preset_data

    return compose_preset_data(
        Path(manager.template_root),
        preset_data_bundle(preset_name),
        preset_facility(preset_name),
    )


def _resolve_preset_bundle(preset: str) -> tuple[str, str]:
    """Resolve a CLI preset spelling to its name and its packaged data bundle.

    The bundle is a preset-side fact — ``app_template:`` is not a profile key —
    so it is read off the preset chain rather than off a resolved profile: a
    preset that inherits the key through ``extends:`` names no bundle of its
    own, and :func:`~.build_profile_presets.preset_data_bundle` follows the
    chain the same way the merge would have.

    Args:
        preset: Preset name as typed, in either spelling
            (``control-assistant`` or ``control_assistant``).

    Returns:
        ``(preset_name, data_bundle)`` — the normalized (hyphenated) preset
        name, and the packaged data bundle the preset chain names.

    Raises:
        click.UsageError: If ``preset`` names no bundled preset — the message
            lists every one that exists — or if resolving it fails.
    """
    from .build_profile import _normalize_preset_name, _preset_exists, list_presets
    from .build_profile_presets import preset_data_bundle

    preset_name = _normalize_preset_name(preset)
    if _preset_exists(preset_name) is None:
        available = ", ".join(list_presets()) or "(none)"
        raise click.UsageError(f"Unknown preset {preset!r}. Available: {available}")
    try:
        bundle = preset_data_bundle(preset_name)
    except BuildProfileError as e:
        raise click.UsageError(f"Cannot resolve preset {preset_name!r}: {e}") from e

    return preset_name, bundle


class _MaterializedProfile(NamedTuple):
    """What :func:`_materialize_profile_directory` produced, for the caller's summary."""

    target: Path
    """The resolved profile directory."""

    skipped_shell_keys: tuple[str, ...]
    """Exported provider keys left out because the profile references none of
    their providers (:func:`_exported_provider_keys`). Reported rather than
    returned as a courtesy: the seeded keys can be read back from the ``.env``,
    the ones that were deliberately not written cannot."""

    profile_name: str
    """The display name written into the profile — the caller's, the directory's,
    or whatever ``--set name=`` said. Returned because the caller cannot
    reconstruct which of those won."""

    deploy_declared: bool
    """Whether the emitted profile carries an ACTIVE ``deploy:`` block, read back
    from the file on disk rather than from the inputs. Presets ship the block
    commented out, so this is normally false on a fresh materialization; it
    decides whether there is anything to render a CI pipeline from."""

    resolved: BuildProfile
    """The materialized profile, resolved back from the written ``profile.yml``
    — the same read-back that validates the round-trip, so what the caller
    summarizes (the composition card) is a fact about what is on disk rather
    than about what the inputs asked for."""

    layer_deltas: dict[str, Mapping[str, Any]]
    """The emitted tier and app files, parsed, keyed by tier or app name — empty
    for a profile that emits none. The per-tier facts a summary needs (each
    tier's panels and its write posture) live in these, not in the host
    profile."""

    seeded: tuple[str, ...]
    """The write-once directories this materialization seeded — empty on every
    re-run, because a name already present in the target is left as the operator
    left it. Returned because only this run knows which of them it created, and
    that is the difference between a directory the caller may still clear and
    one that is now the operator's."""


def _materialize_profile_directory(
    target_dir: Path,
    preset_name: str,
    set_pairs: tuple[str, ...] = (),
    *,
    profile_name: str | None = None,
    seed_dirs: Mapping[str, str] | None = None,
) -> _MaterializedProfile:
    """Materialize an editable, standalone profile directory from ``preset_name``.

    Writes ``profile.yml`` — the preset's fully resolved content as an
    explicit, self-contained profile (comments preserved, no ``extends:``) —
    the ``providers.yml`` catalog beside it (:func:`_plan_provider_catalog`),
    the preset's packaged data (:func:`_preset_data`) copied verbatim, the profile's ``.env`` channel
    (:func:`_write_secret_channel`), and a tutorial ``README.md`` explaining the
    convention directories. ``--set`` pairs are edits of the resolved preset,
    made the way ``osprey set`` makes them to the file afterwards, so a
    validated build one-liner carries into the profile without hand-editing.

    Fail-before-mutating: the preset, its layers, and the rendered profile text
    are all produced before the first ``mkdir``, and anything that fails after
    it removes the target rather than leaving a half-materialized directory.
    With ``force``, an existing target is deleted at that same point — never
    earlier — and only when it is a materialized profile (has ``profile.yml``)
    or an empty directory, so a failed run or a mistyped target never costs an
    unrelated directory.

    Args:
        target_dir: The profile directory to create.
        preset_name: Bundled preset to materialize, in either spelling.
        set_pairs: ``--set`` pairs, each replacing the value at its key.
        profile_name: Display name for the emitted profile. Defaults to one
            derived from the repo directory's own name. ``--set name=`` wins
            over both.
        seed_dirs: Write-once directories to copy out of the bundle, mapping a
            profile-root directory name to the bundle-relative tree it comes
            from (e.g. ``{"mcp_servers": "mcp_servers"}``). Each is copied only
            when the bundle ships it and the target does not already have it, so
            the copy happens on a repo's first init and never again — which is
            why these are NOT in :data:`MATERIALIZED_SOURCE_ENTRIES`: a later
            ``--force`` re-materialization must not delete servers an operator
            has since written.

    Returns:
        What was written, for the caller's summary (:class:`_MaterializedProfile`).

    Raises:
        click.UsageError: For user errors — existing target, an ``extends``
            edit, a roster carrying an unreadable ``access`` value
            (:func:`_unreadable_access_problems`), or layers that produce an
            invalid profile.
        BuildProfileError: For packaging problems (missing seed or data tree).
    """
    import shutil

    from osprey.deployment.dotenv import env_lock_path

    from .build_profile import (
        EXTENDS_OVERRIDE_REFUSAL,
        _normalize_preset_name,
        cli_edit_layer,
        resolve_build_profile,
    )
    from .build_profile_emit import (
        emit_standalone_profile_yaml,
        triggers_layer,
    )
    from .build_profile_presets import preset_data_bundle
    from .templates.manager import TemplateManager

    # Resolving through the public path validates the preset AND its --set
    # edits up front, and names the bundle whose data tree gets copied. It also
    # rejects a user-supplied `data:` in preset mode, which is right: this
    # command materializes the tree, so pointing it elsewhere is a mistake.
    # Everything it rejects is a user error, so it surfaces as one.
    try:
        edit = cli_edit_layer(set_pairs)
        if "extends" in edit:
            # The shared refusal: the same edit must be answered the same way
            # here and on a later `osprey set` into this profile. Asked before
            # anything is resolved, so the answer is about the key and never
            # about whatever the named parent requires.
            raise click.UsageError(EXTENDS_OVERRIDE_REFUSAL)
        resolved, preset_dir = resolve_build_profile(None, preset_name, set_pairs=set_pairs)
    except BuildProfileError as e:
        raise click.UsageError(f"Cannot materialize {preset_name!r}: {e}") from e

    name_override = edit.get("name")

    target = target_dir.resolve()

    normalized_preset = _normalize_preset_name(preset_name)
    # The caller's name, or one read off the directory the operator chose
    # (e.g. "my-profile" → "My Profile"). `--set name=` outranks both: it is an
    # explicit statement of the same fact.
    profile_name_default = profile_name or _directory_derived_name(target.name)
    if name_override is not None:
        profile_name_default = str(name_override)

    # The deployment's own name, recorded in the profile so every host and every
    # clone derives the same compose project, volumes and images from it. The
    # folder name is read here, once, to propose it; `--set project_name=`
    # states it outright (and the resolution above has already held that
    # spelling to compose's).
    from osprey.deployment.compose_generator import _normalize_compose_name

    project_name_override = edit.get("project_name")
    project_name = (
        str(project_name_override)
        if project_name_override is not None
        else _normalize_compose_name(target.name)
    )
    if not project_name:
        raise click.UsageError(
            f"Cannot propose a project_name from the folder name {target.name!r}: it has no "
            f"character a compose project name may use. Pass one with "
            f"`--set project_name=<name>` (lowercase letters, digits, '_' and '-')."
        )

    manager = TemplateManager()
    # The packaged data this repo's `data/` is copied from: the app template's
    # `data/` plus the facility at `data/facility/`. Read off the preset chain,
    # not off `resolved`: both names are preset-side only, so they never reach
    # the profile the emission below writes.
    data_bundle = preset_data_bundle(normalized_preset)
    preset_data = _preset_data(manager, normalized_preset)
    # Read before the first mkdir, like every other input here: a malformed
    # declaration refuses before anything is written.
    try:
        preset_data.placed_files()
    except ValueError as e:
        raise click.UsageError(f"Cannot materialize {preset_name!r}: {e}") from e

    # How the emitted tier and app headers spell their own paths: repo-relative,
    # because the repo root is where a reader stands.
    retired_layer_dirname = ""

    # The trigger config a dispatch profile runs on is facility state, so the
    # profile owns a copy of it (FR-3) and the emitted key names that copy.
    # Located before the first mkdir like everything else here.
    triggers_src = _triggers_source(resolved, preset_dir)

    # The roster's `access` values are resolved before anything else reads
    # them: every later step here parses them strictly, so an unreadable one
    # has to refuse at this altitude — with the message the deploy lint prints
    # for the same entry — rather than reach whichever reader runs first as a
    # traceback. Before the first mkdir, like every other refusal here.
    access_problems = _unreadable_access_problems(resolved.config)
    if access_problems:
        raise click.UsageError(
            "This profile's web-terminal roster carries an access value that does not "
            "resolve:\n  - " + "\n  - ".join(access_problems)
        )

    # A profile that deploys web terminals owns the tiers and apps its roster
    # names: each is materialized beside this one as an explicit file, laid over
    # this profile by its location. Emitted before the first mkdir, like
    # everything else here, so a bad roster reference fails before mutating.
    render_texts = _layer_profile_texts(resolved, profile_name_default, retired_layer_dirname)
    # Parsed here, before the first mkdir and before `--force` replaces
    # anything: the parse is what validates the emitted deltas, so a bad one
    # must cost nothing. Its result is what every later reader sees.
    layer_deltas = _parsed_layer_deltas(render_texts)

    extra_layers: tuple[dict[str, Any], ...] = (
        *((triggers_layer(),) if triggers_src is not None else ()),
    )

    # The catalog to write beside the profile, resolved before the first mkdir
    # like every other input here: it reads the repo's existing `providers.yml`
    # (or the copy a `--force` run holds aside), so a catalog that does not
    # parse refuses before anything is replaced.
    catalog = _plan_provider_catalog(target)

    # Read once, before anything is written: the README rendered below tells the
    # reader whether a `.env` was seeded for them, and the seeding itself happens
    # further down. Two reads of the environment could disagree.
    referenced_providers = _referenced_providers(resolved, layer_deltas, catalog.carried)
    shell_keys = _exported_provider_keys(referenced_providers, catalog.entries)
    exported_keys = shell_keys.seeded

    # Derived once for the same reason: the README lists the per-user context
    # slots and the loop below creates them, and a roster those two disagreed on
    # would document a directory that was never made.
    roster = _roster_user_names(resolved.config)

    # The materialized tree is what the build must read, so `data:` is emitted
    # as an active key — injected through the same --set edit a user would
    # make, rather than through a second path into the resolved content.
    profile_text = emit_standalone_profile_yaml(
        preset_name=normalized_preset,
        set_pairs=(*set_pairs, f"data={_PROFILE_DATA_DIRNAME}"),
        profile_name=profile_name_default,
        project_name=project_name,
        extra_layers=extra_layers,
        include_flow_diagram=True,
        providers_hash=catalog.content_hash,
    )

    # The repo root is allowed to exist — it usually does (an empty clone, the
    # operator's own mkdir). Whether writing into THIS one is acceptable was
    # settled by the caller, which also owns what to clear if this fails.
    target.mkdir(parents=True, exist_ok=True)

    # Bound before the try so both failure arms can hand it to `_cleanup`: the
    # seeds this run created are the only ones it is allowed to remove.
    seeded: list[str] = []

    # `.env` is an operator file, so it is not in MATERIALIZED_SOURCE_ENTRIES
    # and a re-materialization never touches one it found. A run that CREATES
    # one owns it exactly the way it owns a seed: without this, a first init
    # that fails leaves the operator's shell-exported keys sitting in a
    # directory the same refusal calls empty, and the next attempt then finds
    # the root non-empty.
    env_pre_existed = (target / _PROFILE_ENV_FILENAME).exists()
    run_written: list[str] = []

    try:
        # Verbatim copy (D1/FR2): staging subdirectories and any stray `.j2`
        # come across byte-identical — a profile data tree is content, never
        # templates, so nothing here is rendered. The one exclusion is build
        # exhaust the wheel does not ship either (_EXCLUDED_DATA_SUBTREES).
        # The facility, and any files this template shares with another one,
        # land beside the template's own, byte-identical, so the profile
        # cannot tell them apart.
        preset_data.copy_into(
            target / _PROFILE_DATA_DIRNAME, ignore=_data_copy_ignore(preset_data.app_data)
        )
        (target / "profile.yml").write_text(profile_text, encoding="utf-8")
        (target / PROVIDERS_FILENAME).write_text(catalog.text, encoding="utf-8")
        if catalog.carried:
            logger.debug(
                "  Provider catalog: kept %s", ", ".join(f"{name}" for name in catalog.carried)
            )

        if triggers_src is not None:
            shutil.copy2(triggers_src, target / PROFILE_TRIGGERS_FILENAME)
            logger.debug("  Trigger config: %s", PROFILE_TRIGGERS_FILENAME)

        # The profile owns its secrets from the first minute (FR-1) — the
        # documented variable list, whatever the shell already exports, and the
        # .gitignore that keeps the values out of version control.
        secret_files = _write_secret_channel(
            target,
            manager,
            resolved,
            profile_name_default,
            exported_keys,
            referenced_providers,
        )
        logger.debug("  Secrets: %s", ", ".join(secret_files))
        if not env_pre_existed and (target / _PROFILE_ENV_FILENAME).exists():
            # The sibling lock file is created beside the `.env` by the shared
            # writer and deliberately never removed while the `.env` lives; a
            # discarded `.env` takes it along. `_cleanup` joins these names onto
            # `target`, so the name is what is wanted, un-resolved.
            env_path = target / _PROFILE_ENV_FILENAME
            run_written += [_PROFILE_ENV_FILENAME, env_lock_path(env_path).name]
        if shell_keys.skipped:
            # Debug only. `osprey init`'s summary prints the same sentence from
            # the same helper, and this is the only caller, so logging it here
            # at info put the fact on screen twice in one run.
            logger.debug("  %s", _skipped_keys_note(shell_keys.skipped))

        # One empty slot per roster user, so the per-user context a facility
        # writes has an obvious home from the first minute (FR-5). Only the
        # directories are seeded: what goes in them is the facility's, and the
        # build derives the copy from the roster it resolves at the time, so
        # nothing about the roster is frozen here.
        for user in roster:
            user_dir = target / _CONTEXT_CONVENTION_DIRNAME / user
            user_dir.mkdir(parents=True, exist_ok=True)
            (user_dir / ".gitkeep").touch()
        if _stands_up_a_web_tier(resolved.config):
            # The shared baseline, beside the slots — the one seeded entry that
            # carries content, because it IS content the deployment ships: the
            # build copies this file over the framework's fallback, so the text
            # every seeded user starts from lives in the operator's repo where
            # they can read and edit it. Seeded on the module rather than the
            # roster, because the copy side is roster-independent too
            # (plan_convention_copies): a profile that names no users yet still
            # starts from text it can see.
            context_dir = target / _CONTEXT_CONVENTION_DIRNAME
            context_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                _context_baseline_source(manager, data_bundle),
                context_baseline_slot(context_dir),
            )
            logger.debug(
                "  Per-user context: %s (+ shared %s)",
                ", ".join(f"{_CONTEXT_CONVENTION_DIRNAME}/{user}/" for user in roster)
                or "no roster yet",
                CONTEXT_BASELINE_FILENAME,
            )

        # The facility description, in the convention directory the operator
        # edits it in. Seeded here rather than left to the build for the same
        # reason the context baseline is: it is text this deployment ships, and
        # it belongs in the repo from the first minute rather than appearing in
        # a generated tree that `rm -rf build/` is documented to remove.
        # Imported here, not at module scope: `osprey --help` must not pull in
        # the build pipeline (the lazy-import budget test pins this).
        from .build_persistence import FACILITY_RULE_NAME, ensure_profile_facility_rule
        from .init_cmd import FACILITY_RULE_DIR

        if FACILITY_RULE_NAME in resolved.rules:
            rules_dir = target / FACILITY_RULE_DIR
            fresh = not rules_dir.exists()
            moved = ensure_profile_facility_rule(target, enabled_agents=resolved.agents)
            if moved:
                logger.debug("  %s", moved)
                if fresh:
                    # This run created the directory, so this run owns it: a
                    # failure below removes it again rather than leaving half a
                    # profile behind. A directory that was already there is the
                    # operator's, whatever else is in it.
                    run_written.append(FACILITY_RULE_DIR)

        # The tier and app files. Validity was settled by `_parsed_layer_deltas`
        # above, before the first mkdir — nothing reaching here is unparsed, so
        # these writes need no guard of their own.
        for relative, layer_text in render_texts.items():
            layer_path = target / relative
            layer_path.parent.mkdir(exist_ok=True)
            layer_path.write_text(layer_text, encoding="utf-8")
        if render_texts:
            logger.debug("  Tiers and apps: %s", ", ".join(render_texts))
        # Write-once seeds, outside the web-tier branch because they
        # are not a web-tier fact: a directory the deployment OWNS from its first
        # init — an operator's MCP servers, say — copied out of the bundle once
        # and never replaced. An existing one is left exactly as the operator
        # left it, and nothing is written when the bundle ships no such tree, so
        # a bundle gaining or losing one changes only what a fresh init seeds.
        # `__pycache__` and its byte-code are dropped so the seed from a source
        # checkout is byte-identical to the seed from a wheel.
        for name, src_rel in (seed_dirs or {}).items():
            src = Path(manager.template_root) / "apps" / data_bundle / src_rel
            if src.is_dir() and not (target / name).exists():
                # Recorded BEFORE the copy starts, so a copy that dies half-way
                # still leaves `_cleanup` a name to remove.
                seeded.append(name)
                shutil.copytree(
                    src,
                    target / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.py[co]"),
                )
        if seeded:
            logger.debug("  Seeded: %s", ", ".join(f"{name}/" for name in seeded))

        # A profile or app that selects the library facility rule by its bare
        # name selects the copy seeded above instead, by its path.
        from .init_revalidate import name_shipped_shadows

        name_shipped_shadows(target)

        # The round-trip runs last because it validates `data:` against the tree
        # that must already be on disk. Only the host profile is resolved: a
        # tier file is a delta, meaningless on its own, and resolving one is
        # resolving the host with that delta merged in — which the host's own
        # round-trip already covers.
        #
        # Its result is what the caller is told about the profile: read back
        # from the written file, so "declares a deploy block" is a fact about
        # what is on disk rather than about what the inputs asked for.
        written, _written_dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    except BuildProfileError as e:
        # Emission round-trips for every bundled preset (guarded by tests), so
        # with edits present they are the thing to look at; without them this
        # is a framework bug and blaming the user's flags would misdirect.
        blame = (
            "The --set edits produce an invalid profile"
            if set_pairs
            else "The materialized profile does not validate"
        )
        raise click.UsageError(
            f"{blame}: {e}\n{_cleanup(target, seeded=(*seeded, *run_written))}"
        ) from e
    except Exception:
        # Any other failure (a copy error, a full disk) must not leave a
        # half-materialized directory that looks buildable.
        _cleanup(target, seeded=(*seeded, *run_written))
        raise

    logger.debug("Wrote profile directory: %s", target)
    return _MaterializedProfile(
        target,
        shell_keys.skipped,
        profile_name_default,
        written.deploy is not None,
        written,
        layer_deltas,
        tuple(seeded),
    )
