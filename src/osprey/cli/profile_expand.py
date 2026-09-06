"""``osprey profile expand`` — write the keys an older profile never spelled.

A profile emitted before the app template retired says less than it means. Its
``config:`` block held only what the facility had changed, and everything else
came from ``templates/apps/<name>/config.yml.j2``, rendered at build time and
never shown to the operator. That template is gone: the preset now carries the
whole configuration, and a profile is the complete declarative input or it is
not the source of truth. So a profile still carrying the retired
``app_template:`` key is refused by the loader
(:data:`~osprey.cli.build_profile_load._RETIRED_APP_TEMPLATE_REFUSAL`), and this
verb is what makes it loadable again.

What it does, in one pass over the file:

* Reads the document RAW — ruamel round-trip, never through the profile loader,
  which would refuse it for the very key this verb is here to remove.
* Decides which preset the profile came from: ``--from`` if given, else
  ``provenance.preset``, else the retired ``app_template:`` value.
* Writes every ``config:`` leaf the preset documents and the profile does not
  (:func:`~osprey.cli.build_profile_drift.lacking_config_keys`), each under the
  preset's own comment, so the expanded file reads like one ``osprey init``
  emitted rather than like a generated dump.
* Except the two kinds of leaf whose absence is CORRECT: one another block of
  the profile already owns (:func:`_claimed_elsewhere`), and anything below a
  catalog whose entries are the deployment's own data
  (:func:`_data_map_prefixes`). Both are named in the output.
* Drops ``app_template:`` and re-stamps ``provenance:`` — the preset, its hash,
  and the hash of the provider catalog this repo actually holds — and re-points
  the emitted header's prose at the same preset, so the two cannot disagree.

Values already in the file are never touched: the profile is the source of
truth the moment it exists, so expansion only ever ADDS. Running it twice
changes nothing — the second run finds nothing lacking and leaves the file
byte for byte as it was.

The stamp has one consequence worth stating plainly: it turns the preset-drift
check on. A profile with no ``provenance:`` has nothing to be compared with, and
after this verb every structural difference from the preset is something
``osprey validate`` refuses until a marker comment claims it. So the run counts
those differences and says so, rather than leaving them for the next build.

``--providers`` refreshes ``providers.yml`` in the same pass, taking the entries
OSPREY ships fresh and carrying over every entry the operator added.

Usage:
    osprey profile expand
    osprey profile expand --from control-assistant
    osprey profile expand --providers
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from .output import note, report
from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option
from .styles import Styles

if TYPE_CHECKING:
    from ruamel.yaml import CommentedMap

#: The retired top-level key. A profile that still carries it names the app
#: template a build used to render; the value is the packaged bundle name
#: (``control_assistant``), which is the underscore spelling of the preset that
#: replaced it — the same normalization ``--preset`` accepts on the command
#: line, so it needs no table of its own.
RETIRED_TEMPLATE_KEY = "app_template"

#: How the emitted profile opens a section of its ``config:`` block. Recognized
#: rather than re-authored: a copied block that starts with one is a section
#: heading, and a heading wants the blank line above it the emitter gives it.
_SECTION_DIVIDER = "──"

#: The column a ``config:`` block's own keys sit at. Only ever needed for a
#: block this verb CREATES: a mapping built in memory carries no position, and
#: without one every comment copied into it renders flush left, which is the
#: emitted file's section-header convention and reads as a lie inside a block.
_CONFIG_COLUMN = 2

#: The two header lines that name the profile's source in prose. ``--from``
#: re-points the machine-readable stamp; these say the same thing to a person,
#: and a file whose prose and data disagree is worse than either alone.
_HEADER_PRESET = re.compile(r"(Made from the bundled `)[^`]+(` preset)")
_HEADER_HASH = re.compile(r"(#\s+preset content hash: )\S+")


def _comment_text(lines: list[str]) -> str | None:
    """Turn ruamel comment LINES into the text ``anchored_put`` re-renders.

    The two spellings are inverses: a stored line is ``  # prose`` — the
    mapping's indent, a hash, the text — and ``comment=`` takes the text alone
    and writes the indent and hash itself, at the column of the mapping it is
    putting into. So this strips exactly what the writer puts back, which is
    what lets a comment copied out of the emitted preset land at the profile's
    own indentation rather than at the preset's.

    A blank line inside the block survives as a blank line. A line that is a
    bare ``#`` becomes one too: the renderer prefixes ``# `` to every non-empty
    line, so there is no input that produces a lone hash, and a blank separator
    is the honest approximation.

    A block that OPENS with a section divider gets a blank line put back in
    front of it. The emitted profile separates its sections that way, and the
    reader ruamel stores blocks with trims exactly that blank as edge layout —
    so without this a copied section would butt against the key above it and
    the expanded file would read less like the emitted one it is copying.

    Args:
        lines: Raw comment lines as ruamel stored them, hash included.

    Returns:
        The block as text, one line per line, or ``None`` when it is empty.
    """
    text: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped[1:]
            if stripped.startswith(" "):
                stripped = stripped[1:]
        text.append(stripped)
    while text and not text[-1]:
        text.pop()
    if not any(text):
        return None
    if text[0].startswith(_SECTION_DIVIDER):
        text.insert(0, "")
    return "\n".join(text)


def _leaf_comments(
    cm: CommentedMap, parent_entry: list[Any] | None = None, prefix: str = ""
) -> tuple[dict[str, str], list[str]]:
    """Every leaf of ``cm``, dotted, mapped to the comment block above it.

    The emitted ``config:`` block is written flat — one dotted key per rendered
    leaf — so this is usually one call to
    :func:`~osprey.cli.build_profile_emit._extract_visual`. It recurses anyway
    because a preset may spell a subtree nested (``modules.web_terminals:`` in
    the control-assistant presets is), and the leaves under such a branch are
    addressed by the same dotted keys the comparison reports as lacking.

    A leaf with no comment of its own simply has no entry. The comment above a
    nested BRANCH is not inherited by its leaves: it introduces the branch, and
    copying it onto each leaf beneath would repeat one paragraph a dozen times.

    The second return value is what makes the recursion whole. ruamel parks the
    block introducing a key on the key BEFORE it — and when that predecessor is
    a nested branch, on the deepest last leaf inside it, a level the parent's
    own ``_extract_visual`` never looks at. Each level therefore hands its
    caller the lines its last entry holds, and the caller gives them to the next
    sibling. Without it the header of the key that follows a nested branch
    (``execution.execution_method`` after ``modules.web_terminals:``) is lost.

    Args:
        cm: A ``config:`` mapping from an emitted profile.
        parent_entry: The ruamel ``ca`` entry of the key ``cm`` hangs under,
            which is where the block above its FIRST key is stored.
        prefix: Dotted path of ``cm`` itself, for the recursion.

    Returns:
        Dotted key → comment text in the spelling ``anchored_put`` takes, and
        the comment lines ``cm``'s last entry holds for whatever follows ``cm``.
    """
    from .build_profile_emit import _entry_texts, _extract_visual

    pre, _eol = _extract_visual(cm, parent_entry)
    comments: dict[str, str] = {}
    keys = list(cm.keys())
    # Lines the previous sibling held for this one, when the parent level could
    # not see them: they were stored inside that sibling's own subtree.
    spill: list[str] = []
    for index, key in enumerate(keys):
        value = cm[key]
        block, spill = spill + pre.get(key, []), []
        if hasattr(value, "items") and len(value):
            nested, spill = _leaf_comments(value, cm.ca.items.get(key), f"{prefix}{key}.")
            comments.update(nested)
        else:
            text = _comment_text(block)
            if text is not None:
                comments[f"{prefix}{key}"] = text
            if index == len(keys) - 1:
                entry = cm.ca.items.get(key)
                spill = _entry_texts(entry)[1] if entry else []
    return comments, spill


def _leaf_values(mapping: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of ``mapping``, dotted, mapped to its value.

    The value half of :func:`_leaf_comments`, walked the same way so the two
    agree on what a leaf is. Values come back as plain Python — a ruamel
    container carries the comments of the document it was parsed from, and
    putting one into the profile would drag the preset's layout in with it.
    """
    leaves: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if hasattr(value, "items") and len(value):
            leaves.update(_leaf_values(value, f"{dotted}."))
        else:
            leaves[dotted] = _plain(value)
    return leaves


def _plain(value: Any) -> Any:
    """``value`` with every ruamel container replaced by its plain equivalent."""
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _data_map_prefixes() -> frozenset[str]:
    """Config keys whose leaves are a deployment's own data rather than schema.

    Read off the packaged manifest's ``data-map: true`` entries rather than
    listed here, because that is where the fact is already recorded and where
    ``scripts/check_config_keys.py`` reads it: the provider table, the named
    panel layouts and the persona catalog are maps whose KEYS a facility
    invents. (The user roster is a list, so it has no dotted paths below it and
    is already a leaf.)

    They are TERMINI for expansion. A leaf below one of them documents one
    deployment's data — the preset's own five personas, say — so writing it into
    a profile that declares two would invent three the facility never asked for,
    with ``project_path`` values derived from whichever directory the expansion
    happened to run in. Nothing below a data map is ever filled in.
    """
    from .config_cmd import _load_key_manifest

    return frozenset(
        key
        for key, spec in _load_key_manifest().items()
        if isinstance(spec, dict) and spec.get("data-map")
    )


def _claimed_elsewhere(document: Any) -> dict[str, str]:
    """Config leaves whose single home is another block of this same profile.

    A key the preset documents can still be wrong to write here, because the
    profile already states the same fact somewhere the build reads first.
    ``deploy.image_source`` is the shipped case: a build propagates it into the
    rendered config's ``modules.web_terminals.image_source``, and a profile that
    spells that leaf as well gives one fact two homes, which ``osprey validate``
    refuses outright. Expanding it in would break the build of every repo that
    carries deploy coordinates.

    A profile whose ``config:`` block ALREADY spells the leaf is not claimed:
    that profile has a two-homes problem this verb did not create, and silently
    treating the key as handled would hide it from the refusal that reports it.

    Returns:
        Dotted config key → the key that owns the fact instead.
    """
    from .build_profile_deploy import IMAGE_SOURCE_CONFIG_KEY, config_image_source_spelling

    deploy = document.get("deploy")
    if not hasattr(deploy, "items"):
        return {}
    if config_image_source_spelling(document.get("config")) is not None:
        return {}
    return {IMAGE_SOURCE_CONFIG_KEY: "deploy.image_source"}


def _skipped_keys(document: Any, lacking: list[str]) -> dict[str, str]:
    """The lacking keys expand must NOT write, each with the reason.

    Two reasons, both about a key whose absence is correct rather than
    accidental: it sits under a data map (:func:`_data_map_prefixes`), or its
    fact is stated in another block of this profile
    (:func:`_claimed_elsewhere`).
    """
    if not lacking:
        return {}
    data_maps = _data_map_prefixes()
    claimed = _claimed_elsewhere(document)
    skipped: dict[str, str] = {}
    for key in lacking:
        if key in claimed:
            skipped[key] = f"stated by `{claimed[key]}`. The build writes it here for you."
            continue
        parent = next((prefix for prefix in data_maps if key.startswith(f"{prefix}.")), None)
        if parent is not None:
            skipped[key] = f"below `{parent}`, whose entries are this deployment's own data."
    return skipped


def _detach_trailing_section(mapping: Any) -> str | None:
    """Take the comment block trailing *mapping*, so an append can land above it.

    ``anchored_put`` does this itself, through
    :func:`~osprey.utils.config_writer._steal_trailing_block`, which splits the
    trailing token at the first line indented less far than the token starts.
    That is the right split when the entry's own inline comment WRAPS — moving
    its continuation lines would tear one comment in half — but it detaches
    NOTHING when the token starts at column 0, which is exactly where a section
    header introducing the next top-level block sits. Left in place, every key
    this verb appends lands under that header and the block it introduces loses
    it.

    So: the careful split first, and when it declines, the whole token — but
    only once the token's first line is known to be blank, which is what says
    the entry has no inline comment of its own and there is nothing to tear.
    """
    from osprey.utils.config_writer import (
        _deepest_trailing_slot,
        _steal_section_comment,
        _steal_trailing_block,
    )

    moved = _steal_trailing_block(mapping)
    if moved is not None:
        return moved
    found = _deepest_trailing_slot(mapping)
    if found is None:
        return None
    owner, last, slot = found
    entry = owner.ca.items.get(last)
    token = entry[slot] if entry else None
    if token is None or str(token.value).partition("\n")[0].strip():
        return None
    return _steal_section_comment(mapping)


def _put_target(config: Any, dotted: str) -> tuple[Any, str]:
    """Which mapping a lacking leaf goes into, and under which key.

    A dotted key is a SPELLING, not a location, but only at the TOP LEVEL of
    ``config:``: that is where the build's applier expands the dots, so
    ``modules.web_terminals.enabled`` beside a ``modules.web_terminals:``
    mapping and ``enabled`` inside it address one rendered leaf. Writing the
    dotted spelling beside a mapping the profile already nests puts one block at
    two depths — the shape ``osprey validate`` refuses outright for
    ``control_system:`` — and leaves the next reader looking in two places for
    one setting. The nested home wins whenever the profile has one.

    INSIDE a nested mapping the same spelling is not a spelling at all: nothing
    expands it, so ``approval: {tools.channel_write: always}`` renders a literal
    key named ``tools.channel_write`` and the runtime reading
    ``approval['tools']`` never sees the value. So the descent stops at the
    deepest mapping that EXISTS, and a remainder that still carries a dot is
    the caller's signal to create the intermediate mappings
    (:func:`_put_leaf`) rather than to write the dots down as a key.

    Args:
        config: The profile's ``config:`` mapping.
        dotted: The lacking leaf's dotted key.

    Returns:
        The deepest existing mapping on the path and the rest of the dotted key.
        A remainder with dots in it means the path runs through mappings that do
        not exist yet — legal as a key only when the mapping is ``config``
        itself.
    """
    from ruamel.yaml import CommentedMap

    container = config
    segments = dotted.split(".")
    while len(segments) > 1:
        for length in range(len(segments) - 1, 0, -1):
            nested = container.get(".".join(segments[:length]))
            if isinstance(nested, CommentedMap) and len(nested):
                container, segments = nested, segments[length:]
                break
        else:
            break
    return container, ".".join(segments)


#: How much deeper than its parent a nested mapping's keys sit. The emitted
#: profile is two-space indented throughout, and a mapping built in memory has
#: to be told the column its keys will render at or every comment written into
#: it lands flush left (:func:`~osprey.utils.config_writer._mapping_column`).
_INDENT_STEP = 2


def _put_leaf(config: Any, dotted: str, value: Any, comment: str | None) -> None:
    """Write one lacking leaf into ``config:``, under the comment it came with.

    Two homes, decided by :func:`_put_target`:

    * The dotted key at the top level of ``config:`` — the emitted profile's own
      flat spelling, and the only place the build expands dots. A branch the
      profile does not mention AT ALL lands this way, one dotted leaf per line,
      which is what the file would have said had it been emitted today.
    * Inside the mapping the profile nests, when it nests one. If the leaf sits
      below intermediate mappings that do not exist yet, they are created here:
      writing the remainder as a dotted key inside a nested mapping would render
      a literal key with dots in its name, which the runtime never reads.

    The created branch is filled BEFORE it is put, so the single
    :func:`~osprey.utils.config_writer.anchored_put` that attaches it sees a
    complete value and can re-anchor the comment block trailing the mapping
    below it — an empty mapping cannot carry that block, and the section header
    it usually is would end up introducing the branch instead of what follows.
    """
    from ruamel.yaml import CommentedMap

    from osprey.utils.config_writer import _attach_section_comment, anchored_put

    container, remainder = _put_target(config, dotted)
    segments = remainder.split(".")
    branch: Any = value
    key = remainder
    if len(segments) > 1 and container is not config:
        column = _mapping_column(container)
        leaf_depth = len(segments) - 1
        for depth in range(leaf_depth, 0, -1):
            nested = CommentedMap()
            nested.lc.col = column + _INDENT_STEP * depth
            # Only the LEAF carries the preset's prose: the mappings above it
            # are this verb's own scaffolding, which the preset documents no
            # comment for.
            anchored_put(
                nested, segments[depth], branch, comment=comment if depth == leaf_depth else None
            )
            branch = nested
        key, comment = segments[0], None
    # The block trailing the target introduces whatever comes AFTER it, so it
    # has to end up below the append rather than above it.
    moved = _detach_trailing_section(container)
    anchored_put(container, key, branch, comment=comment)
    if moved:
        _attach_section_comment(container, moved)


def _mapping_column(mapping: Any) -> int:
    """The column *mapping*'s keys sit at, or the ``config:`` block's default."""
    column = getattr(getattr(mapping, "lc", None), "col", None)
    return column if isinstance(column, int) else _CONFIG_COLUMN


def _restamp_header(text: str, preset: str, preset_hash: str) -> str:
    """Re-point the emitted header's two preset lines at what the stamp now says.

    ``--from`` moves ``provenance:``; the header says the same thing in prose,
    and a file whose prose and data disagree sends its next reader to the wrong
    preset. Only the leading comment block is touched, and only the two lines
    that name the preset and its hash — the emitting VERSION stays as it was,
    because that line is history and history did not change.
    """
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.startswith("#"):
            break
        line = _HEADER_PRESET.sub(lambda m: m.group(1) + preset + m.group(2), line)
        lines[index] = _HEADER_HASH.sub(lambda m: m.group(1) + preset_hash, line)
    return "".join(lines)


def _reference_document(preset: str, *, repo_name: str, profile_name: str) -> CommentedMap:
    """The profile ``osprey init --preset <preset>`` would write, with comments.

    :func:`~osprey.cli.build_profile_emit.materialized_profile` answers the same
    question as DATA, and the comparison this verb asks its keys of uses it. The
    comments are the other half of the answer here — expansion copies each key's
    documentation along with it — and they only exist in the emitted TEXT, so the
    text is what this parses. The layering is the materializer's, so the two
    documents are the same document read two ways.

    ``repo_name`` is the deployment directory rather than the profile's display
    name: the persona catalog derives each persona's ``project_path`` from it,
    and a value copied out of here has to be the one this repo would have been
    materialized with.
    """
    from ruamel.yaml import YAML

    from .build_profile_emit import (
        emit_standalone_profile_yaml,
        persona_catalog,
        persona_catalog_layer,
        triggers_layer,
    )
    from .build_profile_merge import _load_preset_raw, _resolve_extends

    raw, anchor = _load_preset_raw(preset)
    resolved = _resolve_extends(dict(raw), anchor)
    config = resolved.get("config")
    personas = persona_catalog(config if hasattr(config, "items") else {})
    layers: tuple[dict[str, Any], ...] = (
        *((persona_catalog_layer(personas, repo_name=repo_name),) if personas else ()),
        *((triggers_layer(),) if hasattr(resolved.get("dispatch"), "items") else ()),
    )
    text = emit_standalone_profile_yaml(preset, (), (), profile_name, extra_layers=layers)
    return YAML().load(text)


def _resolve_preset(document: Any, from_preset: str | None, profile_path: Path) -> str:
    """Which preset this profile documents, and where that was read from.

    Three sources, most explicit first. ``--from`` is the operator saying it.
    ``provenance.preset`` is what ``osprey init`` recorded, and it names the
    exact preset — a persona preset included — so it outranks the retired key,
    which only ever named one of the four packaged bundles and would answer
    ``control-assistant`` for a repo materialized from
    ``control-assistant-readonly``. Expanding from the bundle there would write
    the base preset's keys and then re-stamp the profile as if it had come from
    there, which costs drift its baseline. ``app_template:`` is the last resort,
    and the only source a profile emitted before ``provenance:`` existed has.

    ``--from`` does not fall through: a preset the operator named and this
    OSPREY does not ship is a refusal, not a reason to quietly expand from
    something else. The two file-borne sources DO fall through to each other,
    which is how a profile from a release that shipped a preset this one dropped
    still expands from its bundle.

    Args:
        document: The raw profile document.
        from_preset: The ``--from`` value, if given.
        profile_path: Named in the refusal, so an operator knows which file.

    Returns:
        A bundled preset name, in the on-disk spelling.

    Raises:
        click.UsageError: When no source names a preset this installation
            ships. The message asks for ``--from``, which is the answer to
            every version of that.
    """
    from .build_profile_presets import _normalize_preset_name, list_presets

    known = list_presets()
    if from_preset:
        if _normalize_preset_name(from_preset) in known:
            return _normalize_preset_name(from_preset)
        raise click.UsageError(
            f"--from names preset {from_preset!r}, which this OSPREY does not ship. "
            f"Nothing was written.\n\nBundled presets: {', '.join(known)}"
        )

    provenance = document.get("provenance")
    recorded = provenance.get("preset") if hasattr(provenance, "get") else None
    candidates = [
        ("provenance.preset", recorded),
        (f"{RETIRED_TEMPLATE_KEY}:", document.get(RETIRED_TEMPLATE_KEY)),
    ]
    named = [(source, str(value)) for source, value in candidates if value]
    for _source, value in named:
        if _normalize_preset_name(value) in known:
            return _normalize_preset_name(value)

    if named:
        source, value = named[0]
        raise click.UsageError(
            f"{profile_path} names preset {value!r} ({source}), which this OSPREY "
            f"does not ship. Name the preset to expand from with --from PRESET.\n\n"
            f"Bundled presets: {', '.join(known)}"
        )
    raise click.UsageError(
        f"Cannot tell which preset {profile_path} was materialized from: it carries "
        f"no `provenance.preset` and no `{RETIRED_TEMPLATE_KEY}:`. Name the preset to "
        f"expand from with --from PRESET.\n\n"
        f"Bundled presets: {', '.join(known)}"
    )


def _stamp_provenance(document: Any, preset: str, preset_hash: str, providers_hash: str) -> str:
    """Record what the profile now carries: the preset, its hash, the catalog's.

    Written after the expansion rather than before, because that is what makes
    it true: the profile spells every key the preset documents, so the stamp
    the emitter would have written is the honest one. A ``deviation_marker:``
    the operator set is left alone — it is the one key in the block that is
    theirs.

    Returns:
        The deviation marker now in force, which is what a drift finding has to
        be claimed with.
    """
    from ruamel.yaml import CommentedMap

    from osprey.utils.config_writer import anchored_put

    from .build_profile_emit import _PROVENANCE_COMMENT
    from .build_profile_schema import DEFAULT_DEVIATION_MARKER

    stamps = {
        "preset": preset,
        "preset_hash": preset_hash,
        "providers_hash": providers_hash,
    }
    block = document.get("provenance")
    if not isinstance(block, CommentedMap):
        block = CommentedMap()
        for key, value in stamps.items():
            block[key] = value
        anchored_put(
            document, "provenance", block, comment=_comment_text(list(_PROVENANCE_COMMENT))
        )
        return DEFAULT_DEVIATION_MARKER
    for key, value in stamps.items():
        block[key] = value
    marker = block.get("deviation_marker")
    return marker if isinstance(marker, str) and marker else DEFAULT_DEVIATION_MARKER


def _catalog_hash(repo_root: Path) -> str:
    """Hash of the provider catalog a build of this repo would read.

    The repo's own file when it has one, the packaged catalog when it does not
    — which is exactly the fallback
    :func:`~osprey.profiles.providers.load_provider_catalog` applies, so the
    stamp describes the catalog that would actually be rendered rather than a
    file that may not be there.
    """
    from osprey.profiles.providers import (
        PROVIDERS_FILENAME,
        compute_providers_hash,
        packaged_catalog_path,
    )

    repo_catalog = repo_root / PROVIDERS_FILENAME
    return compute_providers_hash(
        repo_catalog if repo_catalog.is_file() else packaged_catalog_path()
    )


def _refresh_provider_catalog(repo_root: Path) -> tuple[str, tuple[str, ...]]:
    """Rewrite ``providers.yml`` with the entries OSPREY ships, keeping the rest.

    The re-materialization ``osprey init --force`` performs, asked for on its
    own: every packaged entry comes back fresh, and every entry the packaged
    catalog does not declare — the operator's own gateways — is carried over
    with the lines they wrote it in. Planning and writing are the init path's,
    unchanged, so a catalog refreshed here is byte-identical to one a re-init
    would have produced.

    Returns:
        The refreshed catalog's content hash, and the names carried over.
    """
    from osprey.profiles.providers import PROVIDERS_FILENAME

    from .profile_cmd import _plan_provider_catalog

    plan = _plan_provider_catalog(repo_root)
    (repo_root / PROVIDERS_FILENAME).write_text(plan.text, encoding="utf-8")
    return plan.content_hash, plan.carried


@click.command(name="expand")
@click.option(
    "--from",
    "from_preset",
    metavar="PRESET",
    default=None,
    help="Preset to expand from. Default: the profile's own provenance.",
)
@click.option(
    "--providers",
    "refresh_providers",
    is_flag=True,
    help="Also refresh providers.yml with the entries OSPREY ships, keeping yours.",
)
@repo_option
def expand(from_preset: str | None, refresh_providers: bool, repo: Path | None) -> None:
    """Fill in every config key this profile leaves to its preset.

    A profile written before the app template retired spells only what its
    facility changed; the rest came from a packaged template the operator never
    saw. This writes those keys into the profile itself, each under the comment
    the preset documents it with, and drops the retired `app_template:` key.

    Nothing already in the file is changed — expansion only adds — so running it
    twice leaves the profile exactly as the first run left it.

    Two kinds of lacking key are deliberately left out, and both are named in
    the output: a key another block of the profile already owns (`deploy:` is
    the home of `image_source`, and the build writes it into the config for
    you), and anything below a catalog whose entries are your own data — the
    persona catalog, the provider table, your named panel layouts.

    Expanding also stamps `provenance:`, which turns the preset-drift check on:
    from then on `osprey validate` refuses every structural difference from that
    preset until a `# DEVIATION: <why>` comment claims it, or the run passes
    `--drift=warn`. A profile that had no stamp before will find differences the
    first time, so this verb counts them for you.

    Examples:

    \b
      $ osprey profile expand
      $ osprey profile expand --from control-assistant
      $ osprey profile expand --providers
    """
    from osprey.errors import BuildProfileError
    from osprey.utils.config_writer import _yaml, anchored_put, load_config_document

    from .build_profile_drift import lacking_config_keys
    from .build_profile_emit import _drop_key_and_pre_comment
    from .build_profile_merge import compute_preset_hash
    from .build_profile_resolve import _atomic_write_bytes

    repo_root = find_repo_root(repo)
    profile_path = repo_root / PROFILE_FILENAME
    before = profile_path.read_text(encoding="utf-8")

    # Read RAW: the loader refuses the very key this verb removes, so going
    # through it would make an expandable profile unexpandable.
    document = load_config_document(profile_path)
    if not hasattr(document, "items"):
        raise click.UsageError(f"{profile_path} is not a YAML mapping. Nothing to expand.")

    preset = _resolve_preset(document, from_preset, profile_path)
    profile_name = str(document.get("name") or repo_root.name)

    try:
        reference = _reference_document(preset, repo_name=repo_root.name, profile_name=profile_name)
        lacking = lacking_config_keys(document, preset)
    except BuildProfileError as e:
        raise click.UsageError(f"Cannot read preset {preset!r} to expand from: {e}") from e

    reference_config = reference.get("config")
    comments = (
        _leaf_comments(reference_config, reference.ca.items.get("config"))[0]
        if hasattr(reference_config, "items")
        else {}
    )
    values = _leaf_values(reference_config) if hasattr(reference_config, "items") else {}

    # Keys whose absence is correct rather than accidental. Removed before the
    # backing check below, because a key that is never written needs no value.
    skipped = _skipped_keys(document, lacking)
    lacking = [key for key in lacking if key not in skipped]

    # The comparison and the document above are two readings of one emitted
    # profile, so every key reported missing has a value here. Checked before
    # anything is written rather than assumed, because the alternative to
    # noticing a disagreement is writing `key: null` into the facility's source
    # of truth and calling it the preset's value.
    unbacked = [key for key in lacking if key not in values]
    if unbacked:
        raise click.UsageError(
            f"Preset {preset!r} reports {len(unbacked)} key(s) as missing that its own "
            f"emitted profile does not carry: {', '.join(unbacked[:5])}. Nothing was "
            f"written. This is a framework inconsistency, not a problem with "
            f"{profile_path}."
        )

    carried: tuple[str, ...] = ()
    if refresh_providers:
        providers_hash, carried = _refresh_provider_catalog(repo_root)
    else:
        providers_hash = _catalog_hash(repo_root)

    if lacking:
        from ruamel.yaml import CommentedMap

        config = document.get("config")
        if not isinstance(config, CommentedMap):
            config = CommentedMap()
            # A mapping built here carries no position, and a comment written
            # into one renders at column 0 — the emitted file's section-header
            # convention, inside a block where it means nothing.
            config.lc.col = _CONFIG_COLUMN
            anchored_put(document, "config", config)
        for key in lacking:
            _put_leaf(config, key, values.get(key), comments.get(key))

    retired = RETIRED_TEMPLATE_KEY in document
    if retired:
        _drop_key_and_pre_comment(document, RETIRED_TEMPLATE_KEY)
    preset_hash = compute_preset_hash(preset) or "(unavailable)"
    marker = _stamp_provenance(document, preset, preset_hash, providers_hash)

    rendered = io.StringIO()
    _yaml.dump(document, rendered)
    after = _restamp_header(rendered.getvalue(), preset, preset_hash)
    changed = after != before
    if changed:
        _atomic_write_bytes(profile_path, after.encode("utf-8"))

    _report(profile_path, preset, lacking, skipped, retired, changed, refresh_providers, carried)
    _report_drift(profile_path, preset, preset_hash, providers_hash, marker)


def _report(
    profile_path: Path,
    preset: str,
    lacking: list[str],
    skipped: dict[str, str],
    retired: bool,
    changed: bool,
    refreshed: bool,
    carried: tuple[str, ...],
) -> None:
    """Say what the file gained, at the altitude the operator asked at."""
    from osprey.profiles.providers import PROVIDERS_FILENAME

    if lacking:
        report(
            f"✓ Wrote {len(lacking)} key(s) from preset '{preset}' into {profile_path}",
            style=Styles.SUCCESS,
        )
        for key in lacking:
            note(key)
        note("Their values are the preset's. That is what this deployment already built with.")
    elif changed:
        report(f"✓ Updated {profile_path}", style=Styles.SUCCESS)
    else:
        report(f"Nothing to expand: {profile_path} already spells every key '{preset}' documents.")

    if skipped:
        report(f"Left {len(skipped)} key(s) the preset documents out of {profile_path}:")
        for key, reason in skipped.items():
            note(f"{key}: {reason}")

    if retired:
        note(f"Dropped the retired `{RETIRED_TEMPLATE_KEY}:` key.")
    if refreshed:
        report(f"✓ Refreshed {PROVIDERS_FILENAME} from the catalog OSPREY ships")
        if carried:
            note(f"Kept your own entries: {', '.join(carried)}")


def _report_drift(
    profile_path: Path, preset: str, preset_hash: str, providers_hash: str, marker: str
) -> None:
    """Count what ``osprey validate`` will now refuse, because this run caused it.

    Stamping ``provenance:`` is what turns the preset-drift check on: before it
    there is nothing to compare the profile with, and after it every structural
    difference from the preset is a refusal until a marker comment claims it. A
    profile that carried no stamp — or one ``--from`` just re-pointed — can
    therefore validate green before this verb runs and be refused after, which
    the operator has to hear from the verb that did it rather than from the next
    build.

    Never fatal: the expansion is written and correct either way, so a
    comparison that cannot be made is simply not reported.
    """
    from osprey.errors import BuildProfileError

    from .build_profile_drift import preset_drift_report
    from .build_profile_schema import ProfileProvenance

    try:
        card = preset_drift_report(
            profile_path,
            ProfileProvenance(
                preset=preset,
                preset_hash=preset_hash,
                providers_hash=providers_hash,
                deviation_marker=marker,
            ),
        )
    except BuildProfileError:
        return
    if not card.refusals:
        return
    report(
        f"⚠ {profile_path} now names preset '{preset}', so `osprey validate` compares "
        f"the two: it differs in {len(card.refusals)} place(s) no marker claims.",
        style=Styles.WARNING,
    )
    note(f"Claim one with a `# {marker}: <why>` comment above its line, or in the key's name.")
    note("`osprey validate --drift=warn` reports them without refusing.")
