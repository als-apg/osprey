"""MATLAB Middle Layer (MML) install commands.

``osprey mml import`` turns one or more MML exports into the canonical
``data/mml/ao.json``, ``ad.json`` and ``PROFILE.md`` of the deployment repo.
The location is fixed, not a flag: the later verbs read the same directory, so
an output that could move would strand the chain. :func:`mml_data_dir` is the
one place that names it.

``osprey mml map --init`` writes the ``data/mml/mapping.yaml`` skeleton beside
them and ``map --check`` checks the reviewed file against the export. The file
is parsed here; the services receive dicts and ``Mapping`` objects.

``osprey mml emit`` turns the export and the checked mapping into the
deployment's artifacts under ``data/``: the middle-layer channel database, the
facility ontology (schema and compiled table), the OKF knowledge pages and the
Turtle corpus. It refuses, before writing anything, while the deployment still
carries the preset's demo tier databases or untouched demo knowledge pages,
and names them in one ``rm`` line. It refuses just as ``map --check`` does on
a judgment the mapping leaves unanswered or the export cannot carry, and on a
signal group of the judged grain with no direction, so nothing is written from
a mapping the check rejects.

A 2.0 export carries a virtual accelerator as well, and ``emit`` writes the
files a served one boots from: the saved deck, the bindings that say what each
coupled address does to it, the two starting-state documents and the write
bands. A 1.0 tree says in one line that the lane was skipped and is otherwise
untouched. The lane's own pre-flight sits beside the judgment one, so a tree
it cannot serve is refused with nothing written.

Note: this module keeps the import services out of its import graph. The
``.mat`` loader pulls numpy and scipy, the JSON loader pulls the channel-finder
preview, and the census pulls the TTL generator model, so every
``osprey.services.mml`` import happens lazily inside a command body.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from .output import report, warn
from .repo_resolver import find_repo_root, repo_option

if TYPE_CHECKING:  # the services stay out of the runtime import graph
    from osprey.services.mml.emit.va import ChannelBand, NominalSeed
    from osprey.services.mml.family import FamilyView
    from osprey.services.mml.judgments import VAPending
    from osprey.services.mml.mapping.check import VAExport
    from osprey.services.mml.mapping.schema import Mapping, VAFamily
    from osprey.services.mml.va.elements import ElementBinding
    from osprey.services.mml.va.verify import VerifyReport

#: File name of the import profile written beside the canonical documents.
PROFILE_FILENAME = "PROFILE.md"

#: File name of the mapping document ``map`` writes and checks.
MAPPING_FILENAME = "mapping.yaml"

#: Input suffixes ``import`` accepts, each dispatched to its loader.
_SUFFIXES = (".json", ".mat")


def mml_data_dir(repo: Path | None) -> Path:
    """The fixed ``data/mml/`` directory of the deployment repo.

    Args:
        repo: The ``--repo`` value, or ``None`` to search from the cwd.

    Returns:
        ``<repo root>/data/mml``; the directory may not exist yet.

    Raises:
        RepoNotFoundError: No deployment repo encloses the search start.
    """
    return find_repo_root(repo) / "data" / "mml"


@click.group()
def mml() -> None:
    """Install a facility from its MATLAB Middle Layer export."""


@mml.command("import")
@click.argument(
    "inputs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--system",
    "systems",
    multiple=True,
    metavar="TOKEN | PATH=TOKEN",
    help=(
        "System token for a flat input. A bare TOKEN only with a single input; "
        "with several inputs, PATH=TOKEN per flat input."
    ),
)
@repo_option
def import_cmd(inputs: tuple[Path, ...], systems: tuple[str, ...], repo: Path | None) -> None:
    """Import MML exports into data/mml/ (ao.json, ad.json, lattice/, PROFILE.md).

    A 2.0 export also writes its va.json and response.json there. The sibling
    files of an export are optional inputs: naming its ao.json is enough when
    they sit beside it under the same name. A lattice is filed only when it is
    the ring its export states it was sampled over, and the siblings of an
    export this import replaces are removed.
    """
    import shutil

    out_dir = mml_data_dir(repo)
    tokens = _pair_system_tokens(inputs, systems)
    for path in inputs:
        if path.suffix.lower() not in _SUFFIXES:
            raise click.UsageError(
                f"Cannot import {path}: only .json and .mat MML exports are accepted."
            )

    from osprey.services.mml.canonical import write_canonical
    from osprey.services.mml.census import take_census
    from osprey.services.mml.directions import vote_directions
    from osprey.services.mml.loaders.json_any import (
        AO_SUFFIX,
        LATTICE_SUFFIX,
        RESPONSE_SUFFIX,
        VA_SUFFIX,
        load_json,
        load_sibling,
        paired_sibling,
    )
    from osprey.services.mml.profile import render_profile
    from osprey.services.mml.systems import (
        IMPORT_ORDER_KEY,
        input_systems,
        merge_inputs,
        resolve_system,
    )
    from osprey.services.mml.va.canonical import (
        RESPONSE_FILENAME,
        VA_FILENAME,
        merge_response_inputs,
        merge_va_inputs,
        sibling_system,
        write_va_canonical,
    )
    from osprey.services.mml.va.fingerprint import check_fingerprint, lattice_fingerprint

    pairs = []
    decks: list[tuple[Path, str | None]] = []
    siblings: dict[str, list[tuple[Path, dict, str | None]]] = {VA_SUFFIX: [], RESPONSE_SUFFIX: []}
    for index, path in enumerate(inputs):
        if path.suffix.lower() == ".mat":
            from osprey.services.mml.loaders.mat import load_mat

            loaded = load_mat(path)
        else:
            sibling_suffix = next((s for s in siblings if path.name.endswith(s)), None)
            if sibling_suffix is not None:
                siblings[sibling_suffix].append((path, load_sibling(path), tokens[index]))
                continue
            loaded = load_json(path)
        if loaded.lattice is not None:
            decks.append((loaded.lattice, tokens[index]))
            continue
        # Resolved per input before merging, so an unresolvable input is
        # reported before any later input is read.
        pairs.append((loaded, resolve_system(loaded, tokens[index])))

    ao, ad = merge_inputs(pairs)

    # Every file of an export is named after its AO, so the AO's stem says
    # which system a lattice or a sibling given beside it belongs to.
    exports: list[tuple[Path, list[str]]] = []
    systems_by_stem: dict[str, list[str]] = {}
    for loaded, token in pairs:
        name = loaded.source.name
        if name.endswith(AO_SUFFIX):
            carried = input_systems(loaded, token)
            exports.append((loaded.source, carried))
            systems_by_stem.setdefault(name[: -len(AO_SUFFIX)], []).extend(carried)

    # The siblings an export was not asked for by name: they belong to its
    # system, so they are only paired for an export that carries exactly one.
    given = {path.resolve() for path in inputs}
    for source, carried in exports:
        if len(carried) != 1:
            continue
        for suffix, found in siblings.items():
            paired_path = paired_sibling(source, suffix)
            if paired_path is not None and paired_path.resolve() not in given:
                found.append((paired_path, load_sibling(paired_path), carried[0]))
        deck = paired_sibling(source, LATTICE_SUFFIX)
        if deck is not None and deck.resolve() not in given:
            from osprey.services.mml.loaders.mat import load_mat

            if load_mat(deck).lattice is not None:
                decks.append((deck, carried[0]))

    documents: dict[str, dict] = {}
    for suffix, filename, merge in (
        (VA_SUFFIX, VA_FILENAME, merge_va_inputs),
        (RESPONSE_SUFFIX, RESPONSE_FILENAME, merge_response_inputs),
    ):
        resolved = [
            (source, document, sibling_system(source, document, token))
            for source, document, token in siblings[suffix]
        ]
        for source, _, system in resolved:
            if system not in ao[IMPORT_ORDER_KEY]:
                raise click.UsageError(
                    f"Cannot import {source}: nothing in this import carries system "
                    f"{system!r}; import that system's export in the same command."
                )
        documents[filename] = merge(resolved)

    lattices: dict[str, Path] = {}
    for source, explicit in decks:
        paired = systems_by_stem.get(source.stem.removesuffix(".lattice"), [])
        if explicit is None and len(paired) != 1:
            raise click.UsageError(
                f"Cannot tell which system the lattice {source} belongs to: no export "
                f"among the inputs is named after it and carries one system; pass "
                f"--system {source.name}=TOKEN for it."
            )
        system = explicit.strip() if explicit is not None else paired[0]
        if system not in ao[IMPORT_ORDER_KEY]:
            raise click.UsageError(
                f"Cannot import the lattice {source}: nothing in this import carries "
                f"system {system!r}; import that system's export in the same command."
            )
        if system in lattices:
            raise click.UsageError(
                f"System {system!r} is given two lattices, {lattices[system]} and "
                f"{source}; give each system one lattice."
            )
        lattices[system] = source

    # A deck and the export sampled over it travel as separate files, and the
    # export's four facts about its ring are the only thing that pairs them: a
    # deck of another ring would bind every calibration to the wrong element.
    for system, source in sorted(lattices.items()):
        block = documents[VA_FILENAME].get(system)
        stated = block.get("lattice") if isinstance(block, dict) else None
        if stated is None:
            continue
        if not isinstance(stated, dict) or "refused" in stated:
            reason = stated.get("refused") if isinstance(stated, dict) else stated
            warn(
                f"The export of system {system!r} states no ring fingerprint "
                f"({reason}), so the lattice {source} is filed unchecked."
            )
            continue
        from osprey.services.mml.loaders.mat import load_lattice

        mismatch = check_fingerprint(stated, lattice_fingerprint(load_lattice(source)))
        if mismatch is not None:
            raise click.UsageError(
                f"The lattice {source} is not the ring system {system!r} was exported "
                f"over: its {mismatch.field} is {mismatch.actual!r} and the export "
                f"states {mismatch.expected!r}; import the lattice the export was "
                "sampled from, or export the facts again over this one."
            )

    try:
        write_canonical(ao, ad, out_dir)
    except (ValueError, TypeError) as exc:
        raise click.ClickException(
            f"The merged export cannot be written as canonical JSON ({exc}); "
            "fix the offending value in the export and import again."
        ) from exc
    except OSError as exc:
        raise click.ClickException(
            f"Cannot write into {out_dir} ({exc}); make the directory writable and import again."
        ) from exc

    try:
        va_path, response_path = write_va_canonical(
            documents[VA_FILENAME], documents[RESPONSE_FILENAME], out_dir
        )
    except (ValueError, TypeError) as exc:
        raise click.ClickException(
            f"The virtual-accelerator export cannot be written as canonical JSON ({exc}); "
            "fix the offending value in the export and import again."
        ) from exc
    except OSError as exc:
        raise click.ClickException(
            f"Cannot write into {out_dir} ({exc}); make the directory writable and import again."
        ) from exc

    lattice_dir = out_dir / "lattice"
    try:
        if lattices:
            lattice_dir.mkdir(parents=True, exist_ok=True)
        for system, source in sorted(lattices.items()):
            shutil.copyfile(source, lattice_dir / f"{system}.mat")
    except OSError as exc:
        raise click.ClickException(
            f"Cannot write into {lattice_dir} ({exc}); make the directory writable "
            "and import again."
        ) from exc

    # The canonical pair holds what this import carried and nothing else, so
    # the rest of the directory says the same: a sibling or a deck of an export
    # this one replaces would be read against families that are no longer there.
    swept: list[Path] = []
    try:
        for filename in (VA_FILENAME, RESPONSE_FILENAME):
            stale = out_dir / filename
            if not documents[filename] and stale.exists():
                stale.unlink()
                swept.append(stale)
        if lattice_dir.is_dir():
            for deck in sorted(lattice_dir.glob("*.mat")):
                if deck.stem not in lattices:
                    deck.unlink()
                    swept.append(deck)
            if not any(lattice_dir.iterdir()):
                lattice_dir.rmdir()
    except OSError as exc:
        raise click.ClickException(
            f"Cannot remove the files of an earlier import from {out_dir} ({exc}); "
            "make the directory writable and import again."
        ) from exc

    census = take_census(ao, ad)
    votes = vote_directions(ao)
    profile_path = out_dir / PROFILE_FILENAME
    try:
        profile_path.write_text(render_profile(census, votes), encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(
            f"Cannot write {profile_path} ({exc}); make it writable and import again."
        ) from exc

    undecided = sum(1 for vote in votes.values() if vote.direction is None)
    totals = census.totals
    report(
        f"Imported {_count(len(census.systems), 'system')}, "
        f"{_count(totals.families, 'family', 'families')}, "
        f"{_count(totals.distinct_pvs, 'distinct PV')} and "
        f"{_count(undecided, 'undecided direction')} into {out_dir}."
    )
    if lattices:
        report(f"Copied {_count(len(lattices), 'lattice')} into {lattice_dir}.")
    if swept:
        names = ", ".join(str(path.relative_to(out_dir)) for path in swept)
        report(f"Removed {_count(len(swept), 'file')} left by an earlier import: {names}.")
    for path, filename in ((va_path, VA_FILENAME), (response_path, RESPONSE_FILENAME)):
        if path is not None:
            report(f"Wrote {filename} for {_count(len(documents[filename]), 'system')}.")


@mml.command("map")
@click.option("--init", "init", is_flag=True, help="Write the mapping.yaml skeleton.")
@click.option("--check", "check", is_flag=True, help="Check mapping.yaml against the export.")
@click.option("--force", is_flag=True, help="With --init, overwrite an existing mapping.yaml.")
@click.option(
    "--force-va",
    "force_va",
    is_flag=True,
    help="With --init, replace the virtual_accelerator block of an existing mapping.yaml.",
)
@click.option(
    "--no-derived",
    "no_derived",
    is_flag=True,
    help="With --check, report every derived slot as a problem.",
)
@repo_option
def map_cmd(
    init: bool,
    check: bool,
    force: bool,
    force_va: bool,
    no_derived: bool,
    repo: Path | None,
) -> None:
    """Write (--init) or check (--check) data/mml/mapping.yaml."""
    if init == check:
        raise click.UsageError("Give exactly one of --init or --check.")
    if force and not init:
        raise click.UsageError("--force only applies to --init.")
    if force_va and not init:
        raise click.UsageError("--force-va only applies to --init.")
    if no_derived and not check:
        raise click.UsageError("--no-derived only applies to --check.")

    out_dir = mml_data_dir(repo)
    mapping_path = out_dir / MAPPING_FILENAME
    ao, ad = _read_import(out_dir)

    from osprey.services.mml.directions import vote_directions

    votes = vote_directions(ao)
    if init:
        _init_mapping(mapping_path, ao, ad, votes, force=force, force_va=force_va)
    else:
        _check_mapping(mapping_path, ao, ad, votes, no_derived=no_derived)


def _read_import(out_dir: Path) -> tuple[dict, dict]:
    """Read the canonical export ``import`` wrote, as click errors on failure."""
    from osprey.services.mml.canonical import read_canonical

    try:
        return read_canonical(out_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(f"{exc}; run osprey mml import on the export first.") from exc
    except (ValueError, OSError) as exc:
        raise click.ClickException(
            f"Cannot read the imported export ({exc}); run osprey mml import again."
        ) from exc


def _init_mapping(
    path: Path, ao: dict, ad: dict, votes: dict, *, force: bool, force_va: bool
) -> None:
    """Write the skeleton to ``path``, refusing to replace a file without ``force``.

    A 2.0 export also carries a ``virtual_accelerator`` block. It is appended
    as text rather than dumped with the rest, because an existing mapping may
    already hold reviewed decisions that re-dumping would rewrite.
    """
    from osprey.services.mml.mapping.skeleton import (
        build_skeleton,
        count_judgment_slots,
        count_va_slots,
        dump_va_block,
        dump_yaml,
    )

    block = _va_block(path.parent, ao, ad)
    if path.exists() and not force:
        _append_va_block(path, block, force_va=force_va)
        return

    document = build_skeleton(ao, ad or None, votes)
    text = dump_yaml(document)
    if block is not None:
        text += dump_va_block(block)
    _write_mapping(path, text)

    undecided = sum(1 for vote in votes.values() if vote.direction is None)
    among = f"{_count(undecided, 'undecided direction')} among them"
    judgments = count_judgment_slots(document)
    if judgments:
        among += f" and {_count(judgments, 'judgment')} to answer"
    if block is not None:
        among += f" and {_count(count_va_slots(block), 'virtual-accelerator slot')}"
    report(f"Wrote {path}; fill its null slots ({among}), then run osprey mml map --check.")


def _write_mapping(path: Path, text: str) -> None:
    """Write mapping text, as a click error naming the file on failure."""
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(
            f"Cannot write {path} ({exc}); make it writable and run map --init again."
        ) from exc


def _va_block(out_dir: Path, ao: dict, ad: dict) -> dict | None:
    """Build the ``virtual_accelerator`` block of this export, or report why not.

    The block exists only for a 2.0 export: it needs the sampled
    ``va.json`` of one system and the deck that export was sampled over. When
    either is missing the caller writes no block, and one line says so.
    """
    import json

    from osprey.services.mml.mapping.skeleton import va_block, va_system
    from osprey.services.mml.va.canonical import VA_FILENAME

    va_path = out_dir / VA_FILENAME
    document: dict = {}
    if va_path.is_file():
        try:
            document = json.loads(va_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise click.ClickException(
                f"Cannot read {va_path} ({exc}); run osprey mml import on the export again."
            ) from exc
    available = {
        system
        for system, block in document.items()
        if isinstance(block, dict) and isinstance(ao.get(system), dict)
    }
    choice = va_system(ao, ad or None, available) if available else None
    source = choice or next(iter(sorted(available)), None)
    if source is None:
        named = va_system(ao, ad or None) or ", ".join(raw for raw in ao if not raw.startswith("_"))
        report(f"no 2.0 export for {named}; VA block not written")
        return None

    deck = out_dir / "lattice" / f"{source}.mat"
    if not deck.is_file():
        report(f"no lattice deck for {source} in {deck.parent}; VA block not written")
        return None

    from osprey.services.mml.family import family_views
    from osprey.services.mml.loaders.mat import load_lattice
    from osprey.services.mml.va.verdicts import propose

    views = {view.raw_name: view for view in family_views(source, ao[source])}
    return va_block(propose(document[source], load_lattice(deck), views), choice)


def _append_va_block(path: Path, block: dict | None, *, force_va: bool) -> None:
    """Add the virtual-accelerator block to a mapping that was already written.

    The rest of the file is never re-dumped: the block's lines are appended
    after the last one, so every key a reviewer touched keeps its bytes.
    """
    from osprey.services.mml.mapping.skeleton import count_va_slots, dump_va_block

    if block is None:
        raise click.ClickException(
            f"{path} already exists and may hold reviewed decisions; "
            "pass --force to replace it with a fresh skeleton."
        )

    mapping = _parse_mapping_file(path)
    if mapping.virtual_accelerator is not None and not force_va:
        raise click.ClickException(
            f"{path} already holds a virtual_accelerator block, which may hold "
            "reviewed answers; pass --force-va to replace it."
        )

    kept = path.read_text(encoding="utf-8")
    if mapping.virtual_accelerator is not None:
        kept = _without_va_block(kept)
    if kept and not kept.endswith("\n"):
        kept += "\n"
    _write_mapping(path, kept + dump_va_block(block))

    slots = count_va_slots(block)
    among = f" ({_count(slots, 'null slot')} to answer)" if slots else ""
    report(
        f"Added a virtual_accelerator block for "
        f"{_count(len(block['families']), 'family', 'families')} to {path}{among}."
    )


def _without_va_block(text: str) -> str:
    """Return the document without its ``virtual_accelerator:`` lines.

    Exactly the block is removed: the key's own line and every line up to the
    next top-level key, so a reviewed prefix and suffix keep their bytes.
    """
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    inside = False
    for line in lines:
        if inside:
            if line[:1].strip() and not line.startswith("#"):
                inside = False
            else:
                continue
        if line.startswith("virtual_accelerator:"):
            inside = True
            continue
        kept.append(line)
    return "".join(kept)


def _va_export(out_dir: Path, ao: dict, ad: dict, mapping: Mapping) -> VAExport | None:
    """Return the virtual accelerator this tree carries, as the check reads it.

    A 2.0 import files a ``va.json`` block per system beside the deck that
    block was sampled over. The one system checked is the one the mapping says
    it decided, so a reviewer's own answer settles which questions the answers
    are held to; with the slot still null the same narrowing that wrote the
    block picks it. A tree carrying no such block is a 1.0 tree, which is asked
    nothing about a virtual accelerator.

    The deck is loaded because an answer binding an element is held to that
    element: without it every stated index reads as past the end of an empty
    ring. A block whose deck was never imported is accepted by ``import`` and
    refused here, where tree completeness belongs.
    """
    import json

    from osprey.services.mml.mapping.check import VAExport
    from osprey.services.mml.mapping.skeleton import va_system
    from osprey.services.mml.va.canonical import VA_FILENAME

    va_path = out_dir / VA_FILENAME
    if not va_path.is_file():
        return None
    try:
        document = json.loads(va_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise click.ClickException(
            f"Cannot read {va_path} ({exc}); run osprey mml import on the export again."
        ) from exc

    available = {
        system
        for system, block in document.items()
        if isinstance(block, dict) and isinstance(ao.get(system), dict)
    }
    if not available:
        return None
    decided = mapping.virtual_accelerator.system if mapping.virtual_accelerator else None
    system = decided if decided in available else va_system(ao, ad or None, available)
    if system is None:
        system = sorted(available)[0]

    deck = out_dir / "lattice" / f"{system}.mat"
    if not deck.is_file():
        return VAExport(system=system)

    from osprey.services.mml.family import family_views
    from osprey.services.mml.judgments import VAPending
    from osprey.services.mml.loaders.mat import load_lattice
    from osprey.services.mml.va.verdicts import propose

    ring = load_lattice(deck)
    views = {view.raw_name: view for view in family_views(system, ao[system])}
    return VAExport(
        system=system,
        pending=VAPending(
            system=system,
            proposed=propose(document[system], ring, views),
            block=document[system],
            ring=ring,
        ),
    )


def _check_mapping(path: Path, ao: dict, ad: dict, votes: dict, *, no_derived: bool) -> None:
    """Parse and check ``path``; print every problem and exit non-zero on any."""
    from osprey.services.mml.mapping.check import check_mapping

    mapping = _parse_mapping_file(path)
    result = check_mapping(
        mapping, ao, votes, no_derived=no_derived, va=_va_export(path.parent, ao, ad, mapping)
    )
    for problem in result.problems:
        report(str(problem))

    levels = ", ".join(f"{count} {level}" for level, count in result.derived_descriptions.items())
    warn(
        f"Derived slots still to review: descriptions ({levels}); "
        f"{_count(result.derived_directions, 'direction')}."
    )

    if result.problems:
        raise click.ClickException(
            f"{_count(len(result.problems), 'problem')} in {path}; fix each and check again."
        )
    report(f"{path} passes the check.")


#: The packaged control-assistant data tree the demo refusals compare against.
_PACKAGED_DATA = (
    Path(__file__).resolve().parents[1] / "templates" / "apps" / "control_assistant" / "data"
)

#: The one tier database emit owns; every other file under ``tiers/`` refuses.
_TIER_DATABASE = Path("tier3") / "middle_layer.json"

#: The saved deck, under ``data/``. It carries no provenance stamp: it stays a
#: pure pyAT document, and its "was this emitted?" is the digest the bindings
#: name.
_VA_LATTICE = Path("simulation") / "lattice.json"

#: The bindings document, under ``data/``: what says a tree serves a virtual
#: accelerator at all, and the file the model's channel set is derived from,
#: since an emitted tree carries no channel manifest.
_VA_BINDINGS = Path("simulation") / "va_bindings.json"

#: The starting-state document, under ``data/``: the channels the simulation
#: serves and the values it boots them at. It is the one document a scenario is
#: resolved against, so it is what a scenario bundle has to fit.
_VA_MACHINE = Path("simulation") / "machine.json"

#: The pair that makes a tree serve a ring: the deck and the bindings that tie
#: channels to it. An export describing no machine displaces them, so a tree
#: harvested from one serves its own channels and no model.
_VA_SERVED_MODEL = (_VA_LATTICE, _VA_BINDINGS)

#: The virtual-accelerator documents this command owns whole, under ``data/``.
#: Each opens with the provenance stamp, which is how the pre-flight tells a
#: file this lane wrote from one a person hand-authored.
_VA_STAMPED = (
    _VA_BINDINGS,
    _VA_MACHINE,
    Path("machine_state_channels.json"),
)

#: The scenario bundles, under ``data/``. Each names channels in its overrides
#: and its archiver events, and the simulation refuses to boot on one naming a
#: channel the served machine does not carry; emit answers that channel set
#: afresh, so every bundle is held against the one this run writes.
_VA_SCENARIOS = Path("simulation") / "scenarios"

#: The write-safety bands, under ``data/``. The file is shared with the
#: facility, so the lane states only the addresses its own bindings name.
_VA_LIMITS = Path("channel_limits.json")


@dataclass(frozen=True)
class _VALane:
    """What the virtual-accelerator lane writes, decided before the first write.

    Every fact the emitters need is read once, by the pre-flight, so a tree the
    lane cannot serve is refused with nothing written rather than half way
    through.

    Attributes:
        system: The raw system token the exported block describes.
        ring: The deck, renamed by the addressing pass -- the elements the
            bindings name are the ones saved, so both come from one pass.
        views: The judged family views of that system.
        verdicts: What the virtual accelerator does with each family, keyed by
            ``(raw system, raw family)``; the reviewer's answers, as the
            mapping's ``virtual_accelerator`` block states them.
        judged_va: Each family's ``va.json`` block in the judged device order,
            keyed the same way.
        element_bindings: What each family's devices drive, keyed by raw family.
        energy_gev: The beam energy the deck was built at.
    """

    system: str
    ring: Any
    views: tuple[FamilyView, ...]
    verdicts: dict[tuple[str, str], VAFamily]
    judged_va: dict[tuple[str, str], dict]
    element_bindings: dict[str, tuple[ElementBinding, ...]]
    energy_gev: float


#: Value ``--duckdb`` takes when given without a path.
_DUCKDB_DEFAULT = ""


@mml.command("emit")
@click.option(
    "--duckdb",
    "duckdb_path",
    is_flag=False,
    flag_value=_DUCKDB_DEFAULT,
    default=None,
    metavar="[PATH]",
    help=(
        "Also import the channel database into DuckDB, at PATH or "
        "data/channel_databases/middle_layer.duckdb."
    ),
)
@repo_option
def emit_cmd(duckdb_path: str | None, repo: Path | None) -> None:
    """Write the channel database, ontology, knowledge pages and corpus into data/."""
    from osprey.services.mml.emit.context import require_knowledge_extra

    require_knowledge_extra()

    root = find_repo_root(repo)
    data = root / "data"
    out_dir = data / "mml"
    db_dir = data / "channel_databases"
    tiers = db_dir / "tiers"
    bundle = data / "facility_knowledge"

    offenders = _demo_offenders(root, tiers, bundle)
    if offenders:
        report(offenders)
        raise click.ClickException(
            "The deployment still holds demo files emit would contradict; "
            "remove them with the rm line above and run osprey mml emit again."
        )

    ao, ad = _read_import(out_dir)
    mapping_path = out_dir / MAPPING_FILENAME
    mapping = _parse_mapping_file(mapping_path)
    # The judgments settle the grain the directions are asked about, so they
    # are checked first and every pre-flight precedes every write. The virtual
    # accelerator is read once, here: its answers are judged beside the rest,
    # and what its lane writes is decided before the first file is written.
    va_pending = _require_va(root, out_dir, mapping, _va_export(out_dir, ao, ad, mapping))
    _require_judgments(mapping, ao, va_pending)
    _require_directions(mapping, ao)
    lane = None if va_pending is None else _va_lane(ao, mapping, va_pending)

    import json

    import yaml

    from osprey.services.channel_finder.tools.validate_database import (
        validate_database_loading,
    )
    from osprey.services.facility_knowledge.ttl_generator import emitter
    from osprey.services.facility_knowledge.ttl_generator.direction import DirectionSource
    from osprey.services.facility_knowledge.ttl_generator.mml_source import build_graph_model
    from osprey.services.facility_knowledge.ttl_generator.ontology_map import (
        OntologyMapError,
        UnknownFamilyError,
    )
    from osprey.services.mml.canonical import AO_FILENAME
    from osprey.services.mml.emit.channel_db import build_channel_db
    from osprey.services.mml.emit.context import build_context
    from osprey.services.mml.emit.okf import write_okf_bundle
    from osprey.services.mml.emit.ontology import build_ontology_yaml, compile_to_json

    try:
        ctx = build_context(out_dir / AO_FILENAME, mapping_path, ao)
    except OSError as exc:
        raise click.ClickException(f"Cannot read the emit inputs ({exc}).") from exc

    # The last pre-flight, and the first point the served machine is knowable:
    # a scenario is held against the machine.json this deployment will serve,
    # which on a tree this run writes one for is the document the lane renders.
    # Still before the first write, so a refused tree is the tree this command
    # was handed.
    _require_scenarios(root, _served_machine_channels(data, lane, mapping, ctx))

    def _mapping_disagrees(exc: Exception) -> click.ClickException:
        return click.ClickException(
            f"{mapping_path} does not fit the export ({exc}); "
            "run osprey mml map --check and fix what it reports."
        )

    # -- channel database -------------------------------------------------------
    try:
        db = build_channel_db(ao, mapping, ctx)
        db_text = (
            json.dumps(db, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False) + "\n"
        )
    except ValueError as exc:
        raise _mapping_disagrees(exc) from exc

    db_path = db_dir / "middle_layer.json"
    db_targets = [db_path]
    if tiers.is_dir():
        db_targets.append(tiers / _TIER_DATABASE)
    for target in db_targets:
        _write_text(target, db_text)
        report(f"Wrote {target}.")

    _success, errors, _stats = validate_database_loading(db_path, "middle_layer")
    if errors:
        for error in errors:
            report(error.rstrip())
        raise click.ClickException(
            f"{db_path} was written but does not load as a middle-layer database; "
            "fix the mapping or export and emit again."
        )

    if duckdb_path is not None:
        from osprey.services.channel_finder.databases.duckdb_import import import_to_duckdb

        duck = (
            db_dir / "middle_layer.duckdb"
            if duckdb_path == _DUCKDB_DEFAULT
            else Path(duckdb_path).resolve()
        )
        try:
            duck.parent.mkdir(parents=True, exist_ok=True)
            import_to_duckdb(str(db_path), str(duck))
        except Exception as exc:  # noqa: BLE001 - duckdb raises its own hierarchy
            raise click.ClickException(
                f"Cannot import {db_path} into {duck} ({exc}); "
                "check the path is writable and emit again."
            ) from exc
        report(f"Wrote {duck}.")
        for line in _duckdb_collapsed_rows(ao, ad or None, mapping):
            report(line)

    # -- ontology ---------------------------------------------------------------
    token = mapping.facility.token
    if not token:
        raise _mapping_disagrees(ValueError("facility.token is empty"))
    schema_path = data / "ontology" / f"{token}.yaml"
    table_path = data / "facility_ontology.json"
    try:
        schema = build_ontology_yaml(mapping, ctx)
    except ValueError as exc:
        raise _mapping_disagrees(exc) from exc
    _write_text(schema_path, yaml.safe_dump(schema, sort_keys=False))
    report(f"Wrote {schema_path}.")
    try:
        ontology_map = compile_to_json(schema_path, table_path)
    except (OntologyMapError, ValueError) as exc:
        raise click.ClickException(
            f"The ontology schema {schema_path} does not compile ({exc}); "
            "run osprey mml map --check and fix what it reports."
        ) from exc
    except OSError as exc:
        raise click.ClickException(f"Cannot write {table_path} ({exc}).") from exc
    report(f"Wrote {table_path}.")

    # -- knowledge pages --------------------------------------------------------
    try:
        pages = write_okf_bundle(ao, ad or None, mapping, ctx, bundle)
    except ValueError as exc:
        raise _mapping_disagrees(exc) from exc
    except OSError as exc:
        raise click.ClickException(f"Cannot write into {bundle} ({exc}).") from exc
    report(f"Wrote {_count(len(pages), 'knowledge file')} under {bundle}.")

    # -- virtual accelerator ----------------------------------------------------
    if lane is not None:
        _emit_va(lane, data, mapping, ctx, _channel_addresses(db))
    else:
        _sweep_served_model(data)

    # -- corpus -----------------------------------------------------------------
    ttl_path = data / f"{token}.ttl"
    try:
        model = build_graph_model(ao, mapping, list(mapping.section_order))
        written = emitter.write_turtle(
            model,
            ontology_map,
            ttl_path,
            direction_source=DirectionSource.MAPPING,
            header_comments=ctx.header_lines,
        )
    except emitter.UndirectedSignalError as exc:
        raise click.ClickException(
            f"The corpus was not written because a signal has no direction: {exc}; "
            "run osprey mml map --check."
        ) from exc
    except UnknownFamilyError as exc:
        raise click.ClickException(
            f"The corpus was not written: {exc}; run osprey mml map --check."
        ) from exc
    except ValueError as exc:
        raise _mapping_disagrees(exc) from exc
    except OSError as exc:
        raise click.ClickException(f"Cannot write {ttl_path} ({exc}).") from exc
    report(f"Wrote {written}.")

    report(
        "Run osprey build to copy these into the deployment; "
        "the running stack keeps its old copy until then."
    )


def _duckdb_collapsed_rows(ao: dict, ad: dict | None, mapping) -> list[str]:
    """Return the report of the bindings the DuckDB copy holds no row of.

    ``channels`` is keyed by PV, so two slots naming one PV and a broadcast row
    naming one PV for every device are one row each. The middle-layer database
    and the corpus keep every binding; this names what the SQL surface does not.
    The census is taken through the mapping, so the bindings counted are the
    ones the judgments settle: a shared PV a reviewer gave an owner is one
    binding of one device, and a family kept whole is reported as the export
    holds it.
    Each owner carries its device ordinal, counted from one as the mapping,
    ``PROFILE.md`` and the judgment keys count devices.
    """
    from osprey.services.mml.census import take_census

    census = take_census(ao, ad, mapping)
    collapsed = census.totals.bindings - census.totals.distinct_pvs
    if collapsed <= 0:
        return []
    lines = [
        f"{collapsed} of {census.totals.bindings} bindings share a PV with another, "
        f"so the DuckDB channels table holds {census.totals.distinct_pvs} rows; "
        "middle_layer.json and the corpus keep every binding."
    ]
    for item in census.shared_pvs:
        owners = ", ".join(f"{o.system}.{o.family}.{o.field}[{o.index + 1}]" for o in item.owners)
        lines.append(f"  {item.pv} is bound by {owners}.")
    for system in census.systems:
        for row in system.hazards.broadcast_rows:
            lines.append(
                f"  {row.system}.{row.family}.{row.field} broadcasts one "
                f"{row.key} entry to every device."
            )
    return lines


def _write_text(path: Path, text: str) -> None:
    """Write ``path`` atomically, creating its directory, as a click error on failure."""
    from osprey.cli.knowledge_cmd import _replace_file

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _replace_file(path, text)
    except OSError as exc:
        raise click.ClickException(f"Cannot write {path} ({exc}).") from exc


def _parse_mapping_file(path: Path):
    """Read and structurally parse ``path``, as click errors on failure."""
    import yaml

    from osprey.services.mml.mapping import MappingError, parse_mapping

    if not path.is_file():
        raise click.ClickException(
            f"{path} does not exist; run osprey mml map --init to write the skeleton."
        )
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise click.ClickException(f"{path} is not valid YAML: {exc}") from exc
    except OSError as exc:
        raise click.ClickException(f"Cannot read {path} ({exc}).") from exc
    if not isinstance(document, dict):
        raise click.ClickException(f"{path} must hold a YAML mapping at the top level.")
    try:
        return parse_mapping(document)
    except MappingError as exc:
        report(f"{exc.key}: {exc}")
        raise click.ClickException(f"{path} is not a valid mapping document.") from exc


def _require_judgments(mapping, ao: dict, va: VAPending | None = None) -> None:
    """Refuse a mapping whose judgment answers are unsettled or impossible.

    Both halves come from the services ``map --check`` refuses with: the slots
    left null, and the answers the export or the mapping cannot carry, judged
    against the raw export as the grain the reviewer was asked about. So emit
    refuses exactly what the check refuses, and no lane is ever handed a grain
    built from an answer the export cannot carry.

    Args:
        mapping: The parsed mapping carrying the answers.
        ao: The canonical export the questions are read off.
        va: What the virtual-accelerator rules ask of the exported system and
            of what, which is what holds an answer binding an element to the
            deck. Without it that half of the block is not judged at all, so
            the emit lane hands over exactly what ``map --check`` does.
    """
    from osprey.services.mml.judgments import (
        all_pending_judgments,
        unanswered_slots,
        validate_answers,
    )

    problems = unanswered_slots(mapping)
    problems.extend(
        (key, message)
        for key, message, _ in validate_answers(all_pending_judgments(ao), mapping, va)
    )
    if problems:
        for key, message in problems:
            report(f"{key}: {message}")
        raise click.ClickException(
            f"{_count(len(problems), 'judgment problem')} in {MAPPING_FILENAME}; "
            "run osprey mml map --check and fix each."
        )


def _require_directions(mapping, ao: dict) -> None:
    """Refuse a mapping whose ``directions`` leave out a signal group of ``ao``.

    The groups are the judged ones, the fields the lanes will emit: a field a
    judgment creates needs a direction of its own, and a field whose every row
    a judgment drops keeps needing one, because an answer moves rows between
    fields and never takes a field away.
    """
    from osprey.services.mml.family import system_bodies
    from osprey.services.mml.judgments import judged_family_views

    missing: list[str] = []
    for system, families in system_bodies(ao):
        for view in judged_family_views(system, families, mapping):
            for field in view.fields:
                key = f"{view.raw_name}.{field}"
                if key not in mapping.directions and key not in missing:
                    missing.append(key)
    if missing:
        for key in missing:
            report(f"directions.{key}: signal group has no directions entry")
        raise click.ClickException(
            f"{_count(len(missing), 'signal group')} without a direction; "
            "run osprey mml map --check and fill each."
        )


def _require_va(root: Path, out_dir: Path, mapping, export: VAExport | None) -> VAPending | None:
    """Refuse a tree whose virtual accelerator cannot be emitted, before any write.

    A tree with nothing to serve is not an error: a 1.0 export carries no
    virtual accelerator, so the lane says so in one line and writes nothing --
    it sweeps the ring such a tree cannot serve instead
    (:func:`_sweep_served_model`). It is an error the moment the mapping carries a
    ``virtual_accelerator:`` block, because that block is written from an
    export and answered against it: with the export gone the files the served
    machine boots from are missing rather than absent. A deployment serving a
    virtual accelerator of its own -- a preset's demo ring -- says nothing
    here: its deck is not the export's, and a 1.0 install onto such a
    deployment is an ordinary install.

    On a tree that does carry one, this refuses what the answers could not be
    judged against in the first place: a mapping deciding nothing about the
    exported virtual accelerator or deciding about another system's, and a deck
    that was never imported. It refuses a starting-state document a person
    wrote by hand too, named in one ``rm`` line rather than overwritten. It
    runs before the judgments so a mapping with no block is answered with the
    one line that adds it, rather than with a refusal per pending family.

    Args:
        root: The deployment repo root.
        out_dir: ``data/mml/``, where the export and the deck live.
        mapping: The parsed mapping, whose block holds the reviewer's verdicts.
        export: The virtual accelerator the tree carries, as
            :func:`_va_export` read it.

    Returns:
        What the rules ask of the exported system, to judge the block's answers
        against; ``None`` when this tree has no virtual accelerator to emit.

    Raises:
        click.ClickException: The tree cannot be served, naming what to fix.
    """
    from osprey.services.mml.va.canonical import VA_FILENAME

    document = mapping.virtual_accelerator

    if export is None:
        va_path = out_dir / VA_FILENAME
        # One phrasing for both readers: a file that is not there and a file
        # that is there without a block for an imported system are different
        # things to go and fix, whether the tree decided to serve one or not.
        absent = (
            "is not in the tree"
            if not va_path.is_file()
            else "carries no virtual accelerator for an imported system"
        )
        if document is not None:
            raise click.ClickException(
                f"{out_dir / MAPPING_FILENAME} decides a virtual accelerator and "
                f"{va_path} {absent}; "
                "re-export with mml_export 2.0 and run osprey mml import again."
            )
        report(
            f"VA lane skipped: data/mml/{VA_FILENAME} {absent}; "
            "re-export with mml_export 2.0 to enable it"
        )
        return None

    if document is None:
        raise click.ClickException(
            f"The {export.system} export carries a virtual accelerator "
            f"{out_dir / MAPPING_FILENAME} decides nothing about; "
            "run osprey mml map --init to add the block, then answer it."
        )
    if document.system is not None and document.system != export.system:
        raise click.ClickException(
            f"The mapping's virtual_accelerator block names {document.system!r}, and the "
            f"export carries a virtual accelerator for {export.system!r}; "
            "run osprey mml map --check and fix the block."
        )
    if export.pending is None:
        raise click.ClickException(
            f"The deck {export.system} was sampled over is not in the tree "
            f"({out_dir / 'lattice' / f'{export.system}.mat'}), so the virtual accelerator "
            "cannot be emitted; import the deck beside the export."
        )

    offenders = _unstamped_va(root)
    if offenders:
        report(offenders)
        raise click.ClickException(
            "The deployment holds virtual-accelerator files this command did not write and "
            "would replace; remove them with the rm line above and run osprey mml emit again."
        )
    return export.pending


def _va_lane(ao: dict, mapping, pending: VAPending) -> _VALane:
    """Read every fact the virtual-accelerator emitters need, before any write.

    It runs once the answers have been judged, because the grain it reads them
    through is the judged one: a device a reviewer dropped is already gone from
    the views and from the export block's rows alike.

    Args:
        ao: The canonical export.
        mapping: The parsed mapping, whose block holds the reviewer's verdicts.
        pending: What the rules ask of the exported system, carrying that
            system's block and the deck it was sampled over.

    Returns:
        Everything the lane writes.

    Raises:
        click.ClickException: The block does not fit the deck -- a position
            outside it, a family with elements and no devices, more element
            rows than devices, a renaming collision -- or it states no beam
            energy.
    """
    import math

    from osprey.services.mml.judgments import judged_family_views, judged_va_block
    from osprey.services.mml.va.elements import address_elements

    document = mapping.virtual_accelerator
    system = pending.system
    block = pending.block
    families = block.get("families")
    exported = families if isinstance(families, dict) else {}
    views = tuple(judged_family_views(system, ao[system], mapping))
    try:
        addressing = address_elements(block, pending.ring, document.families)
    except ValueError as exc:
        raise click.ClickException(
            f"The virtual accelerator does not fit the deck ({exc}); "
            "run osprey mml map --check and fix what it reports."
        ) from exc

    lattice = block.get("lattice")
    energy = lattice.get("energy_gev") if isinstance(lattice, dict) else None
    if (
        not isinstance(energy, (int, float))
        or isinstance(energy, bool)
        or not math.isfinite(energy)
    ):
        raise click.ClickException(
            f"The {system} virtual accelerator states no beam energy "
            "(virtual_accelerator.lattice.energy_gev); re-export with mml_export 2.0."
        )

    return _VALane(
        system=system,
        ring=addressing.ring,
        views=views,
        verdicts={(system, raw): decided for raw, decided in document.families.items()},
        judged_va={
            (system, view.raw_name): judged_va_block(
                system, view.raw_name, {system: block}, mapping, devices=view.n_devices
            )
            for view in views
            if view.raw_name in exported
        },
        element_bindings=dict(addressing.bindings),
        energy_gev=float(energy),
    )


def _unstamped_va(root: Path) -> str | None:
    """The one ``rm`` line naming virtual-accelerator documents emit did not write.

    The three documents the lane owns whole open with the provenance stamp, so
    a file that does not is one a person wrote, and this command replaces
    nothing it cannot prove it wrote itself. ``None`` when there are none.
    """
    import json
    import shlex

    from osprey.services.virtual_accelerator.bindings import PROVENANCE_KEY

    def stamped(path: Path) -> bool:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(document, dict) and next(iter(document), None) == PROVENANCE_KEY

    named = [
        path
        for path in (root / "data" / name for name in _VA_STAMPED)
        if path.is_file() and not stamped(path)
    ]
    if not named:
        return None
    return "rm " + " ".join(shlex.quote(path.relative_to(root).as_posix()) for path in named)


def _scenario_document(bundle: Path) -> tuple[dict, str | None]:
    """One scenario bundle, and what the simulation would refuse it for.

    The simulation reads each bundle's ``scenario.json`` at boot and stops on a
    directory that has none, one it cannot parse, or one that describes no
    scenario. Emit can see all three on the tree it is handed, so it asks the
    same three questions here and answers in the same order.

    Returns:
        The scenario, and why it could not be read; an unreadable bundle comes
        back as an empty scenario and a reason, a readable one as the scenario
        and ``None``.
    """
    import json

    path = bundle / "scenario.json"
    if not path.is_file():
        return {}, "has no scenario.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return {}, "has a scenario.json that cannot be read"
    except ValueError:
        return {}, "has a scenario.json that is not valid JSON"
    if not isinstance(document, dict):
        return {}, "has a scenario.json that describes no scenario"
    return document, None


def _scenario_channels(bundle: Path) -> list[str]:
    """Every channel one scenario bundle names, in the order it names them.

    A scenario reaches the machine through two keys: the overrides it starts a
    channel at and the archiver events it seeds one with. Both are addresses,
    and both are what the simulation resolves against the machine it serves.
    """
    document, _defect = _scenario_document(bundle)
    overrides = document.get("overrides")
    named = list(overrides) if isinstance(overrides, dict) else []
    entries = document.get("archiver")
    for entry in entries if isinstance(entries, list) else ():
        address = entry.get("channel") if isinstance(entry, dict) else None
        if isinstance(address, str) and address:
            named.append(address)
    return named


def _served_machine_channels(data: Path, lane: _VALane | None, mapping, ctx) -> set[str] | None:
    """Every channel the machine this deployment serves carries.

    A scenario is resolved against one document: the ``machine.json`` in the
    directory the simulation is handed. Which document that is depends on what
    this run writes. A tree whose machine this run writes is asked about the
    machine it is about to write, rendered here so a bundle is held against the
    machine it will actually meet; a tree whose machine this run leaves alone
    is asked about the one already there.

    Args:
        data: The deployment's ``data/`` directory.
        lane: What the virtual-accelerator pre-flight decided this tree serves,
            or ``None`` for a tree this run writes no machine for.
        mapping: The parsed mapping, read for the prose each channel carries.
        ctx: The provenance of this emit run.

    Returns:
        The served channel set, or ``None`` when this deployment has no machine
        at all -- a simulation handed none stops for want of it, whatever its
        scenarios say.
    """
    import json

    if lane is not None:
        from osprey.services.mml.emit.va import emit_machine

        try:
            machine_text, _seeds = emit_machine(
                lane.verdicts, lane.views, lane.judged_va, mapping, ctx
            )
        except ValueError:
            # The lane renders this same document, and says what does not fit
            # the export in its own words. There is no machine to hold a
            # bundle against until it does.
            return None
        document = json.loads(machine_text)
    else:
        path = data / _VA_MACHINE
        if not path.is_file():
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    channels = document.get("channels") if isinstance(document, dict) else None
    return set(channels) if isinstance(channels, dict) else None


def _require_scenarios(root: Path, served: set[str] | None) -> None:
    """Refuse the scenarios the machine this deployment serves cannot resolve.

    A scenario is stated against a machine: it starts channels at values and
    seeds archiver events on them, and the simulation stops at boot on one
    naming a channel the served machine does not carry, or on a bundle it
    cannot read at all. Both are visible here, on the tree emit was handed, so
    both are refused before anything is written and each offending scenario is
    named with what is wrong with it. A bundle the served machine can resolve
    is kept, whoever wrote it, and an empty directory has nothing to keep.

    Args:
        root: The deployment repo root.
        served: Every channel the served machine carries, or ``None`` when this
            deployment serves no machine and a scenario has nothing to fit.

    Raises:
        click.ClickException: The deployment holds a scenario the served
            machine cannot resolve, naming each one and why.
    """
    import shlex

    scenarios = root / "data" / _VA_SCENARIOS
    if served is None or not scenarios.is_dir():
        return
    offenders: list[tuple[Path, str]] = []
    for bundle in sorted(path for path in scenarios.iterdir() if path.is_dir()):
        _scenario, defect = _scenario_document(bundle)
        if defect is not None:
            offenders.append((bundle, defect))
            continue
        absent = sorted({name for name in _scenario_channels(bundle) if name not in served})
        if absent:
            offenders.append((bundle, f"names {', '.join(absent)}"))
    if not offenders:
        return
    for bundle, problem in offenders:
        report(f"{bundle.relative_to(root).as_posix()} {problem}")
    report(
        "rm -r "
        + " ".join(shlex.quote(bundle.relative_to(root).as_posix()) for bundle, _ in offenders)
    )
    raise click.ClickException(
        f"The deployment holds {_count(len(offenders), 'scenario')} the simulation cannot boot "
        "on; remove what the rm line above names and run osprey mml emit again."
    )


def _sweep_served_model(data: Path) -> None:
    """Remove the ring an export describing no machine displaces.

    An import means the last export and nothing else, and the ring a
    deployment serves is part of what an export answers: a tree harvested from
    an export that carries no machine has no ring of its own to serve, and one
    left behind would be served over the harvested channel names -- a model of
    some other machine, addressed as if it were this one. So the deck and the
    bindings go, the build derives no lattice from the tree, and the
    deployment serves its channels without physics. A deployment nobody
    harvested onto never reaches here and keeps the ring it shipped with.

    What is removed is named, because it is the one thing this run took away
    rather than wrote, and a second run finds nothing left to remove.

    Args:
        data: The deployment's ``data/`` tree.
    """
    removed = [path for path in (data / name for name in _VA_SERVED_MODEL) if path.is_file()]
    for path in removed:
        try:
            path.unlink()
        except OSError as exc:
            raise click.ClickException(f"Cannot remove {path} ({exc}).") from exc
    if removed:
        report(
            "Removed "
            + ", ".join(path.relative_to(data.parent).as_posix() for path in removed)
            + ": this export describes no machine, so the deployment serves no model."
        )


def _emit_va(
    lane: _VALane, data: Path, mapping, ctx, channel_addresses: Sequence[str]
) -> tuple[tuple[NominalSeed, ...], tuple[ChannelBand, ...]]:
    """Write the virtual-accelerator artifacts, and return what a report reads.

    The order the texts are rendered in is what each one names: the deck first,
    because the bindings stamp its digest; then the bindings, which the bands
    are derived from; the two starting-state documents; and the bands last. The
    digest is taken over the deck's text rather than over a file, so nothing is
    written until every refusal is known: a band a person's own file states
    differently stops the run with not one artifact of this lane's on the tree.

    Args:
        lane: What the pre-flight decided this tree serves.
        data: The deployment's ``data/`` directory.
        mapping: The parsed mapping, read for the prose each channel carries.
        ctx: The provenance of this emit run; the deck's digest lands on it.
        channel_addresses: Every address the channel database carries, which
            is what a tree with no bands of its own is stated over.

    Returns:
        One row per nominal read and one per coupled setpoint banded, for the
        run's report to list.

    Raises:
        click.ClickException: A file could not be written, or the export and
            the mapping do not fit the deck.
    """
    import json

    from osprey.services.mml.canonical import write_if_changed
    from osprey.services.mml.emit.va import (
        emit_bindings,
        emit_channel_limits,
        emit_lattice,
        emit_machine,
        emit_state_channels,
    )
    from osprey.services.virtual_accelerator.bindings import parse_bindings

    def _disagrees(exc: Exception) -> click.ClickException:
        return click.ClickException(
            f"The virtual accelerator does not fit the export ({exc}); "
            "run osprey mml map --check and fix what it reports."
        )

    lattice_path = data / _VA_LATTICE
    try:
        lattice_text = emit_lattice(lane.ring, lattice_path, ctx, write=False)
    except ValueError as exc:
        raise _disagrees(exc) from exc

    try:
        bindings_text = emit_bindings(
            lane.verdicts,
            lane.views,
            lane.element_bindings,
            lane.judged_va,
            ctx,
            system=lane.system,
            energy_gev=lane.energy_gev,
        )
        machine_text, seeds = emit_machine(lane.verdicts, lane.views, lane.judged_va, mapping, ctx)
        state_text = emit_state_channels(lane.views, mapping, ctx)
    except ValueError as exc:
        raise _disagrees(exc) from exc

    limits_path = data / _VA_LIMITS
    existing = None
    if limits_path.is_file():
        try:
            existing = json.loads(limits_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise click.ClickException(
                f"Cannot read {limits_path} ({exc}); fix it or remove it and emit again."
            ) from exc
    try:
        limits_text, bands = emit_channel_limits(
            existing,
            parse_bindings(json.loads(bindings_text)).bindings,
            channel_addresses,
            ctx,
            views=lane.views,
            system=lane.system,
        )
    except ValueError as exc:
        raise _disagrees(exc) from exc

    refused = [band for band in bands if band.refused]
    if refused:
        for band in refused:
            report(f"{band.address}: {band.refused}")
        raise click.ClickException(
            f"{_count(len(refused), 'band')} in {limits_path} that this command did not write "
            "state a coupled setpoint differently; nothing was written, so fix the file or "
            "remove those entries, then run osprey mml emit again."
        )

    # The deck keeps its own writer: identical bytes leave the file, and its
    # modification time, exactly as they were.
    try:
        write_if_changed(lattice_path, lattice_text)
    except OSError as exc:
        raise click.ClickException(f"Cannot write {lattice_path} ({exc}).") from exc
    report(f"Wrote {lattice_path}.")

    written = (
        (data / _VA_STAMPED[0], bindings_text),
        (data / _VA_STAMPED[1], machine_text),
        (data / _VA_STAMPED[2], state_text),
        (limits_path, limits_text),
    )
    for path, text in written:
        _write_text(path, text)
        report(f"Wrote {path}.")
    return seeds, bands


def _channel_addresses(db: dict) -> list[str]:
    """Every address the emitted channel database carries, in document order.

    Read off the database rather than the export, so the bands a tree with no
    file of its own is stated over cover exactly the channels it serves.
    """
    from osprey.services.channel_finder.databases.middle_layer import CHANNEL_KEYS

    return [
        slot
        for system in db.values()
        if isinstance(system, dict)
        for family in system.values()
        if isinstance(family, dict)
        for field in family.values()
        if isinstance(field, dict)
        for key in CHANNEL_KEYS
        for slot in (field.get(key) or ())
        if isinstance(slot, str) and slot
    ]


def _demo_offenders(root: Path, tiers: Path, bundle: Path) -> str | None:
    """The one ``rm`` line naming demo files that must go before emit writes.

    It names the knowledge files byte-identical to the packaged demo bundle and
    the tier databases other than ``tier3/middle_layer.json`` (``tiers/`` itself
    stays: the build and the manifest read it); ``None`` when there are none. A directory
    whose files are all demo copies is named whole with ``rm -r``, so no demo
    sub-index outlives its pages; the root ``index.md`` is exempt, emit
    rewrites it.
    """
    import shlex

    def rel(path: Path) -> str:
        return shlex.quote(path.relative_to(root).as_posix())

    siblings: list[Path] = []
    if tiers.is_dir():
        siblings = sorted(
            path
            for path in tiers.rglob("*")
            if path.is_file() and path.relative_to(tiers) != _TIER_DATABASE
        )
    dirs: list[Path] = []
    singles: list[Path] = []

    packaged = _PACKAGED_DATA / "facility_knowledge"
    if bundle.is_dir() and packaged.is_dir():
        files = sorted(path for path in bundle.rglob("*") if path.is_file())

        def is_demo(path: Path) -> bool:
            relative = path.relative_to(bundle)
            if relative == Path("index.md"):
                return False
            twin = packaged / relative
            try:
                return twin.is_file() and twin.read_bytes() == path.read_bytes()
            except OSError:
                return False

        demo = {path for path in files if is_demo(path)}

        def all_demo(directory: Path) -> bool:
            return all(path in demo for path in files if directory in path.parents)

        for path in sorted(demo):
            # The highest directory below the bundle root holding only demo files.
            whole = None
            for parent in reversed(path.relative_to(bundle).parents):
                if parent == Path("."):
                    continue
                if all_demo(bundle / parent):
                    whole = bundle / parent
                    break
            if whole is None:
                singles.append(path)
            elif whole not in dirs:
                dirs.append(whole)
    named = (*dirs, *singles, *siblings)
    if not named:
        return None
    return ("rm -r " if dirs else "rm ") + " ".join(rel(path) for path in named)


def _pair_system_tokens(inputs: Sequence[Path], values: Sequence[str]) -> list[str | None]:
    """Assign each ``--system`` value to exactly one input, by position.

    A bare ``TOKEN`` is accepted only when there is a single input; otherwise
    every value must be ``PATH=TOKEN`` whose ``PATH`` names one of the inputs,
    and no input may be named twice. Paths are compared by location, so
    ``./flat.json`` and ``flat.json`` name the same input.
    """
    tokens: list[str | None] = [None] * len(inputs)
    locations = [path.resolve() for path in inputs]

    for value in values:
        path_text, sep, token = value.rpartition("=")
        if not sep:
            if len(inputs) != 1:
                raise click.UsageError(
                    f"--system {value} does not say which input it belongs to; "
                    "with several inputs write --system PATH=TOKEN for each flat input."
                )
            index = 0
        else:
            target = Path(path_text).resolve()
            if target not in locations:
                raise click.UsageError(
                    f"--system {value} names {path_text}, which is not one of the inputs; "
                    "spell PATH exactly as an input given on the command line."
                )
            index = locations.index(target)
        if not token.strip():
            raise click.UsageError(
                f"--system {value} gives an empty token; write --system PATH=TOKEN "
                "with a non-empty system name."
            )
        if tokens[index] is not None:
            raise click.UsageError(
                f"{inputs[index]} is given more than one --system; give each input one token."
            )
        tokens[index] = token
    return tokens


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


# --- verify ----------------------------------------------------------------
#
# The step between `emit` and `osprey build`: the emitted tree is booted
# in-process and its orbit response held against the matrix the export carries.
# Everything the comparison decides lives in `services/mml/va/verify.py`; what
# is here is the tree reading the other verbs already do, plus the report file.


#: What an emitted tree must carry for a model to be built over it, under
#: ``data/``. The deck and the bindings describe the ring; the nominals and the
#: bands are what the model boots in.
_VA_SERVED = (
    _VA_LATTICE,
    _VA_BINDINGS,
    _VA_MACHINE,
    _VA_LIMITS,
)


@mml.command("verify")
@repo_option
def verify_cmd(repo: Path | None) -> None:
    """Check the emitted virtual accelerator against the exported response matrix."""
    import json

    from osprey.services.mml.canonical import AO_FILENAME
    from osprey.services.mml.emit.context import build_context
    from osprey.services.mml.emit.va import emit_channel_limits, emit_machine
    from osprey.services.mml.va.verify import (
        REPORT_FILENAME,
        VerifyError,
        render_report,
        verify,
    )
    from osprey.services.virtual_accelerator.bindings import BindingsError, load_bindings
    from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError

    root = find_repo_root(repo)
    data = root / "data"
    out_dir = data / "mml"
    report_path = out_dir / REPORT_FILENAME

    ao, ad = _read_import(out_dir)
    mapping_path = out_dir / MAPPING_FILENAME
    mapping = _parse_mapping_file(mapping_path)
    pending = _verified_export(out_dir, mapping, _va_export(out_dir, ao, ad, mapping))
    response = _response_block(out_dir, pending.system)
    lane = _va_lane(ao, mapping, pending)
    _require_emitted(root, data)

    try:
        ctx = build_context(out_dir / AO_FILENAME, mapping_path, ao)
    except OSError as exc:
        raise click.ClickException(f"Cannot read the emit inputs ({exc}).") from exc

    limits_path = data / _VA_LIMITS
    try:
        bindings = load_bindings(data / _VA_BINDINGS)
        existing = json.loads(limits_path.read_text(encoding="utf-8"))
        _text, seeds = emit_machine(lane.verdicts, lane.views, lane.judged_va, mapping, ctx)
        _limits, bands = emit_channel_limits(
            existing, bindings.bindings, (), ctx, views=lane.views, system=lane.system
        )
    except (BindingsError, ValueError) as exc:
        raise click.ClickException(
            f"The emitted virtual accelerator does not fit the export ({exc}); "
            "run osprey mml emit again."
        ) from exc
    except OSError as exc:
        raise click.ClickException(f"Cannot read {limits_path} ({exc}).") from exc

    try:
        result = verify(
            data,
            system=lane.system,
            response=response,
            views=lane.views,
            verdicts=lane.verdicts,
            judged_va=lane.judged_va,
            seeds=seeds,
            bands=bands,
        )
    except (VerifyError, BindingsError, OrbitSolveError, ValueError) as exc:
        raise click.ClickException(f"The virtual accelerator cannot be verified ({exc}).") from exc
    except OSError as exc:
        raise click.ClickException(f"Cannot read the emitted tree ({exc}).") from exc

    _write_text(report_path, render_report(result, provenance=ctx.provenance_string))
    for line in _verify_lines(result):
        report(line)
    report(f"Wrote {report_path}.")
    report("Read it, then run osprey build to copy the tree into the deployment.")


def _verified_export(out_dir: Path, mapping, export: VAExport | None) -> VAPending:
    """The virtual accelerator to verify, or the refusal that there is none.

    Unlike ``emit``, which passes a 1.0 tree by with one line, this always
    refuses: a person running ``verify`` is asking for evidence about a served
    model, and a tree that carries none has to say so rather than report
    nothing and exit zero.

    Raises:
        click.ClickException: The tree carries no 2.0 export, or one this
            mapping was never answered against.
    """
    from osprey.services.mml.va.canonical import RESPONSE_FILENAME, VA_FILENAME

    if export is None:
        raise click.ClickException(
            f"There is no virtual accelerator to verify: {out_dir / VA_FILENAME} "
            f"and {out_dir / RESPONSE_FILENAME} are what a 2.0 export files here, and "
            "this tree carries no block for an imported system; re-export with "
            "mml_export 2.0 and run osprey mml import again."
        )
    if mapping.virtual_accelerator is None:
        raise click.ClickException(
            f"The {export.system} export carries a virtual accelerator "
            f"{out_dir / MAPPING_FILENAME} decides nothing about; "
            "run osprey mml map --init to add the block, then answer it."
        )
    if export.pending is None:
        raise click.ClickException(
            f"The deck {export.system} was sampled over is not in the tree "
            f"({out_dir / 'lattice' / f'{export.system}.mat'}), so its virtual accelerator "
            "cannot be built; import the deck beside the export."
        )
    return export.pending


def _response_block(out_dir: Path, system: str) -> dict:
    """The exported response matrix of one system, as the file states it.

    Raises:
        click.ClickException: The file is not in the tree, cannot be read, or
            carries no block for the system the virtual accelerator serves.
    """
    import json

    from osprey.services.mml.va.canonical import RESPONSE_FILENAME

    path = out_dir / RESPONSE_FILENAME
    if not path.is_file():
        raise click.ClickException(
            f"{path} is not in the tree, so there is no exported response matrix to hold "
            "the model against; re-export with mml_export 2.0 and run osprey mml import."
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise click.ClickException(
            f"Cannot read {path} ({exc}); run osprey mml import on the export again."
        ) from exc
    block = document.get(system) if isinstance(document, dict) else None
    if not isinstance(block, dict):
        raise click.ClickException(
            f"{path} carries no response matrix for {system!r}, which is the system the "
            "virtual accelerator serves; re-export that sub-machine with mml_export 2.0."
        )
    return block


def _require_emitted(root: Path, data: Path) -> None:
    """Refuse a tree ``emit`` has not written the served model into.

    Raises:
        click.ClickException: A file the model boots from is missing; the
            message names every one of them.
    """
    missing = [data / name for name in _VA_SERVED if not (data / name).is_file()]
    if missing:
        names = ", ".join(path.relative_to(root).as_posix() for path in missing)
        raise click.ClickException(
            f"The virtual accelerator has not been emitted ({names} missing); "
            "run osprey mml emit first."
        )


def _verify_lines(result: VerifyReport) -> list[str]:
    """What the command says about a report it just wrote."""
    checked, agreed = result.signed
    if not result.compared:
        return ["Not one entry of the exported response matrix could be compared."]
    lines = [
        f"{result.passed} of {result.compared} response entries "
        f"({result.pass_ratio * 100:.1f} %) are inside the band.",
        f"Sign agrees on {agreed} of the {checked} entries above the floor.",
    ]
    if agreed != checked:
        lines.append(
            f"{checked - agreed} entries push the beam the opposite way to the export; "
            "the report's outlier tables list the worst of them."
        )
    return lines
