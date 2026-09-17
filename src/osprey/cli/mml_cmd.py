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

Note: this module keeps the import services out of its import graph. The
``.mat`` loader pulls numpy and scipy, the JSON loader pulls the channel-finder
preview, and the census pulls the TTL generator model, so every
``osprey.services.mml`` import happens lazily inside a command body.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import click

from .output import report, warn
from .repo_resolver import find_repo_root, repo_option

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
    """Import MML exports into data/mml/ (ao.json, ad.json, PROFILE.md)."""
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
    from osprey.services.mml.profile import render_profile
    from osprey.services.mml.systems import merge_inputs, resolve_system

    pairs = []
    for index, path in enumerate(inputs):
        if path.suffix.lower() == ".mat":
            from osprey.services.mml.loaders.mat import load_mat

            loaded = load_mat(path)
        else:
            from osprey.services.mml.loaders.json_any import load_json

            loaded = load_json(path)
        # Resolved per input before merging, so an unresolvable input is
        # reported before any later input is read.
        pairs.append((loaded, resolve_system(loaded, tokens[index])))

    ao, ad = merge_inputs(pairs)

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


@mml.command("map")
@click.option("--init", "init", is_flag=True, help="Write the mapping.yaml skeleton.")
@click.option("--check", "check", is_flag=True, help="Check mapping.yaml against the export.")
@click.option("--force", is_flag=True, help="With --init, overwrite an existing mapping.yaml.")
@click.option(
    "--no-derived",
    "no_derived",
    is_flag=True,
    help="With --check, report every derived slot as a problem.",
)
@repo_option
def map_cmd(init: bool, check: bool, force: bool, no_derived: bool, repo: Path | None) -> None:
    """Write (--init) or check (--check) data/mml/mapping.yaml."""
    if init == check:
        raise click.UsageError("Give exactly one of --init or --check.")
    if force and not init:
        raise click.UsageError("--force only applies to --init.")
    if no_derived and not check:
        raise click.UsageError("--no-derived only applies to --check.")

    out_dir = mml_data_dir(repo)
    mapping_path = out_dir / MAPPING_FILENAME
    ao, ad = _read_import(out_dir)

    from osprey.services.mml.directions import vote_directions

    votes = vote_directions(ao)
    if init:
        _init_mapping(mapping_path, ao, ad, votes, force=force)
    else:
        _check_mapping(mapping_path, ao, votes, no_derived=no_derived)


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


def _init_mapping(path: Path, ao: dict, ad: dict, votes: dict, *, force: bool) -> None:
    """Write the skeleton to ``path``, refusing to replace a file without ``force``."""
    if path.exists() and not force:
        raise click.ClickException(
            f"{path} already exists and may hold reviewed decisions; "
            "pass --force to replace it with a fresh skeleton."
        )

    from osprey.services.mml.mapping.skeleton import (
        build_skeleton,
        count_judgment_slots,
        dump_yaml,
    )

    document = build_skeleton(ao, ad or None, votes)
    try:
        path.write_text(dump_yaml(document), encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(
            f"Cannot write {path} ({exc}); make it writable and run map --init again."
        ) from exc

    undecided = sum(1 for vote in votes.values() if vote.direction is None)
    among = f"{_count(undecided, 'undecided direction')} among them"
    judgments = count_judgment_slots(document)
    if judgments:
        among += f" and {_count(judgments, 'judgment')} to answer"
    report(f"Wrote {path}; fill its null slots ({among}), then run osprey mml map --check.")


def _check_mapping(path: Path, ao: dict, votes: dict, *, no_derived: bool) -> None:
    """Parse and check ``path``; print every problem and exit non-zero on any."""
    from osprey.services.mml.mapping.check import check_mapping

    mapping = _parse_mapping_file(path)
    result = check_mapping(mapping, ao, votes, no_derived=no_derived)
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
    # are checked first and both pre-flights precede every write.
    _require_judgments(mapping, ao)
    _require_directions(mapping, ao)

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


def _require_judgments(mapping, ao: dict) -> None:
    """Refuse a mapping whose judgment answers are unsettled or impossible.

    Both halves come from the services ``map --check`` refuses with: the slots
    left null, and the answers the export or the mapping cannot carry, judged
    against the raw export as the grain the reviewer was asked about. So emit
    refuses exactly what the check refuses, and no lane is ever handed a grain
    built from an answer the export cannot carry.
    """
    from osprey.services.mml.judgments import (
        all_pending_judgments,
        unanswered_slots,
        validate_answers,
    )

    problems = unanswered_slots(mapping)
    problems.extend(
        (key, message) for key, message, _ in validate_answers(all_pending_judgments(ao), mapping)
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
