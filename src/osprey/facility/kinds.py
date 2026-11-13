"""Measurement kinds: what a model's measurement file may allow, declared once.

A measurement kind is a measurement a model's measurement file lists under
``kinds:``. Each kind is one :class:`KindDeclaration`, named by an entry point of
the group :data:`ENTRY_POINT_GROUP` and nothing else: the five pyAML kinds
OSPREY ships (``orm``, ``dispersion``, ``trm``, ``crm``,
``chromaticity_monitor``) are entry points of the ``osprey-framework``
distribution, and a deployment registers its own kind the same way, from an
installable package present both where ``osprey build`` runs and in the
project image. "Built-in" means "declared by the ``osprey-framework``
distribution".

The declarations are read by ``osprey build`` alone. The build writes what
every runtime reader needs into the facts document: the per-model kind block
``measurement_models[<model>].kinds[<kind>]``, which holds everything about the
kind on that model (its sub-block and the facts its writer wrote from it), and
the deployment-wide table
``measurement_kinds`` (:func:`kind_table_row`). The approval hook, the MCP
servers and the measure script read those and import neither this module nor a
kind's package.

The fields a declaration states:

* ``name``: the word the measurement file lists;
* ``tool``: the pyAML tool the kind's view section configures, or ``None`` for
  a kind that writes no tool section;
* ``members``: each group or instrument role the kind uses, with its
  requirement (:data:`REQUIREMENTS`), judged per member against the model's
  wiring at build;
* ``member_sorts``: what each member the pyAML view does not ship
  (:data:`SHIPPED_SORTS`) names in the measurement file, one of
  :data:`MEMBER_SORTS`; the file names every member, shipped or not, under
  ``groups:`` or ``instruments:``;
* ``stepped_roles``: the roles whose setpoints a measurement steps; a kind with
  none writes no stepped entry and is never measured;
* ``tool_keys``: the measurement-file keys the kind's tool takes;
* ``single_pass``: whether the kind runs on a model solved single pass;
* ``matrix``, ``takes_by``, ``corrector_roles``: the measure tool's argument
  set: a response-matrix result, whether ``by`` is taken, the roles whose
  setpoints ``correctors`` may name (empty: no correctors);
* ``view_files``: each file the kind adds to the model's pyAML view, by name,
  with its writer;
* ``measure_plan``: what the measure script does for the kind, as data
  (:func:`plan_problems` states the vocabulary);
* ``description``: the one line the agent is told about the kind;
* ``measurement_schema``: the pydantic model the kind's own measurement
  sub-block ``kind_blocks.<kind>`` is validated by, or ``None`` when it has
  none;
* ``view_section``: the writer of the kind's pyAML tool section;
* optional ``facts_writer``, ``build_stops`` and ``settings_from_measurement``
  (``(model_entry, measurement_record) -> {<engine>: {<setting>: value}}``, the
  engine settings the kind's measurement needs; :func:`settings_patch`).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyaml_cs_osprey.names import CHROMATICITY_MONITOR_NAME

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = [
    "BUILTIN_DISTRIBUTION",
    "CRM_FILE",
    "E_DELTA_CAP",
    "E_DELTA_ORBIT",
    "ENTRY_POINT_GROUP",
    "FIT_STRATEGIES",
    "REQUIREMENTS",
    "RESULT_SHAPES",
    "TRM_FILE",
    "KindDeclaration",
    "KindFactsError",
    "KindRegistryError",
    "KindSettingsError",
    "KindStop",
    "KindTool",
    "MEMBER_SORTS",
    "SHIPPED_SORTS",
    "ViewContext",
    "ViewSection",
    "block_problems",
    "builtin_kinds",
    "kind_source",
    "kind_table_row",
    "written_facts",
    "member_sorts",
    "plan_problems",
    "registered",
    "settings_patch",
]

#: The entry-point group every measurement kind registers under.
ENTRY_POINT_GROUP = "osprey.facility.kinds"

#: The distribution whose kinds are the built-in ones.
BUILTIN_DISTRIBUTION = "osprey-framework"

#: How a member is judged against the model: named and wired by the model,
#: named and present in the facility, or neither required.
REQUIREMENTS = ("wired", "exists", "optional")

#: What a member names in the measurement file: a group under ``groups:``
#: standing for the monitors or the setpoints its devices carry, or one channel
#: under ``instruments:``.
MEMBER_SORTS = ("monitors", "setpoints", "instrument")

#: The member roles the pyAML view writes an array or a device for, each with
#: its sort; every other member is sorted by the kind declaring it.
SHIPPED_SORTS: Mapping[str, str] = {
    "bpm": "monitors",
    "hcor": "setpoints",
    "vcor": "setpoints",
    "quad": "setpoints",
    "sext": "setpoints",
    "tune": "instrument",
    "rf": "instrument",
}

#: How the measure script sizes a step: a magnet or group of magnets held to
#: half each setpoint's ``max_step``, the RF span, or a chromaticity monitor's
#: momentum step.
FIT_STRATEGIES = ("magnet-group", "rf", "monitor")

#: The shape of a completed measurement: a response matrix, the two planes of
#: one response (a one-column matrix), or a named pair of numbers.
RESULT_SHAPES = ("matrix", "planes", "scalar")

#: The request values a measure plan's arguments may name, each ``$<word>``.
PLAN_VALUES = (
    "settle",
    "n_avg_meas",
    "sleep_between_meas",
    "names",
    "deltas",
    "step",
    "half_step",
    "negative_step",
    "groups",
)

#: The keys a measure plan states, and those it may leave out.
_PLAN_KEYS = frozenset(
    {
        "tool",
        "fit",
        "magnets",
        "named",
        "delta",
        "rf_plant",
        "monitor",
        "bipolar",
        "unipolar",
        "kwargs",
        "group",
        "bpm_array",
        "result",
        "planes",
        "scalar",
        "missing",
    }
)


class KindRegistryError(ValueError):
    """A kind declaration the registry refuses.

    Attributes:
        kind: The kind's name.
        source: Who declares it: ``built-in`` or ``package <distribution>``.
        detail: What is wrong.
        remedy: What the kind's author changes.
    """

    def __init__(self, kind: str, source: str, detail: str, remedy: str) -> None:
        self.kind = kind
        self.source = source
        self.detail = detail
        self.remedy = remedy
        super().__init__(f"measurement kind {kind} ({source}): {detail}; fix: {remedy}")


class KindFactsError(ValueError):
    """A kind's ``facts_writer`` that fails on a model's sub-block.

    Attributes:
        kind: The kind's name.
        model: The model whose ``kind_blocks.<kind>`` the writer was handed.
        cause: What the writer raised, as ``<type>: <message>``.
    """

    def __init__(self, kind: str, model: str, cause: str) -> None:
        self.kind = kind
        self.model = model
        self.cause = cause
        super().__init__(f"kind {kind}'s facts writer fails on model {model}'s block: {cause}")


class KindSettingsError(ValueError):
    """A kind's ``settings_from_measurement`` that fails on a model's measurement record.

    Attributes:
        kind: The kind's name.
        model: The model whose entry and measurement record the transform was handed.
        cause: What the transform raised, or what is wrong with what it returned,
            as ``<type>: <message>``.
    """

    def __init__(self, kind: str, model: str, cause: str) -> None:
        self.kind = kind
        self.model = model
        self.cause = cause
        super().__init__(
            f"kind {kind}'s settings transform fails on model {model}'s block: {cause}"
        )


@dataclass(frozen=True)
class KindTool:
    """A pyAML tool: its ``type:`` path and its name in the configuration."""

    type: str
    name: str


@dataclass(frozen=True)
class KindStop:
    """One reason a kind's ``build_stops`` hook stops the build.

    Attributes:
        detail: What is wrong, naming the record or address at fault.
        remedy: What the author changes.
    """

    detail: str
    remedy: str


@dataclass(frozen=True)
class ViewContext:
    """What a kind's view writers are handed for one model's pyAML view.

    Attributes:
        model: The model's name.
        kinds: The kinds the model's measurement file allows.
        arrays: Each group role the file names, with its array's name in the view.
        measurement: The model's measurement file, as the facility file holds it.
        devices: The name a tool gives each pyAML device it may name, by its
            word: ``rf_plant``, ``tune_monitor``, ``chromaticity_monitor``.
        present: The words of the devices the view holds.
        alphac: The design momentum compaction, or ``None``.
        tuning_matrix: The design-optics response of a group role's magnets
            read by a named monitor, or ``None`` when the model has no design
            optics or a magnet has no column.
        matrix_drives_a_correction: Whether a matrix can drive a two-plane
            correction.
    """

    model: str
    kinds: tuple[str, ...]
    arrays: Mapping[str, str]
    measurement: Mapping[str, Any]
    devices: Mapping[str, str]
    present: frozenset[str]
    alphac: float | None
    tuning_matrix: Callable[[str, str], Mapping[str, Any] | None]
    matrix_drives_a_correction: Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class ViewSection:
    """What a kind's view-section writer adds to one model's pyAML configuration.

    Attributes:
        fields: The tool section's keys after its ``type`` and ``name``; the
            measurement-file values of the kind's ``tool_keys`` follow them.
        devices: Further devices, written after every kind's tool section.
    """

    fields: Mapping[str, Any] = field(default_factory=dict)
    devices: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True)
class KindDeclaration:
    """One measurement kind, as its package declares it.

    Every field is required except ``member_sorts``, ``facts_writer``,
    ``build_stops`` and ``settings_from_measurement``; :func:`registered`
    refuses a declaration whose values the build cannot use.
    """

    name: str
    tool: KindTool | None
    members: Mapping[str, str]
    stepped_roles: tuple[str, ...]
    tool_keys: tuple[str, ...]
    single_pass: bool
    matrix: bool
    takes_by: bool
    corrector_roles: tuple[str, ...]
    view_files: Mapping[str, Callable[[ViewContext], str | None]]
    measure_plan: Mapping[str, Any]
    description: str
    measurement_schema: type[BaseModel] | None
    view_section: Callable[[ViewContext], ViewSection] | None
    member_sorts: Mapping[str, str] = field(default_factory=dict)
    facts_writer: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None
    build_stops: Callable[[Mapping[str, Any], Any], Iterable[KindStop]] | None = None
    settings_from_measurement: (
        Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None
    ) = None


#: The declaration's field names, the only hooks a mapping declaration may name.
_FIELDS = tuple(f.name for f in fields(KindDeclaration))
_OPTIONAL = frozenset({"member_sorts", "facts_writer", "build_stops", "settings_from_measurement"})


def _source(entry: Any) -> str:
    dist = getattr(entry, "dist", None)
    name = getattr(dist, "name", None) if dist is not None else None
    if name is None:
        return "package unknown"
    return "built-in" if _normal(str(name)) == _normal(BUILTIN_DISTRIBUTION) else f"package {name}"


def _normal(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _declaration(loaded: Any, name: str, source: str) -> KindDeclaration:
    """The declaration an entry point loads to: a :class:`KindDeclaration` or its fields."""
    if isinstance(loaded, KindDeclaration):
        declaration = loaded
    elif isinstance(loaded, Mapping):
        unknown = sorted(set(loaded) - set(_FIELDS))
        if unknown:
            raise KindRegistryError(
                name,
                source,
                f"it names hook {unknown[0]!r}, which OSPREY does not offer",
                f"name only these: {', '.join(_FIELDS)}",
            )
        missing = [key for key in _FIELDS if key not in loaded and key not in _OPTIONAL]
        if missing:
            raise KindRegistryError(
                name, source, f"it states no {missing[0]!r}", f"state {missing[0]!r}"
            )
        declaration = KindDeclaration(**dict(loaded))
    else:
        raise KindRegistryError(
            name,
            source,
            f"its entry point loads a {type(loaded).__name__}, not a KindDeclaration",
            "point the entry point at a KindDeclaration",
        )
    if declaration.name != name:
        raise KindRegistryError(
            name,
            source,
            f"its entry point is named {name} and its declaration {declaration.name}",
            "give the entry point the declaration's name",
        )
    for problem in _declaration_problems(declaration):
        raise KindRegistryError(name, source, problem[0], problem[1])
    return declaration


def _declaration_problems(declaration: KindDeclaration) -> Iterator[tuple[str, str]]:
    for role, requirement in declaration.members.items():
        if requirement not in REQUIREMENTS:
            yield (
                f"member {role} has requirement {requirement!r}",
                f"give it one of {', '.join(REQUIREMENTS)}",
            )
    for role in (*declaration.stepped_roles, *declaration.corrector_roles):
        if role not in declaration.members:
            yield (f"it steps role {role}, which is not one of its members", f"declare {role}")
    for role, sort in declaration.member_sorts.items():
        if role not in declaration.members:
            yield (f"it sorts role {role}, which is not one of its members", f"declare {role}")
        elif sort not in MEMBER_SORTS:
            yield (f"member {role} has sort {sort!r}", f"give it one of {', '.join(MEMBER_SORTS)}")
        elif role in SHIPPED_SORTS and sort != SHIPPED_SORTS[role]:
            yield (
                f"member {role} has sort {sort}; the pyAML view names {role} {SHIPPED_SORTS[role]}",
                f"drop member_sorts[{role!r}]",
            )
    for role in declaration.members:
        if role not in SHIPPED_SORTS and role not in declaration.member_sorts:
            yield (
                f"member {role} states no sort",
                f"state member_sorts[{role!r}] as one of {', '.join(MEMBER_SORTS)}",
            )
    if declaration.stepped_roles or declaration.measure_plan:
        yield from (
            (problem, "use the measure-plan words OSPREY offers")
            for problem in plan_problems(declaration.measure_plan)
        )


def plan_problems(plan: Mapping[str, Any]) -> list[str]:
    """What is wrong with a measure plan, each a sentence; empty when the script can run it.

    A plan is a mapping:

    * ``tool``: ``{"attribute": <holder attribute>}`` or ``{"monitor": <name>}``,
      the chromaticity monitor of that name;
    * ``fit``: one of :data:`FIT_STRATEGIES`;
    * ``magnet-group``: ``magnets`` (``{"members": [<tool attribute>, ...]}``
      or ``{"array": <tool attribute naming the array>}``), ``delta`` (the
      tool attribute holding the step) and ``named`` (``correctors`` picks
      the members);
    * ``rf``: ``rf_plant`` (the tool attribute naming the plant) and ``delta``;
    * ``monitor`` (optional): a tool attribute holding a chromaticity monitor
      whose RF step is fitted first;
    * ``bipolar`` and ``unipolar``: ``{kwargs, set_delta?, factory?,
      per_member?}``: the ``measure`` arguments, whether the fitted step is
      written into ``delta`` first, the ``StepMeasurement`` constructor run
      instead of the tool, whether each member is fitted on its own;
    * ``kwargs``: the arguments of a ``monitor`` fit's ``measure``;
    * ``group`` (optional): ``{matrix_file, basis, factory, kwargs}``, a whole
      group stepped as one knob;
    * ``bpm_array`` (optional): the tool attribute naming the BPM array;
    * ``result``: one of :data:`RESULT_SHAPES`; ``planes`` names the two
      measurement keys, ``scalar`` the tool attribute read;
    * ``missing``: ``variables`` or ``kind``, what a stopped run says it left.

    An argument value ``$<word>`` is a request value (:data:`PLAN_VALUES`),
    ``@<attribute>`` the tool's attribute, anything else a literal.

    Args:
        plan: The plan.
    """
    problems: list[str] = []
    unknown = sorted(set(plan) - _PLAN_KEYS)
    if unknown:
        problems.append(f"its measure plan names {unknown[0]!r}, which OSPREY does not offer")
    tool = plan.get("tool")
    if (
        not isinstance(tool, Mapping)
        or len(tool) != 1
        or not ({"attribute", "monitor"} & set(tool))
    ):
        problems.append("its measure plan's tool is not {attribute} or {monitor}")
    fit = plan.get("fit")
    if fit not in FIT_STRATEGIES:
        problems.append(f"its measure plan fits by {fit!r}, not one of {', '.join(FIT_STRATEGIES)}")
    if fit == "magnet-group":
        magnets = plan.get("magnets")
        if not isinstance(magnets, Mapping) or not ({"members", "array"} & set(magnets)):
            problems.append("its measure plan's magnets are not {members} or {array}")
    if fit in ("magnet-group", "rf") and not isinstance(plan.get("delta"), str):
        problems.append("its measure plan names no delta attribute")
    if fit == "rf" and not isinstance(plan.get("rf_plant"), str):
        problems.append("its measure plan names no rf_plant attribute")
    if plan.get("result") not in RESULT_SHAPES:
        problems.append(
            f"its measure plan's result is {plan.get('result')!r}, "
            f"not one of {', '.join(RESULT_SHAPES)}"
        )
    if plan.get("missing") not in ("variables", "kind"):
        problems.append("its measure plan's missing is not variables or kind")
    calls = [plan.get("kwargs") or {}]
    for name in ("bipolar", "unipolar", "group"):
        part = plan.get(name)
        if part is None:
            if name != "group" and fit != "monitor":
                problems.append(f"its measure plan states no {name}")
            continue
        if not isinstance(part, Mapping):
            problems.append(f"its measure plan's {name} is not a mapping")
            continue
        calls.append(part.get("kwargs") or {})
        factory = part.get("factory")
        if factory is not None and not _step_factory(str(factory)):
            problems.append(
                f"its measure plan's {name} names StepMeasurement.{factory}, "
                "which pyaml_cs_osprey does not offer"
            )
    problems.extend(
        f"its measure plan names {value}, which OSPREY does not offer"
        for kwargs in calls
        for value in kwargs.values()
        if isinstance(value, str) and value.startswith("$") and value[1:] not in PLAN_VALUES
    )
    return problems


def _step_factory(name: str) -> bool:
    from pyaml_cs_osprey.measure import StepMeasurement

    return not name.startswith("_") and callable(getattr(StepMeasurement, name, None))


def registered() -> dict[str, KindDeclaration]:
    """Every registered measurement kind, by name.

    The built-in kinds first, then every other kind, each sorted by name.

    Raises:
        KindRegistryError: Two distributions declare one name, or a declaration
            names a hook OSPREY does not offer or a value it cannot use.
    """
    found, _sources = _load()
    return found


def judged_here(*, mounted: bool, lists_packages: bool) -> bool:
    """Whether a tree is where its listed measurement kinds are judged.

    A tree that mounts a machine root, or lists an application or a dependency,
    is a deployment: a listed kind no installed package registers stops it. A
    tree doing neither is a machine tree on its own, whose kinds are judged
    where the application is mounted.
    """
    return mounted or lists_packages


def listed_kinds(pyprojects: Iterable[Path]) -> dict[str, str]:
    """The kinds a deployment's listed packages declare, read from their project files.

    Each ``pyproject.toml`` is parsed for its ``[project]`` name and its
    ``[project.entry-points."osprey.facility.kinds"]`` table. Nothing is built
    and nothing is imported, so a kind is found whether or not its package is
    installed. Entry points a build backend computes (``dynamic``) are not read.

    Args:
        pyprojects: The ``pyproject.toml`` of each package the deployment lists.

    Returns:
        ``{kind: distribution}``, the distribution by its canonical name; a
        file that cannot be read, or names no project, declares nothing.
    """
    import tomllib

    from packaging.utils import canonicalize_name

    found: dict[str, str] = {}
    for pyproject in pyprojects:
        try:
            project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project")
        except (OSError, ValueError):
            continue
        if not isinstance(project, dict) or not isinstance(project.get("name"), str):
            continue
        groups = project.get("entry-points")
        declared = groups.get(ENTRY_POINT_GROUP) if isinstance(groups, dict) else None
        if not isinstance(declared, dict):
            continue
        for kind in declared:
            found.setdefault(str(kind), str(canonicalize_name(project["name"])))
    return found


def builtin_kinds() -> dict[str, KindDeclaration]:
    """The kinds the ``osprey-framework`` distribution declares, by name."""
    found, sources = _load()
    return {name: declaration for name, declaration in found.items() if sources[name] == "built-in"}


def kind_source(name: str) -> str | None:
    """Who declares kind *name*: ``built-in``, ``package <distribution>``, or ``None``."""
    return _load()[1].get(name)


def _load() -> tuple[dict[str, KindDeclaration], dict[str, str]]:
    from importlib import metadata

    builtin: dict[str, KindDeclaration] = {}
    others: dict[str, KindDeclaration] = {}
    sources: dict[str, str] = {}
    for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
        source = _source(entry)
        name = str(entry.name)
        if name in sources:
            if sources[name] == source:
                continue
            first, second = sorted((sources[name], source), key=lambda s: s != "built-in")
            raise KindRegistryError(
                name,
                second,
                f"{first} declares a kind of the same name",
                "give the deployment's kind a name no other kind has",
            )
        sources[name] = source
        declaration = _declaration(entry.load(), name, source)
        (builtin if source == "built-in" else others)[name] = declaration
    found = {**dict(sorted(builtin.items())), **dict(sorted(others.items()))}
    sorted_by: dict[str, tuple[str, str]] = {}
    for name, declaration in found.items():
        for role, sort in declaration.member_sorts.items():
            sorter, earlier = sorted_by.setdefault(role, (name, sort))
            if earlier != sort:
                raise KindRegistryError(
                    name,
                    sources[name],
                    f"member {role} has sort {sort}; kind {sorter} ({sources[sorter]}) "
                    f"sorts it {earlier}",
                    f"give member {role} the sort {earlier}, or name the role differently",
                )
    return found, sources


def member_sorts(declarations: Iterable[KindDeclaration]) -> dict[str, str]:
    """The sort of every member role the measurement file may name, given the kinds it lists.

    The shipped roles (:data:`SHIPPED_SORTS`) and each listed kind's own; a
    role is named under ``instruments:`` when its sort is ``instrument`` and
    under ``groups:`` otherwise.

    Args:
        declarations: The kinds the file lists.
    """
    sorts = dict(SHIPPED_SORTS)
    for declaration in declarations:
        for role in declaration.members:
            if role not in sorts:
                sorts[role] = declaration.member_sorts[role]
    return sorts


def kind_table_row(declaration: KindDeclaration) -> dict[str, Any]:
    """The kind's row of the facts document's deployment-wide ``measurement_kinds`` table."""
    return {
        "members": dict(declaration.members),
        "stepped_roles": list(declaration.stepped_roles),
        "single_pass": declaration.single_pass,
        "matrix": declaration.matrix,
        "takes_by": declaration.takes_by,
        "corrector_roles": list(declaration.corrector_roles),
        "description": declaration.description,
    }


def written_facts(
    document: Mapping[str, Any], declared: Mapping[str, KindDeclaration]
) -> dict[str, dict[str, Any]]:
    """What each listed kind's ``facts_writer`` writes, for every model whose file lists it.

    The writer is handed the facility file and the model's ``kind_blocks.<kind>``
    (absent: empty); what it writes is the kind's ``facts`` in that model's kind
    block, so two models giving the kind two blocks carry two sets of facts.

    Args:
        document: The facility file.
        declared: The registered kinds, by name.

    Returns:
        ``{kind: {model: facts}}``, the models in name order; a kind with no
        writer, or listed by no model, has no entry.

    Raises:
        KindFactsError: A writer raised, or wrote no mapping, on a model's block.
    """
    written: dict[str, dict[str, Any]] = {}
    for model in sorted(document.get("models") or [], key=lambda record: str(record["name"])):
        measurement = model.get("measurement")
        if not isinstance(measurement, Mapping):
            continue
        blocks = measurement.get("kind_blocks") or {}
        for kind in measurement.get("kinds") or []:
            declaration = declared.get(str(kind))
            if declaration is None or declaration.facts_writer is None:
                continue
            try:
                facts = dict(declaration.facts_writer(document, blocks.get(str(kind)) or {}))
            except Exception as error:
                cause = f"{type(error).__name__}: {error}"
                raise KindFactsError(str(kind), str(model["name"]), cause) from error
            written.setdefault(str(kind), {})[str(model["name"])] = facts
    return written


def settings_patch(
    declaration: KindDeclaration, model: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """What the kind's ``settings_from_measurement`` patches into one model's settings.

    The transform is handed copies of the model's entry and of its measurement
    record, and returns ``{<engine>: {<setting>: value}}`` for the model's own
    engine. It runs only on a block the kind admits: an absent block for a kind
    with no measurement schema, else one its schema accepts (an absent block is
    an empty one); the measurement stage states what is wrong with any other.

    Args:
        declaration: The kind.
        model: The model's entry in the combined document.

    Returns:
        The patch; empty when the kind declares no transform or does not admit
        the model's block.

    Raises:
        KindSettingsError: The transform raised, returned no mapping, a patch of
            an engine that is no mapping, or a patch of another engine.
    """
    transform = declaration.settings_from_measurement
    measurement = model.get("measurement")
    if transform is None or not isinstance(measurement, Mapping):
        return {}
    blocks = measurement.get("kind_blocks")
    block = blocks.get(declaration.name) if isinstance(blocks, Mapping) else None
    if not block_admitted(declaration, block):
        return {}
    name = str(model.get("name"))
    try:
        patch = transform(copy.deepcopy(dict(model)), copy.deepcopy(dict(measurement)))
        if not isinstance(patch, Mapping):
            raise TypeError(
                f"the settings transform returned a {type(patch).__name__}, not a mapping"
            )
        for engine, settings in patch.items():
            if not isinstance(settings, Mapping):
                raise TypeError(
                    f"the settings transform's patch of {engine} is a "
                    f"{type(settings).__name__}, not a mapping"
                )
            if engine != model.get("engine"):
                raise ValueError(
                    f"the settings transform patches engine {engine}; model {name} runs "
                    f"{model.get('engine')}"
                )
    except Exception as error:
        cause = f"{type(error).__name__}: {error}"
        raise KindSettingsError(declaration.name, name, cause) from error
    return {str(engine): dict(settings) for engine, settings in patch.items()}


def block_admitted(declaration: KindDeclaration, block: Any) -> bool:
    """Whether the kind takes this measurement sub-block; an absent block is an empty one.

    A kind with no ``measurement_schema`` takes only an absent block; a kind with
    one takes any block its schema accepts.

    Args:
        declaration: The kind.
        block: The sub-block, as the measurement file states it, or None.
    """
    if declaration.measurement_schema is None:
        return block is None
    return not block_problems(declaration, {} if block is None else block)


def block_problems(declaration: KindDeclaration, block: Any) -> list[str]:
    """What the kind's ``measurement_schema`` refuses in its measurement sub-block.

    Each problem reads ``<kind>.<loc>: <msg>``, ``loc`` the dotted path pydantic
    names.

    Args:
        declaration: The kind.
        block: The sub-block, as the measurement file states it.
    """
    from pydantic import ValidationError

    schema = declaration.measurement_schema
    if schema is None:
        return []
    try:
        schema.model_validate(block)
    except ValidationError as exc:
        return [
            f"{declaration.name}.{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
    return []


# --- the built-in kinds ------------------------------------------------------------

#: The measure-script arguments every built-in shares.
_SETTLE = {
    "sleep_between_step": "$settle",
    "n_avg_meas": "$n_avg_meas",
    "sleep_between_meas": "$sleep_between_meas",
}


def _orm_section(context: ViewContext) -> ViewSection:
    arrays = context.arrays
    return ViewSection(
        {
            "bpm_array_name": arrays["bpm"],
            "hcorr_array_name": arrays["hcor"],
            "vcorr_array_name": arrays["vcor"],
        }
    )


def _dispersion_section(context: ViewContext) -> ViewSection:
    return ViewSection(
        {"bpm_array_name": context.arrays["bpm"], "rf_plant_name": context.devices["rf_plant"]}
    )


#: The response matrices the built-in ``trm`` and ``crm`` add to a view.
TRM_FILE = "trm.json"
CRM_FILE = "crm.json"

#: Upper bound of the chromaticity monitor's relative momentum step.
E_DELTA_CAP = 0.01
#: The orbit shift ``e_delta * alphac`` the momentum step aims for.
E_DELTA_ORBIT = 5.0e-6

_TUNE_CORRECTION = KindTool("pyaml.tuning_tools.tune", "DEFAULT_TUNE_CORRECTION")
_CHROMATICITY_CORRECTION = KindTool(
    "pyaml.tuning_tools.chromaticity", "DEFAULT_CHROMATICITY_CORRECTION"
)
_CHROMATICITY_MONITOR_TYPE = "pyaml.tuning_tools.chromaticity_monitor"


def _trm_matrix(context: ViewContext) -> Mapping[str, Any] | None:
    if "tune_monitor" not in context.present:
        return None
    return context.tuning_matrix("quad", context.devices["tune_monitor"])


def _crm_matrix(context: ViewContext) -> Mapping[str, Any] | None:
    if not {"tune_monitor", "rf_plant"} <= context.present:
        return None
    return context.tuning_matrix("sext", context.devices["chromaticity_monitor"])


def _matrix_text(matrix: Mapping[str, Any] | None) -> str | None:
    import json

    if matrix is None:
        return None
    return json.dumps(matrix, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _trm_section(context: ViewContext) -> ViewSection:
    devices: list[Mapping[str, Any]] = []
    matrix = _trm_matrix(context)
    if matrix is not None and context.matrix_drives_a_correction(matrix):
        devices.append(
            {
                "type": _TUNE_CORRECTION.type,
                "name": _TUNE_CORRECTION.name,
                "quad_array_name": context.arrays["quad"],
                "betatron_tune_name": context.devices["tune_monitor"],
                "response_matrix": f"${{path:{TRM_FILE}}}",
            }
        )
    return ViewSection(
        {
            "quad_array_name": context.arrays["quad"],
            "betatron_tune_name": context.devices["tune_monitor"],
        },
        devices,
    )


def _crm_section(context: ViewContext) -> ViewSection:
    devices: list[Mapping[str, Any]] = []
    names = context.devices
    if "chromaticity_monitor" not in context.kinds and {"tune_monitor", "rf_plant"} <= (
        context.present
    ):
        devices.append(
            {
                "type": _CHROMATICITY_MONITOR_TYPE,
                "name": names["chromaticity_monitor"],
                "betatron_tune_name": names["tune_monitor"],
                "rf_plant_name": names["rf_plant"],
            }
        )
    matrix = _crm_matrix(context)
    if matrix is not None and context.matrix_drives_a_correction(matrix):
        devices.append(
            {
                "type": _CHROMATICITY_CORRECTION.type,
                "name": _CHROMATICITY_CORRECTION.name,
                "sextu_array_name": context.arrays["sext"],
                "chromaticity_monitor_name": names["chromaticity_monitor"],
                "response_matrix": f"${{path:{CRM_FILE}}}",
            }
        )
    return ViewSection(
        {
            "sextu_array_name": context.arrays["sext"],
            "chromaticity_name": names["chromaticity_monitor"],
        },
        devices,
    )


def _monitor_section(context: ViewContext) -> ViewSection:
    import math

    alphac = abs(context.alphac) if context.alphac else 0.0
    e_delta = (
        min(E_DELTA_CAP, E_DELTA_ORBIT / alphac)
        if math.isfinite(alphac) and alphac > 0
        else E_DELTA_CAP
    )
    return ViewSection(
        {
            "betatron_tune_name": context.devices["tune_monitor"],
            "rf_plant_name": context.devices["rf_plant"],
            "e_delta": e_delta,
            "max_e_delta": 2.0 * e_delta,
        }
    )


def _tuning_plan(
    attribute: str, array: str, delta: str, group: Mapping[str, Any], monitor: str | None
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "tool": {"attribute": attribute},
        "fit": "magnet-group",
        "magnets": {"array": array},
        "named": False,
        "delta": delta,
        "bipolar": {"kwargs": {**_SETTLE, delta: "$half_step", "n_step": 2}},
        "unipolar": {"kwargs": {**_SETTLE, delta: "$negative_step", "n_step": 1}},
        "group": group,
        "result": "matrix",
        "missing": "variables",
    }
    if monitor is not None:
        plan["monitor"] = monitor
    return plan


ORM = KindDeclaration(
    name="orm",
    tool=KindTool("pyaml.tuning_tools.orbit_response_matrix", "DEFAULT_ORBIT_RESPONSE_MATRIX"),
    members={"bpm": "wired", "hcor": "wired", "vcor": "wired"},
    stepped_roles=("hcor", "vcor"),
    tool_keys=("corrector_delta",),
    single_pass=True,
    matrix=True,
    takes_by=False,
    corrector_roles=("hcor", "vcor"),
    view_files={},
    measure_plan={
        "tool": {"attribute": "orm"},
        "fit": "magnet-group",
        "magnets": {"members": ["hcorrectors", "vcorrectors"]},
        "named": True,
        "delta": "corrector_delta",
        "bpm_array": "bpm_array_name",
        "bipolar": {"set_delta": True, "kwargs": {"corrector_names": "$names", **_SETTLE}},
        "unipolar": {
            "per_member": True,
            "factory": "orm",
            "kwargs": {
                "bpm_array": "@bpm_array_name",
                "correctors": "$names",
                "deltas": "$deltas",
                "set_wait_time": "$settle",
                "n_avg_meas": "$n_avg_meas",
            },
        },
        "result": "matrix",
        "missing": "kind",
    },
    description="orbit response of the corrector setpoints",
    measurement_schema=None,
    view_section=_orm_section,
)

DISPERSION = KindDeclaration(
    name="dispersion",
    tool=KindTool("pyaml.tuning_tools.dispersion", "DEFAULT_DISPERSION"),
    members={"bpm": "wired", "hcor": "wired", "vcor": "wired", "rf": "wired"},
    stepped_roles=("rf",),
    tool_keys=("frequency_delta",),
    single_pass=False,
    matrix=True,
    takes_by=False,
    corrector_roles=(),
    view_files={},
    measure_plan={
        "tool": {"attribute": "dispersion"},
        "fit": "rf",
        "rf_plant": "rf_plant_name",
        "delta": "frequency_delta",
        "bpm_array": "bpm_array_name",
        "bipolar": {"set_delta": True, "kwargs": {"set_waiting_time": "$settle"}},
        "unipolar": {
            "factory": "dispersion",
            "kwargs": {
                "bpm_array": "@bpm_array_name",
                "rf_plant": "@rf_plant_name",
                "delta": "$step",
                "set_wait_time": "$settle",
                "n_avg_meas": "$n_avg_meas",
            },
        },
        "result": "planes",
        "planes": ["frequency_response_x", "frequency_response_y"],
        "missing": "kind",
    },
    description="orbit response to the RF frequency",
    measurement_schema=None,
    view_section=_dispersion_section,
)

TRM = KindDeclaration(
    name="trm",
    tool=KindTool("pyaml.tuning_tools.tune_response_matrix", "DEFAULT_TUNE_RESPONSE_MATRIX"),
    members={"quad": "wired", "tune": "wired"},
    stepped_roles=("quad",),
    tool_keys=("quad_delta",),
    single_pass=False,
    matrix=True,
    takes_by=True,
    corrector_roles=(),
    view_files={TRM_FILE: lambda context: _matrix_text(_trm_matrix(context))},
    measure_plan=_tuning_plan(
        "trm",
        "quad_array_name",
        "quad_delta",
        {
            "matrix_file": TRM_FILE,
            "basis": "tune excursion",
            "factory": "trm",
            "kwargs": {
                "groups": "$groups",
                "delta": "$step",
                "tune_monitor": "@betatron_tune_name",
                "set_wait_time": "$settle",
                "n_avg_meas": "$n_avg_meas",
                "sleep_between_meas": "$sleep_between_meas",
            },
        },
        None,
    ),
    description="tune response of the quadrupoles",
    measurement_schema=None,
    view_section=_trm_section,
)

CRM = KindDeclaration(
    name="crm",
    tool=KindTool(
        "pyaml.tuning_tools.chromaticity_response_matrix", "DEFAULT_CHROMATICITY_RESPONSE_MATRIX"
    ),
    members={"sext": "wired", "tune": "wired", "rf": "wired"},
    stepped_roles=("sext", "rf"),
    tool_keys=("sextu_delta",),
    single_pass=False,
    matrix=True,
    takes_by=True,
    corrector_roles=(),
    view_files={CRM_FILE: lambda context: _matrix_text(_crm_matrix(context))},
    measure_plan=_tuning_plan(
        "crm",
        "sextu_array_name",
        "sextu_delta",
        {
            "matrix_file": CRM_FILE,
            "basis": "chromaticity excursion",
            "factory": "crm",
            "kwargs": {
                "groups": "$groups",
                "delta": "$step",
                "chromaticity_monitor": "@chromaticity_name",
                "set_wait_time": "$settle",
            },
        },
        "chromaticity_monitor",
    ),
    description="chromaticity response of the sextupoles (steps the RF)",
    measurement_schema=None,
    view_section=_crm_section,
)

CHROMATICITY_MONITOR = KindDeclaration(
    name="chromaticity_monitor",
    tool=KindTool(_CHROMATICITY_MONITOR_TYPE, CHROMATICITY_MONITOR_NAME),
    members={"tune": "wired", "rf": "wired"},
    stepped_roles=("rf",),
    tool_keys=("n_step", "n_avg_meas", "fit_order", "sleep_between_step", "sleep_between_meas"),
    single_pass=False,
    matrix=False,
    takes_by=False,
    corrector_roles=(),
    view_files={},
    measure_plan={
        "tool": {"monitor": CHROMATICITY_MONITOR_NAME},
        "fit": "monitor",
        "kwargs": {},
        "result": "scalar",
        "scalar": "chromaticity",
        "missing": "kind",
    },
    description="the chromaticity [xi_x, xi_y] (steps the RF)",
    measurement_schema=None,
    view_section=_monitor_section,
)
