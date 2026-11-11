"""A deployment registers a measurement kind of its own, and ``osprey build`` takes it unchanged.

The deployment package under ``fixtures/fixture_kind/`` registers the kind
``fixture_kind`` and the engine ``fixture_ring`` through their entry-point
groups, the way an installed package does: each test puts the directory and its
``dist-info`` on ``sys.path`` for its own duration. The deployment names the
package's source as a local-path profile ``dependencies`` entry, which the build
turns into a wheel listed in the deployment-packages manifest the project image
installs; the wheel build itself is stood in for, so no build backend is fetched.

The kind uses each seam a deployment's kind may use: members judged by their
requirement, a monitor group and instruments of its own, the stepped corrector
roles, its own measurement sub-block, a view file, the facts its writer puts in
each model's kind block, a settings patch of its model's engine, a build stop,
and a facility-added signal role; its engine marks a record ``command`` and
refuses a scenario fault on it. The built-in ``orm`` goes through the same
registry beside it. The tree is a ``data/facility/`` tree in the deployment
repository, not a mounted machine repository.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import shutil
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from click.testing import Result

from osprey.facility import kinds
from osprey.facility.errors import FacilityBuildError
from osprey_connectors.simulation.view import VARIABLES_FILE
from tests.facility._fixture_kind import (
    BLOCK,
    DISTRIBUTION,
    ENGINE,
    FACTS,
    FIXTURE_ROLE,
    KIND,
    PACKAGE_DIR,
    PYPROJECT,
    SOURCE,
    fake_wheel,
    fixture_tree,
    install,
    read_entry_points_afresh,
)
from tests.facility._pyaml_trees import built_document, measured_tree

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

pytest.importorskip("pyaml")

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Packages the registry refuses, each a site of its own beside the deployment package.
REFUSED = PACKAGE_DIR / "refused"

#: The built-in kinds, in their distribution's order.
BUILT_INS = ("orm", "dispersion", "trm", "crm", "chromaticity_monitor")

#: What the kind's settings transform asks of the engine: the trigger sample plus its window.
PATCHED = {ENGINE: {"turns": 12 + 64}}


# --- the package on the path -----------------------------------------------------------


@pytest.fixture(autouse=True)
def _entry_points_read_afresh() -> Iterator[None]:
    """Each test reads the entry points of its own path, and leaves none read behind.

    Torn down after the test's ``sys.path`` is restored, so a reading made
    while a fixture package was installed does not outlive the test.
    """
    read_entry_points_afresh()
    yield
    read_entry_points_afresh()


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> kinds.KindDeclaration:
    """The deployment package installed; its kind's declaration."""
    install(monkeypatch, PACKAGE_DIR)
    return kinds.registered()[KIND]


# --- the facility tree -----------------------------------------------------------------


def _sr(tree: dict[str, Any]) -> dict[str, Any]:
    (record,) = [model for model in tree["models.yaml"] if model["name"] == "SR"]
    sr: dict[str, Any] = record
    return sr


def _model(document: dict[str, Any], name: str) -> dict[str, Any]:
    found: dict[str, Any] = next(m for m in document["models"] if m["name"] == name)
    return found


def _channel(document: dict[str, Any], address: str) -> dict[str, Any]:
    found: dict[str, Any] = next(c for c in document["channels"] if c["id"] == address)
    return found


def _stop(tmp_path: Path, tree: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path / "tree", tree)
    return str(stopped.value.format_message())


# --- osprey build ----------------------------------------------------------------------


Deploy = Callable[..., tuple["BuiltProject", Result]]


@pytest.fixture
def deploy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Deploy:
    """``osprey build`` of a repo holding a tree and the package's source under ``kinds/``.

    The returned callable takes the tree and the profile's ``dependencies``
    (the package's local path when omitted).
    """
    from osprey.deployment import wheel_build
    from tests._builds import BuiltProject, init_project, run_build
    from tests.facility._synthetic_trees import write_tree

    monkeypatch.setattr(wheel_build, "_build_wheel", fake_wheel)

    def build(
        tree: dict[str, Any], dependencies: list[str] | None = None
    ) -> tuple[BuiltProject, Result]:
        repo = init_project(tmp_path, "hello-world", "demo")
        facility = repo / "data" / "facility"
        shutil.rmtree(facility)
        write_tree(facility, copy.deepcopy(tree))
        source = repo / SOURCE
        shutil.copytree(
            PACKAGE_DIR / "fixture_kind_deployment",
            source / "fixture_kind_deployment",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (source / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
        profile = repo / "profile.yml"
        document = yaml.safe_load(profile.read_text(encoding="utf-8"))
        document["dependencies"] = [SOURCE] if dependencies is None else dependencies
        profile.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
        return BuiltProject(repo), run_build(repo)

    return build


@pytest.fixture
def built(installed: kinds.KindDeclaration, deploy: Deploy) -> BuiltProject:
    """The fixture deployment, built, with the package installed."""
    del installed
    project, result = deploy(fixture_tree())
    assert result.exit_code == 0, result.output
    return project


def _facts(project: BuiltProject) -> dict[str, Any]:
    from osprey.apps import sdk

    return sdk.read_facts(project.build_dir)


def test_osprey_build_validates_and_renders_the_deployments_kind(built) -> None:
    facts = _facts(built)

    assert set(facts["measurement_models"]["SR"]["kinds"]) == {"trm", KIND}
    assert set(facts["measurement_models"]["LINE"]["kinds"]) == {"orm"}
    assert set(facts["measurement_kinds"]) == {*BUILT_INS, KIND}
    assert facts["measurement_kinds"][KIND] == kinds.kind_table_row(kinds.registered()[KIND])


def test_the_built_in_orm_goes_through_the_same_registry(built) -> None:
    from importlib.metadata import entry_points

    (orm,) = [e for e in entry_points(group=kinds.ENTRY_POINT_GROUP) if e.name == "orm"]
    (own,) = [e for e in entry_points(group=kinds.ENTRY_POINT_GROUP) if e.name == KIND]

    assert orm.dist is not None and own.dist is not None
    assert (orm.dist.name, own.dist.name) == ("osprey-framework", DISTRIBUTION)
    assert (kinds.kind_source("orm"), kinds.kind_source(KIND)) == (
        "built-in",
        f"package {DISTRIBUTION}",
    )
    line = _facts(built)["measurement_models"]["LINE"]["kinds"]["orm"]
    assert line["measure_plan"] == json.loads(json.dumps(dict(kinds.ORM.measure_plan)))


def test_the_package_is_built_to_a_wheel_the_project_image_installs(built) -> None:
    from osprey.deployment.wheel_build import (
        DEPLOYMENT_PACKAGES_DIRNAME,
        DEPLOYMENT_PACKAGES_MANIFEST,
    )

    packages = built.build_dir / DEPLOYMENT_PACKAGES_DIRNAME
    manifest = json.loads((packages / DEPLOYMENT_PACKAGES_MANIFEST).read_text("utf-8"))
    (wheel,) = manifest["packages"]

    assert (wheel["name"], wheel["version"]) == (DISTRIBUTION, "0.1")
    with zipfile.ZipFile(packages / wheel["wheel"]) as archive:
        entry_points = archive.read("fixture_kind_deployment-0.1.dist-info/entry_points.txt")
    assert f"{KIND} = fixture_kind_deployment:FIXTURE_KIND" in entry_points.decode()
    assert f"{DISTRIBUTION}==0.1" in (built.build_dir / "Dockerfile").read_text("utf-8")
    assert not list(built.build_dir.rglob("osprey-local-requirements.txt"))


def test_the_model_block_carries_the_kinds_members_roles_and_sub_block(built) -> None:
    entry = _facts(built)["measurement_models"]["SR"]["kinds"][KIND]

    assert entry["members"]["series_monitors"] == {
        "requirement": "wired",
        "addresses": ["BPM1:X", "BPM1:Y"],
    }
    assert entry["members"]["inject_request"] == {"requirement": "exists", "addresses": ["INJ:REQ"]}
    assert entry["members"]["rf_status"] == {"requirement": "optional", "addresses": []}
    assert (entry["stepped_roles"], entry["corrector_roles"]) == (
        ["hcor", "vcor"],
        ["hcor", "vcor"],
    )
    assert (entry["single_pass"], entry["matrix"], entry["takes_by"]) == (True, True, False)
    assert entry["block"] == BLOCK
    assert entry["facts"] == FACTS


def test_the_view_file_is_written_and_named_in_the_model_block(built) -> None:
    record = _facts(built)["measurement_models"]["SR"]

    assert record["kinds"][KIND]["view_files"] == ["fixture_chain.json"]
    assert "fixture_chain.json" in record["files"]
    chain = built.build_dir / "data" / "pyaml" / "SR" / "fixture_chain.json"
    assert json.loads(chain.read_text("utf-8")) == FACTS["chain"]


def test_an_application_reads_the_kinds_facts_and_tool_keys_through_the_sdk(built) -> None:
    from osprey.apps import sdk

    facts = sdk.read_facts(built.build_dir)

    assert facts["measurement_models"]["SR"]["kinds"][KIND]["facts"] == FACTS
    assert sdk.tool_keys(facts, "SR", KIND) == ["corrector_delta"]


def test_the_settings_patch_reaches_the_facility_file_and_the_simulator_view(built) -> None:
    sr = _model(built.facility, "SR")
    variables = json.loads(
        (built.build_dir / "data" / "simulator" / VARIABLES_FILE).read_text("utf-8")
    )
    (served,) = [model for model in variables["models"] if model["name"] == "SR"]

    assert sr["settings"] == PATCHED
    assert served["settings"] == PATCHED


@pytest.mark.usefixtures("installed")
def test_the_engine_judges_the_setting_the_kind_patches(tmp_path: Path) -> None:
    tree = fixture_tree()
    tree["measurement/SR.yaml"]["kinds"] = ["trm"]
    del tree["measurement/SR.yaml"]["kind_blocks"]
    _sr(tree)["settings"] = {ENGINE: {"turns": 0}}

    assert "settings key fixture_ring.turns is 0, not a positive whole number" in _stop(
        tmp_path, tree
    )


def test_the_facility_added_signal_role_is_carried(built) -> None:
    assert built.facility["signals"] == [FIXTURE_ROLE]
    assert _channel(built.facility, "CLEAN:EN")["signal"] == "fixture_role"
    assert "fixture_role" in json.dumps(_facts(built)["vocabulary"])


# --- the engine's mark and scenario stop -----------------------------------------------


def test_the_engine_marks_the_record_it_fires_command(built) -> None:
    (record,) = [w for w in _model(built.facility, "SR")["wiring"] if w["address"] == "INJ:REQ"]

    assert (record["engine_role"], record["marks"]) == ("setpoint", ["command"])
    assert _channel(built.facility, "INJ:REQ")["command"] is True
    assert "command" not in _channel(built.facility, "CLEAN:EN")


@pytest.mark.usefixtures("installed")
def test_a_command_step_on_a_channel_the_facility_does_not_mark_stops_the_build(
    tmp_path: Path,
) -> None:
    block = copy.deepcopy(BLOCK)
    block["steps"][2]["command"]["address"] = "CLEAN:EN"

    assert _stop(tmp_path, fixture_tree(block)) == (
        f"facility: value-invalid: measurement SR — kind {KIND}: step clean fires CLEAN:EN, "
        "which the facility does not mark command; fix: wire CLEAN:EN where its engine marks "
        "it command, or drop the step's command"
    )


@pytest.mark.usefixtures("installed")
def test_a_scenario_fault_on_a_record_the_engine_fires_stops_the_build(tmp_path: Path) -> None:
    tree = fixture_tree()
    tree["scenarios/fire.yaml"] = {"faults": {"SR": {"INJ:REQ": 1.0}}}

    assert _stop(tmp_path, tree) == (
        "facility: engine-invalid: scenario fire — `faults.SR.INJ:REQ`: INJ:REQ is a command "
        "the engine fires; a scenario holds no command; fix: drop INJ:REQ from the "
        "scenario's faults"
    )


# --- what the runtime readers make of it ----------------------------------------------


def test_the_measure_tool_resolves_and_advertises_the_kind(built) -> None:
    from fastmcp import FastMCP

    from osprey.mcp_server.python_executor.tools import pyaml_measure
    from osprey.mcp_server.python_executor.tools._pyaml_measure_args import resolve_measure_args

    args = resolve_measure_args(
        KIND,
        "SR",
        ["SCOR:V:SP"],
        None,
        None,
        config_path=built.build_dir / "config.yml",
        timeout=60,
    )
    server = FastMCP("probe")
    pyaml_measure.register(server, built.build_dir)
    (tool,) = asyncio.run(server.list_tools())

    assert args.correctors == ("SCOR:V:SP",)
    assert args.matrix is True
    assert args.plan == json.loads(json.dumps(dict(kinds.registered()[KIND].measure_plan)))
    assert KIND in tool.parameters["properties"]["kind"]["enum"]


def test_the_operator_skill_names_the_kind_its_step_keys_and_its_arguments(built) -> None:
    from osprey.cli.templates.manager import TemplateManager

    template = TemplateManager().jinja_env.get_template(
        "claude_code/claude/skills/pyaml-tools/SKILL.md.j2"
    )
    lines = template.render(pyaml_view_present=True, facility_facts=_facts(built)).splitlines()

    (listed,) = [line for line in lines if line.startswith(f"  - `{KIND}`:")]
    assert (
        listed
        == f"  - `{KIND}`: corrector response of the turn-series monitors; `corrector_delta`."
    )
    (correctors,) = [line for line in lines if line.startswith("  - `correctors`:")]
    assert f"`{KIND}`" in correctors
    (by,) = [line for line in lines if line.startswith("  - `by`:")]
    assert f"`{KIND}`" not in by and "`trm`" in by


# --- refusals --------------------------------------------------------------------------


def test_a_kind_naming_a_hook_osprey_does_not_offer_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, PACKAGE_DIR, REFUSED / "hooked")

    assert _stop(tmp_path, measured_tree()).startswith(
        "facility: value-invalid: measurement kind hooked_kind — it names hook 'on_trigger', "
        "which OSPREY does not offer; declared by package fixture-kind-hooked; "
        "fix: name only these: name, tool,"
    )


def test_a_kind_named_like_a_built_in_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, PACKAGE_DIR, REFUSED / "clash")

    assert _stop(tmp_path, measured_tree()) == (
        "facility: value-invalid: measurement kind orm — built-in declares a kind of the "
        "same name; declared by package fixture-kind-clash; fix: give the deployment's "
        "kind a name no other kind has"
    )


def test_a_kind_named_like_another_packages_kind_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, PACKAGE_DIR, REFUSED / "twin")

    message = _stop(tmp_path, measured_tree())

    assert message.startswith(f"facility: value-invalid: measurement kind {KIND} — package ")
    assert f"package {DISTRIBUTION}" in message and "package fixture-kind-twin" in message
    assert message.endswith("fix: give the deployment's kind a name no other kind has")


def test_a_measurement_file_naming_a_kind_no_package_registers_stops_the_build(
    deploy: Deploy,
) -> None:
    tree = fixture_tree()
    sr = _sr(tree)
    sr["engine"] = "pyat"
    sr["wiring"] = [w for w in sr["wiring"] if w["address"] != "INJ:REQ"]

    _project, result = deploy(tree, dependencies=[])

    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert (
        f"kind {KIND} is not registered under osprey.facility.kinds by any installed package; "
        f"fix: install the kind's package, or remove {KIND} from `kinds`"
    ) in output


@pytest.mark.usefixtures("installed")
def test_a_kind_whose_package_the_image_does_not_install_stops_the_build(
    deploy: Deploy, caplog: pytest.LogCaptureFixture
) -> None:
    _project, result = deploy(fixture_tree(), dependencies=[])

    assert result.exit_code != 0
    output = " ".join(caplog.text.split())
    assert (
        f"measurement kind '{KIND}' (package {DISTRIBUTION}): the kind's package is not among "
        "the packages the project image installs"
    ) in output
    assert "'orm'" not in output


def test_the_built_in_kinds_distribution_is_never_refused_by_the_install_check() -> None:
    from osprey.deployment.wheel_build import refuse_unshipped_kinds

    assert {kinds.kind_source(name) for name in kinds.registered()} == {"built-in"}
    refuse_unshipped_kinds([])


# --- no kind is dispatched on by name --------------------------------------------------

#: The modules a kind passes through between the registry and the measure script.
KIND_HANDLING = (
    "src/osprey/facility/views/pyaml.py",
    "src/osprey/mcp_server/python_executor/tools/_pyaml_measure_args.py",
    "src/osprey/mcp_server/python_executor/tools/_pyaml_measure_script.py",
    "src/osprey/mcp_server/python_executor/tools/_pyaml_measure_result.py",
    "src/osprey/mcp_server/python_executor/tools/pyaml_measure.py",
    "src/osprey/templates/claude_code/claude/hooks/osprey_approval.py",
    "src/osprey/facility/validate.py",
    "src/osprey/facility/layers/mml/seed.py",
)

#: Each (module, function, word) where a kind's name is spelled for another reason.
NOT_A_KIND = {
    (
        "src/osprey/facility/views/pyaml.py",
        "_view_context",
        "chromaticity_monitor",
    ): "the ViewContext's device word for the chromaticity monitor device, not the kind",
}


def _spelled(tree: ast.AST, names: frozenset[str]) -> Iterator[tuple[str, int, str]]:
    """Each comparison operand, mapping key or subscript spelling a name: function, line, word."""

    def constants(node: ast.AST) -> Iterator[ast.Constant]:
        if isinstance(node, ast.Constant):
            yield node
        elif isinstance(node, ast.Tuple | ast.List | ast.Set):
            for element in node.elts:
                yield from constants(element)

    def visit(node: ast.AST, function: str) -> Iterator[tuple[str, int, str]]:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        found: list[ast.Constant] = []
        if isinstance(node, ast.Compare):
            for operand in (node.left, *node.comparators):
                found += constants(operand)
        elif isinstance(node, ast.Dict):
            found += [key for key in node.keys if isinstance(key, ast.Constant)]
        elif isinstance(node, ast.Subscript):
            found += constants(node.slice)
        elif isinstance(node, ast.MatchValue):
            found += constants(node.value)
        for constant in found:
            if constant.value in names:
                yield function, constant.lineno, str(constant.value)
        for child in ast.iter_child_nodes(node):
            yield from visit(child, function)

    yield from visit(tree, "<module>")


@pytest.mark.usefixtures("installed")
def test_no_kind_handling_module_dispatches_on_a_registered_kinds_name() -> None:
    names = frozenset(kinds.registered())
    assert {*BUILT_INS, KIND} <= names

    spelled = [
        f"{module}:{line} {word} (in {function})"
        for module in KIND_HANDLING
        for function, line, word in _spelled(
            ast.parse((REPO_ROOT / module).read_text("utf-8")), names
        )
        if (module, function, word) not in NOT_A_KIND
    ]

    assert spelled == []


def test_the_dispatch_guard_sees_a_kind_compared_keyed_or_matched() -> None:
    source = (
        "def f(kind, table):\n"
        "    if kind == 'orm':\n"
        "        return {'trm': 1}\n"
        "    match kind:\n"
        "        case 'crm':\n"
        "            return table['dispersion']\n"
    )

    found = list(_spelled(ast.parse(source), frozenset(BUILT_INS)))

    assert [(line, word) for _function, line, word in found] == [
        (2, "orm"),
        (3, "trm"),
        (5, "crm"),
        (6, "dispersion"),
    ]


def test_the_deployment_package_imports_osprey_only_through_the_kinds_api() -> None:
    from osprey.apps.boundary import check_package

    assert check_package(PACKAGE_DIR) == []
