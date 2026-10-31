"""The file contracts a released framework and this connectors tree share.

The import scanner proves a framework can still load the names it takes from
the connectors. These checks prove the two still agree about the files that
pass between them, which no import catches:

* **The simulator view.** A framework's ``osprey build`` writes
  ``build/data/simulator/``; the connectors' :class:`SimulatorView` is its one
  reader, inside the deployment the build rendered. The check renders a real
  deployment with the installed framework's own CLI (``osprey init`` from a
  bundled preset, then ``osprey build --skip-lifecycle --skip-deps``) and
  opens every document of it through the installed connectors. Driving the
  public CLI rather than the framework's view writer keeps the check
  independent of internals an older release spelled differently, and the
  view it reads is the one an operator's rebuild would produce. A release
  that ships no view writer (``facility/views/simulator.py``; the 2026.9.0
  betas predate the view) has no view to check, and the check says so
  instead of rendering -- unless ``--require-view`` is given, which the lane
  passes for every release that must carry the view, so a writer that went
  missing there is a break, not a skip.
* **The control-context record.** The connectors write the record and the
  per-server reports; a deployment's hooks, rendered from the framework's
  templates, restate the record parser in the standard library. The check
  writes a record with the installed connectors and parses it with the
  installed framework's ``osprey_target_state`` hook, and parses the golden
  files under ``fixtures/control_context/`` with the installed connectors. The
  golden files were written by the connectors' own writer and are never
  regenerated: they stand for the records a deployment already holds in its
  durable ``var/`` when its rebuild picks up a newer connectors. A newer
  writer re-writes such a record in schema 1's normal form
  (:func:`check_golden_rewrite`), so the hook is held to reading that
  re-write as the same record, not as the same bytes. A second golden record
  (``switch_on/``) arms a machine with the switch's legacy ``"on"``: both the
  package and the hook must read it as armed (``"asked"`` where the hook knows
  the three positions), and ``set_switch(..., "on")`` -- the word a framework
  released before the positions sends -- must store ``"asked"``, which the
  hook must read as armed too.

Usage::

    python -I contracts.py all --workdir <scratch dir> [--require-view]
    python -I contracts.py view <render root>
    python -I contracts.py control-context [--hook <osprey_target_state.py>]

``all`` locates the installed framework (``osprey``) and its CLI beside the
running interpreter. Each check prints one line on success and raises
:class:`ContractError` on a break; the CLI exits 1 on any. Standard library
and ``osprey_connectors`` only, for the reason the package's ``__init__``
gives.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

#: The golden control-context files, written once by the connectors' writer.
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "control_context"

#: The preset the view check renders. Bundled by every framework release that
#: depends on this distribution, and it serves physics models besides the
#: texture, so every document of the view carries entries.
PRESET = "control-assistant"

#: Where the framework ships the simulator view's writer, when it ships one.
VIEW_WRITER_RELPATH = Path("facility/views/simulator.py")

#: Where the framework ships the hook that restates the record parser.
HOOK_RELPATH = Path("templates/claude_code/claude/hooks/osprey_target_state.py")

#: What the golden record states, as both parsers must read it.
GOLDEN_RECORD = {"target": "va", "generation": 3, "posture": {"live": "sandbox"}}

#: The golden record that arms a machine with the switch's legacy word.
SWITCH_ON_FIXTURES = FIXTURES / "switch_on"

#: The machine that record arms, and the one the legacy gesture arms beside it.
SWITCH_ON_TARGET = "va"
LEGACY_GESTURE_TARGET = "live"

#: The keys schema 1's first write adds to a record written before them.
NORMAL_FORM_KEYS = frozenset({"switch", "switch_epoch"})


class ContractError(AssertionError):
    """A file one side writes that the other side no longer reads."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


# -- the simulator view ------------------------------------------------------


def render_deployment(workdir: Path, osprey: Path, preset: str = PRESET) -> Path:
    """Render a deployment from ``preset`` with the ``osprey`` CLI given.

    Returns:
        The render root, ``<repo>/build``.
    """
    repo = workdir / "compat-deployment"
    env = {**os.environ, "NO_COLOR": "1"}
    for argv in (
        [str(osprey), "init", str(repo), "--preset", preset, "--no-git"],
        [str(osprey), "build", "--skip-lifecycle", "--skip-deps", "--repo", str(repo)],
    ):
        result = subprocess.run(
            argv,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ContractError(
                f"{' '.join(argv[1:3])} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
            )
    return repo / "build"


def check_simulator_view(render_root: Path) -> str:
    """Open the view under ``render_root`` and read every document of it.

    Returns:
        A one-line summary.

    Raises:
        ContractError: The view is absent, or any read of it raises.
    """
    try:
        return _read_simulator_view(render_root)
    except ContractError:
        raise
    except Exception as error:  # every refusal of the reader is the break this reports
        raise ContractError(
            f"simulator view under {render_root}: {error.__class__.__name__}: {error}"
        ) from error


def _read_simulator_view(render_root: Path) -> str:
    from osprey_connectors.simulation.view import SimulatorView

    view = SimulatorView.of_render(render_root)
    _require(bool(view.code), "the view states no facility code")
    models = view.models()
    _require(bool(models), "the view lists no model")
    for name in view.served():
        view.model(name)
    physics = view.physics_models()
    channels = view.channels()
    _require(bool(channels), "the view lists no channel")
    bound = 0
    for address in channels:
        view.channel(address)
        view.seed(address)
        if view.binding(address) is not None:
            bound += 1
    _require(not physics or bound > 0, "physics models are served but no channel is bound to one")
    list(view.bindings())
    dict(view.status_addresses())
    scenarios = view.scenarios()
    for scenario in scenarios:
        view.scenario_dir(str(scenario["name"]))
    return (
        f"simulator view OK: {len(models)} models ({len(physics)} physics), "
        f"{len(channels)} channels ({bound} bound), {len(scenarios)} scenarios"
    )


# -- the control-context record ---------------------------------------------


def load_hook(path: Path) -> ModuleType:
    """Import a framework's ``osprey_target_state`` hook from its template file.

    The hooks import their siblings as top-level modules, as they do inside a
    rendered deployment, so the template directory is on ``sys.path`` while it
    loads. The siblings it loaded are dropped again afterwards, so a later
    import of a hook by the same name reads its own directory.
    """
    _require(path.is_file(), f"no record-parsing hook at {path}")
    before = set(sys.modules)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("osprey_target_state", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
        for name in set(sys.modules) - before:
            if (
                getattr(sys.modules[name], "__file__", None)
                and Path(sys.modules[name].__file__ or "").parent == path.parent
            ):
                del sys.modules[name]
    return module


def check_control_context_golden(fixtures: Path = FIXTURES) -> str:
    """The golden record and report still parse to what they state."""
    from osprey_connectors import control_context

    raw = json.loads((fixtures / control_context.RECORD_FILENAME).read_text(encoding="utf-8"))
    record = control_context.parse_record(raw)
    _require(record is not None, "the golden record no longer parses")
    assert record is not None
    stated = {
        "target": record.target,
        "generation": record.generation,
        "posture": dict(record.posture),
    }
    _require(stated == GOLDEN_RECORD, f"the golden record reads as {stated}")
    _require(
        record.owner is not None and record.owner.kind == control_context.OWNER_WEB_TERMINAL,
        f"the golden record's owner reads as {record.owner}",
    )
    _require(
        record.last_switch is not None and record.last_switch.get("request_id") == "req-7",
        f"the golden record's last switch reads as {record.last_switch}",
    )

    reports = sorted(fixtures.glob(control_context.REPORT_FILE_GLOB))
    _require(bool(reports), f"no golden report under {fixtures}")
    for path in reports:
        report = control_context.parse_report(json.loads(path.read_text(encoding="utf-8")))
        _require(report is not None, f"the golden report {path.name} no longer parses")
        assert report is not None
        _require(
            (report.applied_target, report.applied_generation) == ("va", 3),
            f"{path.name} reads as applied to {report.applied_target} at {report.applied_generation}",
        )
    check_switch_golden(fixtures / SWITCH_ON_FIXTURES.name)
    return f"control-context golden OK: 2 records, {len(reports)} report(s)"


def check_switch_golden(fixtures: Path = SWITCH_ON_FIXTURES) -> None:
    """The golden record spelling ``"on"`` reads as one machine at ``asked``."""
    from osprey_connectors import control_context
    from osprey_connectors.types import CONTROL_TARGETS

    raw = json.loads((fixtures / control_context.RECORD_FILENAME).read_text(encoding="utf-8"))
    record = control_context.parse_record(raw)
    _require(record is not None, "the golden switch record no longer parses")
    positions = {
        target: control_context.switch_position(record, target) for target in CONTROL_TARGETS
    }
    expected = dict.fromkeys(CONTROL_TARGETS, "off") | {SWITCH_ON_TARGET: "asked"}
    _require(positions == expected, f"the golden switch record reads as {positions}")


def check_set_switch_legacy_word(workdir: Path) -> Path:
    """``set_switch(..., "on")`` arms at ``asked`` and stores no ``"on"``.

    Applied to a copy of the golden switch record under *workdir*.

    Returns:
        The written record's path.
    """
    from osprey_connectors import control_context

    source = SWITCH_ON_FIXTURES / control_context.RECORD_FILENAME
    path = workdir / SWITCH_ON_FIXTURES.name / control_context.RECORD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    epoch = json.loads(path.read_text(encoding="utf-8"))["switch_epoch"]
    try:
        control_context.set_switch([LEGACY_GESTURE_TARGET], "on", epoch=epoch, path=path)
    except Exception as error:  # the released chip route's gesture is refused
        raise ContractError(
            f'set_switch(["{LEGACY_GESTURE_TARGET}"], "on") raised '
            f"{error.__class__.__name__}: {error}"
        ) from error
    stored = json.loads(path.read_text(encoding="utf-8"))["switch"]
    expected = {SWITCH_ON_TARGET: "asked", LEGACY_GESTURE_TARGET: "asked"}
    _require(stored == expected, f'set_switch(..., "on") stored the switch as {stored}')
    return path


def defaulted_posture(posture: dict[str, str]) -> dict[str, str]:
    """*posture* with every machine it leaves out stated as ``sandbox``.

    An absent machine already reads as ``sandbox`` to every reader; the normal
    form only writes that default out, for every machine in
    :data:`~osprey_connectors.types.CONTROL_TARGETS`.
    """
    from osprey_connectors import control_context

    return {**control_context.posture_mirror({}), **posture}


def check_golden_rewrite(written: dict, golden: dict) -> None:
    """*written*, the writer's re-write of *golden*, is the same record in normal form.

    Schema 1's normal form is what the first write of a record that carries no
    ``switch_epoch`` stores: the epoch minted, the switch empty (every machine's
    writes off) and the ``posture`` mirror written out for every machine.
    Everything else -- the schema, the target, the generation, the owner and
    the last switch -- is carried unchanged, and nothing else is added.
    """
    from osprey_connectors import control_context

    _require(control_context.parse_record(written) is not None, "the re-write no longer parses")
    _require(
        set(written) == set(golden) | NORMAL_FORM_KEYS,
        f"the re-write carries the keys {sorted(written)}",
    )
    for key in sorted(set(golden) - {"posture"}):
        _require(
            written[key] == golden[key],
            f"the re-write reads {key} as {written[key]!r}, not {golden[key]!r}",
        )
    _require(
        written["posture"] == defaulted_posture(golden["posture"]),
        f"the re-write reads posture as {written['posture']}",
    )
    _require(written["switch"] == {}, f"the re-write reads switch as {written['switch']}")
    epoch = written["switch_epoch"]
    _require(
        isinstance(epoch, int) and not isinstance(epoch, bool) and epoch > 0,
        f"the re-write minted no switch epoch: {epoch!r}",
    )


def check_control_context_hook(hook: Path, workdir: Path) -> str:
    """A record the connectors write, and the golden one, read alike in the hook.

    The connectors re-write the golden record in schema 1's normal form
    (:func:`check_golden_rewrite`); the hook must read that re-write as the
    golden record with its posture defaulted, and the golden as it stands.
    """
    from osprey_connectors import control_context

    module = load_hook(hook)
    golden = (FIXTURES / control_context.RECORD_FILENAME).read_text(encoding="utf-8")
    written = control_context.parse_record(golden)
    assert written is not None
    path = workdir / control_context.RECORD_FILENAME
    control_context.write_record(written, path=path)
    rewrite = path.read_text(encoding="utf-8")
    raw_golden = json.loads(golden)
    check_golden_rewrite(json.loads(rewrite), raw_golden)
    normal = {**GOLDEN_RECORD, "posture": defaulted_posture(raw_golden["posture"])}
    for label, text, expected in (
        ("written", rewrite, normal),
        ("golden", golden, GOLDEN_RECORD),
    ):
        parsed = module.parse_record(text)
        _require(parsed == expected, f"the hook reads the {label} record as {parsed}")
    if not hasattr(module, "parse_switch_record"):
        return f"control-context hook OK: {hook} (it reads no switch)"
    gestured = check_set_switch_legacy_word(workdir)
    for label, text, armed in (
        (
            "golden switch",
            (SWITCH_ON_FIXTURES / control_context.RECORD_FILENAME).read_text(encoding="utf-8"),
            {SWITCH_ON_TARGET},
        ),
        (
            'set_switch(..., "on")',
            gestured.read_text(encoding="utf-8"),
            {SWITCH_ON_TARGET, LEGACY_GESTURE_TARGET},
        ),
    ):
        record = module.parse_switch_record(text)
        _require(record is not None, f"the hook reads no {label} record")
        _require(
            set(record["switch"]) == armed,
            f"the hook reads the {label} record's switch as {record['switch']}",
        )
        if hasattr(module, "switch_position"):
            positions = {target: module.switch_position(record, target) for target in armed}
            _require(
                set(positions.values()) == {"asked"},
                f"the hook reads the {label} record's positions as {positions}",
            )
    return f"control-context hook OK: {hook}"


# -- the CLI -----------------------------------------------------------------


def _framework_dir() -> Path:
    spec = importlib.util.find_spec("osprey")
    _require(
        spec is not None and bool(spec.submodule_search_locations),
        "no installed osprey framework",
    )
    assert spec is not None and spec.submodule_search_locations is not None
    return Path(next(iter(spec.submodule_search_locations)))


def _osprey_cli() -> Path:
    cli = Path(sys.executable).with_name("osprey")
    _require(cli.is_file(), f"no osprey CLI beside {sys.executable}")
    return cli


def check_installed_view(framework: Path, workdir: Path, *, require_view: bool) -> str:
    """Render with the installed CLI and open its view, if the release writes one.

    Raises:
        ContractError: The view is required and the release ships no writer,
            or the view it writes does not open.
    """
    if not (framework / VIEW_WRITER_RELPATH).is_file():
        _require(
            not require_view,
            f"the framework at {framework} ships no simulator view writer "
            f"({VIEW_WRITER_RELPATH}), and this release must write the view",
        )
        return (
            f"simulator view: not applicable for this release (no {VIEW_WRITER_RELPATH} "
            f"under {framework})"
        )
    return check_simulator_view(render_deployment(workdir, _osprey_cli()))


def run_all(workdir: Path, *, require_view: bool = False) -> list[str]:
    """Every check, against the installed framework and connectors."""
    workdir.mkdir(parents=True, exist_ok=True)
    framework = _framework_dir()
    return [
        check_control_context_golden(),
        check_control_context_hook(framework / HOOK_RELPATH, workdir),
        check_installed_view(framework, workdir, require_view=require_view),
    ]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    every = commands.add_parser(
        "all", help="render with the installed framework and run every check"
    )
    every.add_argument("--workdir", type=Path, required=True)
    every.add_argument(
        "--require-view",
        action="store_true",
        help="fail, rather than skip, when the release ships no simulator view writer",
    )
    view = commands.add_parser("view", help="open an existing render's simulator view")
    view.add_argument("render_root", type=Path)
    record = commands.add_parser("control-context", help="the golden files, and the hook if named")
    record.add_argument("--hook", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "all":
            lines = run_all(args.workdir, require_view=args.require_view)
        elif args.command == "view":
            lines = [check_simulator_view(args.render_root)]
        else:
            lines = [check_control_context_golden()]
            with tempfile.TemporaryDirectory() as scratch:
                check_set_switch_legacy_word(Path(scratch))
                lines.append('set_switch "on" OK')
                if args.hook is not None:
                    lines.append(check_control_context_hook(args.hook, Path(scratch)))
    except ContractError as error:
        print(f"CONTRACT BROKEN: {error}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
