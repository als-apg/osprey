"""The facility-seam regression gate.

The seam is one sentence: **no accelerator-physics imports on the no-lattice
boot path.** ``VA_LATTICE=none`` describes a facility whose channels a
manifest lists and whose physics this process does not have, and the promise
that makes that mode worth having is that such a facility can boot the
virtual accelerator on a host where the physics stack is not installed at
all. So the guard is drawn around the physics, and only the physics: ``at``
(PyAT itself) and ``lume_pyat`` (the ring model built on it). ``lume_pyat``
has to be named separately because its ``__init__`` is PEP 562 lazy -- an
accidental import costs almost nothing and would not announce itself by
dragging anything heavy in.

``lume`` and ``h5py`` are **deliberately not blocked**, and that is a considered
position rather than a relaxation. The serving layer is built around a
:class:`~lume.model.LUMEModel`: the runner derives its PVA namespace and the
shape of its run loop from one, and the no-lattice boot serves the empty
:class:`~osprey.services.virtual_accelerator.serving.model_stub.NullModel`
rather than no model at all. ``lume`` is therefore a dependency of the
*serving* layer, not of the physics behind it, and blocking it would have
this gate assert something the design does not claim. What the seam
protects is what this boot path is allowed to *depend on* -- not what its
dependencies cost.

It does cost, and the cost is worth naming here rather than discovering
later. Timing a ``VA_LATTICE=none`` boot from process spawn to the ``Loading
simulation engine`` line -- the last milestone both the pre-serving and the
serving entrypoint print, so the interval spans exactly the assembly this
seam governs and none of the transport behind it -- on darwin/arm64,
CPython 3.13, eleven paired runs interleaved between the two trees with one
discarded warm-up each, median of each arm:

    pre-serving entrypoint (no ``lume`` imported)    0.518 s
    serving entrypoint (``lume`` for NullModel)      1.216 s
    delta                                           +0.698 s

Nearly all of it is the single import: ``import lume.model`` alone costs
+0.916 s over a bare interpreter on the same host (median of seven), and the
boot pays less than that only because it shares the numpy that import needs
with code it was already loading. Two thirds of a second is a real number on
a boot whose remaining work is standing up a Channel Access server, and it
is not a regression to chase: it is what serving a LUMEModel costs, and this
mode serves one by design.

Two halves, matching the seam's two promises:

* **The tutorial machine is still reachable, by name.** There is no longer
  an unconfigured path to it: a boot that names no channel source is refused
  rather than served the framework's bundled demo namespace. The demo is a
  committed manifest like any other, and asking for it -- the packaged file,
  plus ``VA_LATTICE`` naming the lattice its own tree carries, which is what
  ``scripts/va/run_va.sh`` does -- resolves to that namespace and that PyAT
  lattice. Both halves are files named against the served tree, so there is
  no spelling of either that means "whatever this installation bundles".
  (The deep guarantee is the rest of ``tests/va``, which runs against the
  generated manifest directly, so this half only pins the resolution.)

* **A file-backed facility boots without PyAT.** A manifest of three-part
  addresses (identity carried in the hierarchy keys, not the address text)
  boots the *real* ``entrypoint.main()`` in a subprocess whose import
  machinery makes any ``import at`` fatal. A change that sneaks a physics
  dependency onto the no-lattice path turns that boot into a crash, and this
  test red.

The boot runs in a subprocess of its own because ``main()`` is a process
entry point in the literal sense: it installs SIGINT/SIGTERM handlers, binds
a Channel Access port and then blocks in the runner's loop until signalled.
None of that belongs in pytest's process.

**How far the boot gets depends on the host.** ``serving.runner`` imports the
Channel Access server extension, and a host without it cannot reach the
serving announcement no matter how clean the seam is. So the assertions are
split by what each proves: the seam itself and the whole no-lattice assembly
ahead of the transport are checked unconditionally, and the terminal outcome
is required to be either the serving announcement or a missing *server*
module -- never a physics one, and never anything else. On a host with the
extension this test proves the boot end to end; on one without it, it still
proves everything the seam is about.

Both outcomes have been exercised, so neither is a branch written on faith.
A host carrying none of the server extensions stops at the runner import,
which is the second outcome. In a linux/amd64 container carrying pcaspy,
p4p and the PVA serving package, the same ``main()`` runs to ``virtual
accelerator IOC serving PVs: 4 channels`` with the physics blocked -- so the
seam is known to hold across a complete boot, not merely across the part of
one a development host can reach.

A second, static gate rides in this file, because it guards the same seam from
the other side. The boot test above proves the no-lattice path depends on no
physics; :class:`TestTheServiceNamesNoFacility` proves the whole service names
no facility -- no family, no address token, no constant of one particular
ring, and no reach into the bundled tree from a module that is serving. The
two together are what makes "a manifest and a data directory are the whole
input" a property rather than an intention: one holds at run time on the path
a facility actually boots, the other holds over every line, including the ones
no test on a development host reaches.
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import tokenize
from pathlib import Path

import pytest

from osprey.interfaces._serving import free_port

# Every root the no-lattice path must stay clear of: PyAT, and the ring model
# built on it. See the module docstring for why ``lume`` and ``h5py`` are not
# here, and why ``lume_pyat`` has to be named rather than left to arrive with
# the heavy stack.
_BLOCKED_ROOTS = ("at", "lume_pyat")

# The marker the subprocess raises with, and the string the test greps its
# output for. Distinctive enough that no library's own ImportError can be
# mistaken for it.
_VIOLATION = "SEAM VIOLATION"

# Roots whose absence stops the boot at the transport rather than at the
# seam. ``serving.runner`` imports the Channel Access server extension at
# module scope, so a host without these gets as far as the runner import and
# no further -- a legitimate outcome for this test, and the only failure
# other than a full boot it accepts.
_SERVER_EXTENSION_ROOTS = ("pcaspy", "p4p", "lume_pva_apg")


def _run_seam_ioc_subprocess() -> None:
    """Subprocess entry point: make the physics unimportable, then boot main().

    The blocker sits at the front of ``sys.meta_path``, so even an installed
    PyAT cannot be imported in this process -- equivalent to (and stricter
    than) running on a machine without it.
    """
    import importlib.abc

    class ATBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002 - the importlib.abc.MetaPathFinder signature
            root = fullname.split(".")[0]
            if root in _BLOCKED_ROOTS:
                raise ImportError(
                    f"{_VIOLATION}: {root} imported on the no-lattice path (via {fullname!r})"
                )
            return None

    sys.meta_path.insert(0, ATBlocker())

    from osprey.services.virtual_accelerator import entrypoint

    entrypoint.main()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--run-seam-ioc-subprocess":
    _run_seam_ioc_subprocess()
    sys.exit(0)


import osprey.services.virtual_accelerator as virtual_accelerator  # noqa: E402
from osprey.services.virtual_accelerator import entrypoint  # noqa: E402
from osprey.services.virtual_accelerator.manifest import (  # noqa: E402
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_LONG_STRING,
)
from osprey.services.virtual_accelerator.manifest.classify import (  # noqa: E402
    _BOOLEAN_FIELDS,
    _BOOLEAN_SUBFIELDS,
    READBACK_SUBFIELD,
    SETPOINT_SUBFIELD,
)
from osprey.services.virtual_accelerator.manifest.paths import (  # noqa: E402
    MANIFEST_OUTPUT,
    PACKAGE_PATHS,
)
from osprey.simulation.facility_spec import ALS_U_AR  # noqa: E402

# A three-part-address facility: the address text carries no six-level
# grammar; identity (SP<->RB pairing) rides entirely in the hierarchy keys.
_SEAM_CHANNELS = [
    {
        "address": "ZZSEAM:JET:PRESSURE",
        "ring": "ZZSEAM",
        "system": "JET",
        "family": "TARGET",
        "device": "01",
        "field": "PRESSURE",
        "subfield": "RB",
        "partition": PARTITION_STATIC_NOISY,
        "record_type": RECORD_TYPE_ANALOG,
        "noise": True,
    },
    {
        "address": "ZZSEAM:STAGE:POS:SP",
        "ring": "ZZSEAM",
        "system": "STAGE",
        "family": "MOTOR",
        "device": "01",
        "field": "POS",
        "subfield": "SP",
        "partition": PARTITION_SP_ECHO,
        "record_type": RECORD_TYPE_ANALOG,
        "noise": False,
    },
    {
        "address": "ZZSEAM:STAGE:POS",
        "ring": "ZZSEAM",
        "system": "STAGE",
        "family": "MOTOR",
        "device": "01",
        "field": "POS",
        "subfield": "RB",
        "partition": PARTITION_SP_ECHO,
        "record_type": RECORD_TYPE_ANALOG,
        "noise": False,
    },
    {
        "address": "ZZSEAM:STATUS:MSG",
        "ring": "ZZSEAM",
        "system": "STATUS",
        "family": "MSG",
        "device": "01",
        "field": "TEXT",
        "subfield": "RB",
        "partition": PARTITION_STATIC_NOISY,
        "record_type": RECORD_TYPE_LONG_STRING,
        "noise": False,
    },
]


class TestTheTutorialMachineIsNamedNotAssumed:
    """Built-in half: the bundled namespace and lattice, asked for rather than
    fallen into."""

    def test_no_channel_source_is_refused(self, monkeypatch, tmp_path):
        # The path that used to lead here silently. A boot told nothing has
        # no facility to serve, and the bundled demo namespace is not an
        # answer to that question -- it is one particular facility's
        # addresses wearing whatever name the deployment gave the container.
        monkeypatch.delenv("VA_CHANNELS_FILE", raising=False)
        with pytest.raises(SystemExit, match="VA_CHANNELS_FILE"):
            entrypoint._resolve_channels_file(tmp_path)

    def test_naming_the_packaged_manifest_gives_the_pre_seam_path(self, monkeypatch, tmp_path):
        # Both halves of the demo are asked for the same way, and both are
        # files: the namespace is the committed manifest, named outright, and
        # the physics behind it is the lattice the served tree carries, named
        # relative to that tree. Neither is a mode the service can be in --
        # there is no spelling of either that means "whatever this
        # installation happens to bundle".
        monkeypatch.setenv("VA_CHANNELS_FILE", str(MANIFEST_OUTPUT))
        assert entrypoint._resolve_channels_file(tmp_path) == MANIFEST_OUTPUT

        lattice = tmp_path / PACKAGE_PATHS.lattice_json.name
        lattice.write_text("{}")
        monkeypatch.setenv("VA_LATTICE", lattice.name)
        assert entrypoint._resolve_lattice(tmp_path) == lattice

    def test_no_lattice_named_is_no_physics_rather_than_the_bundled_one(
        self, monkeypatch, tmp_path
    ):
        # The other side of the same rule. A boot that names no lattice gets
        # none -- the demo ring is not the answer to an unanswered question --
        # and that is what makes the no-physics path reachable on a host
        # carrying no physics at all.
        monkeypatch.setenv("VA_LATTICE", entrypoint.LATTICE_NONE)
        assert entrypoint._resolve_lattice(tmp_path) is None

        monkeypatch.delenv("VA_LATTICE", raising=False)
        assert entrypoint._resolve_lattice(tmp_path) is None


class TestSetpointEchoEngineSync:
    """No-lattice physics coupling: sp-echo readback values are synced into
    the engine each tick, so a machine-file expression channel (the camera
    response) follows the latest accepted setpoint."""

    class _FakeRecord:
        def __init__(self, value: float) -> None:
            self._value = value

        def get(self) -> float:
            return self._value

        def set(self, value: float) -> None:
            self._value = value

    def test_expression_channel_follows_synced_setpoint(self, tmp_path: Path):
        from osprey.services.virtual_accelerator.ioc.engine_source import EngineSource
        from osprey.simulation.engine import SimulationEngine

        machine = tmp_path / "machine.json"
        machine.write_text(
            json.dumps(
                {
                    "name": "sync-test",
                    "description": "engine-sync unit machine",
                    "channels": {
                        "ZZSYNC:STAGE:POS": {"value": 0.0, "noise": 0},
                        "ZZSYNC:CAM:INTENSITY": {
                            "expr": "1000 * exp(-((ch('ZZSYNC:STAGE:POS') - 2.0) ** 2))",
                            "noise": 0,
                        },
                    },
                }
            )
        )
        engine = SimulationEngine.from_file(machine)
        stage_rb = self._FakeRecord(0.0)
        intensity = self._FakeRecord(0.0)
        channels = [
            {
                "address": "ZZSYNC:CAM:INTENSITY",
                "partition": PARTITION_STATIC_NOISY,
                "record_type": RECORD_TYPE_ANALOG,
                "noise": False,
            }
        ]
        source = EngineSource(
            engine,
            channels,
            {"ZZSYNC:CAM:INTENSITY": intensity},
            tmp_path,
            setpoint_echo_records={
                "ZZSYNC:STAGE:POS": stage_rb,
                "ZZSYNC:NOT:SERVED": self._FakeRecord(1.0),  # dropped, engine-unknown
            },
        )

        source.poll_once()
        off_peak = intensity.get()

        stage_rb.set(2.0)  # the accepted setpoint echo moves the readback...
        source.poll_once()
        on_peak = intensity.get()

        # ...and the expression channel responds: on-peak beats off-peak by
        # the Gaussian's e^{-4} contrast.
        assert on_peak == pytest.approx(1000.0)
        assert off_peak == pytest.approx(1000.0 * 0.0183156, rel=1e-3)

    def test_without_sync_map_behaviour_is_unchanged(self, tmp_path: Path):
        from osprey.services.virtual_accelerator.ioc.engine_source import EngineSource
        from osprey.simulation.engine import SimulationEngine

        machine = tmp_path / "machine.json"
        machine.write_text(
            json.dumps(
                {
                    "name": "sync-default-test",
                    "description": "no sync map -> pure scenario source",
                    "channels": {"ZZSYNC2:TEMP:RB": {"value": 21.5, "noise": 0}},
                }
            )
        )
        engine = SimulationEngine.from_file(machine)
        temp = self._FakeRecord(0.0)
        channels = [
            {
                "address": "ZZSYNC2:TEMP:RB",
                "partition": PARTITION_STATIC_NOISY,
                "record_type": RECORD_TYPE_ANALOG,
                "noise": False,
            }
        ]
        EngineSource(engine, channels, {"ZZSYNC2:TEMP:RB": temp}, tmp_path).poll_once()
        assert temp.get() == pytest.approx(21.5)


class TestFileBackedBootWithoutPyat:
    """Facility half: real main() boot, file manifest, PyAT import fatal."""

    # Assembly milestones the no-lattice boot must pass on its way to the
    # transport. Each is printed by ``main()`` itself, and together they
    # cover every decision the seam is about: the lattice branch not taken,
    # the manifest turned into a serving database, the scenario engine
    # loaded. Reaching all three with the physics unimportable IS the seam
    # holding, whether or not the host can stand a server up afterwards.
    MILESTONES = (
        "PhysicsBridge skipped",
        f"Built serving database: {len(_SEAM_CHANNELS)} channels",
        "Loading simulation engine from",
    )

    @pytest.fixture()
    def seam_data_dir(self, tmp_path: Path) -> Path:
        (tmp_path / "channels_manifest.json").write_text(json.dumps({"channels": _SEAM_CHANNELS}))
        (tmp_path / "machine.json").write_text(
            json.dumps(
                {
                    "name": "seam-test facility",
                    "description": "minimal machine for the facility-seam boot test",
                    "channels": {
                        "ZZSEAM:JET:PRESSURE": {
                            "label": "jet backing pressure",
                            "value": 30.0,
                        }
                    },
                }
            )
        )
        return tmp_path

    def test_boots_serves_and_never_imports_pyat(self, seam_data_dir: Path):
        # Inherit the ambient environment MINUS the whole VA_ family, so this
        # boot is configured by exactly what is set below. An inherited
        # VA_BPM_ERRORS or VA_CORR_GAIN is a lattice-physics fault, and the
        # entrypoint refuses to serve one unless VA_LATTICE='builtin' -- which
        # a file-backed boot deliberately never sets.
        env = {k: v for k, v in os.environ.items() if not k.startswith("VA_")}
        env.update(
            VA_DATA_DIR=str(seam_data_dir),
            VA_CHANNELS_FILE="channels_manifest.json",
            # VA_LATTICE deliberately unset: file-backed default must be "none".
            EPICS_CA_ADDR_LIST="127.0.0.1",
            EPICS_CA_AUTO_ADDR_LIST="NO",
            EPICS_CA_SERVER_PORT=str(free_port()),
            EPICS_CA_REPEATER_PORT=str(free_port()),
        )

        served, output, returncode = self._boot(env)

        # The seam, and the only assertion here that is about the seam: with
        # the physics unimportable, nothing on this path reached for it. This
        # holds whether the boot completed or stopped at the transport, which
        # is why it is checked before anything else is.
        assert _VIOLATION not in output, f"the no-lattice boot imported physics:\n{output}"

        # ...and it got far enough for that to mean something. A process that
        # died before the lattice branch would satisfy the check above
        # vacuously.
        for milestone in self.MILESTONES:
            assert milestone in output, f"boot never reached {milestone!r}:\n{output}"

        if served:
            # A host with the Channel Access server extension: the boot ran to
            # completion and is serving the manifest whole, four channels and
            # nothing else.
            assert entrypoint._ready_line(len(_SEAM_CHANNELS)) in output
            assert returncode is None, f"the IOC exited instead of serving:\n{output}"
            return

        # A host without it. The boot is required to have stopped for exactly
        # that reason -- a missing *server* module -- and for no other. Any
        # other exception here is a real failure of the no-lattice path,
        # including a physics import that somehow evaded the blocker.
        assert returncode is not None, (
            f"the boot neither served nor exited within the deadline -- it hung:\n{output}"
        )
        missing = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", output)
        assert missing, (
            "the boot stopped short of serving without a missing-module error, "
            f"so something other than the absent server extension ended it:\n{output}"
        )
        root = missing.group(1).split(".")[0]
        assert root in _SERVER_EXTENSION_ROOTS, (
            f"the no-lattice boot stopped on a missing {root!r}, which is not one of the "
            f"server extensions {list(_SERVER_EXTENSION_ROOTS)} this host is allowed to lack"
        )

    def _boot(self, env: dict[str, str]) -> tuple[bool, str, int | None]:
        """Boot ``main()`` in a subprocess and drain it.

        Returns whether the serving announcement was reached, everything the
        process wrote (stderr merged in, so a traceback is part of the
        record), and its exit status -- ``None`` while it is still running,
        which is what a boot that reached the announcement looks like.

        Draining runs to the announcement or to EOF, never on a timer alone:
        the announcement means success and EOF means the process ended, and
        the deadline exists only so a boot that hangs without doing either
        fails this test rather than pytest's whole run.
        """
        proc = subprocess.Popen(
            [sys.executable, __file__, "--run-seam-ioc-subprocess"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            lines: list[str] = []
            deadline = time.monotonic() + 60
            served = False
            while time.monotonic() < deadline:
                line = proc.stdout.readline()
                if not line:
                    break  # process exited; its status says why
                lines.append(line)
                if entrypoint.READY_MARKER in line:
                    served = True
                    break
            if not served:
                # stdout hit EOF, or the deadline expired. Wait for the exit
                # status rather than sampling it: the pipe closes a moment
                # before the process is reapable, so ``poll()`` on its own
                # would report a crashed boot as still running. A timeout
                # here means it really is still running and silent -- a hang,
                # which the caller reports as one.
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            return served, "".join(lines), proc.poll()
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


# The service's own source tree, discovered from the installed package rather
# than climbed with a fixed number of ``__file__`` parents, so this gate reads
# the same files from an editable checkout and from a wheel.
_VA_ROOT = Path(virtual_accelerator.__file__).resolve().parent

# The scripts that derive a served tree's own data and drive its model. They
# ship beside the service rather than inside it, and the same sentence holds
# for them: each is pointed at a tree and works from what that tree declares,
# so a family name here would be one ring's answer baked into the tooling
# every other ring has to use. They live in the checkout, not in the package.
_SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "va"

# The families the bundled demo facility declares, read from the spec that
# declares them: a family added there joins this gate without anyone
# remembering to add it here.
_FAMILY_TOKENS = frozenset(ALS_U_AR.family_names())

# The address vocabulary that facility's namespace is spelled in.
_ADDRESS_TOKENS = frozenset({"SR", "MAG", "DIAG", "CURRENT", "POSITION"})

# The constant naming the facility itself, and the corrector scale its demo
# lattice was calibrated with.
_FACILITY_CONSTANTS = frozenset({"ALS_U_AR", "AMPS_PER_RADIAN_KICK"})

_FACILITY_TOKENS = _FAMILY_TOKENS | _ADDRESS_TOKENS | _FACILITY_CONSTANTS

# Whole words only. A token is a facility's name when the code spells it as a
# name; it is not one when it happens to sit inside a longer identifier, which
# is why ``VA_BPM_ERRORS`` -- the seeded-readout-error grammar, which every
# facility's monitors go through -- is not a hit.
_FACILITY_TOKEN_RE = re.compile(r"\b(" + "|".join(sorted(_FACILITY_TOKENS)) + r")\b")

# The modules that run while the service is serving. ``manifest/`` is
# deliberately outside: it holds the build-time generator, whose job is to read
# the bundled tree.
_SERVING_PATH = ("entrypoint.py", "ioc", "lattice", "model", "serving")


def _executable_lines(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, text)`` for the file's code, with its prose removed.

    Docstring lines are dropped and every comment is cut at its ``#``. What is
    left is what the interpreter acts on -- identifiers, and the string
    literals the code builds addresses, predicates and messages out of. Those
    literals stay in scope on purpose: a hardcoded family name arrives as one
    far more often than as an identifier.
    """
    source = path.read_text()

    prose_lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if ast.get_docstring(node, clean=False) is None:
            continue
        literal = node.body[0]
        prose_lines.update(range(literal.lineno, literal.end_lineno + 1))

    # Taken from the tokenizer rather than by looking for ``#``, so a hash
    # inside a string literal is not mistaken for the start of a comment.
    comment_column: dict[int, int] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comment_column.setdefault(token.start[0], token.start[1])

    lines: list[tuple[int, str]] = []
    for number, text in enumerate(source.splitlines(), start=1):
        if number in prose_lines:
            continue
        cut = comment_column.get(number)
        lines.append((number, text if cut is None else text[:cut]))
    return lines


def _shown(path: Path) -> str:
    """The file, named the way a reader of the failure would go looking for it."""
    root = _VA_ROOT if path.is_relative_to(_VA_ROOT) else _SCRIPT_ROOT.parents[1]
    return str(path.relative_to(root))


def _scan(paths: list[Path], pattern: re.Pattern[str]) -> list[str]:
    """Every match of ``pattern`` in the code of ``paths``, as reportable lines."""
    hits: list[str] = []
    for path in paths:
        for number, text in _executable_lines(path):
            for match in pattern.finditer(text):
                hits.append(f"  {_shown(path)}:{number}  {match.group(0)}  ->  {text.strip()}")
    return hits


def _script_modules() -> list[Path]:
    """The Python scripts under the gate, compiled leftovers of old runs out."""
    return sorted(path for path in _SCRIPT_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _service_modules() -> list[Path]:
    return sorted(_VA_ROOT.rglob("*.py")) + _script_modules()


def _serving_path_modules() -> list[Path]:
    modules: list[Path] = []
    for name in _SERVING_PATH:
        target = _VA_ROOT / name
        modules.extend(sorted(target.rglob("*.py")) if target.is_dir() else [target])
    return modules


class TestTheServiceNamesNoFacility:
    """The service holds no facility's names.

    Everything the virtual accelerator serves arrives as data -- a channel
    manifest, a bindings document, a lattice, the tree's own write bands -- so
    nothing inside it needs to know what a family is called or how an address
    is spelled. This gate holds that line against the one change that always
    looks harmless: a single token dropped into a predicate or a format string
    because the facility at hand happens to spell it that way. One is enough
    to make the service work on one ring and quietly misbehave on the next,
    and it survives review precisely because it reads like domain knowledge.

    The scan covers the Python scripts under ``scripts/va`` as well as the
    service. They are handed a tree the same way the service is -- one derives
    that tree's write bands from its own ring, another drives its model through
    the LUME interface -- so a family name in either is the same mistake made
    one directory further out, where no deployment would meet it and every
    other facility's operator would. The shell scripts beside them are not
    read: a token scan of Python source is what this gate knows how to do, and
    claiming the directory would promise more than it reads.

    The token list is not a curated denylist. The family names come from the
    spec that declares them, so the gate widens when that facility does.
    Beside them sit the address vocabulary its namespace is spelled in, and
    the two constants that name the facility and the corrector scale its demo
    lattice was calibrated with.

    **Grammar is not a facility.** ``SP``, ``RB``, and the boolean field and
    subfield tokens the hierarchical record-type rule matches on, belong to
    the paradigm rather than to any facility: every tree spelled in that
    paradigm carries them, and a facility that does not spell its channels
    that way is simply not in that paradigm. They are not carved out of the
    scan -- carve-outs rot, and each one is a place a real token can be
    parked -- they are held to be outside the token list, which is checked
    here.

    **What the code does, not what it says about itself.** Docstrings and
    comments are stripped before the scan; string literals are not. A token
    the interpreter acts on is a constant, while the same word in prose is the
    field's vocabulary being used to explain a rule that holds for every
    facility -- the physics bridge's own docstring says a facility whose
    monitors are not called ``BPM`` needs no special case, and a gate that
    forbade that sentence would forbid the code from documenting the property
    this gate exists to protect. The tree's committed JSON is out of scope for
    the mirror-image reason: it is one facility's data, and data is supposed
    to name a facility.
    """

    def test_the_exempt_grammar_is_not_a_facility_token(self):
        grammar = (
            {SETPOINT_SUBFIELD, READBACK_SUBFIELD} | set(_BOOLEAN_SUBFIELDS) | set(_BOOLEAN_FIELDS)
        )
        collision = grammar & _FACILITY_TOKENS
        assert not collision, (
            "a token is claimed by both the paradigm's grammar and the demo "
            f"facility's vocabulary: {sorted(collision)}. One of the two has to "
            "give, because a gate cannot both forbid a token and rely on it."
        )

    def test_the_scripts_beside_the_service_are_in_the_scan(self):
        """The scripts are read from the checkout, so a run that cannot see
        them would pass this gate by scanning nothing at all."""
        assert _script_modules(), f"no script was scanned; none was found under {_SCRIPT_ROOT}"

    def test_no_facility_token_reaches_the_service_code(self):
        hits = _scan(_service_modules(), _FACILITY_TOKEN_RE)
        assert not hits, (
            "the virtual accelerator names a facility. Every family, address "
            "token and constant below belongs to one particular ring, and the "
            "service is handed all three as data:\n" + "\n".join(hits)
        )

    def test_the_serving_path_reads_no_packaged_tree(self):
        hits = _scan(_serving_path_modules(), re.compile(r"\bPACKAGE_PATHS\b"))
        assert not hits, (
            "a serving-path module reaches for the bundled tree. PACKAGE_PATHS "
            "anchors the demo data the build-time generator reads; a process "
            "that is serving was handed a data directory, and every file it "
            "needs comes from that one tree. A reference here is a second "
            "source, silently the demo's, on a process serving a facility:\n" + "\n".join(hits)
        )
