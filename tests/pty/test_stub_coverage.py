"""Does the stub runtime answer everything the deploy path asks a runtime?

The pty harness replaces ``docker`` with a script. That is only sound while the
script answers every invocation the code under test makes — and the failure mode
when it does not is the expensive one: a deploy that hangs, or worse, one that
finishes and reports something the harness then asserts against.

So the stub is checked against the source rather than against a wish list. This
module walks :mod:`osprey.deployment` with the ``ast`` module, collects every
runtime argv the package builds, resolves each to a seam (``compose up``,
``image inspect``), and asserts the stub declares it handled — then runs the
stub on each seam and asserts it really is. A new runtime call added anywhere on
the deploy path fails this test on the commit that adds it, in the normal lane,
without a terminal or a container.

Deliberately **not** marked ``pty``: it needs no terminal. It is the pre-check
that makes the pty lane's failures worth reading.

Three things the scanner cannot do, and does not pretend to:

* It reads argv *shapes*, not runtime values. A subcommand that only exists as a
  variable is listed in :data:`DYNAMIC_TOKENS` with the literals it can take,
  and an unlisted one is a failure naming its site — never a silent skip.
* It scans the whole ``deployment`` package, not the ``up`` path alone. That is
  a superset on purpose: reachability analysis would be the thing most likely
  to be wrong, and answering a seam the ``up`` path never reaches costs the stub
  four lines.
* It cannot see what is not a subprocess. ``up`` also opens *network* clients
  (the archiver's mongod, ARIEL's postgres) which no ``docker`` stub can stand
  in for — which is exactly why the exemplar is rendered from a preset whose
  services have none. See ``conftest.py``.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.pty.conftest import StubRuntime, repo_root, stub_runtime_dir
from tests.pty.stub_runtime import _stub_main

#: The package whose runtime conversations must be covered.
DEPLOYMENT_PACKAGE = Path("src/osprey/deployment")

#: Local names that hold a *compose* argv base (``["docker", "compose", …]``),
#: so a subcommand appended to one is a compose seam rather than a plain one.
_COMPOSE_BASE_NAMES = frozenset({"base", "cmd", "web_cmd", "services_base", "runtime_cmd"})

#: Local names that hold the runtime *binary* alone, so the tokens after them
#: are a plain-runtime subcommand.
_RUNTIME_HEAD_NAMES = frozenset({"runtime", "runtime_bin"})

#: Calls that answer the runtime command, so subscripting one is the binary.
#: Half the package resolves the binary into a local first; the other half
#: writes the call inline at the head of the argv it is building, and a scanner
#: that only knew the local spelling would read those sites as no argv at all.
_RUNTIME_CALL_NAMES = frozenset({"get_runtime_command"})

#: Subcommands that are a namespace rather than a verb: the seam is two words.
_NAMESPACES = _stub_main._NAMESPACES

#: Flags that take a value, so the scanner steps over the value too.
_VALUE_FLAGS = _stub_main._COMPOSE_VALUE_FLAGS | _stub_main._PLAIN_VALUE_FLAGS

#: Every subcommand the deploy path spells as a variable, with the literals it
#: can hold, and where. Listed rather than inferred: the point of the gate is
#: that a runtime call cannot be added without someone deciding it is covered,
#: and a scanner that quietly skipped what it could not read would give that up.
DYNAMIC_TOKENS = {
    # container_lifecycle._down_by_label: `for verb in ("stop", "rm")`
    ("container_lifecycle.py", "verb"): ("stop", "rm"),
    # reset.RuntimeProbe._labels_of: one inspect, three kinds of resource.
    ("reset.py", "kind"): ("container", "volume", "image"),
}

#: Seams the scanner must find, or it is not looking at anything. Without this
#: a scanner that silently matched nothing would pass every assertion below.
ANCHOR_SEAMS = frozenset({"compose up", "compose down", "ps", "image inspect", "build"})


# ---------------------------------------------------------------------------
# the scanner
# ---------------------------------------------------------------------------


class _Seam:
    """One runtime invocation the source builds, and where it builds it."""

    def __init__(self, seam: str, module: str, lineno: int) -> None:
        self.seam = seam
        self.module = module
        self.lineno = lineno

    def __repr__(self) -> str:
        return f"{self.seam!r} ({self.module}:{self.lineno})"


def _constants(elements: list[ast.expr]) -> list[str | None]:
    """Each element as its string constant, or ``None`` when it is dynamic."""
    return [
        node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None
        for node in elements
    ]


def _verbs(tokens: list[str | None], module: str, dynamic_names: list[str | None]) -> list[str]:
    """The seam(s) an argv tail resolves to.

    Args:
        tokens: The argv tail as constants, ``None`` for anything dynamic.
        module: File name, for the dynamic-token lookup and for error text.
        dynamic_names: The identifier behind each dynamic token, positionally.

    Returns:
        Zero seams (a tail that names no subcommand — a flag-only append), one,
        or several (a dynamic subcommand expands to its listed literals).

    Raises:
        AssertionError: A dynamic subcommand that :data:`DYNAMIC_TOKENS` does
            not account for.
    """
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token is None:
            name = dynamic_names[index]
            if name is None:
                # A value handed to a flag, or an unnamed expression in a
                # value position; neither can be the subcommand.
                index += 1
                continue
            key = (module, name)
            assert key in DYNAMIC_TOKENS, (
                f"{module} builds a runtime argv whose subcommand is the variable {name!r}. "
                f"Add it to DYNAMIC_TOKENS with the literals it can hold, so the stub is "
                f"checked against them."
            )
            tail = tokens[index + 1 :]
            names = dynamic_names[index + 1 :]
            expanded = []
            for literal in DYNAMIC_TOKENS[key]:
                expanded.extend(_verbs([literal, *tail], module, [None, *names]))
            return expanded
        if token.startswith("-"):
            # A flag. Its value is skipped with it — read off the same flag
            # tables the stub parses argv with, so the scanner and the stub
            # cannot disagree about which flags take one.
            index += 2 if token in _VALUE_FLAGS else 1
            continue
        if token in _NAMESPACES:
            for follower in tokens[index + 1 :]:
                if follower and not follower.startswith("-"):
                    return [f"{token} {follower}"]
            return [token]
        return [token]
    return []


def _plain_seams(elements: list[ast.expr], module: str, lineno: int) -> list[_Seam]:
    tokens = _constants(elements)
    if "compose" in tokens:
        # ``[runtime, "compose", "version"]`` builds the compose argv inline
        # rather than from a base. A bare ``[runtime, "compose"]`` names no
        # subcommand — it is the base itself — and yields nothing.
        after = tokens.index("compose") + 1
        return _compose_seams(elements[after:], module, lineno)
    names = [node.id if isinstance(node, ast.Name) else None for node in elements]
    return [_Seam(verb, module, lineno) for verb in _verbs(tokens, module, names)]


def _compose_seams(elements: list[ast.expr], module: str, lineno: int) -> list[_Seam]:
    tokens = _constants(elements)
    names = [node.id if isinstance(node, ast.Name) else None for node in elements]
    return [_Seam(f"compose {verb}", module, lineno) for verb in _verbs(tokens, module, names)]


def _is_compose_base(node: ast.expr) -> bool:
    """Whether *node* names a local holding a compose argv base."""
    if not isinstance(node, ast.Name):
        return False
    return node.id in _COMPOSE_BASE_NAMES or node.id.endswith("_cmd") or node.id.endswith("_base")


def _is_runtime_head(node: ast.expr) -> bool:
    """Whether *node* is the runtime binary standing at the head of an argv.

    Two spellings, both of them common in this package: a local the caller
    resolved the binary into (``runtime = get_runtime_command(config)[0]``), and
    the same call written inline where the local would have gone
    (``[get_runtime_command(config)[0], "build", …]``). The inline one builds
    real seams, so a scanner that only knew the local would read those sites as
    a list of strings and find nothing.
    """
    if isinstance(node, ast.Name):
        return node.id in _RUNTIME_HEAD_NAMES
    if isinstance(node, ast.Subscript):
        call = node.value
        if not isinstance(call, ast.Call):
            return False
        function = call.func
        if isinstance(function, ast.Name):
            return function.id in _RUNTIME_CALL_NAMES
        if isinstance(function, ast.Attribute):
            return function.attr in _RUNTIME_CALL_NAMES
    return False


def _walk_seams(tree: ast.AST, module: str) -> list[_Seam]:
    """Every runtime seam one parsed module builds.

    Five shapes, which between them are how this package talks to a runtime:

    * ``[runtime, "ps", "-a", …]`` — a plain-runtime argv;
    * ``[*base, "version"]`` — a subcommand spliced onto a compose base;
    * ``base_cmd + ["up", "-d"]`` — the same, written as a concatenation;
    * ``cmd.append("down")`` — and again, written as a mutation;
    * ``cmd.extend(["down", "--volumes"])`` — the mutation, several tokens at once.

    Plus ``RuntimeProbe._capture([...])`` in ``reset.py``, whose argv omits the
    binary because the class supplies it.

    Shared with the scanner's own tests below, which feed it a snippet: two
    walks would drift, and the one that drifted would be the one nothing checks.
    """
    found: list[_Seam] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and node.elts:
            head = node.elts[0]
            if _is_runtime_head(head):
                found.extend(_plain_seams(node.elts[1:], module, node.lineno))
            elif isinstance(head, ast.Starred) and _is_compose_base(head.value):
                found.extend(_compose_seams(node.elts[1:], module, node.lineno))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if _is_compose_base(node.left) and isinstance(node.right, ast.List):
                found.extend(_compose_seams(node.right.elts, module, node.lineno))
        elif isinstance(node, ast.Call):
            found.extend(_call_seams(node, module))
    return found


def scan_module(path: Path) -> list[_Seam]:
    """Every runtime seam one module builds."""
    return _walk_seams(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), path.name)


def _call_seams(node: ast.Call, module: str) -> list[_Seam]:
    """Seams from ``cmd.append("down")``, ``cmd.extend([…])`` and ``self._capture([…])``."""
    function = node.func
    if isinstance(function, ast.Attribute):
        if (
            function.attr == "append"
            and _is_compose_base(function.value)
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            # Constants only: an argv that appends a *value* (a build context,
            # a service name) is appending to a subcommand it already carries,
            # not naming one.
            return _compose_seams(node.args[:1], module, node.lineno)
        if (
            function.attr == "extend"
            and _is_compose_base(function.value)
            and node.args
            and isinstance(node.args[0], ast.List | ast.Tuple)
            and node.args[0].elts
        ):
            # Only the first element, for the same reason ``append`` takes only
            # a constant: everything after a subcommand is its arguments. It is
            # also what keeps the scanner honest about a base it may have
            # misread — ``ps_cmd`` is a plain argv the name test cannot tell
            # from a compose one, and reading its tail would invent a compose
            # subcommand out of a ``--format`` value.
            return _compose_seams(node.args[0].elts[:1], module, node.lineno)
        if function.attr == "_capture" and node.args and isinstance(node.args[0], ast.List):
            return _plain_seams(node.args[0].elts, module, node.lineno)
    return []


def scan_deployment() -> list[_Seam]:
    """Every runtime seam the deployment package builds, module by module."""
    root = repo_root() / DEPLOYMENT_PACKAGE
    found: list[_Seam] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        found.extend(scan_module(path))
    return found


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def discovered_seams() -> list[_Seam]:
    """The scan, once — it parses the whole package."""
    return scan_deployment()


def _run_stub(argv: list[str], stub: StubRuntime, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the stub ``docker`` exactly as a deploy would: found on ``PATH``."""
    return subprocess.run(
        ["docker", *argv],
        cwd=cwd,
        env=stub.env(),
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


class TestScanner:
    """The instrument itself, before anything it measures."""

    def test_scan_finds_the_seams_everyone_knows_about(self, discovered_seams: list[_Seam]) -> None:
        """A scanner that matched nothing would pass every other test here."""
        seams = {found.seam for found in discovered_seams}
        missing = ANCHOR_SEAMS - seams
        assert not missing, (
            f"the AST scan missed seams the deploy path certainly has ({sorted(missing)}); "
            f"it found {sorted(seams)} — the scanner is broken, not the stub"
        )

    def test_scan_reads_a_compose_concatenation(self) -> None:
        """``base_cmd + ["up", "-d", svc]`` is a compose ``up``, not a plain one."""
        source = 'base_cmd = x\nup_cmd = base_cmd + ["up", "-d", service]\n'
        seams = _seams_of_source(source)
        assert seams == ["compose up"]

    def test_scan_reads_a_plain_argv_with_a_namespace(self) -> None:
        """``[runtime, "image", "inspect", …]`` is two words, not one."""
        source = 'cmd = [runtime, "image", "inspect", "--format", "{{.Id}}", image]\n'
        assert _seams_of_source(source) == ["image inspect"]

    def test_scan_skips_a_flag_only_append(self) -> None:
        """``cmd.append("--follow")`` adds a flag to a seam, not a seam."""
        assert _seams_of_source('cmd.append("--follow")\n') == []

    def test_scan_reads_a_compose_extend(self) -> None:
        """``cmd.extend(["down", "--volumes"])`` names a subcommand like an append does."""
        assert _seams_of_source('cmd.extend(["down", "--volumes"])\n') == ["compose down"]

    def test_scan_skips_a_flag_only_extend(self) -> None:
        """``ps_cmd.extend(["--format", "json"])`` adds arguments to a seam, not a seam."""
        assert _seams_of_source('ps_cmd.extend(["--format", "json"])\n') == []

    def test_scan_reads_an_inline_runtime_binary(self) -> None:
        """``[get_runtime_command(config)[0], "build", …]`` is a plain ``build``."""
        source = 'cmd = [get_runtime_command(config)[0], "build", "-t", tag]\n'
        assert _seams_of_source(source) == ["build"]


def _seams_of_source(source: str) -> list[str]:
    """Seams found in a source snippet, for the scanner's own tests."""
    return [entry.seam for entry in _walk_seams(ast.parse(source), "snippet.py")]


class TestStubCoverage:
    """Every seam the source builds, answered by the stub."""

    def test_stub_declares_every_seam_the_source_builds(
        self, discovered_seams: list[_Seam]
    ) -> None:
        """The whole gate, in one assertion: nothing unanswered."""
        gaps: list[_Seam] = []
        for found in discovered_seams:
            if found.seam.startswith("compose "):
                handled = found.seam.removeprefix("compose ") in _stub_main.HANDLED_COMPOSE
            else:
                handled = found.seam in _stub_main.HANDLED_PLAIN
            if not handled:
                gaps.append(found)
        assert not gaps, (
            "the deploy path shells out to runtime subcommands the stub does not answer:\n  "
            + "\n  ".join(repr(gap) for gap in gaps)
            + "\nAdd them to tests/pty/stub_runtime/_stub_main.py (handler + HANDLED_* set)."
        )

    def test_stub_answers_every_seam_it_declares(self, stub: StubRuntime, tmp_path: Path) -> None:
        """Declared is not answered: run each one and read the log back.

        A seam could be listed in ``HANDLED_*`` and fall through every branch of
        the dispatcher, which is precisely the failure the declaration was
        supposed to rule out.
        """
        declared = [seam.split() for seam in sorted(_stub_main.HANDLED_PLAIN)]
        declared += [["compose", *seam.split()] for seam in sorted(_stub_main.HANDLED_COMPOSE)]

        failures = []
        for argv in declared:
            result = _run_stub(argv, stub, tmp_path)
            if result.returncode != 0:
                failures.append(f"{' '.join(argv)} exited {result.returncode}")
        assert not failures, failures

        unhandled = stub.unhandled()
        assert not unhandled, [record["args"] for record in unhandled]
        assert len(stub.invocations()) == len(declared)

    @pytest.mark.allow_unhandled_runtime
    def test_stub_records_an_invocation_it_cannot_answer(
        self, stub: StubRuntime, tmp_path: Path
    ) -> None:
        """The gate can fail. A stub that reported everything handled proves nothing.

        Exit 0 even here, on purpose: the harness must not turn "not scripted
        yet" into "the deploy is broken" — the log carries the fact instead.
        The marker is what keeps the ``stub_runtime`` teardown from failing the
        one test whose whole subject is an unanswered invocation.
        """
        result = _run_stub(["frobnicate", "--hard"], stub, tmp_path)
        assert result.returncode == 0
        assert stub.unhandled(), "an unknown subcommand must be recorded as unhandled"
        assert "unhandled" in result.stderr


class TestStubIsReachable:
    """The wiring the harness depends on, asserted rather than assumed."""

    def test_stub_docker_is_executable(self) -> None:
        """A lost mode bit is a silent fall-through to the real docker."""
        stub_docker = stub_runtime_dir() / "docker"
        assert os.access(stub_docker, os.X_OK), f"{stub_docker} is not executable"

    def test_container_runtime_env_pins_the_probe_to_docker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``CONTAINER_RUNTIME`` beats config — the whole reason the stub is named docker."""
        from osprey.deployment.runtime_helper import _runtimes_to_try

        monkeypatch.setenv("CONTAINER_RUNTIME", "docker")
        assert _runtimes_to_try({"container_runtime": "podman"}) == ("docker",)

    def test_runtime_detection_selects_the_stub(
        self, stub: StubRuntime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``get_runtime_command`` finds the stub and is satisfied by its answers."""
        from osprey.deployment import runtime_helper

        for key, value in stub.env().items():
            monkeypatch.setenv(key, value)
        runtime_helper.reset_runtime_cache()
        try:
            assert runtime_helper.get_runtime_command(None) == ["docker", "compose"]
        finally:
            runtime_helper.reset_runtime_cache()

    def test_compose_provider_probe_accepts_the_stub_banner(
        self, stub: StubRuntime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The provider probe fails closed on a banner it cannot place.

        So the stub's version string is a contract with
        :func:`~osprey.deployment.runtime_helper.detect_compose_provider`, not
        decoration — pinned here rather than discovered as a refusal mid-deploy.
        """
        from osprey.deployment import runtime_helper

        for key, value in stub.env().items():
            monkeypatch.setenv(key, value)
        runtime_helper.reset_runtime_cache()
        try:
            detected = runtime_helper.detect_compose_provider(["docker", "compose"])
        finally:
            runtime_helper.reset_runtime_cache()
        assert detected.provider is runtime_helper.ComposeProvider.DOCKER_V2


class TestStubAnswersPlausibly:
    """Exit 0 is not enough: the answers have to be usable ones."""

    def test_compose_up_starts_the_services_the_exemplar_declares(
        self, stub: StubRuntime, exemplar_repo: Path, tmp_path: Path
    ) -> None:
        """The stub reads the rendered compose files rather than inventing services.

        This is what makes the ``ps`` that follows an ``up`` say something true:
        the containers it lists are the ones the repo under test declares.
        """
        from osprey.deployment.container_lifecycle import (
            as_built_compose_files,
            as_built_config_path,
        )
        from osprey.utils.config import load_project_config

        config = load_project_config(str(as_built_config_path(exemplar_repo)), wrap_errors=True)
        compose_files = as_built_compose_files(config, exemplar_repo)
        assert compose_files, f"the exemplar rendered no compose files into {exemplar_repo}/build"

        argv = ["compose", "--project-directory", str(exemplar_repo)]
        for path in compose_files:
            argv += ["-f", str(path)]

        up = _run_stub([*argv, "-p", "ptyproj", "up", "-d"], stub, tmp_path)
        assert up.returncode == 0
        # Compose reports progress on stderr and keeps stdout for data; the
        # renderer under test depends on that split.
        assert "Started" in up.stderr

        listing = _run_stub(["ps", "--format", "json"], stub, tmp_path)
        records = [json.loads(line) for line in listing.stdout.splitlines() if line.strip()]
        assert records, "ps saw nothing after a compose up"
        assert all(record["Project"] == "ptyproj" for record in records)
        assert {record["Service"] for record in records} == {"openobserve"}

        down = _run_stub([*argv, "-p", "ptyproj", "down"], stub, tmp_path)
        assert down.returncode == 0
        after = _run_stub(["ps", "--format", "json"], stub, tmp_path)
        assert after.stdout.strip() == "", "containers survived a compose down"

    def test_image_and_container_inspect_agree(self, stub: StubRuntime, tmp_path: Path) -> None:
        """Two IDs that disagreed would send every deploy into a recreate.

        Image drift is decided by comparing these two answers, so a stub that
        answered them independently would make the harness measure a
        reconciliation that no real deployment would perform.
        """
        image = _run_stub(["image", "inspect", "--format", "{{.Id}}", "x:local"], stub, tmp_path)
        container = _run_stub(
            ["container", "inspect", "--format", "{{.Image}}", "x-1"], stub, tmp_path
        )
        assert image.stdout.strip().removeprefix("sha256:") == container.stdout.strip()

    def test_build_speaks_buildkit_plain_progress(self, stub: StubRuntime, tmp_path: Path) -> None:
        """``build_progress`` parses these lines, so the stub has to emit them."""
        result = _run_stub(["build", "-t", "x:local", "."], stub, tmp_path)
        assert result.returncode == 0
        assert "DONE" in result.stderr
        assert "CACHED" in result.stderr
        assert "sha256:" in result.stderr

    def test_the_log_carries_argv_and_stdin(self, stub: StubRuntime, tmp_path: Path) -> None:
        """Both halves of an invocation, because ``exec -i`` is one of them."""
        subprocess.run(
            ["docker", "exec", "-i", "container", "sh", "-c", "cat"],
            cwd=tmp_path,
            env=stub.env(),
            input="seeded-payload\n",
            capture_output=True,
            text=True,
            check=False,
        )
        record = stub.invocations()[-1]
        assert record["args"][:2] == ["exec", "-i"]
        assert record["stdin"] == "seeded-payload\n"
        assert record["seam"] == "exec"


@pytest.fixture
def stub(stub_runtime: StubRuntime) -> StubRuntime:
    """The stub, freshly reset — every test here reads its own log."""
    stub_runtime.reset()
    return stub_runtime


def test_stub_runs_under_the_bare_system_interpreter() -> None:
    """The stub is spawned by whatever ``PATH`` finds, not by the test venv.

    Its shebang is ``/usr/bin/env python3``, so it has to keep to the standard
    library — a stub that imported anything installed would work here and fail
    on the first host whose ``python3`` is bare.
    """
    source = (stub_runtime_dir() / "_stub_main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"__future__"}
    assert not third_party, f"the stub imports non-stdlib modules: {sorted(third_party)}"
