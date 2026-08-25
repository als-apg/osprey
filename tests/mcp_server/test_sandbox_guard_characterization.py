"""Characterization suite for the visualization-sandbox filesystem guard.

This suite pins the **current** behaviour of the ``open()`` guard that
``sandbox_executor._create_sandbox_wrapper`` emits into every sandboxed
script (the guard block at ``sandbox_executor.py`` lines 276-325). It is a
*characterization* suite: it documents what the guard does today, not what it
ought to do. Nothing here is an endorsement — several pinned behaviours are
holes, and they are pinned precisely so that the upcoming extraction of the
guard into a shared ``render_fs_guard`` module is a refactor and not a silent
behaviour change.

The load-bearing fact this suite exists to record:

    The original wrapper rebound **only** ``builtins.open``. CPython's
    ``pathlib`` calls ``io.open`` directly, so *every* ``Path.open`` /
    ``Path.read_text`` / ``Path.write_text`` call bypassed the guard
    entirely — reads and writes alike.

Re-wiring the wrapper onto the shared ``render_fs_guard`` closed
**half** of that, deliberately and by design: ``io.open`` is now patched for
*write modes only*, so ``Path.write_text`` and ``Path.open('w')`` are refused
while every pathlib **read** keeps exactly today's behaviour. So the guard is
still a guard for one of the two ways Python *reads* a file, and the read side
is still wide open (:class:`TestReadOutsideAllowedRoots` is where that bites).
Each case below exercises both routes and asserts the asymmetry where it
remains.

Determinism notes — both matter, and both are easy to get wrong:

  * ``tempfile.gettempdir()`` is an allowed root. On macOS ``pytest``'s
    ``tmp_path`` lives *under* the system temp dir, so a naive "outside"
    fixture built from ``tmp_path`` is silently inside an allowed root and
    the refusal cases pass vacuously. Every test here therefore repoints
    ``TMPDIR`` at a dedicated ``tmp_path/tmp`` subdirectory, making the rest
    of ``tmp_path`` genuinely outside every allowed root.
  * ``$HOME``'s ``.matplotlib``/``.config``/``.cache``/``.local`` are added as
    allowed roots *when they exist*. Tests point ``HOME`` at an empty
    directory so the developer's real cache dirs never widen the sandbox.

The guard cases drive ``_create_sandbox_wrapper`` plus a real subprocess
directly rather than going through :func:`execute_sandbox_code`, because the
latter's ``validate_sandbox_code`` pre-check (import whitelist, dangerous
substring scan) is a *separate* static gate that would reject or distort
probe code aimed at the runtime guard. The matplotlib case does go through
the public entry point, since its point is the real end-to-end render.
"""

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.mcp_server.workspace.execution.sandbox_executor import (
    _create_sandbox_wrapper,
    execute_sandbox_code,
)

# The prefix the guard puts on its own refusals. Asserting on it distinguishes
# a sandbox denial from an ordinary OS EACCES, which would also be a
# PermissionError but would mean something completely different.
_GUARD_PREFIX = "Sandbox:"


@dataclass
class _SandboxRun:
    """Outcome of one wrapped-script subprocess."""

    success: bool
    stdout: str
    stderr: str
    error: str | None
    returncode: int


@dataclass
class _Sandbox:
    """A fully isolated set of roots for one characterization case."""

    project_root: Path
    workspace_root: Path
    execution_folder: Path
    outside_dir: Path
    temp_dir: Path
    home_dir: Path
    env: dict[str, str]

    def run(self, user_code: str) -> _SandboxRun:
        """Generate the sandbox wrapper around *user_code* and run it for real."""
        wrapper = _create_sandbox_wrapper(
            textwrap.dedent(user_code),
            self.execution_folder,
            self.workspace_root,
            self.project_root,
        )
        script = self.execution_folder / "wrapped_script.py"
        script.write_text(wrapper, encoding="utf-8")
        proc = subprocess.run(  # noqa: S603 - fixed argv, test-controlled script
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            env=self.env,
            timeout=120,
        )
        meta_path = self.execution_folder / "execution_metadata.json"
        assert meta_path.exists(), (
            "wrapper did not write execution_metadata.json; "
            f"rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return _SandboxRun(
            success=meta["success"],
            stdout=meta["stdout"],
            stderr=meta["stderr"],
            error=meta.get("error"),
            returncode=proc.returncode,
        )


def _isolated_env(temp_dir: Path, home_dir: Path) -> dict[str, str]:
    """Child environment with the temp dir and HOME pinned to test-owned dirs."""
    env = dict(os.environ)
    env["TMPDIR"] = str(temp_dir)
    env["HOME"] = str(home_dir)
    env["MPLBACKEND"] = "Agg"
    # TEMP/TMP would outrank nothing on POSIX but are checked by
    # tempfile._candidate_tempdir_list before the hard-coded fallbacks; the
    # XDG_*/MPLCONFIGDIR pair would drag the developer's real cache dirs back
    # in through the wrapper's HOME-derived allowed roots.
    for stale in ("TEMP", "TMP", "MPLCONFIGDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
        env.pop(stale, None)
    return env


@pytest.fixture
def sandbox(tmp_path: Path) -> _Sandbox:
    """Isolated project/workspace/outside/temp/home roots for one case.

    Layout (siblings, so none contains another):

        tmp_path/project                 <- project_root (read-only per the guard)
        tmp_path/project/var/agent_data  <- workspace_root (writable)
        tmp_path/project/var/agent_data/exec  <- execution folder
        tmp_path/outside                 <- outside every allowed root
        tmp_path/tmp                     <- the child's tempfile.gettempdir()
        tmp_path/home                    <- the child's $HOME, empty
    """
    project_root = tmp_path / "project"
    (project_root / "data").mkdir(parents=True)
    (project_root / "data" / "machine_data.csv").write_text("pv,value\nSR:BPM1,1.25\n")

    workspace_root = project_root / "var" / "agent_data"
    workspace_root.mkdir(parents=True)
    execution_folder = workspace_root / "exec"
    execution_folder.mkdir()

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("OUTSIDE-SECRET")

    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    home_dir = tmp_path / "home"
    home_dir.mkdir()

    return _Sandbox(
        project_root=project_root,
        workspace_root=workspace_root,
        execution_folder=execution_folder,
        outside_dir=outside_dir,
        temp_dir=temp_dir,
        home_dir=home_dir,
        env=_isolated_env(temp_dir, home_dir),
    )


# ---------------------------------------------------------------------------
# Fixture self-check — without this the refusal cases can pass vacuously
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestFixtureIsolation:
    """The refusal cases are only meaningful if the isolation actually took."""

    def test_child_sees_test_owned_tempdir_and_home(self, sandbox: _Sandbox):
        result = sandbox.run(
            """
            import tempfile
            print("TEMPDIR:", Path(tempfile.gettempdir()).resolve())
            print("HOME:", Path.home().resolve())
            """
        )
        assert result.success, result.stderr
        assert f"TEMPDIR: {sandbox.temp_dir.resolve()}" in result.stdout
        assert f"HOME: {sandbox.home_dir.resolve()}" in result.stdout

    def test_outside_dir_is_not_inside_any_allowed_root(self, sandbox: _Sandbox):
        # The whole suite's refusal cases rest on this. tmp_path is under the
        # system temp dir on macOS, which IS an allowed root, so the TMPDIR
        # repoint in the fixture is what makes "outside" mean outside.
        outside = sandbox.outside_dir.resolve()
        for root in (
            sandbox.project_root.resolve(),
            sandbox.workspace_root.resolve(),
            sandbox.execution_folder.resolve(),
            sandbox.temp_dir.resolve(),
            sandbox.home_dir.resolve(),
        ):
            assert not outside.is_relative_to(root)


# ---------------------------------------------------------------------------
# Case 1 — site-packages / Python environment reads
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestSitePackagesRead:
    """Reading the Python environment is allowed, by both routes.

    ``builtins.open`` is allowed by an explicit bypass (a ``site-packages`` or
    ``lib/python`` substring, or a ``sys.prefix`` prefix) that short-circuits
    ahead of the allowed-root loop. pathlib is allowed because it never
    reaches the guard at all. Same outcome, two entirely different reasons.
    """

    def test_site_packages_read_allowed_via_both_routes(self, sandbox: _Sandbox):
        # pytest's own source file: guaranteed present whenever this suite
        # runs, guaranteed under site-packages, and resolved in the parent so
        # the child needs no import of its own to reach it.
        target = Path(pytest.__file__)
        assert "site-packages" in str(target.resolve()), target
        result = sandbox.run(
            f"""
            target = Path(r"{target}")
            print("BUILTINS_OK:", bool(open(str(target)).read()))
            print("PATHLIB_READ_TEXT_OK:", bool(target.read_text()))
            with target.open() as fh:
                print("PATHLIB_OPEN_OK:", bool(fh.read()))
            """
        )
        assert result.success, result.stderr
        assert "BUILTINS_OK: True" in result.stdout
        assert "PATHLIB_READ_TEXT_OK: True" in result.stdout
        assert "PATHLIB_OPEN_OK: True" in result.stdout


# ---------------------------------------------------------------------------
# Case 2 — temp directory writes
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestTempDirWrite:
    """``tempfile.gettempdir()`` is an unconditionally writable allowed root."""

    def test_tempdir_write_allowed_via_both_routes(self, sandbox: _Sandbox):
        result = sandbox.run(
            """
            import tempfile
            tmp = Path(tempfile.gettempdir())
            with open(str(tmp / "builtins.txt"), "w") as fh:
                fh.write("builtins-wrote-this")
            print("BUILTINS_WRITE_OK")
            (tmp / "pathlib.txt").write_text("pathlib-wrote-this")
            print("PATHLIB_WRITE_OK")
            """
        )
        assert result.success, result.stderr
        assert "BUILTINS_WRITE_OK" in result.stdout
        assert "PATHLIB_WRITE_OK" in result.stdout
        assert (sandbox.temp_dir / "builtins.txt").read_text() == "builtins-wrote-this"
        assert (sandbox.temp_dir / "pathlib.txt").read_text() == "pathlib-wrote-this"


# ---------------------------------------------------------------------------
# Case 3 — project-root reads
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestProjectRootRead:
    """Project root is readable — that is the point of the read-only clause."""

    def test_project_root_read_allowed_via_both_routes(self, sandbox: _Sandbox):
        data_file = sandbox.project_root / "data" / "machine_data.csv"
        result = sandbox.run(
            f"""
            target = Path(r"{data_file}")
            print("BUILTINS:", open(str(target)).read().splitlines()[1])
            print("PATHLIB:", target.read_text().splitlines()[1])
            """
        )
        assert result.success, result.stderr
        assert "BUILTINS: SR:BPM1,1.25" in result.stdout
        assert "PATHLIB: SR:BPM1,1.25" in result.stdout


# ---------------------------------------------------------------------------
# Case 4 — project-root writes: refused via builtins, NOT via pathlib
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestProjectRootIsReadOnly:
    """Project root is read-only, by both routes.

    This used to be the asymmetry at its most consequential: the read-only
    clause is the guard's only *write* protection for the project tree, and
    pathlib walked straight past it and created the file. The re-wiring closed that
    by patching ``io.open`` for write modes, so both routes now refuse.
    """

    def test_project_root_write_refused_via_builtins_open(self, sandbox: _Sandbox):
        target = sandbox.project_root / "data" / "written_by_builtins.txt"
        result = sandbox.run(
            f"""
            target = Path(r"{target}")
            try:
                with open(str(target), "w") as fh:
                    fh.write("should not land")
                print("UNEXPECTED_WRITE_ALLOWED")
            except PermissionError as exc:
                print("DENIED:", exc)
            print("EXISTS:", target.exists())
            """
        )
        assert result.success, result.stderr
        assert "UNEXPECTED_WRITE_ALLOWED" not in result.stdout
        assert f"DENIED: {_GUARD_PREFIX}" in result.stdout
        assert "write denied" in result.stdout
        assert "EXISTS: False" in result.stdout
        assert not target.exists()

    def test_project_root_write_refused_via_pathlib(self, sandbox: _Sandbox):
        # DELIBERATE FLIP. This case previously pinned the opposite
        # verdict — pathlib's write_text/open("w") never touched
        # builtins.open, so the read-only project root was not read-only in
        # practice. 3.5 closed that hole by design: the shared guard patches
        # ``io.open`` for WRITE MODES ONLY, which is the one route pathlib
        # takes, so both write forms now raise the same refusal
        # ``builtins.open`` already raised. The read half of the same patch is
        # untouched on purpose — see TestReadOutsideAllowedRoots, which still
        # pins pathlib reads as passing.
        by_write_text = sandbox.project_root / "data" / "written_by_write_text.txt"
        by_open_w = sandbox.project_root / "data" / "written_by_path_open.txt"
        result = sandbox.run(
            f"""
            write_text_target = Path(r"{by_write_text}")
            open_w_target = Path(r"{by_open_w}")
            try:
                write_text_target.write_text("pathlib-write_text")
                print("UNEXPECTED_PATHLIB_WRITE_TEXT_ALLOWED")
            except PermissionError as exc:
                print("WRITE_TEXT_DENIED:", exc)
            try:
                with open_w_target.open("w") as fh:
                    fh.write("pathlib-open-w")
                print("UNEXPECTED_PATHLIB_OPEN_W_ALLOWED")
            except PermissionError as exc:
                print("OPEN_W_DENIED:", exc)
            """
        )
        assert result.success, result.stderr
        assert "UNEXPECTED_PATHLIB_WRITE_TEXT_ALLOWED" not in result.stdout
        assert "UNEXPECTED_PATHLIB_OPEN_W_ALLOWED" not in result.stdout
        assert f"WRITE_TEXT_DENIED: {_GUARD_PREFIX}" in result.stdout
        assert f"OPEN_W_DENIED: {_GUARD_PREFIX}" in result.stdout
        assert result.stdout.count("write denied") == 2
        assert not by_write_text.exists()
        assert not by_open_w.exists()


# ---------------------------------------------------------------------------
# Case 5 — unrelated paths
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestUnrelatedPathRefusal:
    """A path under no allowed root is refused by ``builtins.open``.

    Also pins that the refusal is an ordinary ``PermissionError`` raised
    *inside* the child, catchable by user code rather than killing the run.
    """

    def test_system_path_refused_via_builtins_open(self, sandbox: _Sandbox):
        result = sandbox.run(
            """
            try:
                with open("/etc/hosts") as fh:
                    fh.read()
                print("UNEXPECTED_READ_ALLOWED")
            except PermissionError as exc:
                print("DENIED:", exc)
            """
        )
        assert result.success, result.stderr
        assert "UNEXPECTED_READ_ALLOWED" not in result.stdout
        assert f"DENIED: {_GUARD_PREFIX}" in result.stdout
        assert "access denied" in result.stdout

    def test_refusal_does_not_abort_the_script(self, sandbox: _Sandbox):
        # The guard raises rather than exits; a caught denial leaves the run
        # successful, which is what makes it a usable error for agent code.
        result = sandbox.run(
            """
            try:
                open("/etc/hosts").read()
            except PermissionError:
                pass
            print("STILL_RUNNING")
            """
        )
        assert result.success, result.stderr
        assert result.error is None
        assert "STILL_RUNNING" in result.stdout


# ---------------------------------------------------------------------------
# Case 6 — the asymmetry, stated explicitly
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestReadOutsideAllowedRoots:
    """THE asymmetry: one absolute path, outside everything, two verdicts.

    ``builtins.open`` refuses it. ``Path.read_text`` and ``Path.open`` return
    its contents. Both statements are true of the code as it stands today,
    and both are part of the contract the ``render_fs_guard`` extraction must
    keep green. The re-wiring in particular must **not** change the
    pathlib read verdict — and it did not: ``io.open`` is patched for write
    modes only, so a pathlib read never reaches the guard.
    """

    def test_outside_path_refused_via_builtins_open(self, sandbox: _Sandbox):
        secret = sandbox.outside_dir / "secret.txt"
        result = sandbox.run(
            f"""
            secret = Path(r"{secret}")
            try:
                with open(str(secret)) as fh:
                    print("LEAKED_VIA_BUILTINS:", fh.read())
            except PermissionError as exc:
                print("DENIED:", exc)
            """
        )
        assert result.success, result.stderr
        assert "LEAKED_VIA_BUILTINS" not in result.stdout
        assert f"DENIED: {_GUARD_PREFIX}" in result.stdout
        assert "access denied" in result.stdout
        assert "OUTSIDE-SECRET" not in result.stdout

    def test_outside_path_READS_FINE_via_pathlib_today(self, sandbox: _Sandbox):
        # Characterization, not endorsement. If this test ever starts failing
        # because pathlib reads became refused, that is a deliberate
        # behaviour change and belongs in its own task with its own review —
        # it is explicitly out of scope for the guard extraction (3.3) and
        # the re-wiring (3.5).
        secret = sandbox.outside_dir / "secret.txt"
        result = sandbox.run(
            f"""
            secret = Path(r"{secret}")
            print("READ_TEXT:", secret.read_text())
            with secret.open() as fh:
                print("PATH_OPEN:", fh.read())
            """
        )
        assert result.success, result.stderr
        assert "READ_TEXT: OUTSIDE-SECRET" in result.stdout
        assert "PATH_OPEN: OUTSIDE-SECRET" in result.stdout


# ---------------------------------------------------------------------------
# Case 7 — real end-to-end render on a first-run HOME
# ---------------------------------------------------------------------------
def _matplotlib_available() -> bool:
    from importlib.util import find_spec

    return find_spec("matplotlib") is not None


@pytest.mark.integration
@pytest.mark.skipif(
    not _matplotlib_available(),
    reason="matplotlib is a project dependency; only skipped in a stripped environment",
)
class TestFirstRunHomeStillRenders:
    """A HOME with no ``~/.cache`` still renders a figure through ``save_artifact``.

    The success criterion has not moved — a freshly provisioned container
    renders — but the *reason* has. The wrapper used to add the
    four HOME cache dirs only ``if _cfg_dir.exists()``, so a cold HOME added
    none of them, matplotlib's font-cache write landed outside every root, and
    the guard refused it; matplotlib degraded to an in-memory cache and the
    render survived. 3.5 appends those four roots **unconditionally**, so the
    first-run write now simply succeeds. Both outcomes satisfy the contract;
    this case asserts the render, and asserts that the refusal that used to
    accompany it is gone.

    This case runs through the public :func:`execute_sandbox_code` entry
    point rather than the wrapper harness, so it also covers the static
    ``validate_sandbox_code`` gate, the env scrub, and artifact collection.
    """

    async def test_cold_home_renders_and_collects_artifact(
        self, sandbox: _Sandbox, monkeypatch: pytest.MonkeyPatch
    ):
        for key, value in (
            ("TMPDIR", str(sandbox.temp_dir)),
            ("HOME", str(sandbox.home_dir)),
            ("MPLBACKEND", "Agg"),
        ):
            monkeypatch.setenv(key, value)
        for stale in ("TEMP", "TMP", "MPLCONFIGDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
            monkeypatch.delenv(stale, raising=False)

        assert not (sandbox.home_dir / ".cache").exists()
        assert not (sandbox.home_dir / ".matplotlib").exists()

        code = textwrap.dedent(
            """
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot([0, 1, 2], [0, 1, 4])
            ax.set_title("cold home")
            save_artifact(fig, title="Cold HOME Figure", description="first-run cache dirs absent")
            print("RENDER_DONE")
            """
        )

        with (
            patch(
                "osprey.utils.workspace.resolve_workspace_root", return_value=sandbox.workspace_root
            ),
            patch("osprey.utils.workspace.resolve_project_root", return_value=sandbox.project_root),
            patch("osprey.utils.workspace.load_osprey_config", return_value={}),
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=sandbox.execution_folder,
                timeout=180,
            )

        assert result.success, f"stderr={result.stderr}\nerror={result.error_message}"
        assert "RENDER_DONE" in result.stdout
        assert len(result.artifacts) == 1, result.artifacts
        artifact = result.artifacts[0]
        assert artifact["title"] == "Cold HOME Figure"
        assert Path(artifact["path"]).exists()
        assert Path(artifact["path"]).stat().st_size > 0

        # The HOME cache dirs are permitted roots now, so nothing under this
        # cold HOME may be refused. Asserted against the HOME path rather than
        # the bare prefix: an unrelated refusal elsewhere is a different fact
        # and belongs to a different test.
        assert f"{_GUARD_PREFIX} write denied for '{sandbox.home_dir}" not in result.stderr
        assert f"{_GUARD_PREFIX} access denied for '{sandbox.home_dir}" not in result.stderr
