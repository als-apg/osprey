"""Unit tests for the emitted filesystem guard (``fs_guard.render_fs_guard``).

The guard is *source code*, not a callable, so testing it by calling the
renderer and inspecting the string would prove almost nothing — the thing that
can be wrong is what the emitted code does once it has rebound
``builtins.open`` and friends in a live interpreter. Every behavioural test
here therefore writes the rendered guard plus a probe into a script and runs it
in a **real subprocess**, exactly as the wrapper and the sandbox will, and
asserts on what the probe printed. That also keeps the patching out of the test
process, where a leaked ``builtins.open`` would poison every test after it.

The two postures are parameterized over one shared fixture matrix
(:class:`TestBothPostures`) so that "allowed read / allowed write / refused
write" is asserted identically for both, and the cases that only exist in one
posture get their own classes.

Fixture-isolation note, same trap the sandbox characterization suite documents:
on macOS ``tmp_path`` lives under the system temp dir. Nothing here adds the
temp dir as a root, so that is harmless, but every root is ``resolve()``d in
the fixture because the guard resolves the candidate path before comparing —
an unresolved ``/var/...`` root would never match a resolved ``/private/var``
candidate.
"""

import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from osprey.services.python_executor.execution.fs_guard import (
    DEFAULT_ALLOWLIST_PREFIX,
    DEFAULT_DENYLIST_PREFIX,
    EXECUTOR_PATCH_TARGETS,
    SANDBOX_PATCH_TARGETS,
    SANDBOX_WRITE_MODES_ONLY_TARGETS,
    render_fs_guard,
)

pytestmark = pytest.mark.unit

# The Python environment has to stay readable or the child cannot import
# anything under the allowlist posture. Same three markers the sandbox passes.
_BYPASS = ("site-packages", "lib/python", sys.prefix)


@dataclass
class _Roots:
    """One isolated set of roots, plus the machinery to run a probe against it."""

    tmp: Path
    permitted: Path
    readonly: Path
    protected: Path
    outside: Path

    def run(self, guard: str, probe: str) -> str:
        """Run *guard* + *probe* in a real subprocess and return its stdout."""
        script = self.tmp / "probe_script.py"
        script.write_text(guard + "\n" + textwrap.dedent(probe) + "\n", encoding="utf-8")
        proc = subprocess.run(  # noqa: S603 - fixed argv, test-authored script
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"probe script failed rc={proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
        return proc.stdout


@pytest.fixture
def roots(tmp_path: Path) -> _Roots:
    """Four sibling roots — none contains another, so no case is ambiguous."""
    tmp = tmp_path.resolve()

    permitted = tmp / "permitted"
    permitted.mkdir()
    (permitted / "data.txt").write_text("PERMITTED-DATA")

    readonly = tmp / "readonly"
    readonly.mkdir()
    (readonly / "data.txt").write_text("READONLY-DATA")

    protected = tmp / "protected"
    protected.mkdir()
    (protected / "data.txt").write_text("PROTECTED-DATA")
    (protected / "victim.txt").write_text("VICTIM")
    # Empty, so the directory-removal forms would actually succeed if the guard
    # let them through — a non-empty target would refuse for the wrong reason.
    (protected / "empty_dir").mkdir()

    outside = tmp / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("OUTSIDE-SECRET")

    return _Roots(
        tmp=tmp, permitted=permitted, readonly=readonly, protected=protected, outside=outside
    )


def _allowlist_guard(roots: _Roots, **overrides) -> str:
    kwargs = {
        "default_deny": True,
        "permitted_roots": (roots.permitted,),
        "protected_roots": (),
        "read_roots": (roots.readonly,),
        "bypass_prefixes": _BYPASS,
        "patch_targets": EXECUTOR_PATCH_TARGETS,
        "write_modes_only_targets": (),
    }
    kwargs.update(overrides)
    return render_fs_guard(**kwargs)


def _denylist_guard(roots: _Roots, **overrides) -> str:
    kwargs = {
        "default_deny": False,
        "permitted_roots": (roots.permitted,),
        "protected_roots": (roots.protected,),
        "read_roots": (),
        "bypass_prefixes": (),
        "patch_targets": EXECUTOR_PATCH_TARGETS,
        "write_modes_only_targets": (),
    }
    kwargs.update(overrides)
    return render_fs_guard(**kwargs)


@dataclass
class _Posture:
    """A rendered guard plus the paths whose verdicts that posture fixes."""

    name: str
    guard: str
    refused_write: Path
    refusal_prefix: str


@pytest.fixture(params=["allowlist", "denylist"])
def posture(request, roots: _Roots) -> _Posture:
    """Both postures, over the same roots, for the shared matrix."""
    if request.param == "allowlist":
        return _Posture(
            name="allowlist",
            guard=_allowlist_guard(roots),
            # Under the allowlist the project-tree analogue is readable but
            # not writable; that is this posture's refused write.
            refused_write=roots.readonly / "written.txt",
            refusal_prefix=DEFAULT_ALLOWLIST_PREFIX,
        )
    return _Posture(
        name="denylist",
        guard=_denylist_guard(roots),
        refused_write=roots.protected / "written.txt",
        refusal_prefix=DEFAULT_DENYLIST_PREFIX,
    )


# ---------------------------------------------------------------------------
# The shared matrix — asserted identically for both postures
# ---------------------------------------------------------------------------
class TestBothPostures:
    """Read-allowed, write-allowed, write-refused: the same three in each mode."""

    def test_allowed_read_under_permitted_root(self, roots: _Roots, posture: _Posture):
        out = roots.run(
            posture.guard,
            f"""
            target = r"{roots.permitted / "data.txt"}"
            print("BUILTINS_READ:", open(target).read())
            """,
        )
        assert "BUILTINS_READ: PERMITTED-DATA" in out

    def test_allowed_write_under_permitted_root(self, roots: _Roots, posture: _Posture):
        target = roots.permitted / "written.txt"
        out = roots.run(
            posture.guard,
            f"""
            target = r"{target}"
            with open(target, "w") as fh:
                fh.write("WROTE")
            print("WRITE_ALLOWED")
            """,
        )
        assert "WRITE_ALLOWED" in out
        assert target.read_text() == "WROTE"

    def test_refused_write_raises_permission_error(self, roots: _Roots, posture: _Posture):
        target = posture.refused_write
        out = roots.run(
            posture.guard,
            f"""
            target = r"{target}"
            try:
                with open(target, "w") as fh:
                    fh.write("SHOULD NOT LAND")
                print("UNEXPECTED_WRITE_ALLOWED")
            except PermissionError as exc:
                print("DENIED:", exc)
            print("STILL_RUNNING")
            """,
        )
        assert "UNEXPECTED_WRITE_ALLOWED" not in out
        assert f"DENIED: {posture.refusal_prefix}" in out
        assert "write denied" in out
        # The refusal is a catchable exception inside the child, not a kill.
        assert "STILL_RUNNING" in out
        assert not target.exists()


# ---------------------------------------------------------------------------
# Allowlist-only behaviour (default_deny=True) — the sandbox's posture
# ---------------------------------------------------------------------------
class TestAllowlistPosture:
    """Bypass prefixes, the read-only root, and refusal of everything else."""

    def test_bypass_prefix_read_is_allowed(self, roots: _Roots):
        # pytest's own source: guaranteed present, guaranteed under
        # site-packages, and outside every root this fixture defines.
        target = Path(pytest.__file__).resolve()
        assert "site-packages" in str(target), target
        out = roots.run(
            _allowlist_guard(roots),
            f"""
            print("BYPASS_READ_OK:", bool(open(r"{target}").read()))
            """,
        )
        assert "BYPASS_READ_OK: True" in out

    def test_read_root_is_readable(self, roots: _Roots):
        out = roots.run(
            _allowlist_guard(roots),
            f"""
            print("READ:", open(r"{roots.readonly / "data.txt"}").read())
            """,
        )
        assert "READ: READONLY-DATA" in out

    def test_read_root_write_is_refused_with_write_denied(self, roots: _Roots):
        target = roots.readonly / "nope.txt"
        out = roots.run(
            _allowlist_guard(roots),
            f"""
            try:
                open(r"{target}", "w")
                print("UNEXPECTED_WRITE_ALLOWED")
            except PermissionError as exc:
                print("DENIED:", exc)
            """,
        )
        assert "UNEXPECTED_WRITE_ALLOWED" not in out
        assert f"DENIED: {DEFAULT_ALLOWLIST_PREFIX} write denied" in out
        assert not target.exists()

    def test_path_outside_every_root_is_refused_with_access_denied(self, roots: _Roots):
        out = roots.run(
            _allowlist_guard(roots),
            f"""
            try:
                print("LEAKED:", open(r"{roots.outside / "secret.txt"}").read())
            except PermissionError as exc:
                print("DENIED:", exc)
            """,
        )
        assert "LEAKED" not in out
        assert "OUTSIDE-SECRET" not in out
        assert f"DENIED: {DEFAULT_ALLOWLIST_PREFIX} access denied" in out

    def test_read_mode_variants_are_not_treated_as_writes(self, roots: _Roots):
        # 'rt'/'rb' are reads. The inline guard this module replaces tested
        # ``mode not in ("r", "rb")``, which called 'rt' a write; mode-character
        # detection is the deliberate correction.
        out = roots.run(
            _allowlist_guard(roots),
            f"""
            target = r"{roots.readonly / "data.txt"}"
            print("RT:", open(target, "rt").read())
            print("RB:", open(target, "rb").read().decode())
            """,
        )
        assert "RT: READONLY-DATA" in out
        assert "RB: READONLY-DATA" in out


# ---------------------------------------------------------------------------
# Denylist-only behaviour (default_deny=False) — the executor's posture
# ---------------------------------------------------------------------------
class TestDenylistPosture:
    """Everything passes except a write landing in a protected root."""

    def test_unrelated_path_is_allowed_for_read_and_write(self, roots: _Roots):
        target = roots.outside / "free.txt"
        out = roots.run(
            _denylist_guard(roots),
            f"""
            print("READ:", open(r"{roots.outside / "secret.txt"}").read())
            with open(r"{target}", "w") as fh:
                fh.write("UNPROTECTED")
            print("WRITE_ALLOWED")
            """,
        )
        assert "READ: OUTSIDE-SECRET" in out
        assert "WRITE_ALLOWED" in out
        assert target.read_text() == "UNPROTECTED"

    def test_protected_root_stays_readable(self, roots: _Roots):
        out = roots.run(
            _denylist_guard(roots),
            f"""
            print("READ:", open(r"{roots.protected / "data.txt"}").read())
            """,
        )
        assert "READ: PROTECTED-DATA" in out

    def test_permitted_root_inside_protected_root_still_writable(self, tmp_path: Path):
        # The wrapper's own metadata and artifacts live under a protected
        # parent; permitted_roots is what keeps them writable.
        base = tmp_path.resolve()
        protected = base / "build"
        nested = protected / "var" / "agent_data"
        nested.mkdir(parents=True)
        guard = render_fs_guard(
            default_deny=False,
            permitted_roots=(nested,),
            protected_roots=(protected,),
            read_roots=(),
        )
        roots = _Roots(tmp=base, permitted=nested, readonly=base, protected=protected, outside=base)
        target = nested / "metadata.json"
        refused = protected / "config.yml"
        out = roots.run(
            guard,
            f"""
            with open(r"{target}", "w") as fh:
                fh.write("{{}}")
            print("NESTED_WRITE_ALLOWED")
            try:
                open(r"{refused}", "w")
                print("UNEXPECTED_WRITE_ALLOWED")
            except PermissionError as exc:
                print("DENIED:", exc)
            """,
        )
        assert "NESTED_WRITE_ALLOWED" in out
        assert "UNEXPECTED_WRITE_ALLOWED" not in out
        assert f"DENIED: {DEFAULT_DENYLIST_PREFIX} write denied" in out
        assert target.read_text() == "{}"
        assert not refused.exists()

    @pytest.mark.parametrize(
        ("label", "call"),
        [
            ("os.open", "os.open(VICTIM, os.O_WRONLY | os.O_TRUNC)"),
            ("os.remove", "os.remove(VICTIM)"),
            ("os.unlink", "os.unlink(VICTIM)"),
            ("os.truncate", "os.truncate(VICTIM, 0)"),
            ("os.rename", "os.rename(VICTIM, VICTIM + '.moved')"),
            ("os.replace", "os.replace(VICTIM, VICTIM + '.moved')"),
            ("os.mkdir", "os.mkdir(NEW_DIR)"),
            ("os.makedirs", "os.makedirs(NEW_DIR)"),
            ("os.makedirs-kwarg", "os.makedirs(name=NEW_DIR)"),
            ("shutil.copy", "shutil.copy(SRC, VICTIM)"),
            ("shutil.copy2", "shutil.copy2(SRC, VICTIM)"),
            ("shutil.copyfile", "shutil.copyfile(SRC, VICTIM)"),
            ("shutil.copyfile-kwarg", "shutil.copyfile(src=SRC, dst=VICTIM)"),
            ("shutil.move", "shutil.move(SRC, VICTIM)"),
            ("shutil.rmtree", "shutil.rmtree(PROTECTED_DIR)"),
            ("io.open", "io.open(VICTIM, 'w')"),
            # Link creation: the *link* is the new entry inside the protected
            # root, so it is the write. The source it points at is untouched.
            ("os.symlink", "os.symlink(SRC, NEW_LINK)"),
            ("os.symlink-kwarg", "os.symlink(SRC, dst=NEW_LINK)"),
            ("os.link", "os.link(SRC, NEW_LINK)"),
            # Directory removal: dropping a directory out of the render zone is
            # as much a write to it as overwriting a file in it.
            ("os.rmdir", "os.rmdir(EMPTY_DIR)"),
            ("os.removedirs", "os.removedirs(EMPTY_DIR)"),
        ],
    )
    def test_protected_write_refused_through_every_patched_route(
        self, roots: _Roots, label: str, call: str
    ):
        victim = roots.protected / "victim.txt"
        out = roots.run(
            _denylist_guard(roots),
            f"""
            import io
            import os
            import shutil
            VICTIM = r"{victim}"
            NEW_DIR = r"{roots.protected / "new_dir"}"
            NEW_LINK = r"{roots.protected / "new_link"}"
            EMPTY_DIR = r"{roots.protected / "empty_dir"}"
            PROTECTED_DIR = r"{roots.protected}"
            SRC = r"{roots.permitted / "data.txt"}"
            try:
                {call}
                print("UNEXPECTED_ALLOWED")
            except PermissionError as exc:
                print("DENIED:", exc)
            """,
        )
        assert "UNEXPECTED_ALLOWED" not in out, f"{label} was not guarded"
        assert f"DENIED: {DEFAULT_DENYLIST_PREFIX} write denied" in out, label
        # Nothing was created, removed or overwritten on the way to the refusal.
        assert victim.read_text() == "VICTIM"
        assert (roots.protected / "data.txt").exists()
        assert not (roots.protected / "new_dir").exists()
        assert not (roots.protected / "victim.txt.moved").exists()
        # ``exists()`` follows the link, so a broken link would slip past it.
        assert not (roots.protected / "new_link").is_symlink()
        assert not (roots.protected / "new_link").exists()
        assert (roots.protected / "empty_dir").is_dir()

    def test_same_routes_still_work_outside_protected_roots(self, roots: _Roots):
        out = roots.run(
            _denylist_guard(roots),
            f"""
            import os
            import shutil
            base = r"{roots.outside}"
            src = os.path.join(base, "secret.txt")
            os.makedirs(os.path.join(base, "sub"))
            shutil.copy(src, os.path.join(base, "copied.txt"))
            os.rename(os.path.join(base, "copied.txt"), os.path.join(base, "renamed.txt"))
            os.remove(os.path.join(base, "renamed.txt"))
            shutil.rmtree(os.path.join(base, "sub"))
            fd = os.open(os.path.join(base, "viafd.txt"), os.O_WRONLY | os.O_CREAT)
            os.close(fd)
            os.symlink(src, os.path.join(base, "link.txt"))
            os.link(src, os.path.join(base, "hard.txt"))
            os.mkdir(os.path.join(base, "rd"))
            os.rmdir(os.path.join(base, "rd"))
            os.makedirs(os.path.join(base, "rd", "inner"))
            # Prunes "inner" then "rd", then stops on the non-empty base.
            os.removedirs(os.path.join(base, "rd", "inner"))
            print("ALL_ROUTES_OK")
            """,
        )
        assert "ALL_ROUTES_OK" in out
        assert (roots.outside / "viafd.txt").exists()
        assert (roots.outside / "link.txt").is_symlink()
        assert (roots.outside / "hard.txt").exists()
        assert not (roots.outside / "rd").exists()


# ---------------------------------------------------------------------------
# write_modes_only_targets — the pathlib asymmetry the sandbox depends on
# ---------------------------------------------------------------------------
class TestWriteModesOnlyTargets:
    """``io.open`` patched for writes only: pathlib reads keep today's verdict."""

    def test_io_open_read_passes_and_write_is_refused(self, roots: _Roots):
        guard = _allowlist_guard(
            roots,
            patch_targets=SANDBOX_PATCH_TARGETS,
            write_modes_only_targets=SANDBOX_WRITE_MODES_ONLY_TARGETS,
        )
        secret = roots.outside / "secret.txt"
        target = roots.outside / "written.txt"
        out = roots.run(
            guard,
            f"""
            import io
            print("IO_READ:", io.open(r"{secret}").read())
            try:
                io.open(r"{target}", "w")
                print("UNEXPECTED_WRITE_ALLOWED")
            except PermissionError as exc:
                print("DENIED:", exc)
            """,
        )
        # A read through io.open never reaches the check at all — that is what
        # keeps pathlib reads outside the allowlist working as they do today.
        assert "IO_READ: OUTSIDE-SECRET" in out
        assert "UNEXPECTED_WRITE_ALLOWED" not in out
        assert f"DENIED: {DEFAULT_ALLOWLIST_PREFIX} access denied" in out
        assert not target.exists()

    def test_pathlib_reads_pass_while_pathlib_writes_are_refused(self, roots: _Roots):
        # The designed end state for the sandbox: read behaviour unchanged
        # from the characterization suite, write hole closed.
        guard = _allowlist_guard(
            roots,
            patch_targets=SANDBOX_PATCH_TARGETS,
            write_modes_only_targets=SANDBOX_WRITE_MODES_ONLY_TARGETS,
        )
        secret = roots.outside / "secret.txt"
        target = roots.readonly / "by_pathlib.txt"
        out = roots.run(
            guard,
            f"""
            from pathlib import Path
            print("PATHLIB_READ:", Path(r"{secret}").read_text())
            try:
                Path(r"{target}").write_text("SHOULD NOT LAND")
                print("UNEXPECTED_WRITE_ALLOWED")
            except PermissionError as exc:
                print("DENIED:", exc)
            """,
        )
        assert "PATHLIB_READ: OUTSIDE-SECRET" in out
        assert "UNEXPECTED_WRITE_ALLOWED" not in out
        assert f"DENIED: {DEFAULT_ALLOWLIST_PREFIX} write denied" in out
        assert not target.exists()


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------
class TestRestore:
    """``_restore_patched_targets`` has to cover every name that was rebound."""

    def test_restore_returns_every_patched_name_to_working_order(self, roots: _Roots):
        guard = _allowlist_guard(
            roots,
            patch_targets=EXECUTOR_PATCH_TARGETS,
        )
        refused = roots.outside / "after_restore.txt"
        out = roots.run(
            guard,
            f"""
            import builtins
            import io
            import os
            import shutil
            base = r"{roots.outside}"
            target = r"{refused}"

            try:
                open(target, "w")
                print("UNEXPECTED_WRITE_ALLOWED")
            except PermissionError:
                print("REFUSED_WHILE_INSTALLED")

            _restore_patched_targets()
            print("PATCHED_NAMES_LEFT:", len(_osprey_fs_originals))

            with builtins.open(target, "w") as fh:
                fh.write("RESTORED")
            print("BUILTINS_OPEN_OK:", open(target).read())
            print("IO_OPEN_OK:", io.open(target).read())
            copied = os.path.join(base, "copied.txt")
            shutil.copy(target, copied)
            os.rename(copied, os.path.join(base, "renamed.txt"))
            os.remove(os.path.join(base, "renamed.txt"))
            os.makedirs(os.path.join(base, "sub", "deep"))
            shutil.rmtree(os.path.join(base, "sub"))
            fd = os.open(os.path.join(base, "viafd.txt"), os.O_WRONLY | os.O_CREAT)
            os.close(fd)
            os.truncate(os.path.join(base, "viafd.txt"), 0)
            print("ALL_RESTORED_ROUTES_OK")

            # Restore is idempotent — a second call in a finally must not fail.
            _restore_patched_targets()
            print("SECOND_RESTORE_OK")
            """,
        )
        assert "REFUSED_WHILE_INSTALLED" in out
        assert "PATCHED_NAMES_LEFT: 0" in out
        assert "BUILTINS_OPEN_OK: RESTORED" in out
        assert "IO_OPEN_OK: RESTORED" in out
        assert "ALL_RESTORED_ROUTES_OK" in out
        assert "SECOND_RESTORE_OK" in out

    def test_restore_covers_write_modes_only_targets_too(self, roots: _Roots):
        guard = _allowlist_guard(
            roots,
            patch_targets=SANDBOX_PATCH_TARGETS,
            write_modes_only_targets=SANDBOX_WRITE_MODES_ONLY_TARGETS,
        )
        target = roots.outside / "after_restore.txt"
        out = roots.run(
            guard,
            f"""
            import io
            print("PATCHED:", sorted(_osprey_fs_originals))
            _restore_patched_targets()
            with io.open(r"{target}", "w") as fh:
                fh.write("RESTORED")
            print("IO_OPEN_WRITE_OK:", open(r"{target}").read())
            """,
        )
        assert "PATCHED: ['builtins.open', 'io.open']" in out
        assert "IO_OPEN_WRITE_OK: RESTORED" in out


# ---------------------------------------------------------------------------
# The tamper limit — characterization, not endorsement
# ---------------------------------------------------------------------------
class TestTamperLimit:
    """The guard is defense in depth. This pins exactly how far that goes.

    Characterization, in the same spirit as the sandbox suite's
    ``*_TODAY`` cases: the behaviour below is a hole, and it is pinned so that
    the project's stance on it stays a written-down fact instead of an
    assumption. The guard is rendered *into* the child and installs itself in
    the same module namespace as the user code, restore handle included, so
    code that knows it is there disarms it in one call. Nothing rendered into
    the child can be hidden from the child.

    If this test ever starts failing because a disarm became impossible, that
    is a deliberate change of posture and wants its own task and its own
    review — not a quiet edit here. The boundary that *does* hold against
    adversarial code is the OS one: the container's privilege split, where the
    render zone belongs to a different user than the one running agent code.
    """

    def test_knowing_user_code_CAN_DISARM_the_guard_via_its_restore_handle_today(
        self, roots: _Roots
    ):
        target = roots.protected / "tampered.txt"
        out = roots.run(
            _denylist_guard(roots),
            f"""
            target = r"{target}"

            # While the guard is installed, the write is refused — including
            # the spelled-around form the static pre-check cannot see.
            try:
                open(target, "w").close()
                print("UNEXPECTED_WRITE_ALLOWED")
            except PermissionError:
                print("REFUSED_WHILE_INSTALLED")

            # The guard's own uninstall is in scope for the code it guards.
            _restore_patched_targets()
            with open(target, "w") as fh:
                fh.write("TAMPERED")
            print("DISARMED_WRITE_LANDED:", open(target).read())
            """,
        )
        assert "REFUSED_WHILE_INSTALLED" in out
        assert "UNEXPECTED_WRITE_ALLOWED" not in out
        # The hole, pinned. Not an endorsement — see the class docstring.
        assert "DISARMED_WRITE_LANDED: TAMPERED" in out
        assert target.read_text() == "TAMPERED"


# ---------------------------------------------------------------------------
# Renderer-level contract (no subprocess needed)
# ---------------------------------------------------------------------------
class TestRendererContract:
    """What the renderer itself promises about its output and its arguments."""

    def test_roots_are_embedded_as_repr_of_string_tuples(self):
        guard = render_fs_guard(
            default_deny=False,
            permitted_roots=[Path("/a/permitted")],
            protected_roots=[Path("/a/protected"), "/b/protected"],
            read_roots=(),
        )
        assert "_OSPREY_FS_PERMITTED = ('/a/permitted',)" in guard
        assert "_OSPREY_FS_PROTECTED = ('/a/protected', '/b/protected')" in guard
        assert "_OSPREY_FS_READ_ROOTS = ()" in guard

    def test_default_refusal_prefix_follows_the_posture(self):
        allowlist = render_fs_guard(
            default_deny=True, permitted_roots=(), protected_roots=(), read_roots=()
        )
        denylist = render_fs_guard(
            default_deny=False, permitted_roots=(), protected_roots=(), read_roots=()
        )
        assert f"_OSPREY_FS_PREFIX = {DEFAULT_ALLOWLIST_PREFIX!r}" in allowlist
        assert f"_OSPREY_FS_PREFIX = {DEFAULT_DENYLIST_PREFIX!r}" in denylist

    def test_refusal_prefix_is_overridable(self):
        guard = render_fs_guard(
            default_deny=False,
            permitted_roots=(),
            protected_roots=(),
            read_roots=(),
            refusal_prefix="Refused (readonly execution mode):",
        )
        assert "_OSPREY_FS_PREFIX = 'Refused (readonly execution mode):'" in guard

    def test_empty_refusal_prefix_is_rejected(self):
        with pytest.raises(ValueError, match="refusal_prefix"):
            render_fs_guard(
                default_deny=True,
                permitted_roots=(),
                protected_roots=(),
                read_roots=(),
                refusal_prefix="  ",
            )

    def test_unknown_patch_target_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported fs guard target"):
            render_fs_guard(
                default_deny=True,
                permitted_roots=(),
                protected_roots=(),
                read_roots=(),
                patch_targets=("os.system",),
            )

    def test_write_modes_only_rejects_a_target_without_a_mode(self):
        with pytest.raises(ValueError, match="mode-bearing"):
            render_fs_guard(
                default_deny=True,
                permitted_roots=(),
                protected_roots=(),
                read_roots=(),
                patch_targets=(),
                write_modes_only_targets=("shutil.rmtree",),
            )

    def test_a_name_cannot_be_in_both_target_sets(self):
        with pytest.raises(ValueError, match="both patch_targets"):
            render_fs_guard(
                default_deny=True,
                permitted_roots=(),
                protected_roots=(),
                read_roots=(),
                patch_targets=("io.open",),
                write_modes_only_targets=("io.open",),
            )

    def test_emitted_source_is_valid_python_and_left_aligned(self):
        import ast

        guard = render_fs_guard(
            default_deny=True,
            permitted_roots=("/a",),
            protected_roots=(),
            read_roots=("/b",),
            bypass_prefixes=_BYPASS,
        )
        ast.parse(guard)
        assert not guard.startswith((" ", "\n"))
