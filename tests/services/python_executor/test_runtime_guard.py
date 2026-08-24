"""Tests for the runtime filesystem guard the execution wrapper emits.

The guard exists because the pre-execution walker only sees *literal* paths in
*standard* spellings. ``open(Path('bui' + 'ld') / 'config.yml', 'w')`` gets past
it; the emitted guard does not care how the path was spelled, because it runs in
the child and inspects the resolved path at the moment of the call.

Two properties are load-bearing and are asserted separately here:

* **Unconditional.** The guard is installed in ``readwrite`` runs exactly as in
  ``readonly`` ones. Approving a run to move a magnet is not approving it to
  rewrite the profile that decides what the *next* run may do, so the mode
  changes the refusal's wording and nothing else.
* **Auditable in readonly.** The readonly wording carries
  :data:`READONLY_REFUSAL_MARKER`, which is the only thing
  ``report_runtime_refusal`` scans the child's stderr for. Lose the marker and
  the refusal still holds but silently stops reaching the operator alert and the
  audit ledger.

Behavioural tests run the emitted source in a **real subprocess**, like the
sibling ``test_fs_guard.py`` suite: the guard is source code whose whole point is
what it does to a live interpreter's ``builtins.open``, and running it in-process
would leak patched entry points into every test after it. Two tests run the
*whole* generated wrapper, which is the only way to prove that the wrapper's own
metadata, results and figure writes still land with the guard installed.

Fixture-isolation note: every root is ``resolve()``d, because the guard resolves
the candidate path before comparing — on macOS an unresolved ``/var/...`` root
would never match a resolved ``/private/var/...`` candidate.
"""

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from osprey.mcp_server.python_executor.executor import (
    PROFILE_SOURCE_ENTRIES,
    resolve_permitted_roots,
    resolve_protected_roots,
)
from osprey.services.python_executor.execution.fs_guard import (
    DEFAULT_DENYLIST_PREFIX,
    EXECUTOR_PATCH_TARGETS,
)
from osprey.services.python_executor.execution.wrapper import (
    READONLY_REFUSAL_MARKER,
    ExecutionWrapper,
)

pytestmark = pytest.mark.unit

_GUARD_BANNER = "OSPREY filesystem guard"


# --------------------------------------------------------------------------
# Fixture: a deployment repo with both zones in it
# --------------------------------------------------------------------------


@dataclass
class _Project:
    """A throwaway deployment repo, plus the machinery to probe it."""

    root: Path
    build: Path
    agent_data: Path
    audit: Path
    personas: Path
    execution_folder: Path
    source_file: Path

    def wrapper(self, execution_mode: str = "readonly") -> ExecutionWrapper:
        return ExecutionWrapper(
            execution_mode=execution_mode,
            protected_roots=resolve_protected_roots(self.root),
            permitted_roots=(self.agent_data,),
        )

    def guard(self, execution_mode: str = "readonly") -> str:
        return self.wrapper(execution_mode)._get_filesystem_guard(self.execution_folder)

    def probe(self, statement: str, execution_mode: str = "readonly") -> str:
        """Run the guard plus one write *statement* in a child; return its stdout.

        The child's cwd is the repo root, as it is in production, so a relative
        path in the probe resolves the way the agent's would.
        """
        body = textwrap.dedent(
            """
            import io
            import os
            import shutil
            from pathlib import Path

            try:
            {statement}
            except PermissionError as exc:
                print("REFUSED:", exc)
            else:
                print("ALLOWED")
            """
        ).format(statement=textwrap.indent(textwrap.dedent(statement).strip(), "    "))

        script = self.root / "probe_script.py"
        script.write_text(self.guard(execution_mode) + "\n" + body + "\n", encoding="utf-8")
        proc = subprocess.run(  # noqa: S603 - fixed argv, test-authored script
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(self.root),
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"probe failed rc={proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
        return proc.stdout.strip()


@pytest.fixture
def project(tmp_path: Path) -> _Project:
    """A repo root with a render zone, a profile source set and a state zone."""
    tmp = tmp_path.resolve()
    root = tmp / "repo"
    build = root / "build"
    agent_data = root / "var" / "agent_data"
    audit = root / "var" / "audit"
    personas = root / "personas"
    execution_folder = agent_data / "data" / "python_executions" / "run_0001"

    for directory in (build, agent_data, audit, personas, execution_folder):
        directory.mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text("project: demo\n", encoding="utf-8")

    # One pre-existing victim per zone, for the delete/overwrite forms.
    for zone in (build, agent_data, personas, audit):
        (zone / "victim.txt").write_text("original\n", encoding="utf-8")

    # Copy source lives outside the repo entirely: a copy's *source* is a read
    # and must never be what makes a case refuse.
    source_file = tmp / "source.txt"
    source_file.write_text("payload\n", encoding="utf-8")

    return _Project(
        root=root,
        build=build,
        agent_data=agent_data,
        audit=audit,
        personas=personas,
        execution_folder=execution_folder,
        source_file=source_file,
    )


# --------------------------------------------------------------------------
# The write forms, exercised against both zones
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _WriteForm:
    """One way to modify a file, and the shape of path it needs."""

    name: str
    #: ``{target}`` is the path literal; ``{source}`` the external copy source.
    statement: str
    #: "new" — a path that must not exist; "existing" — the zone's victim file.
    target: str


_WRITE_FORMS: tuple[_WriteForm, ...] = (
    _WriteForm("builtins.open", "open({target!r}, 'w').close()", "new"),
    _WriteForm("io.open", "io.open({target!r}, 'w').close()", "new"),
    # pathlib routes through io.open, not builtins.open — patching only the
    # builtin would leave this form wide open.
    _WriteForm("Path.write_text", "Path({target!r}).write_text('x')", "new"),
    _WriteForm("os.remove", "os.remove({target!r})", "existing"),
    _WriteForm("os.makedirs", "os.makedirs({target!r})", "new"),
    _WriteForm("shutil.copyfile", "shutil.copyfile({source!r}, {target!r})", "new"),
    # Planting a link is a write to the directory that receives it, even though
    # nothing is written *through* it. Refusing it is what stops a run from
    # aiming a new entry into the render zone.
    _WriteForm("os.symlink", "os.symlink({source!r}, {target!r})", "new"),
)


def _statement(form: _WriteForm, zone: Path, project: _Project) -> str:
    target = zone / "victim.txt" if form.target == "existing" else zone / f"new_{form.name}"
    return form.statement.format(target=str(target), source=str(project.source_file))


@pytest.mark.parametrize("form", _WRITE_FORMS, ids=lambda f: f.name)
def test_render_zone_writes_are_refused(project: _Project, form: _WriteForm):
    """Every write form is refused under ``build/`` — the render zone."""
    out = project.probe(_statement(form, project.build, project))
    assert out.startswith("REFUSED:"), out
    assert "build" in out


@pytest.mark.parametrize("form", _WRITE_FORMS, ids=lambda f: f.name)
def test_agent_data_writes_are_allowed(project: _Project, form: _WriteForm):
    """The same forms all succeed under ``var/agent_data`` — the agent's zone.

    This is the half that a guard which merely refused everything would also
    pass, which is why it is asserted form by form rather than once.
    """
    assert project.probe(_statement(form, project.agent_data, project)) == "ALLOWED"


@pytest.mark.parametrize("form", _WRITE_FORMS, ids=lambda f: f.name)
def test_render_zone_writes_are_refused_in_readwrite_too(project: _Project, form: _WriteForm):
    """Readwrite approval does not buy a write into the render zone.

    Only the wording differs, and the readwrite wording must not claim the run
    was readonly — the agent would read that as "resubmit as readwrite", which
    it already did.
    """
    out = project.probe(_statement(form, project.build, project), execution_mode="readwrite")
    assert out.startswith("REFUSED:"), out
    assert DEFAULT_DENYLIST_PREFIX in out
    assert READONLY_REFUSAL_MARKER not in out


def test_profile_yml_write_is_refused(project: _Project):
    """A relative ``profile.yml`` write resolves against cwd and is refused."""
    out = project.probe("open('profile.yml', 'w').close()")
    assert out.startswith("REFUSED:"), out
    assert "profile.yml" in out


def test_profile_yml_write_is_refused_in_readwrite(project: _Project):
    out = project.probe("open('profile.yml', 'w').close()", execution_mode="readwrite")
    assert out.startswith("REFUSED:"), out


def test_persona_write_is_refused(project: _Project):
    """Personas are the posture itself — a run that rewrites one re-grants it."""
    out = project.probe(f"open({str(project.personas / 'operator.md')!r}, 'w').close()")
    assert out.startswith("REFUSED:"), out


def test_audit_ledger_write_is_refused(project: _Project):
    """The refusal ledger is not the agent's to edit, state zone or not."""
    out = project.probe(f"os.remove({str(project.audit / 'victim.txt')!r})")
    assert out.startswith("REFUSED:"), out


def test_readonly_refusal_carries_the_audit_marker(project: _Project):
    """The readonly wording is what ``report_runtime_refusal`` matches on.

    Without the marker the write is still refused, but the operator alert and
    the audit record never happen — a silent failure of the reporting path, not
    of the guard.
    """
    out = project.probe(f"open({str(project.build / 'x.txt')!r}, 'w').close()")
    assert READONLY_REFUSAL_MARKER in out


def test_reads_under_a_protected_root_still_work(project: _Project):
    """A denylist refuses *writes*. Reading the render zone stays ordinary work."""
    out = project.probe(
        f"""
        _text = open({str(project.build / "victim.txt")!r}).read()
        assert _text == "original\\n", _text
        _text = Path({str(project.build / "victim.txt")!r}).read_text()
        assert _text == "original\\n", _text
        """
    )
    assert out == "ALLOWED"


def test_execution_folder_writes_are_allowed(project: _Project):
    """The run's own output folder is permitted even under a protected parent."""
    out = project.probe(f"open({str(project.execution_folder / 'artifact.json')!r}, 'w').close()")
    assert out == "ALLOWED"


def test_unrelated_paths_are_untouched(project: _Project):
    """Denylist, not allowlist: everything outside the protected set is fine."""
    out = project.probe(f"open({str(project.source_file.parent / 'scratch.txt')!r}, 'w').close()")
    assert out == "ALLOWED"


# --------------------------------------------------------------------------
# Where the guard sits in the generated script
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["readonly", "readwrite"])
def test_guard_is_emitted_unconditionally(project: _Project, mode: str):
    script = project.wrapper(mode).create_wrapper("x = 1", project.execution_folder)
    assert _GUARD_BANNER in script


def test_guard_is_independent_of_the_readonly_guard(project: _Project):
    """A readwrite run emits no readonly guard and the filesystem guard anyway."""
    wrapper = project.wrapper("readwrite")
    assert wrapper._get_readonly_guard() == ""
    assert _GUARD_BANNER in wrapper._get_filesystem_guard(project.execution_folder)


def test_guard_precedes_user_code_and_restores_after_it(project: _Project):
    """Install before the user code, restore in the finally that follows it."""
    script = project.wrapper().create_wrapper("x = 1", project.execution_folder)
    install = script.index("_install_patched_targets()")
    user_code = script.index("# Execute user code")
    restore = script.index("    _restore_patched_targets()")
    assert install < user_code < restore

    # The restore sits inside the finally block (4-space body indent), not at
    # module level: an early return path must not leave the patches installed.
    finally_at = script.index("\nfinally:")
    assert finally_at < restore


def test_metadata_persistence_runs_after_the_restore(project: _Project):
    """The wrapper's own record is written on unpatched entry points.

    Belt and braces with the permitted roots: if a protected root is ever
    misjudged, the operator still gets the execution metadata that says so.
    """
    script = project.wrapper().create_wrapper("x = 1", project.execution_folder)
    assert script.index("    _restore_patched_targets()") < script.index(
        "open('execution_metadata.json', 'w'"
    )


def test_roots_are_baked_in_not_re_derived(project: _Project):
    """The child gets literals. A child that re-derived could be misdirected."""
    guard = project.guard()
    assert repr(str(project.build)) in guard
    assert repr(str(project.execution_folder)) in guard
    for forbidden in ("resolve_project_root", "load_osprey_config", "import osprey"):
        assert forbidden not in guard


def test_every_widened_patch_target_is_installed(project: _Project):
    """The widened set, not just ``builtins.open``."""
    guard = project.guard()
    for target in EXECUTOR_PATCH_TARGETS:
        assert repr(target) in guard, target
    assert len(EXECUTOR_PATCH_TARGETS) == 19


def test_wrapper_without_roots_still_emits_the_guard():
    """A bare wrapper (no project layout known) refuses nothing but is wired."""
    guard = ExecutionWrapper()._get_filesystem_guard(None)
    assert _GUARD_BANNER in guard
    assert "_OSPREY_FS_PROTECTED = ()" in guard


# --------------------------------------------------------------------------
# The whole generated wrapper, in a real subprocess
# --------------------------------------------------------------------------


def _run_full_wrapper(project: _Project, user_code: str, mode: str = "readonly"):
    """Execute a complete generated wrapper the way the adapter does."""
    script = project.execution_folder / "wrapped_script.py"
    script.write_text(
        project.wrapper(mode).create_wrapper(user_code, project.execution_folder),
        encoding="utf-8",
    )

    env = os.environ.copy()
    # osprey must be importable in the child: the wrapper's persistence section
    # imports its serializer. In production the child inherits the deployment's
    # environment; here the test's own source tree stands in for it.
    src_root = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src_root, env.get("PYTHONPATH")]))

    return subprocess.run(  # noqa: S603 - fixed argv, generated script
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(project.root),
        env=env,
        timeout=300,
    )


def test_full_wrapper_writes_its_own_outputs_in_readonly(project: _Project):
    """Metadata, results and a user artifact all land with the guard installed."""
    artifact = project.execution_folder / "user_output.txt"
    proc = _run_full_wrapper(
        project,
        f"open({str(artifact)!r}, 'w').write('hello')\nresults = {{'value': 41}}",
    )

    assert proc.returncode == 0, proc.stderr
    assert artifact.read_text() == "hello"

    metadata = json.loads((project.execution_folder / "execution_metadata.json").read_text())
    assert metadata["success"] is True
    assert json.loads((project.execution_folder / "results.json").read_text()) == {"value": 41}


def test_full_wrapper_refuses_render_zone_write_and_reports_it(project: _Project):
    """The refusal reaches the parent's stderr, marker and all.

    That stderr string is the whole input ``report_runtime_refusal`` has, so
    this asserts on the exact surface the audit path reads.
    """
    proc = _run_full_wrapper(project, "open('build/config.yml', 'w').write('owned')")

    assert READONLY_REFUSAL_MARKER in proc.stderr
    assert not (project.build / "config.yml").exists()

    metadata = json.loads((project.execution_folder / "execution_metadata.json").read_text())
    assert metadata["success"] is False
    assert metadata["error_type"] == "PermissionError"


def test_full_wrapper_refuses_render_zone_write_in_readwrite_too(project: _Project):
    """The readwrite twin: same refusal, different wording, no audit marker.

    The probe-level tests already prove the refusal survives readwrite. This
    asserts the two things only the whole wrapper can show — that the readwrite
    wording is what actually reaches the parent's stderr, and that the run is
    recorded as a failed one — so that a mode-dependent regression in either
    the prefix or the metadata is caught rather than inferred.
    """
    proc = _run_full_wrapper(
        project, "open('build/config.yml', 'w').write('owned')", mode="readwrite"
    )

    assert DEFAULT_DENYLIST_PREFIX in proc.stderr
    # Not audited, on purpose: the marker is a factual claim about the run, and
    # this run was approved for writes. See ``_get_filesystem_guard``.
    assert READONLY_REFUSAL_MARKER not in proc.stderr
    assert not (project.build / "config.yml").exists()

    metadata = json.loads((project.execution_folder / "execution_metadata.json").read_text())
    assert metadata["success"] is False
    assert metadata["error_type"] == "PermissionError"


# --------------------------------------------------------------------------
# Parent-side root resolution
# --------------------------------------------------------------------------


def test_protected_roots_cover_render_zone_sources_and_ledger(project: _Project):
    protected = resolve_protected_roots(project.root)
    for expected in (
        project.root / "build",
        project.root / "var" / "audit",
        project.root / "profile.yml",
        project.root / "personas",
        project.root / "project",
        project.root / "skills",
    ):
        assert expected.resolve() in protected, expected


def test_protected_roots_exclude_the_agent_data_zone(project: _Project):
    protected = resolve_protected_roots(project.root)
    assert project.agent_data not in protected


def test_protected_roots_are_absolute_resolved_and_unique(project: _Project):
    protected = resolve_protected_roots(project.root)
    assert len(protected) == len(set(protected))
    assert all(path.is_absolute() and path == path.resolve() for path in protected)


def test_permitted_roots_default_to_the_agent_data_zone(project: _Project):
    assert resolve_permitted_roots(project.root, {}) == (project.agent_data,)


def test_permitted_roots_follow_a_relocated_data_root(project: _Project):
    """A project that moved its data root must keep it writable."""
    config = {"agent_data": {"base_dir": "state/agent"}}
    assert resolve_permitted_roots(project.root, config) == ((project.root / "state/agent"),)


def test_profile_source_entries_match_the_convention_table():
    """Pins the executor's local copy against the module that owns the table.

    The copy exists because ``osprey.cli.profile_conventions`` cannot be
    imported from the runtime path (it drags in the whole Click command group).
    A convention directory added there and forgotten here would be a silently
    writable hole, so the drift is a test failure instead.
    """
    from osprey.cli.profile_conventions import (  # noqa: PLC0415 - test-only import
        _SOURCE_ZONE_ENTRIES,
        CONVENTION_SOURCES,
    )

    canonical = set(CONVENTION_SOURCES) | set(_SOURCE_ZONE_ENTRIES)
    assert set(PROFILE_SOURCE_ENTRIES) == canonical
    assert len(PROFILE_SOURCE_ENTRIES) == len(set(PROFILE_SOURCE_ENTRIES))
