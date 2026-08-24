"""Tests for the container entrypoint that ships into every render.

``osprey build`` emits ``entrypoint.sh`` beside the Dockerfile, inside the
render, and the image installs it as the container's ENTRYPOINT. It runs as
root and does three things in one fixed order — drift-regen, scaffold restore,
then ``gosu`` down to the unprivileged ``osprey`` user — because every one of
them is a write into a render that this image makes root-owned, and the process
that serves requests must not hold the privilege to repeat them.

The order is the security property, so it is asserted twice: once on the text
of the shipped script, and once by actually running it against stub ``python``,
``gosu`` and ``id`` executables that record what was invoked and when. The
stubs make the run hermetic — no container, no docker, no real interpreter —
while still exercising the branch logic (root vs. not, gosu present vs. not,
maintenance succeeding vs. failing) that a textual assertion cannot reach.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from osprey.cli.templates.manager import TemplateManager

#: The shipped template, before it is copied into a render.
TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "project"
    / "entrypoint.sh"
)

#: The privilege drop, spelled exactly. ``exec`` so the served process is PID 1
#: and receives the orchestrator's signals directly; ``"$@"`` quoted so a CMD
#: argument containing a space survives.
_EXEC_FORM = re.compile(r'^[ \t]*exec gosu osprey "\$@"$', re.MULTILINE)


@pytest.fixture(scope="module")
def render(tmp_path_factory) -> Path:
    """A rendered deployment, so the assertions run on what a build emits."""
    return TemplateManager().create_project(
        project_name="entrypoint-demo",
        output_dir=tmp_path_factory.mktemp("entrypoint-render"),
        data_bundle="hello_world",
    )


@pytest.fixture(scope="module")
def script(render: Path) -> Path:
    return render / "entrypoint.sh"


@pytest.fixture(scope="module")
def text(script: Path) -> str:
    return script.read_text()


class TestShipsIntoTheRender:
    """The entrypoint is part of the deployment, not a sidecar beside it."""

    def test_lands_beside_config_yml(self, render: Path, script: Path):
        """In the render, next to the config it maintains and the Dockerfile
        that installs it — which is what puts it inside the tree the image
        copies in verbatim, with no second path for the build to remember."""
        assert script.is_file(), "entrypoint.sh missing from the render"
        assert (render / "config.yml").is_file()
        assert (render / "Dockerfile").is_file()

    def test_copied_verbatim(self, text: str):
        """Static copy, not a Jinja render: the script derives everything it
        needs from its own location, so there is no context key to substitute
        and none that can go missing."""
        assert text == TEMPLATE.read_text()

    def test_no_unrendered_jinja(self, text: str):
        assert "{{" not in text
        assert "{%" not in text

    def test_rendered_copy_is_executable(self, script: Path):
        """A JSON-array ENTRYPOINT execs the file directly — without the exec
        bit every container start dies with permission denied, and invoking
        the script as `sh script` in tests would never notice."""
        assert os.access(script, os.X_OK), "rendered entrypoint.sh lacks the exec bit"


class TestScriptShape:
    """Invariants of the shipped text."""

    def test_posix_syntax(self, script: Path):
        """The gate that catches a typo before an image is built around it."""
        result = subprocess.run(
            ["/bin/sh", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_posix_shebang_and_strict_mode(self, text: str):
        """``sh``, not ``bash``: this runs as PID 1 in a slim image that has no
        bash. ``set -eu`` so an unset variable or a failing step is not silently
        stepped over on the way to handing out a shell."""
        assert text.startswith("#!/bin/sh\n")
        assert re.search(r"^set -eu$", text, flags=re.MULTILINE)

    def test_regen_then_restore_then_drop(self, text: str):
        """The order that makes the render's root ownership meaningful."""
        # The call sites, not the prose: the header comment names both steps
        # while explaining the order, in whatever sequence reads best there.
        regen = text.index("regen_if_drift(render_dir)")
        restore = text.index("restore_scaffold_bodies(render_dir)")
        drop = _EXEC_FORM.search(text)
        assert drop is not None, 'no `exec gosu osprey "$@"` privilege drop'
        assert regen < restore < drop.start()

    def test_exec_form(self, text: str):
        assert len(_EXEC_FORM.findall(text)) == 1

    def test_regen_goes_through_regen_if_drift(self, text: str):
        """Not ``regenerate_claude_code`` directly. Only ``regen_if_drift``
        previews first and stamps ``settings.json`` when nothing changed, and
        that stamp is what keeps the SessionStart drift hook from warning at
        every session of an untouched deployment."""
        assert "regen_if_drift(render_dir)" in text
        assert "regenerate_claude_code(" not in text

    def test_restore_goes_through_the_shared_function(self, text: str):
        """A private reimplementation here would be a second gate to keep in
        step with the reserved-path refusal inside ``restore_scaffold_bodies``
        — and the copy running as root is the worst one to let drift."""
        assert "from osprey.interfaces.web_terminal.scaffold_gallery_service import" in text
        assert "restore_scaffold_bodies(render_dir)" in text

    def test_render_dir_derived_from_the_script_location(self, text: str):
        """No baked path: the same script has to be correct at any
        ``/app/<name>``, so the render it maintains is the directory it sits
        in."""
        assert re.search(r'^RENDER_DIR=.*dirname -- "\$0"', text, flags=re.MULTILINE)
        code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        assert not [line for line in code if "/app/" in line], "a container path is baked in"


# ── behaviour ────────────────────────────────────────────────────────────────


def _stub_bin(tmp_path: Path, *, uid: int, with_gosu: bool, python_rc: int = 0) -> Path:
    """A PATH holding only stubs, so a run touches nothing on the host.

    ``date`` and ``dirname`` are stubbed too, not because the test cares about
    them, but because a PATH with real directories on it would decide the
    "gosu is missing" case from whatever the host happens to have installed.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()

    def write(name: str, body: str) -> None:
        path = bindir / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)

    write("date", 'printf "1970-01-01T00:00:00Z\\n"\n')
    write(
        "dirname",
        'if [ "$1" = "--" ]; then shift; fi\n'
        'case "$1" in\n'
        '  */*) printf "%s\\n" "${1%/*}" ;;\n'
        '  *) printf ".\\n" ;;\n'
        "esac\n",
    )
    write("id", f'if [ "$1" = "-u" ]; then printf "{uid}\\n"; fi\nexit 0\n')
    # Records the invocation and captures the program it was handed on stdin,
    # so the regen-before-restore order can be checked on what actually ran
    # rather than only on the script's source.
    # stdin is drained with shell builtins rather than `cat`, because the PATH
    # this builds is the only one the run has and nothing else is on it.
    write(
        "python",
        'printf "python\\n" >> "$ORDER_LOG"\n'
        'while IFS= read -r line; do printf "%s\\n" "$line"; done > "$PY_PROGRAM"\n'
        f"exit {python_rc}\n",
    )
    # The hand-back is a `find ... -exec chown`, so both halves are stubbed:
    # `find` records the tree it was pointed at plus its full predicate (to
    # "$ORDER_LOG.find", so the run signature stays a 3-tuple) and then stands
    # in for the -exec by invoking chown once on that tree; `chown` records the
    # last path it was handed. Recording both makes the ORDER relative to the
    # drop observable — a hand-back after `exec gosu` would never run at all.
    write(
        "find",
        'target="$1"\n'
        'printf "find %s\\n" "$target" >> "$ORDER_LOG"\n'
        'printf "%s\\n" "$*" > "$ORDER_LOG.find"\n'
        'case "$*" in\n'
        '  *-exec*) chown osprey:osprey "$target" ;;\n'
        "esac\n"
        "exit 0\n",
    )
    write(
        "chown",
        'for a in "$@"; do last="$a"; done\nprintf "chown %s\\n" "$last" >> "$ORDER_LOG"\nexit 0\n',
    )
    if with_gosu:
        write("gosu", 'printf "gosu %s\\n" "$1" >> "$ORDER_LOG"\nshift\nexec "$@"\n')
    return bindir


def _run(script: Path, tmp_path: Path, bindir: Path, *args: str):
    order_log = tmp_path / "order.log"
    py_program = tmp_path / "program.py"
    order_log.touch()
    result = subprocess.run(
        ["/bin/sh", str(script), *args],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": str(bindir),
            "ORDER_LOG": str(order_log),
            "PY_PROGRAM": str(py_program),
        },
    )
    return result, order_log.read_text().split(), py_program


class TestBehaviour:
    """What the script does when it runs, not only what it says."""

    def _cmd(self, tmp_path: Path) -> list[str]:
        """A CMD stand-in that proves it was reached. Absolute, because the
        stub PATH deliberately cannot resolve anything else."""
        return ["/bin/sh", "-c", f'printf "cmd\\n" >> "{tmp_path / "order.log"}"']

    def test_root_runs_maintenance_then_drops(self, script: Path, tmp_path: Path):
        bindir = _stub_bin(tmp_path, uid=0, with_gosu=True)
        result, order, py_program = _run(script, tmp_path, bindir, *self._cmd(tmp_path))

        assert result.returncode == 0, result.stderr
        assert order == ["python", "gosu", "osprey", "cmd"], result.stderr

        # The maintenance program itself sequences regen ahead of restore.
        program = py_program.read_text()
        assert program.index("regen_if_drift") < program.index("restore_scaffold_bodies")

    def test_the_command_owns_stdout(self, script: Path, tmp_path: Path):
        """Diagnostics go to stderr, so the command keeps its own stdout.

        This script runs in front of whatever the image was given, and that
        output is read by things that cannot tolerate a preamble: `docker run
        <image> whoami` has to print `osprey` alone, and a package probe, a
        version query or a JSON payload all break on a prepended progress line.
        """
        bindir = _stub_bin(tmp_path, uid=0, with_gosu=True)
        cmd = ["/bin/sh", "-c", 'printf "only-the-command\\n"']
        result, _, _ = _run(script, tmp_path, bindir, *cmd)

        assert result.returncode == 0, result.stderr
        assert result.stdout == "only-the-command\n", (
            f"the entrypoint wrote into the command's stdout: {result.stdout!r}"
        )
        assert "[osprey-entrypoint]" in result.stderr, (
            f"the diagnostics reached neither stream: {result.stderr!r}"
        )

    def test_maintenance_failure_still_drops(self, script: Path, tmp_path: Path):
        """Fail open. A container that will not boot because an artifact could
        not be re-rendered is worse than one running stale artifacts loudly."""
        bindir = _stub_bin(tmp_path, uid=0, with_gosu=True, python_rc=1)
        result, order, _ = _run(script, tmp_path, bindir, *self._cmd(tmp_path))

        assert result.returncode == 0, result.stderr
        assert order == ["python", "gosu", "osprey", "cmd"]
        assert "WARNING" in result.stderr

    def test_missing_gosu_is_fatal(self, script: Path, tmp_path: Path):
        """Fail closed, and before the maintenance step: the only alternatives
        are refusing to start and running the agent as root."""
        bindir = _stub_bin(tmp_path, uid=0, with_gosu=False)
        result, order, _ = _run(script, tmp_path, bindir, *self._cmd(tmp_path))

        assert result.returncode != 0
        assert "gosu" in result.stderr
        assert order == [], "maintenance ran for a container that cannot start"

    def test_already_unprivileged_skips_maintenance(self, script: Path, tmp_path: Path):
        """Run with ``--user``: neither write can succeed against a root-owned
        render, so run the command and say what was skipped."""
        bindir = _stub_bin(tmp_path, uid=1000, with_gosu=True)
        result, order, _ = _run(script, tmp_path, bindir, *self._cmd(tmp_path))

        assert result.returncode == 0, result.stderr
        assert order == ["cmd"]
        assert "WARNING" in result.stderr

    def test_no_command_is_fatal(self, script: Path, tmp_path: Path):
        """An entrypoint with nothing to exec has no useful thing to do, and
        `gosu osprey` with no command would hand out a root-started shell."""
        bindir = _stub_bin(tmp_path, uid=0, with_gosu=True)
        result, order, _ = _run(script, tmp_path, bindir)

        assert result.returncode != 0
        assert order == []


class TestStateZoneHandBack:
    """Root writes into ``var/`` on the way through, and must not keep it.

    Both maintenance steps write there: the scaffold restore appends its
    reserved-path refusals to ``var/audit/protected-writes.jsonl``, and on a
    fresh deployment root is the FIRST writer, so the file is created root-owned
    0644. The server then runs as ``osprey`` and cannot append to it — and the
    refusal recorder never raises, so every refusal the running app makes after
    that is dropped in silence. An audit log only root can write is worse than
    no audit log, because it looks like one.
    """

    def _cmd(self, tmp_path: Path) -> list[str]:
        marker = tmp_path / "ran"
        return ["/bin/sh", "-c", f"printf ran > {marker}"]

    @pytest.fixture()
    def state_zone(self, script: Path) -> Path:
        """The zone as a build leaves it: created, beside the render."""
        zone = script.parent.parent / "var"
        (zone / "audit").mkdir(parents=True, exist_ok=True)
        return zone

    def test_the_state_zone_is_handed_back_before_the_drop(
        self, script: Path, tmp_path: Path, state_zone: Path
    ):
        """Order is the property: after ``exec gosu`` nothing else runs."""
        bindir = _stub_bin(tmp_path, uid=0, with_gosu=True)

        result, order, _ = _run(script, tmp_path, bindir, *self._cmd(tmp_path))

        assert result.returncode == 0, result.stderr
        assert "chown" in order, f"the state zone was never handed back: {order}\n{result.stderr}"
        assert order.index("chown") < order.index("gosu"), (
            f"the hand-back runs after the privilege drop, so it never runs: {order}"
        )

    def test_the_hand_back_targets_the_state_zone_beside_the_render(
        self, script: Path, tmp_path: Path, state_zone: Path
    ):
        """``var/`` is the render's SIBLING, not a directory inside it.

        The render is ``<repo>/build`` and every runtime reader resolves the
        state zone one level up from it. A hand-back aimed inside the render
        would chown part of the tree whose whole point is that the agent's user
        cannot write it.
        """
        bindir = _stub_bin(tmp_path, uid=0, with_gosu=True)

        _, order, _ = _run(script, tmp_path, bindir, *self._cmd(tmp_path))

        target = order[order.index("find") + 1]
        assert target.endswith("/var"), target
        assert not target.startswith(str(script.parent) + "/"), (
            f"the hand-back reaches into the render zone: {target}"
        )
        assert target == f"{script.parent.parent}/var"

    def test_the_hand_back_touches_only_what_root_left_behind(
        self, script: Path, tmp_path: Path, state_zone: Path
    ):
        """Not ``chown -R``: an operator's storage under ``var/`` keeps its owner.

        A recursive chown would rewrite a bind-mounted dataset on every single
        start, and deliberate foreign ownership down there is a choice rather
        than damage. Only the paths root actually left behind are wrong, and
        ``! -user osprey`` is exactly that set.
        """
        bindir = _stub_bin(tmp_path, uid=0, with_gosu=True)

        _run(script, tmp_path, bindir, *self._cmd(tmp_path))

        predicate = (tmp_path / "order.log.find").read_text()
        assert "! -user osprey" in predicate, predicate
        assert "-exec chown osprey:osprey" in predicate, predicate

    def test_a_non_root_start_hands_nothing_back(
        self, script: Path, tmp_path: Path, state_zone: Path
    ):
        """Nothing was written as root, so there is nothing to give back — and
        a non-root process could not chown it anyway."""
        bindir = _stub_bin(tmp_path, uid=1000, with_gosu=True)

        _, order, _ = _run(script, tmp_path, bindir, *self._cmd(tmp_path))

        assert "chown" not in order, order
        assert "find" not in order, order

    def test_a_deployment_with_no_state_zone_still_starts(self, script: Path, tmp_path: Path):
        """No ``var/`` to hand back is not an error — the drop still happens.

        Run against a COPY in a bare tree rather than the shared render: the
        script derives everything from its own location, and the module-scoped
        render acquires a ``var/`` as soon as any other test here asks for one.
        """
        bare = tmp_path / "repo" / "build"
        bare.mkdir(parents=True)
        copy = bare / "entrypoint.sh"
        copy.write_text(script.read_text())
        copy.chmod(0o755)
        bindir = _stub_bin(tmp_path, uid=0, with_gosu=True)

        result, order, _ = _run(copy, tmp_path, bindir, *self._cmd(tmp_path))

        assert result.returncode == 0, result.stderr
        assert "find" not in order, order
        assert "gosu" in order, order
