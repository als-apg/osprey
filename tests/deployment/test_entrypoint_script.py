"""The entrypoint's root phase: mounted-group membership and the writer marker.

``entrypoint.sh`` runs as root and drops to ``osprey`` with ``gosu``. Two
properties of that root phase are load-bearing for the audit trail and are
asserted here:

**The group join.** ``gosu`` re-derives the dropped process's supplementary
groups from the image's ``/etc/group``, so a group granted at runtime (compose
``group_add:``) reaches the container's initial process and is discarded at the
drop. The bind-mounted audit subdir and facility bundle are owned by a HOST
group, and the dropped process reaches them through membership rather than
ownership — so the group has to be in ``/etc/group`` and ``osprey`` has to be
in it before ``exec gosu``. The set of directories is named by the render
(``OSPREY_AUDIT_DIR``, ``OSPREY_FACILITY_BUNDLE_DIR``, ``OSPREY_ARIEL_MIRROR_DIR``)
and the gid is stat'd off the mount, never passed in. Root and system gids are refused, because a mount
that presents one must not hand the agent's user a privileged group. And a
join is not claimed until it is VERIFIED: membership is the mechanism, while
write access to the mount is the property the audit trail depends on, so the
success line is emitted only after ``gosu osprey test -w`` says so.

**The writer marker.** ``OSPREY_AUDIT_WRITER=maintenance`` is a per-command
prefix on the maintenance interpreter, never a shell ``export`` — an export
would survive ``exec gosu`` (which preserves the environment) and stamp every
app-side record with the root phase's writer.

Everything here is script logic, so nothing is Linux-gated and nothing needs a
container: the run happens against stub ``stat``/``getent``/``groupadd``/
``usermod``/``gosu`` executables on a PATH that holds nothing else, with
fabricated gids in a stub's lookup table rather than real ``chown``ed
directories (which would need root). ``tests/cli/test_entrypoint_script.py``
owns the complementary assertions — that this script ships into the render
verbatim, and the regen/restore/hand-back order.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

#: The shipped template, which is what a build copies into the render.
TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "project"
    / "entrypoint.sh"
)

#: The variables the render names, and the only ones the join may read.
AUDIT_VAR = "OSPREY_AUDIT_DIR"
BUNDLE_VAR = "OSPREY_FACILITY_BUNDLE_DIR"
MIRROR_VAR = "OSPREY_ARIEL_MIRROR_DIR"
RENDER_NAMED_VARS = frozenset({AUDIT_VAR, BUNDLE_VAR, MIRROR_VAR})

#: The marker that routes the root phase's records to their own file.
WRITER_VAR = "OSPREY_AUDIT_WRITER"

#: Stand-in the stubs print when the variable is not in their environment, so
#: "absent" and "empty" are distinguishable in an assertion.
UNSET = "<unset>"

#: The host's real ``find``, which the ``find`` stub re-runs with the ``-exec``
#: action replaced by ``-print`` so the prune EXPRESSION is evaluated rather
#: than merely recorded. Resolved before PATH is replaced by the stub dir.
REAL_FIND = shutil.which("find") or ""

#: Separator in the ``stat`` stub's gid table. A tab rather than a space
#: because mount paths in these tests deliberately contain spaces.
GID_MAP_SEP = "\t"


@pytest.fixture(scope="module")
def text() -> str:
    return TEMPLATE.read_text()


# ── the shipped text ─────────────────────────────────────────────────────────


class TestScriptShape:
    """Invariants that are true of the source, container or no container."""

    def test_posix_syntax(self):
        """``sh -n`` is the gate that catches a typo before an image is built
        around it. This runs as PID 1 in a slim image with no bash, so a
        bashism is a boot failure, not a style problem."""
        result = subprocess.run(
            ["/bin/sh", "-n", str(TEMPLATE)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_no_bashisms_in_the_added_step(self, text: str):
        """``sh -n`` under a shell that happens to be bash would accept several
        of these, so they are named explicitly."""
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        for bashism in ("[[", "declare ", "local ", "function ", "+=("):
            assert bashism not in code, f"bashism in a POSIX sh script: {bashism!r}"

    def test_the_marker_is_a_per_command_prefix(self, text: str):
        """On the interpreter invocation itself, so only that command carries
        it. This is the spelling the whole one-uid-per-file invariant rests on."""
        assert re.search(
            r"^\s*" + WRITER_VAR + r"=maintenance \"\$PYTHON\" - \"\$RENDER_DIR\" <<'PY'$",
            text,
            flags=re.MULTILINE,
        ), "the maintenance heredoc does not carry the writer marker as a per-command prefix"

    def test_the_marker_is_never_exported(self, text: str):
        """An ``export`` sets it for the whole shell, ``run_maintenance`` runs
        in that same shell, and ``gosu`` preserves the environment across the
        drop — so an exported marker would stamp every record the app writes
        afterwards with the root phase's writer."""
        assert f"export {WRITER_VAR}" not in text
        assert not re.search(r"^\s*export\b.*" + WRITER_VAR, text, flags=re.MULTILINE)
        assert len(re.findall(WRITER_VAR + "=", text)) == 1, (
            "more than one assignment of the writer marker"
        )

    def test_the_marker_is_unset_before_the_drop(self, text: str):
        """Belt to the prefix's brace, and it has to come first in the file to
        come first in the run — after ``exec gosu`` nothing runs at all."""
        unset = text.index(f"unset {WRITER_VAR}")
        drop = text.index('exec gosu osprey "$@"')
        assert unset < drop, "the marker is unset after the privilege drop, i.e. never"

    def test_the_join_precedes_the_drop(self, text: str):
        join = text.index("\n    join_mounted_groups")
        drop = text.index('exec gosu osprey "$@"')
        assert join < drop, "the group join runs after gosu, where it grants nothing"

    def test_the_join_reads_only_the_render_named_variables(self, text: str):
        """Named by the render, never guessed: this script reads no config.

        Scanned over the WHOLE mounted-group section — both the dispatcher and
        ``join_mounted_group``, which is where every read that decides what
        gets stat'd and joined actually lives. A slice of the dispatcher alone
        would let a fourth variable steer the join from inside the worker.
        The set is a deliberate allowlist: widening it means adding exactly
        one name here, never loosening the scan."""
        start = text.index("join_mounted_group() {")
        end = text.index("# \u2500\u2500 state-zone hand-back")
        section = text[start:end]
        assert "join_mounted_groups() {" in section, "the slice lost the dispatcher"
        code = "\n".join(line for line in section.splitlines() if not line.lstrip().startswith("#"))
        read = set(re.findall(r"\bOSPREY_[A-Z_]+\b", code))
        assert RENDER_NAMED_VARS <= read, f"a render-named mount is never joined: {read}"
        others = read - RENDER_NAMED_VARS
        assert not others, f"the join reads variables the render does not name: {others}"

    def test_the_gid_is_stat_ed_off_the_mount(self, text: str):
        """Not passed in. A gid rendered into the compose file is the host's
        gid at render time; the mount's is the only one true at start."""
        assert "stat -c '%g'" in text
        assert not re.search(r"OSPREY_[A-Z_]*GID", text), "a gid is being taken from the env"

    def test_the_sweep_excludes_the_audit_mount(self, text: str):
        """The one deliberate exception to the ownership hand-back."""
        assert re.search(
            r'find "\$STATE_DIR" "\$@" ! -user osprey -exec chown osprey:osprey \{\} \+',
            text,
        ), "the hand-back no longer applies its optional prune expression"
        assert f'set -- -path "${AUDIT_VAR}" -prune -o' in text

    def test_the_exclusion_says_why(self, text: str):
        """A future reader deleting the prune as dead weight is the failure
        mode, so the comment names the mount and the reason by name."""
        comment = "\n".join(line for line in text.splitlines() if line.lstrip().startswith("#"))
        assert AUDIT_VAR in comment
        assert "membership" in comment.lower()
        assert "ownership" in comment.lower()


# ── the running script ───────────────────────────────────────────────────────


@dataclass
class Run:
    returncode: int
    stderr: str
    order: list[str]
    group_ops: list[str]
    find_args: str
    #: Paths the REAL ``find`` reached when the sweep's own expression was
    #: re-run with ``-print`` in place of the ``-exec`` action: the prune as a
    #: traversal outcome rather than as argv spelling.
    visited: list[str]
    maintenance_marker: str
    drop_marker: str


class Sandbox:
    """A rendered-shaped tree plus a PATH that holds only stubs.

    Nothing here touches the host: the gids are entries in a lookup table the
    ``stat`` stub reads, and ``/etc/group`` is a file the ``getent`` and
    ``groupadd`` stubs share. That is what keeps a test about root-only
    mechanics runnable in the ordinary unit lane, on any platform.
    """

    def __init__(self, tmp_path: Path):
        # Resolved, because the script derives STATE_DIR through `pwd -P`
        # while OSPREY_AUDIT_DIR is passed in as given. On macOS pytest hands
        # out /var/folders/... (a symlink to /private/var/...), and an
        # unresolved sandbox would compare a prune pattern against a path that
        # spells the same directory differently — which is exactly the state
        # in which the prune looks right and traverses the audit mount anyway.
        self.tmp = tmp_path.resolve()
        tmp_path = self.tmp
        self.repo = tmp_path / "repo"
        self.render = self.repo / "build"
        self.render.mkdir(parents=True)
        self.script = self.render / "entrypoint.sh"
        self.script.write_text(TEMPLATE.read_text())
        self.script.chmod(0o755)
        self.state = self.repo / "var"
        (self.state / "audit").mkdir(parents=True)

        self.gid_map = tmp_path / "gid-map"
        self.gid_map.write_text("")
        self.group_db = tmp_path / "group-db"
        self.group_db.write_text("root:x:0:\nosprey:x:1000:\n")

    # ── mounts ──────────────────────────────────────────────────────────────

    def audit_mount(self, gid: int, identity: str = "alice") -> Path:
        """The per-identity audit subdir, where compose binds it: under the
        state zone, so the hand-back sweep would reach it but for the prune."""
        return self._mount(self.state / "audit" / identity, gid)

    def bundle_mount(self, gid: int, name: str = "bundle") -> Path:
        """The facility bundle. Its host path is ordinary operator config
        (``facility_knowledge.bundle_path``) and passes through no charset
        normalizer, so it is the mount most likely to carry a space or a glob
        character — and the one these tests use to pin quoting."""
        return self._mount(self.tmp / name, gid)

    def mirror_mount(self, gid: int, name: str = "mirror") -> Path:
        """The deployment's ARIEL qmd mirror: a host-group mount for exactly
        the reason the bundle is, written by this container's own exporter
        after the drop and indexed by the sidecar on the host."""
        return self._mount(self.tmp / name, gid)

    def _mount(self, path: Path, gid: int) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        with self.gid_map.open("a") as handle:
            handle.write(f"{path}{GID_MAP_SEP}{gid}\n")
        return path

    def existing_group(self, name: str, gid: int) -> None:
        with self.group_db.open("a") as handle:
            handle.write(f"{name}:x:{gid}:\n")

    # ── the run ─────────────────────────────────────────────────────────────

    def run(
        self,
        *,
        uid: int = 0,
        audit_dir: Path | str | None = None,
        bundle_dir: Path | str | None = None,
        mirror_dir: Path | str | None = None,
        groupadd_rc: int = 0,
        usermod_rc: int = 0,
        stat_rc: int | None = None,
        stat_out: str | None = None,
        probe_rc: int = 0,
    ) -> Run:
        bindir = self._stubs(
            uid=uid,
            groupadd_rc=groupadd_rc,
            usermod_rc=usermod_rc,
            stat_rc=stat_rc,
            stat_out=stat_out,
            probe_rc=probe_rc,
        )
        order_log = self.tmp / "order.log"
        group_log = self.tmp / "group.log"
        find_args = self.tmp / "find.args"
        find_visited = self.tmp / "find.visited"
        py_env = self.tmp / "py.env"
        drop_env = self.tmp / "drop.env"
        for path in (order_log, group_log, find_args, find_visited, py_env, drop_env):
            path.write_text("")

        env = {
            "PATH": str(bindir),
            "ORDER_LOG": str(order_log),
            "GROUP_LOG": str(group_log),
            "FIND_ARGS": str(find_args),
            "FIND_VISITED": str(find_visited),
            "REAL_FIND": REAL_FIND,
            "PY_ENV": str(py_env),
            "DROP_ENV": str(drop_env),
            "GID_MAP": str(self.gid_map),
            "GROUP_DB": str(self.group_db),
        }
        if audit_dir is not None:
            env[AUDIT_VAR] = str(audit_dir)
        if bundle_dir is not None:
            env[BUNDLE_VAR] = str(bundle_dir)
        if mirror_dir is not None:
            env[MIRROR_VAR] = str(mirror_dir)

        command = ["/bin/sh", "-c", f'printf "cmd\\n" >> "{order_log}"']
        result = subprocess.run(
            ["/bin/sh", str(self.script), *command],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        return Run(
            returncode=result.returncode,
            stderr=result.stderr,
            order=order_log.read_text().split(),
            group_ops=group_log.read_text().splitlines(),
            find_args=find_args.read_text(),
            visited=find_visited.read_text().splitlines(),
            maintenance_marker=py_env.read_text().strip() or UNSET,
            drop_marker=drop_env.read_text().strip() or UNSET,
        )

    def _stubs(
        self,
        *,
        uid: int,
        groupadd_rc: int,
        usermod_rc: int,
        stat_rc: int | None,
        stat_out: str | None,
        probe_rc: int,
    ) -> Path:
        bindir = self.tmp / "bin"
        bindir.mkdir(exist_ok=True)

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
        write(
            "python",
            'printf "python\\n" >> "$ORDER_LOG"\n'
            f'printf "%s\\n" "${{{WRITER_VAR}-{UNSET}}}" > "$PY_ENV"\n'
            "while IFS= read -r line; do :; done\n"
            "exit 0\n",
        )
        # `find <state> [-path <audit> -prune -o] ! -user osprey -exec chown ...`
        # The argv is recorded, and then the SAME expression is re-run through
        # the host's real find with everything from the ownership predicate on
        # replaced by `-print`: the prune is evaluated, so a prune that is
        # spelled right and matches nothing (an unresolved path, a glob
        # character in the audit path) shows up as a visited audit mount
        # rather than as a green test. The ownership predicate cannot come
        # along — `! -user osprey` needs an `osprey` account, which no test
        # host has.
        write(
            "find",
            'target="$1"\n'
            'printf "find\\n" >> "$ORDER_LOG"\n'
            'printf "%s\\n" "$*" > "$FIND_ARGS"\n'
            'case "$*" in\n'
            '  *-exec*) chown osprey:osprey "$target" ;;\n'
            "esac\n"
            '[ -n "$REAL_FIND" ] || exit 0\n'
            "n=$#\n"
            "i=0\n"
            'while [ "$i" -lt "$n" ]; do\n'
            "  a=$1\n"
            "  shift\n"
            "  i=$((i + 1))\n"
            '  if [ "$a" = "!" ]; then\n'
            '    while [ "$i" -lt "$n" ]; do\n'
            "      shift\n"
            "      i=$((i + 1))\n"
            "    done\n"
            "    break\n"
            "  fi\n"
            '  set -- "$@" "$a"\n'
            "done\n"
            '"$REAL_FIND" "$@" -print > "$FIND_VISITED" 2> /dev/null\n'
            "exit 0\n",
        )
        write("chown", 'printf "chown\\n" >> "$ORDER_LOG"\nexit 0\n')
        # Two callers, and they must stay distinguishable. `gosu osprey test
        # -w <dir>` is the join's post-condition probe, answered from a knob
        # because really switching users needs root; anything else is the
        # privilege drop at the end of main().
        write(
            "gosu",
            'if [ "$2" = "test" ]; then\n'
            '  printf "probe %s %s %s\\n" "$1" "$3" "$4" >> "$GROUP_LOG"\n'
            f"  exit {probe_rc}\n"
            "fi\n"
            'printf "gosu %s\\n" "$1" >> "$ORDER_LOG"\n'
            f'printf "%s\\n" "${{{WRITER_VAR}-{UNSET}}}" > "$DROP_ENV"\n'
            'shift\nexec "$@"\n',
        )
        # `stat -c %g <dir>`: the fabricated gid table, so no test needs root.
        # TAB-separated, because the paths in it deliberately contain spaces.
        if stat_rc is not None:
            stat_body = f"exit {stat_rc}\n"
        elif stat_out is not None:
            stat_body = f"printf '%s\\n' {shlex.quote(stat_out)}\nexit 0\n"
        else:
            stat_body = (
                'for a in "$@"; do target="$a"; done\n'
                "while IFS='" + GID_MAP_SEP + "' read -r path gid; do\n"
                '  if [ "$path" = "$target" ]; then printf "%s\\n" "$gid"; exit 0; fi\n'
                'done < "$GID_MAP"\n'
                "exit 1\n"
            )
        write("stat", stat_body)
        # `getent group <gid>`: the shared /etc/group stand-in.
        write(
            "getent",
            'printf "getent %s\\n" "$2" >> "$GROUP_LOG"\n'
            "while IFS=: read -r name pw gid rest; do\n"
            '  if [ "$gid" = "$2" ]; then printf "%s:%s:%s:%s\\n" "$name" "$pw" "$gid" "$rest"; exit 0; fi\n'
            'done < "$GROUP_DB"\n'
            "exit 2\n",
        )
        # `groupadd --gid <gid> <name>`
        write(
            "groupadd",
            'printf "groupadd %s %s\\n" "$2" "$3" >> "$GROUP_LOG"\n'
            'printf "groupadd\\n" >> "$ORDER_LOG"\n'
            + ('printf "%s:x:%s:\\n" "$3" "$2" >> "$GROUP_DB"\n' if groupadd_rc == 0 else "")
            + f"exit {groupadd_rc}\n",
        )
        # `usermod --append --groups <group> <user>`
        write(
            "usermod",
            'printf "usermod %s %s\\n" "$3" "$4" >> "$GROUP_LOG"\n'
            'printf "usermod\\n" >> "$ORDER_LOG"\n'
            f"exit {usermod_rc}\n",
        )
        return bindir


@pytest.fixture()
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def _groupadds(run: Run) -> list[str]:
    return [op for op in run.group_ops if op.startswith("groupadd ")]


def _usermods(run: Run) -> list[str]:
    return [op for op in run.group_ops if op.startswith("usermod ")]


def _probes(run: Run) -> list[str]:
    return [op for op in run.group_ops if op.startswith("probe ")]


def _warnings(run: Run) -> list[str]:
    return [line for line in run.stderr.splitlines() if "WARNING:" in line]


class TestTheGroupJoin:
    """``osprey`` gets the mount's group before ``gosu``, or not at all."""

    def test_the_audit_mount_group_is_created_and_joined(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit)

        assert run.returncode == 0, run.stderr
        assert _groupadds(run) == ["groupadd 3000 osprey-mount-3000"]
        assert _usermods(run) == ["usermod osprey-mount-3000 osprey"]

    def test_the_join_happens_before_the_privilege_drop(self, sandbox: Sandbox):
        """After ``exec gosu`` nothing runs, and gosu has already read
        ``/etc/group`` — a join afterwards would grant the served process
        nothing at all."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit)

        assert "usermod" in run.order, run.order
        assert run.order.index("usermod") < run.order.index("gosu"), run.order

    def test_an_existing_group_entry_is_reused(self, sandbox: Sandbox):
        """The image may already know the gid — a second boot of the same
        container certainly does. Re-running must add membership, not fail."""
        audit = sandbox.audit_mount(gid=3000)
        sandbox.existing_group("hostshared", 3000)

        run = sandbox.run(audit_dir=audit)

        assert run.returncode == 0, run.stderr
        assert _groupadds(run) == [], "a group was created for a gid that already had one"
        assert _usermods(run) == ["usermod hostshared osprey"]

    def test_the_bundle_mount_is_joined_too(self, sandbox: Sandbox):
        """The step is generalized: the facility bundle is a host-group mount
        for exactly the same reason the audit subdir is."""
        audit = sandbox.audit_mount(gid=3000)
        bundle = sandbox.bundle_mount(gid=4000)

        run = sandbox.run(audit_dir=audit, bundle_dir=bundle)

        assert run.returncode == 0, run.stderr
        assert _groupadds(run) == [
            "groupadd 3000 osprey-mount-3000",
            "groupadd 4000 osprey-mount-4000",
        ]
        assert _usermods(run) == [
            "usermod osprey-mount-3000 osprey",
            "usermod osprey-mount-4000 osprey",
        ]

    def test_one_entry_per_distinct_gid(self, sandbox: Sandbox):
        """The audit subdir and the bundle usually share the host group; the
        second one must not be joined twice."""
        audit = sandbox.audit_mount(gid=3000)
        bundle = sandbox.bundle_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, bundle_dir=bundle)

        assert run.returncode == 0, run.stderr
        assert _groupadds(run) == ["groupadd 3000 osprey-mount-3000"]
        assert _usermods(run) == ["usermod osprey-mount-3000 osprey"]

    def test_the_mirror_mount_is_joined_too(self, sandbox: Sandbox):
        """The ARIEL mirror is the third host-group mount: the exporter that
        writes it runs as the DROPPED user, so a `group_add:` alone (which
        gosu discards) left every entry enhanced from a terminal unwritable
        or root-owned on the host."""
        audit = sandbox.audit_mount(gid=3000)
        mirror = sandbox.mirror_mount(gid=5000)

        run = sandbox.run(audit_dir=audit, mirror_dir=mirror)

        assert run.returncode == 0, run.stderr
        assert _groupadds(run) == [
            "groupadd 3000 osprey-mount-3000",
            "groupadd 5000 osprey-mount-5000",
        ]
        assert _usermods(run) == [
            "usermod osprey-mount-3000 osprey",
            "usermod osprey-mount-5000 osprey",
        ]
        assert f"for {MIRROR_VAR}={mirror}" in run.stderr, run.stderr

    def test_mirror_and_bundle_on_one_gid_join_once(self, sandbox: Sandbox):
        """The deploy provisions both shared directories as the same user, so
        they usually carry one group; the dedup covers the third mount too."""
        audit = sandbox.audit_mount(gid=3000)
        bundle = sandbox.bundle_mount(gid=4000)
        mirror = sandbox.mirror_mount(gid=4000)

        run = sandbox.run(audit_dir=audit, bundle_dir=bundle, mirror_dir=mirror)

        assert run.returncode == 0, run.stderr
        assert _groupadds(run) == [
            "groupadd 3000 osprey-mount-3000",
            "groupadd 4000 osprey-mount-4000",
        ]
        assert _usermods(run) == [
            "usermod osprey-mount-3000 osprey",
            "usermod osprey-mount-4000 osprey",
        ]


class TestTheJoinIsVerifiedNotAssumed:
    """A join is claimed only once the write it exists for is known to work."""

    def test_the_probe_runs_after_the_usermod_and_asks_as_osprey(self, sandbox: Sandbox):
        """Membership is the mechanism; write access to the mount is the
        property the audit trail depends on, and they are not the same thing —
        a 2700 directory, a read-only bind and an id-mapped mount all accept
        the usermod and refuse the write. ``gosu`` re-reads ``/etc/group``, so
        the probe sees the membership just granted."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit)

        assert run.group_ops == [
            "getent 3000",
            "groupadd 3000 osprey-mount-3000",
            "usermod osprey-mount-3000 osprey",
            f"probe osprey -w {audit}",
        ], run.group_ops

    def test_a_verified_join_says_joined(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit)

        assert (
            f"joined osprey to group osprey-mount-3000 (gid 3000) for {AUDIT_VAR}={audit}"
            in run.stderr
        ), run.stderr
        assert _warnings(run) == [], run.stderr

    def test_membership_without_write_access_warns_instead_of_claiming_success(
        self, sandbox: Sandbox
    ):
        """The reachable steady state this catches: a container that logs a
        success line at boot, serves normally, and writes no audit record at
        all — because every layer below this one (the join, the sweep, the
        envelope writer) also degrades to a log line."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, probe_rc=1)

        assert run.returncode == 0, run.stderr
        assert "joined osprey" not in run.stderr, "success was claimed for an unwritable mount"
        warnings = _warnings(run)
        assert len(warnings) == 1, warnings
        assert "cannot write" in warnings[0]
        assert str(audit) in run.stderr, "the warning does not name the directory"
        assert "gosu" in run.order, "a failed probe blocked the container's boot"

    def test_a_failed_join_does_not_silence_the_next_directory(self, sandbox: Sandbox):
        """The audit subdir and the bundle usually share the host gid. The gid
        is recorded as joined only once a usermod has succeeded, so a failure
        on the first does not make the second look like a duplicate — each
        gets its own warning naming its own variable."""
        audit = sandbox.audit_mount(gid=3000)
        bundle = sandbox.bundle_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, bundle_dir=bundle, usermod_rc=1)

        assert _usermods(run) == [
            "usermod osprey-mount-3000 osprey",
            "usermod osprey-mount-3000 osprey",
        ], "the second directory was skipped as already joined"
        warnings = _warnings(run)
        assert any(AUDIT_VAR in line for line in warnings), warnings
        assert any(BUNDLE_VAR in line for line in warnings), warnings


class TestPathsAndNamesAreOneArgument:
    """Every env-derived value reaches stat/test/usermod as a single word.

    ``OSPREY_FACILITY_BUNDLE_DIR`` is the live case: it comes from
    ``facility_knowledge.bundle_path``, which is ordinary operator config and
    passes through no charset normalizer, unlike the audit path. A lost pair of
    quotes turns one directory into two arguments and the whole join degrades
    to a warning about a directory that is right there.
    """

    def test_a_mount_path_with_a_space_is_joined(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=3000, identity="alice smith")

        run = sandbox.run(audit_dir=audit)

        assert run.returncode == 0, run.stderr
        assert _groupadds(run) == ["groupadd 3000 osprey-mount-3000"], run.stderr
        assert _usermods(run) == ["usermod osprey-mount-3000 osprey"]
        assert _probes(run) == [f"probe osprey -w {audit}"], "the probe saw a truncated path"
        assert f"{AUDIT_VAR}={audit}" in run.stderr

    def test_a_bundle_path_with_shell_metacharacters_is_joined(self, sandbox: Sandbox):
        """A space, a glob character and a ``$`` in one path. None of them may
        be expanded, split or matched — this script runs no ``eval`` and the
        value is data at every one of its five uses."""
        bundle = sandbox.bundle_mount(gid=4000, name="facility bundle *[1] $HOME")

        run = sandbox.run(bundle_dir=bundle)

        assert run.returncode == 0, run.stderr
        assert _usermods(run) == ["usermod osprey-mount-4000 osprey"], run.stderr
        assert _probes(run) == [f"probe osprey -w {bundle}"]
        assert f"{BUNDLE_VAR}={bundle}" in run.stderr

    def test_a_group_name_from_etc_group_is_one_argument(self, sandbox: Sandbox):
        """The reuse branch takes the name out of ``/etc/group``, which this
        image does not write. Whatever the name column holds goes to
        ``usermod`` as exactly one argument."""
        audit = sandbox.audit_mount(gid=3000)
        sandbox.existing_group("host shared", 3000)

        run = sandbox.run(audit_dir=audit)

        assert run.returncode == 0, run.stderr
        assert _usermods(run) == ["usermod host shared osprey"], run.stderr


class TestSkippedTopologies:
    """Absent mounts are ordinary deployments, not errors."""

    def test_unset_variables_join_nothing(self, sandbox: Sandbox):
        """A bare ``docker run`` names neither directory."""
        run = sandbox.run()

        assert run.returncode == 0, run.stderr
        assert run.group_ops == []
        assert "gosu" in run.order

    def test_a_dispatch_worker_without_a_bundle_needs_no_special_case(self, sandbox: Sandbox):
        """Audit mount, no bundle mount: the unset skip covers it."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit)

        assert _groupadds(run) == ["groupadd 3000 osprey-mount-3000"]

    def test_a_variable_naming_a_missing_path_is_skipped_not_fatal(self, sandbox: Sandbox):
        """Said once, plainly, because a variable pointing at nothing is the
        shape a dropped bind takes — but a container that refuses to boot over
        it is strictly worse than one that logs and starts."""
        missing = sandbox.tmp / "not-mounted"

        run = sandbox.run(audit_dir=missing)

        assert run.returncode == 0, run.stderr
        assert run.group_ops == []
        assert f"{AUDIT_VAR}={missing} is not a directory" in run.stderr
        assert "gosu" in run.order

    def test_a_non_root_start_joins_nothing(self, sandbox: Sandbox):
        """Run with ``--user``: there is no privilege to edit ``/etc/group``
        and no drop to survive."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(uid=1000, audit_dir=audit)

        assert run.returncode == 0, run.stderr
        assert run.group_ops == []
        assert run.order == ["cmd"], run.order


class TestTheGidFloor:
    """Root and system gids are refused, loudly, without failing the boot."""

    @pytest.mark.parametrize("gid", [0, 1, 20, 99])
    def test_root_and_system_gids_are_refused(self, sandbox: Sandbox, gid: int):
        """A bind that appears root- or system-owned inside the container is
        the signature of an ownership remap (Docker Desktop's file sharing
        does exactly this), not of a group the deployment meant to share. The
        mount degrades to non-group access rather than growing privilege."""
        audit = sandbox.audit_mount(gid=gid)

        run = sandbox.run(audit_dir=audit)

        assert run.returncode == 0, run.stderr
        assert run.group_ops == [], f"osprey was added to gid {gid}"
        assert "gosu" in run.order, "the refusal blocked the container's boot"

    def test_the_refusal_is_one_warning_naming_variable_path_and_gid(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=0)

        run = sandbox.run(audit_dir=audit)

        warnings = [line for line in run.stderr.splitlines() if "WARNING:" in line]
        assert len(warnings) == 1, warnings
        assert AUDIT_VAR in warnings[0]
        assert str(audit) in warnings[0]
        assert "gid 0" in warnings[0]

    def test_the_floor_is_a_floor_not_a_blocklist(self, sandbox: Sandbox):
        """100 is the first gid a distribution hands to a real group, and a
        host group is what these mounts carry."""
        audit = sandbox.audit_mount(gid=100)

        run = sandbox.run(audit_dir=audit)

        assert _usermods(run) == ["usermod osprey-mount-100 osprey"]

    def test_a_refused_audit_gid_does_not_block_the_bundle(self, sandbox: Sandbox):
        """Each directory is judged on its own gid."""
        audit = sandbox.audit_mount(gid=0)
        bundle = sandbox.bundle_mount(gid=4000)

        run = sandbox.run(audit_dir=audit, bundle_dir=bundle)

        assert _usermods(run) == ["usermod osprey-mount-4000 osprey"]


class TestTheJoinFailsOpen:
    """Every failure mode logs and continues to the drop."""

    def test_an_unreadable_gid_warns_and_boots(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, stat_rc=1)

        assert run.returncode == 0, run.stderr
        assert "WARNING" in run.stderr
        assert run.group_ops == []
        assert "gosu" in run.order

    def test_a_non_numeric_gid_is_refused(self, sandbox: Sandbox):
        """The digits-only whitelist is what keeps a stat result that is not a
        number out of ``-lt``, out of ``groupadd --gid`` and out of the group
        name. An unreadable gid and an unparseable one are the same refusal."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, stat_out="? (unknown)")

        assert run.returncode == 0, run.stderr
        assert run.group_ops == [], "a non-numeric gid reached the group tools"
        assert len(_warnings(run)) == 1, run.stderr
        assert "gosu" in run.order

    def test_a_malformed_group_line_does_not_reach_usermod(self, sandbox: Sandbox):
        """``usermod --append --groups "" osprey`` would fail, and its warning
        would read as a usermod bug rather than as the corrupt ``/etc/group``
        it is. The reuse branch says which it is."""
        audit = sandbox.audit_mount(gid=3000)
        sandbox.existing_group("", 3000)

        run = sandbox.run(audit_dir=audit)

        assert run.returncode == 0, run.stderr
        assert _usermods(run) == [], "an empty group name reached usermod"
        assert _groupadds(run) == []
        assert "malformed" in run.stderr, run.stderr
        assert "gosu" in run.order

    def test_a_failing_groupadd_warns_and_boots(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, groupadd_rc=1)

        assert run.returncode == 0, run.stderr
        assert "WARNING" in run.stderr
        assert _usermods(run) == [], "membership was claimed for a group that was not created"
        assert "gosu" in run.order

    def test_a_failing_usermod_warns_and_boots(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, usermod_rc=1)

        assert run.returncode == 0, run.stderr
        assert "WARNING" in run.stderr
        assert "gosu" in run.order


class TestTheWriterMarker:
    """One uid per audit file, enforced by where the marker is and is not."""

    def test_the_maintenance_step_carries_it(self, sandbox: Sandbox):
        run = sandbox.run()

        assert run.maintenance_marker == "maintenance", run.stderr

    def test_the_dropped_command_does_not(self, sandbox: Sandbox):
        """``gosu`` preserves the environment across the drop, so anything set
        here reaches the app. A leaked marker would file every record the app
        writes under the root phase's writer — the exact inversion of the
        invariant it exists to enforce."""
        run = sandbox.run()

        assert run.drop_marker == UNSET, (
            f"the writer marker survived the privilege drop as {run.drop_marker!r}"
        )


class TestTheOwnershipSweep:
    """The hand-back skips the one directory group membership already covers."""

    def test_the_audit_mount_is_pruned(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit)

        assert f"-path {audit} -prune -o" in run.find_args, run.find_args
        assert "! -user osprey" in run.find_args, run.find_args
        assert "-exec chown osprey:osprey" in run.find_args, run.find_args

    def test_the_rest_of_the_state_zone_is_still_handed_back(self, sandbox: Sandbox):
        """The prune is one directory, not an abandonment of the sweep: root
        writes elsewhere under ``var/`` on the way through and must not keep
        it."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit)

        assert run.find_args.split()[0] == str(sandbox.state)
        assert "chown" in run.order, run.order
        assert run.order.index("chown") < run.order.index("gosu"), run.order

    def test_the_pruned_mount_is_never_traversed(self, sandbox: Sandbox):
        """The argv above is spelling; this is the outcome. ``-path`` is an
        fnmatch PATTERN rather than a path, so a prune can be textually
        perfect and match nothing — which leaves root chowning the operator's
        host directory on every start, the exact harm the prune exists to
        prevent."""
        audit = sandbox.audit_mount(gid=3000)
        record = audit / "record.jsonl"
        record.write_text("{}\n")
        elsewhere = sandbox.state / "state.json"
        elsewhere.write_text("{}")

        run = sandbox.run(audit_dir=audit)

        assert str(elsewhere) in run.visited, run.visited
        assert str(audit) not in run.visited, run.visited
        assert str(record) not in run.visited, "the sweep walked into the audit mount"

    def test_without_the_prune_the_sweep_would_reach_it(self, sandbox: Sandbox):
        """Non-vacuity for the test above: the same tree, the same sweep, and
        the only difference is that no variable named the mount."""
        audit = sandbox.audit_mount(gid=3000)
        record = audit / "record.jsonl"
        record.write_text("{}\n")

        run = sandbox.run()

        assert str(record) in run.visited, run.visited

    def test_no_prune_when_no_audit_mount(self, sandbox: Sandbox):
        """Nothing to exclude, so the sweep is exactly what it was before."""
        run = sandbox.run()

        assert "-prune" not in run.find_args, run.find_args
        assert "! -user osprey" in run.find_args

    def test_no_prune_for_a_variable_naming_a_missing_path(self, sandbox: Sandbox):
        """A prune expression built from a path that is not there would be
        dead weight in the predicate and would hide the misconfiguration."""
        run = sandbox.run(audit_dir=sandbox.tmp / "not-mounted")

        assert "-prune" not in run.find_args, run.find_args
