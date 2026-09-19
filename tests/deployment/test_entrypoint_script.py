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
(``OSPREY_AUDIT_DIR``, ``OSPREY_CONTROL_CONTEXT_DIR``,
``OSPREY_CONTROL_CONTEXT_TREE``, ``OSPREY_FACILITY_BUNDLE_DIR``,
``OSPREY_ARIEL_MIRROR_DIR``)
and the gid is stat'd off the mount, never passed in. Root and system gids are refused, because a mount
that presents one must not hand the agent's user a privileged group. The join
grants membership and says only that. Whether a directory actually grants the
access the render named it for is a separate question, asked per VARIABLE by
``probe_mounted_dir`` — outside the join, so that the join's seen-gid early
return cannot swallow the second bind of a pair carved out of one host tree —
and answered by access rather than membership: ``-w`` and ``-x`` for the
written binds, ``-r`` and ``-x`` for the read-only tree.

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

#: The script's own last line. Dropping it is what turns the shipped file into
#: something a test can source for one of its functions.
MAIN_INVOCATION = 'main "$@"'

#: The variables the render names, and the only ones the join may read.
AUDIT_VAR = "OSPREY_AUDIT_DIR"
BUNDLE_VAR = "OSPREY_FACILITY_BUNDLE_DIR"
MIRROR_VAR = "OSPREY_ARIEL_MIRROR_DIR"
#: The control-target bind this container WRITES: one owner directory, the
#: account this service acts for.
CONTROL_DIR_VAR = "OSPREY_CONTROL_CONTEXT_DIR"
#: The control-target bind this container only READS: the whole tree, bound
#: read-only, out of which lanes and the dispatch worker resolve owners.
CONTROL_TREE_VAR = "OSPREY_CONTROL_CONTEXT_TREE"
RENDER_NAMED_VARS = frozenset(
    {AUDIT_VAR, CONTROL_DIR_VAR, CONTROL_TREE_VAR, BUNDLE_VAR, MIRROR_VAR}
)

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

#: Where the script reads the container's user-namespace mapping. The sandbox
#: points the script's copy at a fabricated file, because no test host is a
#: rootless container and macOS has no ``/proc`` at all.
UID_MAP_PATH = "/proc/self/uid_map"

#: The kernel's spelling of "no user namespace": container root is host root.
IDENTITY_UID_MAP = "         0          0 4294967295\n"

#: rootless podman with the default subuid layout: the invoking host user
#: (here uid 1000) is container root; everything else maps into a subuid range.
ROOTLESS_UID_MAP = "         0       1000          1\n         1     100000      65536\n"


@pytest.fixture(scope="module")
def text() -> str:
    return TEMPLATE.read_text()


def _dispatcher_body(text: str) -> str:
    """``join_mounted_groups``'s body: the one place that decides which
    directories are joined and probed, and with which access."""
    start = text.index("join_mounted_groups() {")
    return text[start : text.index("\n}", start)]


def _hand_back_body(text: str) -> str:
    """``hand_back_state_zone``'s body: the one place that decides which
    directories the ownership sweep is allowed to reach."""
    start = text.index("hand_back_state_zone() {")
    return text[start : text.index("\n}", start)]


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

    def test_every_render_named_variable_is_joined_and_probed(self, text: str):
        """The dispatcher is the allowlist spelled twice, and both halves have
        to name every variable.

        A join with no probe is the failure this separation exists to end: the
        join returns early for a gid it has already joined, so a directory
        sharing another's group would be reported as fine without anything
        ever asking it a question. A probe with no join is the milder inverse
        — a mount whose group was never granted, warning about an access it
        was never given a chance at.
        """
        body = _dispatcher_body(text)
        for var in sorted(RENDER_NAMED_VARS):
            assert f'join_mounted_group {var} "${{{var}:-}}"' in body, var
            assert re.search(
                rf'probe_mounted_dir {var} "\$\{{{var}:-\}}" (read|write)$',
                body,
                flags=re.MULTILINE,
            ), var

    def test_the_read_only_tree_is_the_only_read_probe(self, text: str):
        """The mode is per VARIABLE, not per container: the tree is bound
        read-only, so a write probe on it would report today's "records will be
        dropped" warning for a directory nothing in this container writes —
        which is the false alarm that made the probe a separate helper."""
        body = _dispatcher_body(text)
        assert f'probe_mounted_dir {CONTROL_TREE_VAR} "${{{CONTROL_TREE_VAR}:-}}" read' in body
        for var in sorted(RENDER_NAMED_VARS - {CONTROL_TREE_VAR}):
            assert f'probe_mounted_dir {var} "${{{var}:-}}" write' in body, var

    def test_the_probe_is_not_nested_in_the_join(self, text: str):
        """Structural, because the reason is structural: the seen-gid early
        return is above the point a nested probe would sit, so a probe inside
        ``join_mounted_group`` silently never runs for the second bind of any
        pair that shares a group — which is every deployment."""
        join = text.index("join_mounted_group() {")
        probe = text.index("probe_mounted_dir() {")
        dispatcher = text.index("join_mounted_groups() {")
        assert join < probe < dispatcher, "the helpers are out of order"
        join_body = "\n".join(
            line for line in text[join:probe].splitlines() if not line.lstrip().startswith("#")
        )
        assert "gosu" not in join_body, "the join still probes; the dispatcher must"

    def test_the_uid_map_is_the_kernels(self, text: str):
        """Read from the one path the kernel states it at — not from the env,
        which is the render's to name and this script's to trust for mounts
        only, and not guessed from the runtime."""
        assert f"UID_MAP={UID_MAP_PATH}\n" in text
        assert not re.search(r"OSPREY_[A-Z_]*UID_MAP", text), "the map path is taken from the env"

    def test_the_gid_is_stat_ed_off_the_mount(self, text: str):
        """Not passed in. A gid rendered into the compose file is the host's
        gid at render time; the mount's is the only one true at start."""
        assert "stat -c '%g'" in text
        assert not re.search(r"OSPREY_[A-Z_]*GID", text), "a gid is being taken from the env"

    def test_the_sweep_prunes_every_group_shared_bind(self, text: str):
        """The deliberate exceptions to the ownership hand-back: every bind
        under ``var/`` the dropped process reaches through group membership
        rather than through ownership, and the read-only tree, which a chown
        could not rewrite even if it wanted to."""
        assert re.search(
            r'find "\$STATE_DIR" "\$@" ! -user osprey -exec chown osprey:osprey \{\} \+',
            text,
        ), "the hand-back no longer applies its optional prune expression"
        body = _hand_back_body(text)
        assert '-path "$_hb_dir" -prune -o' in body, body
        for var in (AUDIT_VAR, CONTROL_DIR_VAR, CONTROL_TREE_VAR):
            assert f'"${{{var}:-}}"' in body, var

    def test_the_sweep_prunes_only_what_lives_under_the_state_zone(self, text: str):
        """The bundle and the ARIEL mirror are group-shared too, and are
        deliberately NOT pruned: neither is under ``var/``, so the sweep never
        reaches them and a prune for them would be dead weight in the predicate
        that reads like a working exclusion."""
        body = _hand_back_body(text)
        assert BUNDLE_VAR not in body, body
        assert MIRROR_VAR not in body, body

    def test_the_exclusion_says_why(self, text: str):
        """A future reader deleting a prune as dead weight is the failure
        mode, so the comment names every pruned mount and the reason by name."""
        comment = "\n".join(line for line in text.splitlines() if line.lstrip().startswith("#"))
        for var in (AUDIT_VAR, CONTROL_DIR_VAR, CONTROL_TREE_VAR):
            assert var in comment, var
        assert "membership" in comment.lower()
        assert "ownership" in comment.lower()
        assert "read-only" in comment.lower(), "the tree's own reason is not stated"


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

    def control_dir_mount(self, gid: int, user: str = "alice") -> Path:
        """The per-owner control-target directory this container writes, where
        compose binds it: under the state zone, beside the audit subdir and
        provisioned the same way (setgid 2770, the host's shared group)."""
        return self._mount(self.state / "agent_data" / "control_target" / user, gid)

    def control_tree_mount(self, gid: int, name: str = "control-tree") -> Path:
        """The whole control-target tree, bound READ-ONLY at a path outside
        ``var/`` — never over this container's own state dir, because the
        hand-back sweep would chown across a read-only bind and fail."""
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
        control_dir: Path | str | None = None,
        control_tree: Path | str | None = None,
        bundle_dir: Path | str | None = None,
        mirror_dir: Path | str | None = None,
        groupadd_rc: int = 0,
        usermod_rc: int = 0,
        stat_rc: int | None = None,
        stat_out: str | None = None,
        probe_rc: int = 0,
        read_probe_rc: int | None = None,
        traverse_probe_rc: int | None = None,
        uid_map: str | None = None,
    ) -> Run:
        # The map is read off a path the script hardcodes, so the sandbox's
        # copy of the script is re-pointed at a file holding the fabricated
        # map — the same move as the stubbed `stat`, for the same reason: the
        # real one answers for the test host, never for the case under test.
        # `None` leaves the real path in, which no test host can read.
        script_text = TEMPLATE.read_text()
        if uid_map is not None:
            uid_map_file = self.tmp / "uid_map"
            uid_map_file.write_text(uid_map)
            assert UID_MAP_PATH in script_text
            script_text = script_text.replace(UID_MAP_PATH, str(uid_map_file))
        self.script.write_text(script_text)
        bindir = self._stubs(
            uid=uid,
            groupadd_rc=groupadd_rc,
            usermod_rc=usermod_rc,
            stat_rc=stat_rc,
            stat_out=stat_out,
            probe_rc=probe_rc,
            read_probe_rc=read_probe_rc,
            traverse_probe_rc=traverse_probe_rc,
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
        if control_dir is not None:
            env[CONTROL_DIR_VAR] = str(control_dir)
        if control_tree is not None:
            env[CONTROL_TREE_VAR] = str(control_tree)
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

    def call_probe(
        self, var: str, directory: Path | str, mode: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Call ``probe_mounted_dir`` directly, with no dispatcher above it.

        The mode arms the dispatcher cannot reach — every mode it passes is a
        literal — are reachable only from a caller that gets the mode wrong,
        which is what a future edit of this file is. So the shipped script is
        sourced instead of run, with its final ``main "$@"`` invocation
        removed (the one and only edit made to it, and the reason the function
        is still defined when the source returns), and the helper is then
        called the way that edit would call it. ``set -eu`` comes along with
        the source, so a call that aborts the shell aborts this one too.
        """
        sourced = self.tmp / "sourced-entrypoint.sh"
        text = TEMPLATE.read_text().rstrip()
        assert text.endswith(MAIN_INVOCATION), "the script no longer ends in its own call"
        sourced.write_text(text[: -len(MAIN_INVOCATION)])
        bindir = self._stubs(
            uid=0,
            groupadd_rc=0,
            usermod_rc=0,
            stat_rc=None,
            stat_out=None,
            probe_rc=0,
        )
        group_log = self.tmp / "group.log"
        group_log.write_text("")
        call = 'probe_mounted_dir "$2" "$3"' + (' "$4"' if mode is not None else "")
        argv = ["_", str(sourced), var, str(directory)]
        if mode is not None:
            argv.append(mode)
        return subprocess.run(
            ["/bin/sh", "-c", f'. "$1"\n{call}\n', *argv],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": str(bindir),
                "GROUP_LOG": str(group_log),
                "GID_MAP": str(self.gid_map),
                "GROUP_DB": str(self.group_db),
            },
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
        read_probe_rc: int | None = None,
        traverse_probe_rc: int | None = None,
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
        # -<bit> <dir>` is `probe_mounted_dir` asking whether the mount grants
        # the access the render named it for, answered from a knob because
        # really switching users needs root; anything else is the privilege
        # drop at the end of main(). One knob per BIT, because the read probe
        # asks two questions and a directory can answer them differently — a
        # traversable tree whose names cannot be listed is a real mode, and
        # one rc for both would make it untestable. Unset bits inherit
        # ``probe_rc``, so "this mount grants nothing" stays one argument.
        read_rc = probe_rc if read_probe_rc is None else read_probe_rc
        traverse_rc = probe_rc if traverse_probe_rc is None else traverse_probe_rc
        write(
            "gosu",
            'if [ "$2" = "test" ]; then\n'
            '  printf "probe %s %s %s\\n" "$1" "$3" "$4" >> "$GROUP_LOG"\n'
            '  case "$3" in\n'
            f"    -w) exit {probe_rc} ;;\n"
            f"    -r) exit {read_rc} ;;\n"
            f"    *) exit {traverse_rc} ;;\n"
            "  esac\n"
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


def _joins(run: Run) -> list[str]:
    """Every group-table operation, probes excluded.

    "No join happened" can no longer be spelled ``group_ops == []``: the probe
    is outside the join and runs for every render-named directory that exists,
    including one whose join was refused — which is exactly the case an
    operator most needs told about.
    """
    return [op for op in run.group_ops if not op.startswith("probe ")]


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


class TestMembershipIsNotReportedAsAccess:
    """Membership is never reported as the access it was granted for.

    The two are separate lines from separate steps: the join says what it did
    to ``/etc/group``, and ``probe_mounted_dir`` says whether the directory
    actually grants the access the render named it for.
    """

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
            f"probe osprey -x {audit}",
        ], run.group_ops

    def test_a_successful_join_says_joined(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit)

        assert (
            f"joined osprey to group osprey-mount-3000 (gid 3000) for {AUDIT_VAR}={audit}"
            in run.stderr
        ), run.stderr
        assert _warnings(run) == [], run.stderr

    def test_an_unwritable_mount_warns_beside_the_membership_line(self, sandbox: Sandbox):
        """The reachable steady state this catches: a container that logs a
        success line at boot, serves normally, and writes no audit record at
        all — because every layer below this one (the join, the sweep, the
        envelope writer) also degrades to a log line.

        The membership line stays, because the membership really was granted
        and naming the group is how an operator finds the host directory to
        fix. What must not survive is a log that reads as if the write works.
        """
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, probe_rc=1)

        assert run.returncode == 0, run.stderr
        warnings = _warnings(run)
        assert len(warnings) == 1, warnings
        assert "cannot write" in warnings[0]
        assert str(audit) in run.stderr, "the warning does not name the directory"
        assert "gosu" in run.order, "a failed probe blocked the container's boot"

    def test_a_mount_that_cannot_be_entered_fails_the_write_probe_too(self, sandbox: Sandbox):
        """``w`` without ``x`` is the mode a 2760 directory presents, and it is
        not enough: creating an entry needs both, so every record this
        container writes there fails with EACCES while ``test -w`` says yes.
        A one-bit probe reports that directory as good — the silent success
        this helper exists to end."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, traverse_probe_rc=1)

        assert run.returncode == 0, run.stderr
        assert _probes(run) == [
            f"probe osprey -w {audit}",
            f"probe osprey -x {audit}",
        ], _probes(run)
        warnings = _warnings(run)
        assert len(warnings) == 1, warnings
        assert "cannot write/enter" in warnings[0]
        assert str(audit) in warnings[0], "the warning does not name the directory"
        assert "gosu" in run.order, "a failed probe blocked the container's boot"

    def test_the_write_warning_names_the_remap_in_both_its_shapes(self, sandbox: Sandbox):
        """The operator-facing half, as the read arm's warning carries it too.

        An ownership remap is the overwhelmingly common cause of a bind that
        arrives unwritable, and it reaches the container in two shapes that
        look nothing alike in the log: gid 0, refused by name at the privilege
        floor, and the nobody and overflow gids, which pass that floor and die
        at ``groupadd``. An operator told about one shape and stuck on the
        other reads the warning as not applying to them.
        """
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, probe_rc=1)

        assert "Docker Desktop" in run.stderr
        assert "gid 0" in run.stderr
        assert "nobody/nogroup" in run.stderr
        assert "overflow" in run.stderr

    def test_the_membership_line_claims_a_membership_and_nothing_more(self, sandbox: Sandbox):
        """It is printed by the join, so it says only what the join did. Any
        wording that also asserted writability would be a claim the join is in
        no position to make — the probe that can make it runs afterwards, once
        per variable, and says so itself when the answer is no."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, probe_rc=1)

        joined = [line for line in run.stderr.splitlines() if "joined osprey" in line]
        assert len(joined) == 1, run.stderr
        assert f"group osprey-mount-3000 (gid 3000) for {AUDIT_VAR}={audit}" in joined[0]
        assert "write" not in joined[0], joined[0]

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


class TestTheControlContextProbes:
    """Each control-target bind is probed for the access it exists to grant.

    The owner directory is written — every owned write in this container
    stamps it — while the tree is only read, by the lanes and the dispatch
    worker that resolve an owner out of it. They are two mounts of one host
    tree, so they carry one group, which is precisely why the probe cannot
    live inside the join.
    """

    def test_the_owner_directory_is_probed_for_write(self, sandbox: Sandbox):
        control = sandbox.control_dir_mount(gid=3000)

        run = sandbox.run(control_dir=control)

        assert run.returncode == 0, run.stderr
        assert _usermods(run) == ["usermod osprey-mount-3000 osprey"]
        assert _probes(run) == [
            f"probe osprey -w {control}",
            f"probe osprey -x {control}",
        ], _probes(run)
        assert _warnings(run) == [], run.stderr

    def test_the_tree_is_probed_for_read_and_traverse(self, sandbox: Sandbox):
        """Both bits, because either alone leaves the tree unusable: readable
        without traversable lists owner names that cannot be opened, and
        traversable without readable opens paths that cannot be enumerated."""
        tree = sandbox.control_tree_mount(gid=3000)

        run = sandbox.run(control_tree=tree)

        assert run.returncode == 0, run.stderr
        assert _probes(run) == [f"probe osprey -r {tree}", f"probe osprey -x {tree}"]
        assert _warnings(run) == [], run.stderr

    def test_the_tree_is_never_write_probed(self, sandbox: Sandbox):
        """It is a read-only bind. A write probe would fail by construction and
        log today's "writes to this path will fail" for a directory nothing in
        this container writes — a warning that is false and, worse, drowns the
        one below it that is true."""
        tree = sandbox.control_tree_mount(gid=3000)

        run = sandbox.run(control_tree=tree, probe_rc=1, read_probe_rc=0, traverse_probe_rc=0)

        assert [op for op in _probes(run) if " -w " in op] == [], _probes(run)
        assert _warnings(run) == [], run.stderr

    def test_an_unreadable_tree_names_the_refusal_the_container_will_serve(self, sandbox: Sandbox):
        """The steady state without this line: every owned write refuses with
        ``control_context_unavailable`` and the boot log looks clean, because
        the bind mounted, the group joined and nothing ever read the tree."""
        tree = sandbox.control_tree_mount(gid=3000)

        run = sandbox.run(control_tree=tree, read_probe_rc=1)

        assert run.returncode == 0, run.stderr
        assert _probes(run) == [f"probe osprey -r {tree}"], (
            "the traverse bit was asked after the read had already refused"
        )
        warnings = _warnings(run)
        assert len(warnings) == 1, warnings
        assert "cannot read/traverse" in warnings[0]
        assert f"{CONTROL_TREE_VAR}={tree}" in warnings[0]
        assert "control_context_unavailable" in run.stderr, run.stderr
        assert "gosu" in run.order, "a failed probe blocked the container's boot"

    def test_a_tree_that_cannot_be_traversed_fails_the_read_probe_too(self, sandbox: Sandbox):
        """``-r`` alone is the mode a 0640 directory presents, and it is not
        enough: nothing under the tree can be opened."""
        tree = sandbox.control_tree_mount(gid=3000)

        run = sandbox.run(control_tree=tree, traverse_probe_rc=1)

        assert _probes(run) == [f"probe osprey -r {tree}", f"probe osprey -x {tree}"]
        assert len(_warnings(run)) == 1, run.stderr
        assert "cannot read/traverse" in _warnings(run)[0]

    def test_the_read_warning_names_the_remap_in_both_its_shapes(self, sandbox: Sandbox):
        """The operator-facing half. A Docker Desktop ownership remap is the
        overwhelmingly common cause, and it arrives in two shapes that look
        nothing alike in the log above: gid 0, which the floor refuses by
        name, and the nobody/overflow gids, which sail past the floor and die
        at ``groupadd``. Naming only one leaves half the operators stuck."""
        tree = sandbox.control_tree_mount(gid=3000)

        run = sandbox.run(control_tree=tree, read_probe_rc=1)

        assert "Docker Desktop" in run.stderr
        assert "gid 0" in run.stderr
        assert "nobody/nogroup" in run.stderr
        assert "overflow" in run.stderr

    def test_both_binds_are_probed_when_they_share_one_gid(self, sandbox: Sandbox):
        """The tree is the owner directory's parent on the host, so they carry
        one group in every deployment this provisions. The join runs once —
        the seen-gid return is doing its job — and both probes still run,
        which is the whole reason the probe is not inside the join."""
        control = sandbox.control_dir_mount(gid=3000)
        tree = sandbox.control_tree_mount(gid=3000)

        run = sandbox.run(control_dir=control, control_tree=tree)

        assert run.returncode == 0, run.stderr
        assert _usermods(run) == ["usermod osprey-mount-3000 osprey"], "joined twice"
        assert _probes(run) == [
            f"probe osprey -w {control}",
            f"probe osprey -x {control}",
            f"probe osprey -r {tree}",
            f"probe osprey -x {tree}",
        ], _probes(run)

    def test_the_tree_is_still_probed_when_the_audit_mount_took_the_gid(self, sandbox: Sandbox):
        """The live shape: the audit subdir is joined first and records the
        gid, so the tree's join returns before it reaches any group work."""
        audit = sandbox.audit_mount(gid=3000)
        tree = sandbox.control_tree_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, control_tree=tree)

        assert _usermods(run) == ["usermod osprey-mount-3000 osprey"]
        assert _probes(run) == [
            f"probe osprey -w {audit}",
            f"probe osprey -x {audit}",
            f"probe osprey -r {tree}",
            f"probe osprey -x {tree}",
        ], _probes(run)
        assert _warnings(run) == [], run.stderr

        refused = sandbox.run(audit_dir=audit, control_tree=tree, read_probe_rc=1)

        assert "cannot read/traverse" in refused.stderr, refused.stderr

    def test_a_refused_join_is_still_probed(self, sandbox: Sandbox):
        """Access, never membership. A gid the floor refuses is exactly the
        directory an operator most needs an answer about, and a bind can be
        reachable with no group at all — already owned by ``osprey``, or
        world-readable — so a refused join with a passing probe is not a
        failure and is reported as none."""
        tree = sandbox.control_tree_mount(gid=0)

        run = sandbox.run(control_tree=tree)

        assert run.returncode == 0, run.stderr
        assert _joins(run) == [], "osprey was added to gid 0"
        assert _probes(run) == [f"probe osprey -r {tree}", f"probe osprey -x {tree}"]
        assert len(_warnings(run)) == 1, "the passing probe added a warning of its own"
        assert "gid 0" in _warnings(run)[0]

    def test_an_unset_control_bind_probes_nothing(self, sandbox: Sandbox):
        """A single-user deployment mounts neither, and a dispatch worker only
        the tree. Silence is the correct output, not a warning per absence."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit)

        assert _probes(run) == [
            f"probe osprey -w {audit}",
            f"probe osprey -x {audit}",
        ], _probes(run)

    def test_a_control_bind_naming_a_missing_path_probes_nothing(self, sandbox: Sandbox):
        """The join has already said the path is not there; a second line from
        the probe would be noise, and probing a path that does not exist would
        warn about permissions that are not the problem."""
        missing = sandbox.tmp / "never-bound"

        run = sandbox.run(control_tree=missing)

        assert run.returncode == 0, run.stderr
        assert _probes(run) == []
        assert f"{CONTROL_TREE_VAR}={missing} is not a directory" in run.stderr

    def test_a_control_path_with_a_space_reaches_the_probe_whole(self, sandbox: Sandbox):
        """The owner segment is an identity, and identities carry spaces. A
        lost pair of quotes turns one directory into two arguments and the
        probe answers for neither."""
        control = sandbox.control_dir_mount(gid=3000, user="alice smith")
        tree = sandbox.control_tree_mount(gid=3000, name="control tree *[1] $HOME")

        run = sandbox.run(control_dir=control, control_tree=tree)

        assert run.returncode == 0, run.stderr
        assert _probes(run) == [
            f"probe osprey -w {control}",
            f"probe osprey -x {control}",
            f"probe osprey -r {tree}",
            f"probe osprey -x {tree}",
        ], "a probe saw a truncated path"


class TestTheProbeSurvivesItsOwnMisuse:
    """A probe called wrongly logs and returns; it never stops the boot.

    Both arms exist for a caller that does not exist yet — the dispatcher
    passes literal modes — so they are exercised by calling the helper the way
    a future edit would get it wrong. Fail-open is the rule for every arm of
    this step: a container that boots with one unchecked directory is strictly
    better than one that refuses to start over an internal mistake.
    """

    def test_an_unknown_mode_is_said_out_loud_and_checks_nothing(self, sandbox: Sandbox):
        """Silence would be indistinguishable in the log from a probe that
        passed, which is the failure the helper was split out to end."""
        control = sandbox.control_dir_mount(gid=3000)

        result = sandbox.call_probe(CONTROL_DIR_VAR, control, "sideways")

        assert result.returncode == 0, result.stderr
        warnings = [line for line in result.stderr.splitlines() if "WARNING:" in line]
        assert len(warnings) == 1, result.stderr
        assert "sideways" in warnings[0], warnings[0]
        assert f"{CONTROL_DIR_VAR}={control}" in result.stderr, result.stderr
        assert "no access was checked" in result.stderr, result.stderr

    def test_a_missing_mode_takes_the_same_arm_rather_than_aborting(self, sandbox: Sandbox):
        """An omitted mode is the likelier spelling of a misspelled one, and
        under ``set -u`` an undefaulted ``$3`` would exit the whole entrypoint
        before the privilege drop — a container that never starts, from a typo
        in a call site."""
        control = sandbox.control_dir_mount(gid=3000)

        result = sandbox.call_probe(CONTROL_DIR_VAR, control)

        assert result.returncode == 0, result.stderr
        assert "no access was checked" in result.stderr, result.stderr


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
        assert _probes(run) == [
            f"probe osprey -w {audit}",
            f"probe osprey -x {audit}",
        ], "the probe saw a truncated path"
        assert f"{AUDIT_VAR}={audit}" in run.stderr

    def test_a_bundle_path_with_shell_metacharacters_is_joined(self, sandbox: Sandbox):
        """A space, a glob character and a ``$`` in one path. None of them may
        be expanded, split or matched — this script runs no ``eval`` and the
        value is data at every one of its five uses."""
        bundle = sandbox.bundle_mount(gid=4000, name="facility bundle *[1] $HOME")

        run = sandbox.run(bundle_dir=bundle)

        assert run.returncode == 0, run.stderr
        assert _usermods(run) == ["usermod osprey-mount-4000 osprey"], run.stderr
        assert _probes(run) == [
            f"probe osprey -w {bundle}",
            f"probe osprey -x {bundle}",
        ]
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
        assert _joins(run) == [], f"osprey was added to gid {gid}"
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


class TestTheFloorInARootlessContainer:
    """gid 0 is joinable exactly when container root is the host user.

    Under rootless podman the invoking host user maps to uid/gid 0 inside, so
    the audit subdir and the bundle that user provisioned (setgid, the user's
    own group) present as gid 0 — the floor's signature for a remap, for a
    reason that does not hold there. The script tells the two apart by the
    kernel's uid map, which is fabricated here: no test host is a rootless
    container, and the logic under test is the guard, not the kernel.
    """

    def test_gid_0_is_joined_when_the_uid_map_is_not_the_identity(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=0)

        run = sandbox.run(audit_dir=audit, uid_map=ROOTLESS_UID_MAP)

        assert run.returncode == 0, run.stderr
        assert _groupadds(run) == [], "gid 0 is `root` in every image's /etc/group"
        assert _usermods(run) == ["usermod root osprey"]
        assert _warnings(run) == [], run.stderr

    def test_a_gid_0_join_is_probed_like_any_other(self, sandbox: Sandbox):
        """Membership in gid 0 is the mechanism; the probe still asks whether
        the write the audit trail depends on actually works."""
        audit = sandbox.audit_mount(gid=0)

        run = sandbox.run(audit_dir=audit, uid_map=ROOTLESS_UID_MAP)

        assert _probes(run) == [
            f"probe osprey -w {audit}",
            f"probe osprey -x {audit}",
        ]
        assert "joined osprey to group root (gid 0)" in run.stderr

    def test_the_log_says_why_the_floor_was_lifted(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=0)

        run = sandbox.run(audit_dir=audit, uid_map=ROOTLESS_UID_MAP)

        lifted = [line for line in run.stderr.splitlines() if "rootless container" in line]
        assert len(lifted) == 1, run.stderr
        assert AUDIT_VAR in lifted[0]
        assert str(audit) in lifted[0]

    def test_gid_0_stays_refused_under_the_identity_map(self, sandbox: Sandbox):
        """A rootful daemon — Docker Desktop's remap included — runs under the
        identity map, and there gid 0 still means the host's root group."""
        audit = sandbox.audit_mount(gid=0)

        run = sandbox.run(audit_dir=audit, uid_map=IDENTITY_UID_MAP)

        assert _joins(run) == [], "osprey was added to gid 0 under the identity map"
        assert len(_warnings(run)) == 1, run.stderr
        assert "gid 0" in _warnings(run)[0]

    def test_gid_0_stays_refused_with_no_readable_map(self, sandbox: Sandbox):
        """No map, no evidence: the default is the answer that grants nothing.
        (This is also what every test above this class runs under.)"""
        audit = sandbox.audit_mount(gid=0)

        run = sandbox.run(audit_dir=audit, uid_map=None)

        assert _joins(run) == []
        assert len(_warnings(run)) == 1, run.stderr

    def test_an_empty_map_is_not_evidence_either(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=0)

        run = sandbox.run(audit_dir=audit, uid_map="")

        assert _joins(run) == []

    @pytest.mark.parametrize("gid", [1, 20, 99])
    def test_the_system_range_stays_refused_in_a_rootless_container(
        self, sandbox: Sandbox, gid: int
    ):
        """Only gid 0 is the host user in a rootless container; a mount that
        presents `daemon` or `dialout` is still not a group the render meant."""
        audit = sandbox.audit_mount(gid=gid)

        run = sandbox.run(audit_dir=audit, uid_map=ROOTLESS_UID_MAP)

        assert _joins(run) == [], f"osprey was added to gid {gid}"
        assert len(_warnings(run)) == 1, run.stderr

    def test_a_host_group_joins_the_same_way_in_a_rootless_container(self, sandbox: Sandbox):
        """The map changes nothing above the floor."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, uid_map=ROOTLESS_UID_MAP)

        assert _usermods(run) == ["usermod osprey-mount-3000 osprey"]

    def test_gid_0_and_a_host_gid_are_each_joined_once(self, sandbox: Sandbox):
        """The audit subdir at gid 0 and a bundle at a host gid — the mixed
        shape a rootless host with one shared bundle group produces."""
        audit = sandbox.audit_mount(gid=0)
        bundle = sandbox.bundle_mount(gid=0)
        mirror = sandbox.mirror_mount(gid=3000)

        run = sandbox.run(
            audit_dir=audit, bundle_dir=bundle, mirror_dir=mirror, uid_map=ROOTLESS_UID_MAP
        )

        assert _usermods(run) == ["usermod root osprey", "usermod osprey-mount-3000 osprey"]


class TestTheJoinFailsOpen:
    """Every failure mode logs and continues to the drop."""

    def test_an_unreadable_gid_warns_and_boots(self, sandbox: Sandbox):
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, stat_rc=1)

        assert run.returncode == 0, run.stderr
        assert "WARNING" in run.stderr
        assert _joins(run) == []
        assert "gosu" in run.order

    def test_a_non_numeric_gid_is_refused(self, sandbox: Sandbox):
        """The digits-only whitelist is what keeps a stat result that is not a
        number out of ``-lt``, out of ``groupadd --gid`` and out of the group
        name. An unreadable gid and an unparseable one are the same refusal."""
        audit = sandbox.audit_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, stat_out="? (unknown)")

        assert run.returncode == 0, run.stderr
        assert _joins(run) == [], "a non-numeric gid reached the group tools"
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

    def test_every_group_shared_bind_gets_its_own_prune(self, sandbox: Sandbox):
        """The owner directory is provisioned exactly as the audit subdir is —
        a host directory at setgid 2770 the container reaches through the group
        — so it earns the same exclusion, and so does the read-only tree."""
        audit = sandbox.audit_mount(gid=3000)
        control = sandbox.control_dir_mount(gid=3000)
        tree = sandbox.control_tree_mount(gid=3000)

        run = sandbox.run(audit_dir=audit, control_dir=control, control_tree=tree)

        for path in (audit, control, tree):
            assert f"-path {path} -prune -o" in run.find_args, run.find_args
        assert "! -user osprey" in run.find_args, run.find_args
        assert "-exec chown osprey:osprey" in run.find_args, run.find_args

    def test_the_pruned_owner_directory_is_never_traversed(self, sandbox: Sandbox):
        """The outcome rather than the spelling. Without it the first start
        chowns the operator's host directory to the container account and drops
        the setgid group the owner resolution depends on — and the next
        ``osprey up`` flips it back, forever."""
        control = sandbox.control_dir_mount(gid=3000)
        record = control / "server_1.json"
        record.write_text("{}\n")
        elsewhere = sandbox.state / "state.json"
        elsewhere.write_text("{}")

        run = sandbox.run(control_dir=control)

        assert str(elsewhere) in run.visited, run.visited
        assert str(control) not in run.visited, run.visited
        assert str(record) not in run.visited, "the sweep walked into the owner directory"

    def test_without_the_prune_the_sweep_would_reach_the_owner_directory(self, sandbox: Sandbox):
        """Non-vacuity for the test above: the same tree, the same sweep, and
        the only difference is that no variable named the mount."""
        control = sandbox.control_dir_mount(gid=3000)
        record = control / "server_1.json"
        record.write_text("{}\n")

        run = sandbox.run()

        assert str(record) in run.visited, run.visited

    def test_no_prune_for_a_control_variable_naming_a_missing_path(self, sandbox: Sandbox):
        """Every pruned variable is guarded the same way, not just the audit
        subdir: a bind the compose file names but the runtime never made is a
        misconfiguration the predicate must not paper over."""
        run = sandbox.run(
            control_dir=sandbox.tmp / "no-owner-dir",
            control_tree=sandbox.tmp / "no-tree",
        )

        assert "-prune" not in run.find_args, run.find_args

    def test_one_missing_bind_does_not_drop_the_other_prunes(self, sandbox: Sandbox):
        """The prunes are independent clauses. A deployment with no tree bind
        (a web terminal writes its owner directory and reads no tree) still
        excludes the two binds it does have."""
        audit = sandbox.audit_mount(gid=3000)
        control = sandbox.control_dir_mount(gid=3000)

        run = sandbox.run(
            audit_dir=audit,
            control_dir=control,
            control_tree=sandbox.tmp / "no-tree",
        )

        assert f"-path {audit} -prune -o" in run.find_args, run.find_args
        assert f"-path {control} -prune -o" in run.find_args, run.find_args
        assert "no-tree" not in run.find_args, run.find_args
