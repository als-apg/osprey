"""E2E: the audit mount's MECHANISM, across two real containers on one host.

Every other test of this feature can only reach the side effect — a record
appears where the render said it would. This one is about *how* it got there,
because the how is the whole safety property and it is made of three things no
in-process test can observe:

1. **The privilege drop.** ``gosu`` re-derives the dropped process's
   supplementary groups from the image's ``/etc/group`` and discards whatever
   the runtime granted the initial root process. That is why compose's
   ``group_add:`` is only the belt, and why the entrypoint's ``/etc/group``
   step exists at all. The probe below runs as the container's actual command,
   so it *is* the post-``gosu`` process — never a ``docker exec``, which starts
   a fresh process that re-applies the runtime's GroupAdd and would prove
   nothing about the one that serves requests. ``--entrypoint``/``--user``
   are the same mistake wearing different flags, and every use of them here is
   scaffolding rather than subject: the two root containers that bracket the
   run (one creates the group-only directory, one widens it back in the
   teardown) and the negative control, which must hold no membership at all.
   Nothing about the drop is observed in any of them.
2. **setgid on the mount.** The host provisions ``var/audit/<identity>/`` at
   ``2770`` with the host's own group. Every record written inside the
   container therefore takes the DIRECTORY's group whichever uid wrote it,
   which is what keeps the operator's host-side purge working on files a
   container's uid 1000 created.
3. **Membership, not ownership.** The dropped process reaches the mount
   through the group the entrypoint joined. Proving that needs a directory
   whose OWNER triad cannot answer in the group's place, so the shared bundle
   carries a ``group-only/`` subdirectory that a throwaway ROOT container
   creates at ``2070`` — owned by uid 0, group rwx, nothing for other.
   Neither the container's uid 1000 nor the host account can be its owner, so
   the group is the only door left on every host (see the note above
   :data:`GROUP_ONLY_RELPATH` for the two ways an earlier draft of this,
   which stripped the owner triad off the MOUNT SOURCE, inverted instead).
   The other half is a negative control: a uid-1000 process that never joined
   the group is REFUSED that same mount. The refusal can arrive at either of
   two levels and the control accepts both, because a non-member holds
   nothing on either: at the group-only directory itself, or already at the
   ``2770`` mount root it would have to traverse to reach it, whose owner is
   the host account and whose group is the one it does not hold. Which one
   fires is a property of the runtime, not of the mechanism; the finer fact
   that the child really is ``0:2070`` is asserted from the SUBJECT, which
   could reach it and reported it.

Two containers, because isolation and sharing are opposite halves of the same
render and each needs the other to mean anything:

* **alice** — the web-terminal topology: ``OSPREY_TERMINAL_USER`` set, its own
  ``var/audit/alice/`` bound read-write, and the facility bundle mounted.
* **dispatch-worker-0** — the framework-service topology, with
  ``OSPREY_TERMINAL_USER`` DELIBERATELY UNSET. Its records must still be filed
  under a real name (its service identity, via ``acting_identity()``'s second
  rung) rather than under the process account or ``unknown``, and they must
  land in its own subdirectory and nowhere near alice's.

The second container also reads what the first wrote into the shared bundle and
writes its own file beside it, which is the sharing half: one host directory,
two containers, no ownership in common.

**Linux only, and that is the point.** The reasoning is the one
``tests/deployment/web_terminals/test_bundle_mount.py`` already records for the
same mechanism: setgid propagation "is a Linux behaviour that macOS does not
reproduce, so asserting it would pin a platform rather than the property".
Docker Desktop compounds it by remapping bind-mount ownership, which is exactly
the condition the entrypoint's ``gid < 100`` refusal exists to survive — a
green run there would be evidence of nothing. The Linux CI lane is the proof.
:data:`ALLOW_NON_LINUX_ENV` bypasses the guard for local diagnosis and is
never the proof; the module says so in its skip reason rather than quietly
passing.

Two further gates keep a green run honest rather than lucky:

* **A joinable group.** The entrypoint refuses gid 0 and the system range
  (< 100) by design. A host session whose only group is a system one cannot
  exercise the join at all, so the module skips rather than asserting a
  degraded path — but on a Linux CI runner that condition is a lane
  misconfiguration rather than an ordinary host, so there it fails instead.
  Every one of these skips names the gids the host actually offered, because
  a skip line and a pass line look alike in a CI summary.
* **Id-mapped runtimes.** Under a user-namespaced or rootless runtime the
  container's uid 1000 is some other uid on the host, so the HOST-side
  ownership assertions would be testing the id map rather than the drop. The
  fixture detects the remap by comparing the container's view of the mount's
  owner against the host's, and the two host-side ownership tests skip with
  that reason. The container-side half — uid 1000, the record's own uid/gid,
  the absent writer marker — holds under every runtime. The two claims about
  the JOIN are the one exception, and for a different reason: a runtime that
  presents a bind mount as owned by the CONTAINER's user leaves the entrypoint
  with no group to join at all, so their absence would be a fact about the
  runtime (:func:`_require_a_group_to_join`). On the Linux lane the mount
  carries the host's gid, which is never the image's 1000, and they run.

Cost discipline: ONE image build (``hello-world``, the lightest preset) and
ONE module-scoped fixture that runs the containers, snapshots every host-side
fact, and then performs the purge. Each test reads one field of that snapshot,
so a failure still names exactly which property broke. The build is cheap only
in the relative sense: on a developer host it reuses the layers
``test_dockerfile_e2e.py`` already pulled and compiled, but in CI it is COLD
every time — that module runs on its own ephemeral runner and shares no layer
cache with this one. Measured cold, ``osprey init`` + ``osprey build`` + the
dev-wheel stage + ``docker build`` is about seven minutes. That is why this
module runs as a second step of the ``privilege-split-e2e`` job, whose budget
already absorbs a cold image build and whose cleanup step reaps this module's
``osprey-e2e-audit2c`` prefix, rather than in the shared ``e2e-tests`` lane
(where it is ``--ignore``d, as every ``dockerbuild``-marked module is). Riding
as a step rather than a job means it has no ``needs:`` or ``check_pr_lane``
entry of its own, so what keeps it honest lives in
``tests/deployment/test_ci_workflow_wiring.py``: the step must exist, must carry
``if: success() || failure()`` (the default would skip it behind a red first
step, and skipped reads like passed), and the lane fails outright if either
module collected nothing or skipped in full. A PARTIAL skip stays a warning —
the runtime-shape guards below deliberately degrade rather than lie, and
:func:`_degraded_host` already fails on a CI runner for the class that is a lane
misconfiguration rather than a host. The
container runs are seconds each and share the one image: two probes, one
negative control, and the two root containers that bracket them — one to
create the group-only directory, one in the teardown to widen it back.

The image is built with the local dev wheel, not the PyPI release: the audit
writer under test here is unreleased, and without the staged wheel this module
would build an image that does not contain the code it claims to test and
report green.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.deployment.compose_generator import (
    audit_identity_dir,
    ensure_audit_dir,
    ensure_shared_corpus_dir,
)
from osprey.deployment.wheel_build import _copy_local_framework_for_override
from osprey.utils.workspace import AUDIT_DIR_RELPATH, container_image_context

#: Escape hatch for local diagnosis on a non-Linux host. Deliberately named as
#: a diagnostic: a Docker Desktop run remaps bind-mount ownership, so it can
#: report green while proving nothing, and nothing in CI sets this.
ALLOW_NON_LINUX_ENV = "OSPREY_E2E_AUDIT_MECHANISM_ANYWAY"

#: ``test_bundle_mount.py``'s recorded reasoning, reused verbatim so the two
#: places that decline to assert this on macOS give one answer.
LINUX_ONLY_REASON = (
    "the audit mount's mechanism is Linux behaviour that macOS does not "
    "reproduce (setgid propagation, and bind-mount ownership that Docker "
    "Desktop remaps), so asserting it here would pin a platform rather than "
    f"the property — set {ALLOW_NON_LINUX_ENV}=1 to run it anyway as a "
    "diagnostic, never as the proof"
)


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _linux_enough() -> bool:
    return sys.platform.startswith("linux") or os.environ.get(ALLOW_NON_LINUX_ENV) == "1"


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.dockerbuild,
    pytest.mark.skipif(not _linux_enough(), reason=LINUX_ONLY_REASON),
    pytest.mark.skipif(not _docker_available(), reason="docker binary or daemon not available"),
]

PRESET = "hello-world"
REPO_NAME = "audit2c"
BUILD_TIMEOUT = 1800
RUN_TIMEOUT = 420

#: Tag and container-name prefix, so a stray left by a killed run is
#: recognizable (and greppable) rather than anonymous.
TAG_PREFIX = "osprey-e2e-audit2c"

#: The two topologies. ``alice`` is a roster user (``OSPREY_TERMINAL_USER``
#: set); ``dispatch-worker-0`` is a framework service, spelled the way
#: ``compose_generator.DISPATCH_WORKER_SERVICE_PREFIX`` renders it, and run
#: with no terminal user at all.
IDENTITY_TERMINAL = "alice"
IDENTITY_SERVICE = "dispatch-worker-0"

#: The surface the probe records on, and therefore the ledger stem
#: (``<identity>/<surface>.jsonl``) both containers write.
PROBE_SURFACE = "e2e_audit_mechanism"

#: Where the shared bundle mounts, mirroring the shipped
#: ``facility_knowledge.bundle_path`` default anchored on the container project
#: directory (``render._container_bundle_dir``). Named here rather than read
#: from the render because the entrypoint reads no config either: this module
#: plays the render's role and hands the container the same two variables
#: compose does.
BUNDLE_RELPATH = "data/facility_knowledge"

#: The gid floor the entrypoint enforces. Below this it refuses to join —
#: deliberately, because a bind that looks system-owned inside the container is
#: the signature of an ownership remap and not a group the deployment granted.
MIN_JOINABLE_GID = 100

#: Gids used when the test process is root and its own group is in the system
#: range. Two of them, because the bundle assertion is only worth making when
#: the bundle's group is distinguishable from the audit mount's.
ROOT_FALLBACK_GIDS = (60000, 60001)

#: The uid the image pins for the agent's user (``OSPREY_RUNTIME_UID``), and so
#: the uid every record a dropped process writes must carry.
RUNTIME_UID = 1000

#: The group-only directory, a child INSIDE the mounted bundle rather than the
#: mount source itself, created by a throwaway ROOT container (see
#: :func:`_make_group_only_dir`) and owned by uid 0.
#:
#: Both halves of that are load-bearing, and an earlier draft of this module
#: got both wrong in one stroke by chmod'ing the mount source to ``2070``.
#: Linux consults the owner triad EXCLUSIVELY when the accessing euid equals
#: the owner's uid — it never falls through to the group — so stripping the
#: owner triad off a directory the accessor owns DENIES it rather than
#: narrowing it, in two places at once:
#:
#: * **The mount source.** A runtime that resolves the bind source as the host
#:   account (Docker Desktop, rootless Docker) then cannot create the mount at
#:   all: ``docker run`` dies with "error while creating mount source path
#:   ...: permission denied" before a single assertion runs, after paying for
#:   the whole image build. The mount source therefore keeps the mode
#:   ``ensure_shared_corpus_dir`` ships (:data:`SHARED_MODE`, host-owned) and
#:   only this child goes group-only.
#: * **The container.** On a host whose invoking uid is 1000 — the default
#:   first user on Debian/Ubuntu, and the uid this image pins for ``osprey`` —
#:   the dropped process IS the owner of a host-owned directory, so ``2070``
#:   denies its write and a healthy mechanism reads as broken. Owned by uid 0,
#:   the container's 1000 can never be the owner, and the assertion is
#:   unconditional rather than host-account-dependent.
GROUP_ONLY_RELPATH = "group-only"

#: Mode of that directory: setgid, group rwx, and NOTHING for owner or other.
#: A write that succeeds under it came through the group.
GROUP_ONLY_MODE = 0o2070

#: The mode ``ensure_shared_corpus_dir`` provisions, and the one the mount
#: source keeps throughout, so both the host and the runtime's file-sharing
#: layer can resolve it.
SHARED_MODE = 0o2770

#: A small public image used for exactly one thing, BEFORE the expensive
#: build: proving this runtime can resolve the bundle directory as a
#: bind-mount source at all. A pull failure leaves the pre-flight inconclusive
#: and the run continues — an offline host is not a defect in the mechanism.
PREFLIGHT_IMAGE = "busybox:stable"

#: The probe: the container's actual command, and therefore the post-``gosu``
#: process itself. It reports its own credentials, drives the REAL writer (so
#: the path under test is the one the framework resolves, not one this test
#: composed), and touches the shared bundle. It prints exactly one line of JSON
#: to stdout — the entrypoint's own diagnostics all go to stderr, which is what
#: keeps a container command's stdout its own.
PROBE = r"""
import json
import os
from pathlib import Path

out = {
    "uid": os.getuid(),
    "gid": os.getgid(),
    "groups": sorted(os.getgroups()),
    # The entrypoint unsets this before the drop; anything else here means the
    # maintenance phase's marker survived gosu and every app record would claim
    # to have come from the root phase.
    "writer_marker": os.environ.get("OSPREY_AUDIT_WRITER"),
}

from osprey.audit.writer import audit_dir, record
from osprey.utils.identity import acting_identity

out["identity"] = acting_identity()
out["audit_dir"] = str(audit_dir())

mounted = os.environ.get("OSPREY_AUDIT_DIR", "")
out["mounted_audit_dir"] = mounted
if mounted and os.path.isdir(mounted):
    st = os.stat(mounted)
    out["audit_dir_stat"] = {
        "uid": st.st_uid,
        "gid": st.st_gid,
        "mode": oct(st.st_mode & 0o7777),
    }

path = record(
    surface=os.environ["OSPREY_E2E_SURFACE"],
    posture="sandbox",
    posture_source="process",
    session=None,
    subject="tests/e2e/test_audit_two_container.py",
    decision="refused",
    reason="e2e_mechanism_probe",
    detail=os.environ["OSPREY_E2E_MARKER"],
)
out["record_path"] = str(path) if path is not None else None
if path is not None:
    st = os.stat(path)
    out["record_stat"] = {"uid": st.st_uid, "gid": st.st_gid}

bundle = os.environ.get("OSPREY_FACILITY_BUNDLE_DIR", "")
out["bundle"] = {}
if bundle:
    if os.path.isdir(bundle):
        st = os.stat(bundle)
        out["bundle"]["dir_stat"] = {
            "uid": st.st_uid,
            "gid": st.st_gid,
            "mode": oct(st.st_mode & 0o7777),
        }
    read_name = os.environ.get("OSPREY_E2E_BUNDLE_READ", "")
    if read_name:
        try:
            out["bundle"]["read"] = Path(bundle, read_name).read_text(encoding="utf-8")
        except OSError as exc:
            out["bundle"]["read_error"] = repr(exc)
    write_name = os.environ.get("OSPREY_E2E_BUNDLE_WRITE", "")
    if write_name:
        target = Path(bundle, write_name)
        # The directory the write actually lands in — a child of the mount,
        # not the mount root — reported BEFORE the attempt so a denial still
        # says what it was denied on.
        try:
            dst = os.stat(target.parent)
            out["bundle"]["write_dir_stat"] = {
                "uid": dst.st_uid,
                "gid": dst.st_gid,
                "mode": oct(dst.st_mode & 0o7777),
            }
        except OSError as exc:
            out["bundle"]["write_dir_error"] = repr(exc)
        try:
            target.write_text(os.environ["OSPREY_E2E_MARKER"], encoding="utf-8")
            fst = target.stat()
            out["bundle"]["write"] = {"uid": fst.st_uid, "gid": fst.st_gid}
        except OSError as exc:
            out["bundle"]["write_error"] = repr(exc)

print(json.dumps(out))
"""

#: The negative control's script. Deliberately tiny and deliberately NOT the
#: probe above: it runs with ``--user 1000:1000 --entrypoint python``, which
#: bypasses the entrypoint entirely — no ``/etc/group`` join, no supplementary
#: groups — so ``osprey.audit.writer`` is not even in play. It reports the
#: groups it holds (so "denied" can be attributed to the missing membership
#: rather than assumed) and then tries the write that must fail.
#:
#: It stats the target's parent BEFORE writing, but that stat is instrumentation
#: rather than a precondition: on a faithful Linux runtime a non-member cannot
#: traverse the ``2770`` mount root either, so the stat is itself refused and
#: the report carries ``dir_error`` instead of ``dir_stat``. Both spellings are
#: the same refusal of the same mount — see the caller.
NEG_PROBE = r"""
import json
import os
from pathlib import Path

target = Path(os.environ["OSPREY_E2E_TARGET"])
out = {"uid": os.getuid(), "gid": os.getgid(), "groups": sorted(os.getgroups())}
try:
    st = os.stat(target.parent)
    out["dir_stat"] = {"uid": st.st_uid, "gid": st.st_gid, "mode": oct(st.st_mode & 0o7777)}
except OSError as exc:
    out["dir_error"] = repr(exc)
try:
    target.write_text("negative-control", encoding="utf-8")
    out["write"] = "succeeded"
except OSError as exc:
    out["write_error"] = repr(exc)

print(json.dumps(out))
"""


# ── host-side helpers ────────────────────────────────────────────────────────


def _degraded_host(reason: str) -> None:
    """Skip on an ordinary host; FAIL on a Linux CI runner.

    A session that cannot offer the groups this proof needs is an unremarkable
    condition on a developer machine and a lane misconfiguration on a runner —
    the module is wired into a job precisely so that it RUNS there, and in a CI
    summary a skipped test and a passing one are the same line. Every caller
    passes a reason that names the gids the host actually offered, so the log
    says what was on hand rather than only that something was missing.
    """
    if sys.platform.startswith("linux") and os.environ.get("CI"):
        pytest.fail(f"{reason} — on a CI runner that is a lane misconfiguration, not a host")
    pytest.skip(reason)


def _joinable_gids() -> tuple[int, int, bool, tuple[int, ...]]:
    """Two gids this host can hand a mount, and whether they are distinct.

    The entrypoint refuses anything below :data:`MIN_JOINABLE_GID`, so a gid
    the test process happens to run under is only usable if it clears that
    floor. Root may name any gid; an unprivileged process may only chgrp to a
    group it is already in.

    :return: ``(audit_gid, bundle_gid, distinct, offered)``. When the host
        offers only one usable group both gids are the same and *distinct* is
        ``False``, which the bundle-membership test reads to skip (or, on CI,
        fail via :func:`_degraded_host`) rather than assert something the
        audit join already guarantees. *offered* is every gid this SESSION
        carries, usable or not, and travels with the run so that a degraded
        lane reports what the host had rather than only what was missing.
    """
    offered = tuple(sorted({os.getgid(), *os.getgroups()}))
    if os.geteuid() == 0:
        # Root may chgrp to any gid, so the session's own groups do not limit
        # it — they are still carried, because they are what a CI log needs to
        # explain a runner whose account changed underneath the lane.
        return (*ROOT_FALLBACK_GIDS, True, offered)

    usable = [gid for gid in offered if gid >= MIN_JOINABLE_GID]
    if not usable:
        _degraded_host(
            "this session's groups are all in the system range (< "
            f"{MIN_JOINABLE_GID}), which the entrypoint refuses to join by "
            f"design — there is no joinable group to provision a mount with "
            f"(this host offered {list(offered)})"
        )
    if len(usable) == 1:
        return usable[0], usable[0], False, offered
    return usable[0], usable[1], True, offered


def _regroup(target: Path, gid: int, mode: int) -> None:
    """Put *target* in *gid* at *mode*, in that order.

    The chmod follows the chown because a chown may clear the setgid bit, and
    setgid is the half that makes a container's record inherit the host group.
    """
    os.chown(target, -1, gid)
    os.chmod(target, mode)


def _docker(*args: str, timeout: int = RUN_TIMEOUT):
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _run_probe(
    tag: str,
    *,
    env: dict[str, str],
    mounts: list[tuple[Path, str]],
    script: str = PROBE,
    extra_args: tuple[str, ...] = (),
    command: tuple[str, ...] = ("python", "-"),
) -> dict:
    """Run a probe script as the container's command and return its report.

    ``docker run`` in the foreground with the script on stdin, so the process
    under observation is the one the entrypoint dropped to. ``--rm`` removes
    the container on exit; the ``finally`` covers the case where the run timed
    out and ``--rm`` never fired.

    *extra_args* exists for the ONE caller that legitimately bypasses the
    entrypoint — the negative control, which must hold no membership at all.
    Every other caller leaves it empty: see the module's note on ``docker
    exec``, which is the same mistake wearing a different flag. That caller
    also passes *command*, because ``--entrypoint python`` already supplies
    the interpreter and the default would run ``python python -``.
    """
    name = f"{TAG_PREFIX}-{uuid.uuid4().hex[:8]}"
    args = ["run", "--rm", "-i", "--name", name, *extra_args]
    for key, value in env.items():
        args += ["-e", f"{key}={value}"]
    for source, target in mounts:
        args += ["-v", f"{source}:{target}"]
    args += [tag, *command]
    try:
        proc = subprocess.run(
            ["docker", *args],
            input=script,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
        )
    finally:
        _docker("rm", "-f", name, timeout=120)

    assert proc.returncode == 0, (
        f"probe container exited {proc.returncode}\n--- stdout ---\n{proc.stdout[-3000:]}"
        f"\n--- stderr ---\n{proc.stderr[-4000:]}"
    )
    report = _parse_report(proc.stdout)
    report["_stderr"] = proc.stderr
    return report


def _parse_report(stdout: str) -> dict:
    """The last JSON object the probe printed.

    Read from the end rather than by position: the container's stdout is its
    own by contract (the entrypoint logs to stderr), but a dependency that
    prints a deprecation warning on import must not be able to break this.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    raise AssertionError(f"probe printed no JSON report:\n{stdout[-3000:]}")


def _make_group_only_dir(tag: str, bundle_host: Path, bundle_container: str, gid: int) -> None:
    """Create the group-only directory INSIDE the bundle, as uid 0.

    A throwaway container with ``--entrypoint sh`` rather than the host,
    because the point of the directory is an owner the host account cannot be
    (see :data:`GROUP_ONLY_RELPATH`) and an unprivileged host process cannot
    ``chown`` anything to root. Bypassing the entrypoint here is setup, not
    subject: nothing about the privilege drop is observed in this run.

    ``chown`` before ``chmod``, for the reason :func:`_regroup` records — a
    chown can clear the setgid bit, and setgid is the half that makes the two
    containers' files share one group.
    """
    target = f"{bundle_container}/{GROUP_ONLY_RELPATH}"
    name = f"{TAG_PREFIX}-setup-{uuid.uuid4().hex[:8]}"
    try:
        proc = _docker(
            "run",
            "--rm",
            "--name",
            name,
            "--entrypoint",
            "sh",
            "-v",
            f"{bundle_host}:{bundle_container}",
            tag,
            "-c",
            f'set -e; mkdir -p "{target}"; chown 0:{gid} "{target}"; '
            f'chmod {oct(GROUP_ONLY_MODE)[2:]} "{target}"; stat -c "%u %g %a" "{target}"',
        )
    finally:
        _docker("rm", "-f", name, timeout=120)
    assert proc.returncode == 0, (
        f"could not create the group-only directory as root:\n{proc.stdout}\n{proc.stderr}"
    )


def _restore_host_access(tag: str, bundle_host: Path, bundle_container: str) -> None:
    """Widen the group-only directory back to :data:`SHARED_MODE`, as uid 0.

    The group-only mode is needed only WHILE the subjects run — every fact
    about it is reported from inside the containers. Afterwards the host has
    to read what they wrote and eventually delete it, and it may be unable to:
    a rootful daemon leaves the directory owned by root, and a rootless one
    leaves it owned by the host account with no owner triad, which denies the
    owner outright (the same exclusivity :data:`GROUP_ONLY_RELPATH` records).
    ``2770`` clears both cases at once — owner rwx if the host owns it, group
    rwx if root does — and only a container can set it, because the host is
    not the owner in the case that matters.

    Best-effort by design: a runtime that ignores the mode leaves the host
    unable to stat the files, which the snapshot records and the host-side
    bundle test skips on, rather than turning into an unreadable teardown
    error.
    """
    target = f"{bundle_container}/{GROUP_ONLY_RELPATH}"
    name = f"{TAG_PREFIX}-restore-{uuid.uuid4().hex[:8]}"
    try:
        _docker(
            "run",
            "--rm",
            "--name",
            name,
            "--entrypoint",
            "sh",
            "-v",
            f"{bundle_host}:{bundle_container}",
            tag,
            "-c",
            f'chmod {oct(SHARED_MODE)[2:]} "{target}" || true',
        )
    finally:
        _docker("rm", "-f", name, timeout=120)


def _stat_or_error(path: Path) -> tuple[os.stat_result | None, str | None]:
    """``(stat, None)``, ``(None, None)`` when absent, ``(None, repr)`` when the
    host cannot look — which is itself a fact worth carrying, not an exception
    to raise inside a fixture that still has cleanup to do."""
    try:
        return path.stat(), None
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, repr(exc)


def _mount_source_usable(bundle_host: Path) -> tuple[bool, str]:
    """Can this runtime resolve *bundle_host* as a bind-mount source?

    One throwaway container against a tiny public image, run BEFORE the image
    fixture. It exists because the failure it catches costs a full cold image
    build to discover otherwise, and surfaces as an unrelated-looking exit 126
    ("error while creating mount source path ...: permission denied") rather
    than as anything about auditing. Structurally this should no longer be
    reachable — the mount source keeps its owner triad now — so a failure here
    means the runtime is denying something more fundamental.

    :return: ``(usable, detail)``. A pull failure returns ``True`` with a note:
        an offline host makes the pre-flight inconclusive, not failed.
    """
    pulled = _docker("pull", PREFLIGHT_IMAGE, timeout=180)
    if pulled.returncode != 0:
        return True, f"inconclusive: could not pull {PREFLIGHT_IMAGE}"
    # Named with the module's prefix like every other container it starts, so
    # a stray from a killed run is greppable. The IMAGE is left alone: it is a
    # public one this host may well have had already, and removing someone
    # else's cached layer is not this module's business.
    name = f"{TAG_PREFIX}-preflight-{uuid.uuid4().hex[:8]}"
    try:
        probe = _docker(
            "run",
            "--rm",
            "--name",
            name,
            "-v",
            f"{bundle_host}:/probe",
            PREFLIGHT_IMAGE,
            "true",
            timeout=120,
        )
    finally:
        _docker("rm", "-f", name, timeout=120)
    if probe.returncode != 0:
        return False, f"exit {probe.returncode}: {probe.stderr.strip()[-500:]}"
    return True, "ok"


def _ledger(repo: Path, identity: str) -> Path:
    """The host-side path a record on :data:`PROBE_SURFACE` must appear at.

    Spelled through the same helper the deploy path provisions with, so this
    test cannot agree with itself about a path the framework would resolve
    differently.
    """
    return audit_identity_dir(repo, identity) / f"{PROBE_SURFACE}.jsonl"


# ── the image ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> Path:
    """One ``osprey init`` + ``osprey build`` of the lightest preset.

    ``--no-git`` because nothing reads history and the tmp tree may sit inside
    another repository; ``--skip-deps``/``--skip-lifecycle`` because the image
    installs its own dependencies and no service is started here.
    """
    target = tmp_path_factory.mktemp("audit-two-container") / REPO_NAME
    runner = CliRunner()
    created = runner.invoke(cli, ["init", str(target), "--preset", PRESET, "--no-git"])
    assert created.exit_code == 0, created.output
    built = runner.invoke(cli, ["build", "--repo", str(target), "--skip-deps", "--skip-lifecycle"])
    assert built.exit_code == 0, built.output
    return target


@pytest.fixture(scope="module")
def project(repo: Path) -> str:
    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    return config["project_name"]


@pytest.fixture(scope="module")
def host_zone(repo: Path):
    """The host half of the render: the directories, their gids, their modes.

    Provisioned by the same helpers ``osprey up`` calls, so the directories
    under test are the ones the deploy path makes. Split out of the run
    fixture below so the mount pre-flight can look at the real bundle
    directory BEFORE the image build spends its budget.
    """
    audit_gid, bundle_gid, distinct, offered = _joinable_gids()

    ensure_audit_dir(repo, IDENTITY_TERMINAL)
    ensure_audit_dir(repo, IDENTITY_SERVICE)
    bundle_host = repo / BUNDLE_RELPATH
    ensure_shared_corpus_dir(bundle_host)

    alice_host = audit_identity_dir(repo, IDENTITY_TERMINAL)
    worker_host = audit_identity_dir(repo, IDENTITY_SERVICE)
    _regroup(alice_host, audit_gid, SHARED_MODE)
    _regroup(worker_host, audit_gid, SHARED_MODE)
    # The mount source stays exactly as shipped. Only its `group-only/` child
    # goes 2070, and a root container makes it — see GROUP_ONLY_RELPATH.
    _regroup(bundle_host, bundle_gid, SHARED_MODE)

    return SimpleNamespace(
        audit_gid=audit_gid,
        bundle_gid=bundle_gid,
        distinct_gids=distinct,
        offered_gids=offered,
        bundle_host=bundle_host,
        alice_host=alice_host,
        worker_host=worker_host,
    )


@pytest.fixture(scope="module")
def mount_preflight(host_zone) -> str:
    """Prove the runtime can bind-mount the bundle at all, before the build.

    Seconds against a tiny public image, ahead of a build that costs minutes:
    a runtime that cannot resolve the mount source produces an exit-126
    message about mount paths that says nothing about auditing, and paying for
    the whole image first to read it is the wrong order.
    """
    usable, detail = _mount_source_usable(host_zone.bundle_host)
    if not usable:
        pytest.skip(
            f"this container runtime cannot use {host_zone.bundle_host} as a "
            f"bind-mount source ({detail}), so every container run below would "
            "fail on the mount rather than on anything this module asserts"
        )
    return detail


@pytest.fixture(scope="module")
def image(repo: Path, project: str, mount_preflight: str):
    """The deployment's own image, built with the local dev wheel staged in."""
    context = container_image_context(repo, project)
    tag = f"{TAG_PREFIX}:{uuid.uuid4().hex[:8]}"
    staged = _copy_local_framework_for_override(str(context))
    assert staged, f"could not stage the local dev wheel into {context}"
    build = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            str(context / "build" / "Dockerfile"),
            "-t",
            tag,
            "--build-arg",
            "OSPREY_DEV=1",
            "--label",
            f"com.osprey.project={context.name}",
            ".",
        ],
        cwd=context,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
        env={**os.environ, "DOCKER_BUILDKIT": "1"},
    )
    assert build.returncode == 0, (
        f"docker build failed for {tag}:\n--- stdout ---\n{build.stdout[-4000:]}"
        f"\n--- stderr ---\n{build.stderr[-4000:]}"
    )
    try:
        yield tag
    finally:
        _docker("rmi", "-f", tag, timeout=120)


# ── the one run all the containers share ─────────────────────────────────────


@pytest.fixture(scope="module")
def two_containers(repo: Path, project: str, image: str, host_zone):
    """Run the containers, snapshot every host-side fact, then purge.

    One fixture rather than one per test because the containers are the
    expensive step and every assertion below is about a different property of
    the same runs. The purge happens here, last, for the same reason: it
    destroys the very files the ownership assertions read, so it cannot be a
    test that happens to run after them.

    Five container runs, all from the one image: a root setup container that
    makes the group-only directory, the two subjects, the negative control
    that must be refused where the subjects are allowed, and the root teardown
    container that widens the directory back.
    """
    audit_gid = host_zone.audit_gid
    bundle_gid = host_zone.bundle_gid
    bundle_host = host_zone.bundle_host

    marker_alice = f"audit2c-alice-{uuid.uuid4().hex[:12]}"
    marker_worker = f"audit2c-worker-{uuid.uuid4().hex[:12]}"
    # Written into the group-only child, not into the mount root: the mount
    # root is reachable by its owner and would prove nothing about the group.
    alice_file = f"{GROUP_ONLY_RELPATH}/from-alice.txt"
    worker_file = f"{GROUP_ONLY_RELPATH}/from-worker.txt"
    control_file = f"{GROUP_ONLY_RELPATH}/from-a-non-member.txt"

    container_root = f"/app/{project}"
    bundle_container = f"{container_root}/{BUNDLE_RELPATH}"
    alice_container = f"{container_root}/{AUDIT_DIR_RELPATH}/{IDENTITY_TERMINAL}"
    worker_container = f"{container_root}/{AUDIT_DIR_RELPATH}/{IDENTITY_SERVICE}"

    audit_root = repo / AUDIT_DIR_RELPATH
    purge = {"error": None, "survived": True}
    group_only_made = False
    host_access_restored = False

    def hand_the_directory_back() -> None:
        """Widen the group-only directory back to :data:`SHARED_MODE`, once.

        Called from two places, deliberately. On the success path it has to run
        BEFORE the host-side snapshot below: under a runtime that resolves the
        mount source as the host account, the host owns a directory with an
        empty owner triad and cannot so much as traverse it, and the snapshot
        would record stat errors for files that are perfectly fine. And it is
        called again from the outer ``finally``, because a probe that raises —
        every non-zero exit inside :func:`_run_probe` is an assertion — must
        not leave a root-owned ``2070`` directory behind for the tmp reaper.
        The flag makes the second call a no-op after the first.
        """
        nonlocal host_access_restored
        if host_access_restored:
            return
        host_access_restored = True
        _restore_host_access(image, bundle_host, bundle_container)

    try:
        _make_group_only_dir(image, bundle_host, bundle_container, bundle_gid)
        group_only_made = True

        alice = _run_probe(
            image,
            env={
                # The web-terminal topology: a named user, its own audit
                # identity, and the bundle it is entitled to.
                "OSPREY_TERMINAL_USER": IDENTITY_TERMINAL,
                "OSPREY_AUDIT_IDENTITY": IDENTITY_TERMINAL,
                "OSPREY_AUDIT_DIR": alice_container,
                "OSPREY_FACILITY_BUNDLE_DIR": bundle_container,
                "OSPREY_E2E_SURFACE": PROBE_SURFACE,
                "OSPREY_E2E_MARKER": marker_alice,
                "OSPREY_E2E_BUNDLE_WRITE": alice_file,
            },
            mounts=[(host_zone.alice_host, alice_container), (bundle_host, bundle_container)],
        )
        worker = _run_probe(
            image,
            env={
                # The framework-service topology: NO terminal user at all, so
                # the identity has to come off the second rung of the ladder.
                "OSPREY_AUDIT_IDENTITY": IDENTITY_SERVICE,
                "OSPREY_AUDIT_DIR": worker_container,
                "OSPREY_FACILITY_BUNDLE_DIR": bundle_container,
                "OSPREY_E2E_SURFACE": PROBE_SURFACE,
                "OSPREY_E2E_MARKER": marker_worker,
                "OSPREY_E2E_BUNDLE_READ": alice_file,
                "OSPREY_E2E_BUNDLE_WRITE": worker_file,
            },
            mounts=[(host_zone.worker_host, worker_container), (bundle_host, bundle_container)],
        )
        # The negative control. Same image, same mount, same uid — and no
        # group, because `--entrypoint` skips the join and `--user` grants no
        # supplementary groups. It turns "reachable only through membership"
        # from an argument about the permission model into an observation.
        control = _run_probe(
            image,
            env={"OSPREY_E2E_TARGET": f"{bundle_container}/{control_file}"},
            mounts=[(bundle_host, bundle_container)],
            script=NEG_PROBE,
            extra_args=("--user", f"{RUNTIME_UID}:{RUNTIME_UID}", "--entrypoint", "python"),
            command=("-",),
        )
        # Every container that had to be refused has now run, so hand the
        # directory back before the host is asked to read what they wrote.
        hand_the_directory_back()

        ledgers = {
            IDENTITY_TERMINAL: _ledger(repo, IDENTITY_TERMINAL),
            IDENTITY_SERVICE: _ledger(repo, IDENTITY_SERVICE),
        }
        host_stat = {
            identity: (path.stat() if path.exists() else None) for identity, path in ledgers.items()
        }
        host_text = {
            identity: (path.read_text(encoding="utf-8") if path.exists() else "")
            for identity, path in ledgers.items()
        }
        bundle_files = {}
        bundle_stat_errors = {}
        for name in (alice_file, worker_file, control_file):
            bundle_files[name], error = _stat_or_error(bundle_host / name)
            if error is not None:
                bundle_stat_errors[name] = error

        # An id-mapped runtime (rootless, or a userns-remapped daemon) makes
        # the HOST-side uid/gid a function of the map rather than of the drop.
        # Detected by asking the container what it sees the mount's owner as:
        # equal means no map is in play and the host-side ownership assertions
        # mean what they say. Asserted present rather than defaulted: the
        # audit mount is bound unconditionally, so a missing stat is a broken
        # run, not a remapped one, and must not be reported as an id map.
        seen = alice.get("audit_dir_stat")
        assert seen is not None, (
            "the probe reported no audit_dir_stat, but OSPREY_AUDIT_DIR is "
            "always bound — the mount is missing inside the container, or is "
            f"not a directory: {alice}"
        )
        id_mapped = seen["uid"] != os.getuid() or seen["gid"] != audit_gid
    finally:
        # Both halves of the cleanup, last and UNCONDITIONAL. The widening
        # first, because a failed probe above skipped its success-path call and
        # would otherwise strand a root-owned 2070 directory in the tmp tree.
        if group_only_made:
            hand_the_directory_back()
        # Then the purge: it is the operator's host-side `osprey reset
        # --purge-audit`, and it is also this module's cleanup. On the failure
        # path the assertion below no longer runs, but the container-written
        # tree still has to go.
        try:
            if audit_root.exists():
                shutil.rmtree(audit_root)
        except OSError as exc:  # pragma: no cover — the failure this catches
            purge["error"] = repr(exc)
        purge["survived"] = audit_root.exists()

    yield SimpleNamespace(
        alice=alice,
        worker=worker,
        control=control,
        audit_gid=audit_gid,
        bundle_gid=bundle_gid,
        distinct_gids=host_zone.distinct_gids,
        offered_gids=host_zone.offered_gids,
        id_mapped=id_mapped,
        markers={IDENTITY_TERMINAL: marker_alice, IDENTITY_SERVICE: marker_worker},
        ledgers=ledgers,
        host_stat=host_stat,
        host_text=host_text,
        bundle_host=bundle_host,
        bundle_files=bundle_files,
        bundle_stat_errors=bundle_stat_errors,
        alice_file=alice_file,
        worker_file=worker_file,
        control_file=control_file,
        audit_root_after_purge=purge["survived"],
        purge_error=purge["error"],
        container_audit_dirs={
            IDENTITY_TERMINAL: alice_container,
            IDENTITY_SERVICE: worker_container,
        },
    )


def _require_direct_ids(snapshot) -> None:
    """Skip when the runtime remaps ids between the container and the host."""
    if snapshot.id_mapped:
        pytest.skip(
            "this container runtime remaps ids between container and host "
            f"(the mount the host owns as {os.getuid()}:{snapshot.audit_gid} "
            f"appears inside as {snapshot.alice.get('audit_dir_stat')}), so a "
            "host-side uid assertion would be testing the id map rather than "
            "the privilege drop — the container-side half above still holds"
        )


def _require_a_group_to_join(snapshot) -> None:
    """Skip when the runtime handed the mounts to the container's OWN user.

    Docker Desktop presents a bind mount as owned by the container's user, so
    the gid the entrypoint stats off it *is* the dropped process's primary gid:
    there is no group left to join, no "joined osprey to group" line, and
    ``os.getgroups()`` comes back empty — none of which says anything about
    whether the join works. A Linux host that provisions the mount with gid
    1000 lands in the same place for the same reason.

    Never fires on the lane this module is wired into: there the mount carries
    the host's own gid, which is not the image's 1000. And it cannot mask a
    broken join, because a join that stopped working leaves the mount's gid
    foreign and the assertion still bites.
    """
    for probe in (snapshot.alice, snapshot.worker):
        mounted_gid = probe["audit_dir_stat"]["gid"]
        if mounted_gid == probe["gid"]:
            pytest.skip(
                "this runtime shows the audit mount as owned by the dropped "
                f"process's own primary group (gid {mounted_gid}), so the "
                "entrypoint had no group to join and its absence is a fact "
                "about the runtime rather than about the /etc/group step"
            )


def _require_the_host_can_read_the_bundle(snapshot) -> None:
    """Skip when the host cannot look inside the group-only directory.

    Only reachable when the teardown container's ``chmod`` did not take —
    i.e. the runtime rewrites bind-mount ownership on its own terms, and a
    host-side assertion would be about that rewriting rather than about the
    mount. The container-side half of the same property still holds.
    """
    if snapshot.bundle_stat_errors:
        pytest.skip(
            "the host cannot stat what the containers wrote into the "
            f"group-only directory ({snapshot.bundle_stat_errors}); this "
            "runtime did not honour the teardown chmod, so a host-side "
            "ownership assertion would be testing its ownership rewriting"
        )


def _group_only_dir(probe: dict) -> dict:
    """The group-only directory as the container itself saw it."""
    return probe["bundle"]["write_dir_stat"]


def _is_permission_denied(rendered: str | None) -> bool:
    """Is *rendered* the ``repr`` of an EACCES the probe caught?

    The probes report an ``OSError`` as its ``repr``, which carries both the
    class and the strerror. Both are checked: ``PermissionError`` alone would
    also match EPERM, and the message alone would match a class this test does
    not mean (``FileNotFoundError`` never says it, but a future probe change
    should not be able to widen this silently).
    """
    return bool(rendered) and "PermissionError" in rendered and "Permission denied" in rendered


def _require_the_group_is_the_only_door(snapshot) -> None:
    """Skip when the runtime did not give the directory a foreign owner.

    The proof rests on the container's uid NOT being the directory's owner:
    Linux consults the owner triad exclusively when they match, so an owner of
    1000 would answer in the group's place. A rootful daemon honours the
    setup container's ``chown 0``; a runtime that rewrites bind-mount
    ownership (Docker Desktop) may not, and there the run proves nothing.
    """
    seen = _group_only_dir(snapshot.alice)
    if seen["uid"] == RUNTIME_UID:
        pytest.skip(
            "this runtime did not honour the setup container's chown: the "
            f"group-only directory appears inside as {seen}, owned by the "
            "same uid the dropped process runs as, so the OWNER triad would "
            "answer and a successful write would say nothing about the group"
        )
    # The entrypoint stats the gid off the directory OSPREY_FACILITY_BUNDLE_DIR
    # names — the mount root — and joins that. A runtime that rewrites a bind
    # mount's apparent ownership (Docker Desktop presents the mount root as the
    # container's own user) hands it a different gid from the one on the
    # directory under test, so the join could not have matched and a denial
    # says nothing about the mechanism. Detected by comparing the mount root
    # with its OWN CHILD as the container sees them: a faithful runtime shows
    # one gid on both. A regression in the entrypoint's join still bites,
    # because it leaves the two gids equal and the write denied.
    mount_root = snapshot.alice["bundle"]["dir_stat"]
    if seen["gid"] != mount_root["gid"]:
        pytest.skip(
            "this runtime rewrote the bind mount's apparent ownership: the "
            f"bundle root appears inside as {mount_root} while its own child "
            f"appears as {seen}, so the gid the entrypoint stat'ed off the "
            "mount is not the gid on the directory under test — the "
            "container-side half above still holds"
        )


# ── the drop itself ──────────────────────────────────────────────────────────


class TestTheDroppedProcess:
    """What the container's real command is, once the entrypoint is done."""

    def test_both_containers_run_the_command_as_the_runtime_uid(self, two_containers):
        """Not root, and not by a ``USER`` line — this image has none. The
        entrypoint's ``exec gosu`` is the only thing that makes it true."""
        assert two_containers.alice["uid"] == RUNTIME_UID, two_containers.alice
        assert two_containers.worker["uid"] == RUNTIME_UID, two_containers.worker

    def test_the_dropped_process_joined_the_audit_mounts_group(self, two_containers):
        """The grant that survives ``gosu``. Asserted against the gid the
        container itself reports for the mount, so it is the same number the
        entrypoint stat'ed rather than one this test assumed."""
        _require_a_group_to_join(two_containers)
        for probe in (two_containers.alice, two_containers.worker):
            mounted_gid = probe["audit_dir_stat"]["gid"]
            assert mounted_gid in probe["groups"], (
                f"the dropped process is not in the audit mount's group: {probe}"
            )

    def test_the_entrypoint_says_which_groups_it_joined(self, two_containers):
        """The mechanism named in the container's own log, so a future change
        that grants access some other way does not read as this one working."""
        _require_a_group_to_join(two_containers)
        assert "joined osprey to group" in two_containers.alice["_stderr"], two_containers.alice[
            "_stderr"
        ][-3000:]
        assert "dropping privileges to the osprey user" in two_containers.alice["_stderr"]

    def test_no_writer_marker_survives_the_drop(self, two_containers):
        """``OSPREY_AUDIT_WRITER`` is a per-command prefix on the root
        maintenance step and is unset before the drop. If it crossed, every
        record the app wrote would be filed as the root phase's — one uid per
        file inverted, in the shipped image rather than in a stub."""
        assert two_containers.alice["writer_marker"] is None, two_containers.alice
        assert two_containers.worker["writer_marker"] is None, two_containers.worker


# ── the record, and where it landed ──────────────────────────────────────────


class TestTheRecordOnTheHost:
    """Written inside, read outside — the whole point of the bind."""

    def test_each_container_files_under_its_own_identity(self, two_containers):
        """The record the container wrote is on the HOST, under the subdir the
        render named for that identity, with that container's marker in it."""
        for identity in (IDENTITY_TERMINAL, IDENTITY_SERVICE):
            assert two_containers.host_stat[identity] is not None, (
                f"no record at {two_containers.ledgers[identity]}; probe said "
                f"{(two_containers.alice if identity == IDENTITY_TERMINAL else two_containers.worker)}"
            )
            assert two_containers.markers[identity] in two_containers.host_text[identity]

    def test_neither_container_can_reach_the_others_records(self, two_containers):
        """Isolation is at the MOUNT, not at a permission: each container binds
        only its own subdirectory, so the other's marker cannot appear in it."""
        assert (
            two_containers.markers[IDENTITY_SERVICE]
            not in two_containers.host_text[IDENTITY_TERMINAL]
        )
        assert (
            two_containers.markers[IDENTITY_TERMINAL]
            not in two_containers.host_text[IDENTITY_SERVICE]
        )

    def test_the_writer_resolved_the_path_the_render_mounted(self, two_containers):
        """The record's path is not this test's arithmetic: the writer resolved
        it from the container's project root, and it has to equal the directory
        compose bound there. A drift makes records accumulate in the
        container's writable layer while the mount stays empty."""
        for identity, probe in (
            (IDENTITY_TERMINAL, two_containers.alice),
            (IDENTITY_SERVICE, two_containers.worker),
        ):
            expected = f"{two_containers.container_audit_dirs[identity]}/{PROBE_SURFACE}.jsonl"
            assert probe["record_path"] == expected, probe

    def test_the_unset_terminal_user_topology_files_under_the_service_identity(
        self, two_containers
    ):
        """A framework service has no terminal user. Its records must still
        name somebody real — its own service identity, off the ladder's second
        rung — rather than the process account or ``unknown``."""
        assert two_containers.worker["identity"] == IDENTITY_SERVICE, two_containers.worker
        stored = json.loads(two_containers.host_text[IDENTITY_SERVICE].splitlines()[-1])
        assert stored["actor"] == IDENTITY_SERVICE, stored

    def test_the_record_is_written_by_the_dropped_process_into_the_mounts_group(
        self, two_containers
    ):
        """The mechanism, seen from inside: the uid on the record is the
        dropped process's, and the gid is the mounted directory's — setgid,
        not the writer's primary group. True under every runtime, id maps
        included, because both numbers come from the same namespace."""
        for probe in (two_containers.alice, two_containers.worker):
            assert probe["record_stat"]["uid"] == RUNTIME_UID, probe
            assert probe["record_stat"]["gid"] == probe["audit_dir_stat"]["gid"], probe

    def test_the_host_sees_that_same_ownership(self, two_containers):
        """And from outside: uid 1000 wrote it, the host's own group owns it.
        The second half is what keeps a host-side purge working on a file the
        operator's account never created."""
        _require_direct_ids(two_containers)
        for identity in (IDENTITY_TERMINAL, IDENTITY_SERVICE):
            st = two_containers.host_stat[identity]
            assert st.st_uid == RUNTIME_UID, f"{identity}: {st}"
            assert st.st_gid == two_containers.audit_gid, f"{identity}: {st}"

    def test_the_host_can_purge_the_zone_the_containers_wrote(self, two_containers):
        """The operator's ``osprey reset --purge-audit`` runs as the host
        account, over files a container's uid owns. It works because the
        DIRECTORIES are the operator's and group-writable — remove that and the
        purge starts failing on records nobody on the host can delete."""
        assert two_containers.purge_error is None, two_containers.purge_error
        assert not two_containers.audit_root_after_purge


# ── the shared bundle: two containers, one directory ─────────────────────────


class TestTheSharedBundle:
    """Reached through membership alone — the directory grants nothing else."""

    def test_the_dropped_process_carries_the_bundle_gid(self, two_containers):
        """The bundle half of the generalized ``/etc/group`` step: the
        entrypoint iterates both variables the render names, so a service with
        a bundle mounted carries that group too.

        Only meaningful when the bundle's group differs from the audit mount's
        — otherwise the audit join alone would satisfy it and the test would be
        a tautology dressed as a proof.
        """
        _require_the_group_is_the_only_door(two_containers)
        if not two_containers.distinct_gids:
            _degraded_host(
                "this host offers only one joinable group (gid "
                f"{two_containers.bundle_gid}, carried by both the audit and "
                "the bundle mount), so the bundle membership is already "
                "implied by the audit join — nothing distinguishable to "
                f"assert (this host offered {list(two_containers.offered_gids)}, "
                f"of which only one clears the gid {MIN_JOINABLE_GID} floor)"
            )
        bundle_gid = two_containers.alice["bundle"]["dir_stat"]["gid"]
        assert bundle_gid in two_containers.alice["groups"], two_containers.alice
        assert bundle_gid != two_containers.alice["audit_dir_stat"]["gid"], two_containers.alice

    def test_the_second_container_writes_where_only_the_group_reaches(self, two_containers):
        """``group-only/`` is ``2070`` and owned by uid 0: no triad but the
        group's can answer for a process running as 1000. A file that appears
        in it came through the group the entrypoint joined.

        Every leg of that is asserted from the container's own view rather
        than from what the host set up, because the host's numbers and the
        container's are only the same numbers when no id map is in play."""
        _require_the_group_is_the_only_door(two_containers)
        seen = _group_only_dir(two_containers.worker)
        assert seen["mode"] == oct(GROUP_ONLY_MODE), seen
        assert seen["uid"] != RUNTIME_UID, seen
        assert seen["gid"] in two_containers.worker["groups"], two_containers.worker
        bundle = two_containers.worker["bundle"]
        assert "write_error" not in bundle, bundle
        assert bundle["write"]["uid"] == RUNTIME_UID, bundle
        assert bundle["write"]["gid"] == seen["gid"], bundle

    def test_a_process_outside_the_group_is_denied_the_same_directory(self, two_containers):
        """The negative control: the one container that bypasses the entrypoint
        and is the SUBJECT rather than scaffolding — same image, same mount,
        same uid 1000, no ``/etc/group`` join and no supplementary groups, and
        the write must fail. Without it "reachable only through membership" is an
        argument about the permission model rather than something observed,
        and deleting the entrypoint's join would still leave this class green
        on the positive case alone.

        The claim under test is exactly that — *same image, same uid, no
        membership, refused* — so the refusal is accepted at whichever level
        the runtime raises it. A non-member holds nothing on either directory:
        the ``2070`` child grants only the group, and the ``2770`` mount root
        it must traverse to reach that child is owned by the host account and
        carries the same group, so ``other`` is ``---`` there too. On a
        faithful Linux runtime the traversal is refused first and the probe
        never gets to stat the child at all — which is the property, not a
        gap in it. The finer fact that the child really is ``0:2070`` is
        asserted next door from the SUBJECT's view in
        :meth:`test_the_second_container_writes_where_only_the_group_reaches`,
        by the one process that could reach it.
        """
        _require_the_group_is_the_only_door(two_containers)
        control = two_containers.control
        assert control["uid"] == RUNTIME_UID, control
        # Attribution: refused because it is not in the group, not because it
        # is some other uid or holds some other membership.
        assert two_containers.bundle_gid not in control["groups"], control
        assert "write" not in control, (
            "a process that never joined the mount's group wrote into a 2070 "
            f"directory — the group is not the door it is claimed to be: {control}"
        )
        if "dir_stat" in control:
            # This runtime let it traverse the mount root, so the refusal has
            # to have come from the group-only directory's own mode — and the
            # mode it was refused by is worth pinning while it is on hand.
            assert control["dir_stat"]["mode"] == oct(GROUP_ONLY_MODE), control
            assert _is_permission_denied(control.get("write_error")), control
        else:
            # It could not even stat the directory: refused one level up, at
            # the mount root, which is the same mount and the same missing
            # membership. The write below it necessarily failed too.
            assert _is_permission_denied(control.get("dir_error")), control
            assert _is_permission_denied(control.get("write_error")), control
        if not two_containers.bundle_stat_errors:
            assert two_containers.bundle_files[two_containers.control_file] is None, (
                "the negative control's file is on the host, so its write landed"
            )

    def test_the_second_container_reads_what_the_first_wrote(self, two_containers):
        """One host directory, two containers, no ownership in common: the
        sharing this whole strategy exists for."""
        _require_the_group_is_the_only_door(two_containers)
        assert (
            two_containers.worker["bundle"].get("read") == two_containers.markers[IDENTITY_TERMINAL]
        ), two_containers.worker

    def test_both_files_are_on_the_host_under_the_directorys_group(self, two_containers):
        """setgid again, on the shared side: whichever container wrote it, the
        file belongs to the directory's group and the host can reach it."""
        _require_direct_ids(two_containers)
        _require_the_group_is_the_only_door(two_containers)
        _require_the_host_can_read_the_bundle(two_containers)
        for name in (two_containers.alice_file, two_containers.worker_file):
            st = two_containers.bundle_files[name]
            assert st is not None, f"{name} never appeared on the host"
            assert st.st_gid == two_containers.bundle_gid, f"{name}: {st}"
            assert stat.S_ISREG(st.st_mode), f"{name}: {st}"
