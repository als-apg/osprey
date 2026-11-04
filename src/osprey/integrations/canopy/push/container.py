"""The fixed scripts the review containers run, their argv, and running one."""

from __future__ import annotations

import contextlib
import os
import subprocess
from collections.abc import Sequence

from osprey.integrations.canopy.library import DEPLOYMENT_IGNORED_NAMES, IGNORED_NAMES_ENV

from .refusals import EXIT_CONTAINER_START, EXIT_CONTAINER_TIMEOUT, Refusal

#: The in-container mount points.
VOLUME_MOUNT = "/v"
STAGE_MOUNT = "/s"
CONTAINER_RUNTIME_UID = "1000:1000"
#: How long a container may run before it is killed.
CONTAINER_TIMEOUT_S = 600


# --- the fixed scripts the containers run ---------------------------------------------------
# Each is passed whole as the ``-c`` argument. They take their roots as positional
# arguments, so nothing in them names a host path.

LIST_SCRIPT = r"""
import json, os, stat, sys
root, max_dirs, max_bytes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = []
try:
    names = sorted(os.listdir(root))
except OSError:
    names = []
for name in names[:max_dirs]:
    entry = {"name": name}
    try:
        dfd = os.open(os.path.join(root, name), os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)
    except OSError as exc:
        entry["error"] = type(exc).__name__
        out.append(entry)
        continue
    try:
        fd = os.open("request.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
    except OSError as exc:
        entry["error"] = type(exc).__name__
        os.close(dfd)
        out.append(entry)
        continue
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
            entry["error"] = "not a regular file"
        else:
            data = b""
            while len(data) <= max_bytes:
                chunk = os.read(fd, max_bytes + 1 - len(data))
                if not chunk:
                    break
                data += chunk
            if len(data) > max_bytes:
                entry["error"] = "request.json too large"
            else:
                entry["request"] = data.decode("utf-8", "replace")
    finally:
        os.close(fd)
        os.close(dfd)
    out.append(entry)
sys.stdout.write(json.dumps({"bundles": out, "truncated": len(names) > max_dirs}) + "\n")
"""

STREAM_SCRIPT = r"""
import json, os, stat, sys, tarfile
bundle, max_files, max_total, max_file = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
def fail(code, message):
    sys.stderr.write(json.dumps({"error": message}) + "\n")
    sys.exit(code)
try:
    top = os.open(bundle, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)
except FileNotFoundError:
    fail(5, "bundle missing")
except OSError as exc:
    fail(3, "bundle is not a directory: " + type(exc).__name__)
state = {"files": 0, "total": 0}
tar = tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.PAX_FORMAT)
def walk(dfd, rel):
    for name in sorted(os.listdir(dfd)):
        path = rel + "/" + name if rel else name
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError as exc:
            fail(3, path + ": " + type(exc).__name__)
        try:
            st = os.fstat(fd)
            if stat.S_ISDIR(st.st_mode):
                walk(fd, path)
                continue
            if not stat.S_ISREG(st.st_mode):
                fail(3, path + ": not a regular file")
            if st.st_nlink > 1:
                fail(3, path + ": hard link")
            if st.st_size > max_file:
                fail(4, path + ": over the per-file cap")
            state["files"] += 1
            state["total"] += st.st_size
            if state["files"] > max_files:
                fail(4, "too many files")
            if state["total"] > max_total:
                fail(4, "bundle too large")
            info = tarfile.TarInfo(path)
            info.size = st.st_size
            info.mode = 0o644
            with os.fdopen(os.dup(fd), "rb") as handle:
                tar.addfile(info, handle)
        finally:
            os.close(fd)
walk(top, "")
tar.close()
"""

CHECK_SCRIPT = r"""
import asyncio, json, os, subprocess, sys
from pathlib import Path
from canopy_mcp.config import Settings
from canopy_mcp.check.refs import BufferTree
from canopy_mcp.listing import ignored_names_from_env, judge, scan
from canopy_mcp.library_screens import PushRefused, push_closure
from canopy_mcp.store import looks_like_image
from canopy_mcp.tools.check import run_check
stage, target_dir, top = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
SUFFIX = ".screen.json"
out = {"ok": False, "refusals": [], "wouldBe": None, "current": None, "specUnavailable": []}
def refuse(code, message, **extra):
    out["refusals"].append({"code": code, "message": message, **extra})
overlay = stage / "overlay"
files = {p.relative_to(overlay).as_posix(): p.read_bytes() for p in sorted(overlay.rglob("*")) if p.is_file()}
for path, data in files.items():
    if not path.endswith(SUFFIX) and not looks_like_image(data, Path(path.lower()).suffix):
        refuse("not_image", path + " does not hold an image of its suffix", path=path)
wouldbe = stage / "would-be"
listing = judge(scan(wouldbe), ignored_names=ignored_names_from_env()).paths
tree = BufferTree({p: (wouldbe / p).read_bytes() for p in listing})
settings = Settings(base_url="http://127.0.0.1:9", drafts_dir=wouldbe)
async def checks():
    for path in sorted(files):
        if not path.endswith(SUFFIX):
            continue
        draft_id = path[len("screens/"):-len(SUFFIX)]
        try:
            result = await run_check(draft_id=draft_id, tree=tree, served_at="library", settings=settings)
        except Exception as exc:
            refuse("check_errors", draft_id + ": check failed: " + str(exc), screen=draft_id)
            continue
        errors = [f for group in (result.schema, result.semantic, result.lint) for f in group if f.severity == "error"]
        for f in (*result.schema, *result.lint):
            if f.code == "spec_unavailable":
                out["specUnavailable"].append({"screen": draft_id, "message": f.message})
        if errors:
            refuse("check_errors", draft_id + " has check errors", screen=draft_id, errors=[f.to_dict() for f in errors])
asyncio.run(checks())
try:
    closure = push_closure(tree, top, target_dir, listing=listing)
except PushRefused as exc:
    for item in exc.refusals:
        refuse(item["code"], item["message"], doc=item.get("doc"), nodeId=item.get("nodeId"), ref=item.get("ref"))
else:
    if set(closure) != set(files):
        refuse("closure_mismatch", "the closure recomputed at the destination differs from the bundle",
               extra=sorted(set(closure) - set(files)), missing=sorted(set(files) - set(closure)))
for key, root in (("wouldBe", wouldbe), ("current", stage / "current")):
    done = subprocess.run([sys.executable, "-I", "-m", "canopy_mcp.report", str(root), "--served-at", "library", "--emit"],
                          capture_output=True, text=True, encoding="utf-8", errors="backslashreplace")
    if done.returncode != 0:
        refuse("report", key + ": canopy_mcp.report exited " + str(done.returncode) + ": " + done.stderr.strip()[-500:])
        continue
    try:
        out[key] = json.loads(done.stdout)["json"]
    except (ValueError, KeyError, TypeError):
        refuse("report", key + ": canopy_mcp.report printed no report")
out["ok"] = not out["refusals"]
sys.stdout.write(json.dumps(out, ensure_ascii=True) + "\n")
"""

EMIT_SCRIPT = r"""
import subprocess, sys
root = sys.argv[1]
done = subprocess.run([sys.executable, "-I", "-m", "canopy_mcp.report", root, "--served-at", "library", "--emit"],
                      capture_output=True)
sys.stderr.buffer.write(done.stderr)
sys.stdout.buffer.write(done.stdout)
sys.exit(done.returncode)
"""


# --- container argv ------------------------------------------------------------------------


def container_argv(
    runtime: str,
    verb: str,
    *,
    nonce: str,
    image: str,
    script: str,
    args: Sequence[str],
    volume: str | None = None,
    stage: str | os.PathLike[str] | None = None,
    payload_dir: str | None = None,
    host_ids: tuple[int, int],
) -> list[str]:
    """The argv of one review container; every container of this module is built here.

    ``runtime`` is the binary (``docker``). Exactly one of ``volume`` (a user's
    agent-data volume, mounted read-only at ``/v``, container run as the
    runtime uid) and ``stage`` (a host stage, bind-mounted read-only at ``/s``,
    container run as ``host_ids``) is given. With ``payload_dir`` (the staged
    payload inside the image) the CANOPY viewer path is passed; the widget
    catalog is the viewer's ``ir-map.json`` and the staged spec is found beside
    the viewer, so neither needs a variable of its own.
    ``script`` and ``args`` follow ``-I -c``.
    """
    if (volume is None) == (stage is None):
        raise ValueError("a review container mounts exactly one of a user volume and a host stage")
    if runtime != "docker":
        raise ValueError(f"push review requires docker in this release (not {runtime!r})")
    uid, gid = host_ids
    user = CONTAINER_RUNTIME_UID if volume is not None else f"{uid}:{gid}"
    argv = [
        runtime,
        "run",
        "--rm",
        "--name",
        f"osprey-canopy-push-{verb}-{nonce}",
        "--label",
        "osprey.canopy.push=1",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "-e",
        "HOME=/tmp",
        "-e",
        f"{IGNORED_NAMES_ENV}={' '.join(DEPLOYMENT_IGNORED_NAMES)}",
        "--user",
        user,
        "--entrypoint",
        "python3",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        "1g",
        "--pids-limit",
        "256",
    ]
    if volume is not None:
        argv += ["--mount", f"type=volume,src={volume},dst={VOLUME_MOUNT},readonly"]
    else:
        assert stage is not None
        argv += ["--mount", f"type=bind,src={os.fspath(stage)},dst={STAGE_MOUNT},readonly"]
    if payload_dir is not None:
        argv += ["-e", f"CANOPY_VIEWER_DIST={payload_dir}/viewer"]
    return [*argv, image, "-I", "-c", script, *args]


def _container_name(argv: Sequence[str]) -> str:
    return argv[argv.index("--name") + 1]


def _kill(runtime: str, name: str) -> None:
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run([runtime, "kill", name], capture_output=True, check=False, timeout=60)


def _run_container(
    runtime: str, argv: Sequence[str], *, timeout: float = CONTAINER_TIMEOUT_S
) -> subprocess.CompletedProcess[bytes]:
    """Run a container to completion; refuse a start failure or a timeout with their own codes."""
    try:
        return subprocess.run(argv, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill(runtime, _container_name(argv))
        raise Refusal(
            EXIT_CONTAINER_TIMEOUT, f"container {_container_name(argv)} timed out"
        ) from None
    except OSError as exc:
        raise Refusal(EXIT_CONTAINER_START, f"cannot start {runtime}: {exc}") from None


def _started(done: subprocess.CompletedProcess[bytes], runtime: str) -> None:
    if done.returncode in (125, 126, 127):
        said = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise Refusal(
            EXIT_CONTAINER_START,
            f"{runtime} could not start the container: {said[-1] if said else ''}",
        )
