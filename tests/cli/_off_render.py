"""Render a deployment from any source tree, in a form two renders can be compared in.

The comparison against ``main`` in ``test_pv_directory_off_render.py`` stores nothing
rendered: the baseline is rendered live, from a detached scratch worktree of
:func:`baseline_ref`, so it never depends on the order tasks landed in. A stored tree
would be half a gigabyte and go red on every later template change on ``main``.

Every render happens in its own process from the source tree it is about, at one repo
path that is re-created for it, which removes the ``project_root``, absolute-path and
repo-id differences. A difference between two snapshots of that path can then only
come from the sources, or from what is blanked here, each item for its stated reason:

- Framework and connectors versions, which name the commit a tree was rendered from:
  the build argument in compose files and the ``emitted by OSPREY`` stamp in emitted
  profiles (:data:`_VERSION_STAMPS`). No render stamps a connectors version today. A
  version pinned as data, such as ``requires_osprey_version``, the Canopy library's own
  or ``osprey-connectors==``, is content and stays.
- In ``.osprey-manifest.json``: :data:`VOLATILE_MANIFEST_KEYS`,
  :data:`VOLATILE_CREATION_KEYS`, and the checksum rows of files whose raw bytes the
  comparison does not see. Those files are still compared, normalised or as present.
- The values of :data:`AGENT_RECORD_SECRETS`, generated afresh by every ``osprey init``.
  Their paths and key names are compared.
- The bytes of every :data:`EXCLUDED` file. Its path and mode are compared.

Everything else in the repo is compared: ``var/``, ``.env``, empty directories, symlink
targets and permission bits. On two demo renders at one path, the only raw differences
were the 14 manifests' ``creation.timestamp``, the 14 graph indexes and these secrets.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: The folder every render lives in, so the deployment is always named ``demo``.
REPO_NAME = "demo"

#: What a baseline ref has to hold for the three configurations to render from it.
DEMO_PRESET = "src/osprey/profiles/presets/demo.yml"
#: Where the demo preset lives until the rename reaches ``origin/main``.
BASELINE_REFS = ("origin/main", "origin/feature/rename-control-assistant-to-demo")

#: Removed from every render's environment: either would stage or check Canopy's payload.
UNSET_ENV = ("OSPREY_CANOPY_PAYLOAD", "OSPREY_CANOPY_PAYLOAD_CHECK")

GRAPH_INDEX = "data/channel_databases/graph.duckdb"
#: Path suffix left out of every comparison, to the reason it is left out.
EXCLUDED = {
    GRAPH_INDEX: "binary index, not reproducible; its input facility.json is compared",
}
#: What stands in for the bytes of an excluded file, so its path and mode still compare.
EXCLUDED_MARK = "<excluded>"

MANIFEST_NAME = ".osprey-manifest.json"
#: Top-level manifest keys left out of the comparison, to the reason each is left out.
VOLATILE_MANIFEST_KEYS = {
    "build_args": "how the build was called, not what it rendered; rendered files are compared",
    "reproducible_command": "how to repeat the build, not what it rendered",
    "timestamp": "the moment of the render; today only under creation",
    "preset_hash": "the preset as shipped, before the strip; today only under creation",
    "watchdogs": "a provenance digest of its source; today only under creation's digests",
}
#: Keys under the manifest's ``creation`` left out of the comparison, to the reason.
VOLATILE_CREATION_KEYS = {
    "timestamp": "moves on every repeat render",
    "osprey_version": "names the commit a tree was rendered from",
    "preset_hash": "the preset as shipped, before the strip; the strip proof covers its content",
}

#: The repo-root files holding generated secrets.
AGENT_RECORD_FILE = re.compile(r"\.env\.agent-record\.[^/]+")
#: The keys in those files whose values move on every repeat render, as generated secrets.
AGENT_RECORD_SECRETS = (
    "CLICKHOUSE_PASSWORD",
    "OSPREY_AGENT_RECORD_CLICKHOUSE_PASSWORD",
    "OSPREY_AGENT_RECORD_LOGIN",
    "OSPREY_AGENT_RECORD_SECRETS",
    "OSPREY_AGENT_RECORD_VERIFIER",
    "OSPREY_TERMINAL_SECRET_HISTORY",
)
SECRET_MARK = "<secret>"
_SECRET_VALUE = re.compile(rf"^((?:{'|'.join(AGENT_RECORD_SECRETS)})=).*$", re.MULTILINE)

VERSION_MARK = "<version>"
_VERSION = r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?(?:\+[0-9A-Za-z.]+)?"
#: Each place a framework or connectors version is stamped; group 1 is what precedes it.
_VERSION_STAMPS = tuple(
    re.compile(rf"({context}){_VERSION}")
    for context in (
        r"\bOSPREY_(?:CONNECTORS_)?VERSION[\"']?\s*[:=]\s*[\"']?",
        r"\bemitted by OSPREY ",
    )
)


@dataclass(frozen=True)
class Configuration:
    """One deployment to render: a preset, and what ``osprey init --set`` adds to it."""

    name: str
    preset: str
    overrides: tuple[str, ...] = ()


DEMO = Configuration("demo", "demo")
DEMO_HIERARCHICAL = Configuration(
    "demo-hierarchical", "demo", ("channel_finder_mode=hierarchical",)
)
HELLO_WORLD = Configuration("hello-world", "hello-world")
#: None adds a ``pv_directory`` section. A preset that ships one has it removed by the
#: caller's ``prepare`` before the build, which is what makes the render an off render.
CONFIGURATIONS = (DEMO, DEMO_HIERARCHICAL, HELLO_WORLD)

_CHECK_SOURCE = """
import sys
from pathlib import Path

import osprey

source = Path(sys.argv[1]).resolve()
if source not in Path(osprey.__file__).resolve().parents:
    sys.exit(f"rendering from {osprey.__file__}, not {source}")
"""

INIT = (
    _CHECK_SOURCE
    + """
from osprey.cli.init_cmd import init

init.main([sys.argv[2], "--preset", sys.argv[3], "--no-git", *sys.argv[4:]])
"""
)

BUILD = (
    _CHECK_SOURCE
    + """
from osprey.cli.build_cmd import build

build.main(["--repo", sys.argv[2], "--skip-deps", "--skip-lifecycle"])
"""
)

PRESET_HASH = (
    _CHECK_SOURCE
    + """
from osprey.cli.build_profile_merge import compute_preset_hash

print(compute_preset_hash(sys.argv[2]))
"""
)


def _git(*args: str, cwd: Path = ROOT) -> str:
    done = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args], cwd=cwd, capture_output=True, text=True
    )
    if done.returncode != 0:
        pytest.fail(f"git {' '.join(args)}: {done.stderr}")
    return done.stdout.strip()


def _merge_base(ref: str, root: Path) -> str | None:
    done = subprocess.run(["git", "merge-base", "HEAD", ref], cwd=root, capture_output=True)
    return done.stdout.decode().strip() if done.returncode == 0 else None


def _holds(commit: str, path: str, root: Path) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{path}"], cwd=root, capture_output=True
        ).returncode
        == 0
    )


def baseline_ref(root: Path = ROOT) -> str:
    """The commit a branch's render is compared against, resolved to a full id.

    The merge-base with ``origin/main`` when that commit holds the demo preset, otherwise
    the merge-base with the rename's branch. Fails when neither does, because no baseline
    can then render the configurations; silently comparing against something else would
    prove nothing about ``main``.
    """
    bases = {ref: _merge_base(ref, root) for ref in BASELINE_REFS}
    if bases[BASELINE_REFS[0]] is None:
        pytest.fail("the render baseline needs origin/main")
    for base in bases.values():
        if base is not None and _holds(base, DEMO_PRESET, root):
            return base
    pytest.fail(f"no merge-base holds {DEMO_PRESET}: {bases}")


@contextmanager
def scratch_worktree(ref: str, path: Path, root: Path = ROOT) -> Iterator[Path]:
    """A detached worktree of *ref* at *path*, removed on the way out."""
    _git("worktree", "add", "--detach", "--quiet", str(path), ref, cwd=root)
    try:
        yield path
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)], cwd=root, capture_output=True
        )


def render_environment(source: Path) -> dict[str, str]:
    """The environment a render runs in: *source* on the path, no Canopy payload."""
    env = {name: value for name, value in os.environ.items() if name not in UNSET_ENV}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source / "src"), str(source / "packages/osprey-connectors/src")]
    )
    return env


def _run(script: str, source: Path, *args: str, cwd: Path) -> str:
    done = subprocess.run(
        [sys.executable, "-c", script, str(source), *args],
        cwd=cwd,
        env=render_environment(source),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
    return done.stdout


def initialise(source: Path, repo: Path, configuration: Configuration = DEMO) -> Path:
    """Re-create *repo* with ``osprey init`` from *source*, as far as its source zone."""
    shutil.rmtree(repo, ignore_errors=True)
    repo.parent.mkdir(parents=True, exist_ok=True)
    overrides = [item for pair in configuration.overrides for item in ("--set", pair)]
    _run(INIT, source, str(repo), configuration.preset, *overrides, cwd=repo.parent)
    return repo


def render(
    source: Path,
    repo: Path,
    configuration: Configuration = DEMO,
    *,
    prepare: Callable[[Path], None] | None = None,
) -> Path:
    """Re-create *repo* and render *configuration* in it with the framework in *source*.

    Args:
        source: The tree whose ``src/`` and connectors package do the rendering.
        repo: The one repo path; anything already there is replaced.
        configuration: The deployment to render.
        prepare: Runs on the initialised repo before the build, to change its source zone.

    Returns:
        The repo, ready for :func:`snapshot`.
    """
    initialise(source, repo, configuration)
    if prepare is not None:
        prepare(repo)
    _run(BUILD, source, str(repo), cwd=repo)
    return repo


def preset_hash(source: Path, preset: str = "demo") -> str:
    """``compute_preset_hash(preset)`` as the tree in *source* resolves that preset."""
    return _run(PRESET_HASH, source, preset, cwd=source).strip()


def is_excluded(relative: str) -> bool:
    """Whether the repo-relative path is one of the copies left out of comparison."""
    return relative.endswith(tuple(f"/{suffix}" for suffix in EXCLUDED)) or relative in EXCLUDED


def _mask_versions(text: str) -> str:
    for stamp in _VERSION_STAMPS:
        text = stamp.sub(rf"\g<1>{VERSION_MARK}", text)
    return text


def _stable(document: dict[str, Any]) -> dict[str, Any]:
    stable = {key: item for key, item in document.items() if key not in VOLATILE_MANIFEST_KEYS}
    if "creation" in stable:
        stable["creation"] = {
            key: item
            for key, item in stable["creation"].items()
            if key not in VOLATILE_CREATION_KEYS
        }
    return stable


def normalise(relative: str, content: bytes) -> bytes:
    """*content* of the file at *relative* without what names the render that made it.

    A manifest comes back without its volatile keys and in a canonical form. Every other
    text file has its version stamps replaced, an agent-record file also its generated
    secrets, and a binary file is returned as it is.
    """
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    if PurePosixPath(relative).name == MANIFEST_NAME:
        text = json.dumps(_stable(json.loads(text)), sort_keys=True, indent=1)
    if AGENT_RECORD_FILE.fullmatch(relative):
        text = _SECRET_VALUE.sub(rf"\g<1>{SECRET_MARK}", text)
    return _mask_versions(text).encode("utf-8")


#: The manifest's per-key digests of the source zone: of the profile, and of the
#: material beside it, which a comparison of the tree sees file by file.
SOURCE_ZONE_DIGESTS = "profile_key_digests"


def manifest_view(
    manifest: str | bytes,
    directory: PurePosixPath,
    unseen_rows: Collection[str],
    *,
    source_zone: bool,
) -> str:
    """A normalised manifest without the checksum rows of files compared another way.

    Args:
        manifest: The manifest after :func:`normalise`.
        directory: The manifest's repo-relative directory, which its row keys are under.
        unseen_rows: Repo-relative paths whose checksum rows are dropped.
        source_zone: Whether the source zone is compared; without it the manifest also
            loses :data:`SOURCE_ZONE_DIGESTS`.
    """
    document = json.loads(manifest)
    rows: Mapping[str, str] = document.get("file_checksums", {})
    document["file_checksums"] = {
        key: row for key, row in rows.items() if (directory / key).as_posix() not in unseen_rows
    }
    if not source_zone:
        document.get("creation", {}).pop(SOURCE_ZONE_DIGESTS, None)
    return json.dumps(document, sort_keys=True, indent=1)


def snapshot(repo: Path, *, unseen_rows: Collection[str] = ()) -> dict[str, str]:
    """Repo-relative path to what the comparison sees of it, for one rendered repo.

    A symlink is its target, without being followed. A directory is its mode, so an empty
    one counts. A file is its mode and the hash of its normalised content.

    Args:
        repo: The rendered repo.
        unseen_rows: Repo-relative paths the caller compares its own way, whose checksum
            rows are dropped from every manifest as those of normalised files are.
    """
    digests: dict[str, str] = {}
    unseen = set(unseen_rows)
    manifests: dict[str, tuple[str, bytes]] = {}
    for path in sorted(repo.rglob("*")):
        relative = path.relative_to(repo).as_posix()
        kind = path.lstat().st_mode
        mode = stat.filemode(kind)
        if stat.S_ISLNK(kind):
            digests[relative] = f"link:{os.readlink(path)}"
        elif not stat.S_ISREG(kind):
            digests[relative] = mode
        elif is_excluded(relative):
            unseen.add(relative)
            digests[relative] = f"{mode} {EXCLUDED_MARK}"
        else:
            content = path.read_bytes()
            normalised = normalise(relative, content)
            if path.name == MANIFEST_NAME:
                manifests[relative] = (mode, normalised)
            else:
                if normalised != content:
                    unseen.add(relative)
                digests[relative] = f"{mode} {hashlib.sha256(normalised).hexdigest()}"
    for relative, (mode, manifest) in manifests.items():
        view = manifest_view(manifest, PurePosixPath(relative).parent, unseen, source_zone=True)
        digests[relative] = f"{mode} {hashlib.sha256(view.encode()).hexdigest()}"
    return digests


def differing(left: Mapping[str, str], right: Mapping[str, str]) -> list[str]:
    """The paths two snapshots disagree on, including a path only one of them has."""
    return sorted(path for path in left.keys() | right.keys() if left.get(path) != right.get(path))
