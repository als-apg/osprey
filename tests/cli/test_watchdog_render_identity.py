"""With no watchdog on, a build renders what it rendered before the feature.

The whole ``build/`` tree is compared by file list and content hash, against two
references. A disabled ``watchdogs:`` block against no block at all, both from this
checkout, is the permanent guard. Against the merge-base with ``origin/main``, plus
the Jev port, only the allowlisted hooks may differ, and each must carry the change
that puts it on the list.

Every render runs in its own process from the source tree it is about, at one repo
path, so a difference between two snapshots can only come from the sources or the
profile.

Volatile content, left out of the comparison:

- The image context's copy of the repo's source zone, ``.image/<project>/profile.yml``
  and ``.image/<project>/watchdogs/``: the operator's own files, verbatim, which
  differ between a repo with a block and one without because the repo itself does.
  The image's ``build/`` is outside this and compared like the rest.
- In ``.osprey-manifest.json``: the render's timestamp, the profile fingerprint and
  the ``watchdogs`` source digest, which restate the same source; and the checksums
  of the allowlisted hooks, whose files are compared themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests._preset_data import copy_bundle_data

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_HOOKS = "src/osprey/templates/claude_code/claude/hooks"
TRIGGER = f"{TEMPLATE_HOOKS}/osprey_watchdog_trigger.py"
BUILT_HOOKS = ".claude/hooks"

#: The trailer naming the one commit that carries the Jev port.
PORT_TRAILER = "^Ported-From: worktree-jev-channel-finder c8c1513a26"

#: What of an uncommitted port can reach a render. The two YAML files also carry
#: other tasks' additions, which only declare keys a baseline render never sets.
PORT_RENDER_PATHS = (
    "src/osprey/models/jev",
    "src/osprey/models/providers/openrouter.py",
    "src/osprey/models/provider_registry.py",
    "src/osprey/models/providers/litellm_delegating.py",
    "src/osprey/profiles/providers.yml",
    "src/osprey/profiles/config_key_manifest.yml",
)

SOURCE_ZONE_COPIES = (".image/w/profile.yml", ".image/w/watchdogs/")
MANIFEST_NAME = ".osprey-manifest.json"
VOLATILE_MANIFEST_KEYS = frozenset({"timestamp", "preset_hash", "watchdogs"})

#: Allowed difference from the baseline: hook file, then a line only the baseline has
#: and a line only the feature has.
ALLOWED_CHANGES = {
    "osprey_target_state.py": (None, "def execution_verdict("),
    "osprey_writes_check.py": (
        "_target_state.write_verdict(section",
        "_target_state.execution_verdict(section",
    ),
}
#: Each allowlisted hook, in the build and in the image context's copy of the build.
ALLOWED_PATHS = sorted(
    f"{prefix}{BUILT_HOOKS}/{name}"
    for name in ALLOWED_CHANGES
    for prefix in ("", ".image/w/build/")
)

PROFILE: dict[str, Any] = {
    "name": "w",
    "project_name": "w",
    "provider": "anthropic",
    "model": "claude-haiku-4-5",
    "channel_finder_mode": "hierarchical",
    "data": "data",
    "hooks": ["turn-state", "memory-guard", "writes-check"],
    "config": {
        "control_system.type": "virtual_accelerator",
        "control_system.connector.virtual_accelerator.serving": "in_process",
        "archiver.type": "mock_archiver",
        "approval.enabled": False,
        "approval.default_policy": "always",
        "hooks.debug": False,
    },
}

DECISION = {
    "engine": "decision",
    "threshold": 0.5,
    "options": [
        {"id": "risky", "text": "A risky prompt", "flag": True},
        {"id": "routine", "text": "A routine prompt", "flag": False},
    ],
}
AGENT = {"engine": "agent", "tools": ["Read"], "timeout_s": 30}

RENDER = """
import sys
from pathlib import Path

import osprey
from osprey.cli.build_cmd import build

source = Path(sys.argv[1]).resolve()
if source not in Path(osprey.__file__).resolve().parents:
    sys.exit(f"rendering from {osprey.__file__}, not {source}")
build.main(["--repo", sys.argv[2], "--skip-deps", "--skip-lifecycle"])
"""


def _git(*args: str, cwd: Path = ROOT, stdin: bytes | None = None) -> bytes:
    done = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
    )
    if done.returncode != 0:
        pytest.fail(f"git {' '.join(args)}: {done.stderr.decode(errors='replace')}")
    return done.stdout


def merge_base(root: Path) -> str:
    """The merge-base of HEAD with ``origin/main``; a failure when there is none."""
    done = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/main"], cwd=root, capture_output=True, text=True
    )
    if done.returncode != 0:
        pytest.fail(f"the render baseline needs origin/main: {done.stderr.strip()}")
    return done.stdout.strip()


def _has_path(commit: str, path: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{path}"], cwd=ROOT, capture_output=True
        ).returncode
        == 0
    )


def _apply_port(base: str, worktree: Path, root: Path = ROOT) -> None:
    """Bring the Jev port onto *worktree*: its commit, else the checkout's port files."""
    commits = (
        _git("log", "--format=%H", f"--grep={PORT_TRAILER}", f"{base}..HEAD", cwd=root)
        .decode()
        .split()
    )
    if commits:
        assert len(commits) == 1, f"the port is one commit, found {commits}"
        patch = _git("show", "--format=", "--binary", commits[0], cwd=root)
        _git("apply", "--whitespace=nowarn", "-", cwd=worktree, stdin=patch)
        return
    if not (root / PORT_RENDER_PATHS[0]).is_dir():
        pytest.fail(f"no commit matches {PORT_TRAILER} and the checkout holds no port")
    for path in PORT_RENDER_PATHS:
        source, target = root / path, worktree / path
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, target)


def _write_repo(repo: Path, block: dict[str, Any] | None) -> None:
    """Point *repo* at the profile with *block*, with the source each entry names."""
    shutil.rmtree(repo / "watchdogs", ignore_errors=True)
    profile = dict(PROFILE)
    if block is not None:
        profile["watchdogs"] = block
        for name in block:
            directory = repo / "watchdogs" / name
            directory.mkdir(parents=True)
            (directory / "instructions.md").write_text(f"Judge as {name}.\n", encoding="utf-8")
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


def _stable_manifest(content: bytes) -> bytes:
    """A provenance manifest without its volatile keys, in a canonical form."""

    def stable(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: stable(item)
                for key, item in value.items()
                if key not in VOLATILE_MANIFEST_KEYS
                and not (key.startswith(BUILT_HOOKS) and Path(key).name in ALLOWED_CHANGES)
            }
        return value

    return json.dumps(stable(json.loads(content)), sort_keys=True).encode()


def _render(source: Path, repo: Path, block: dict[str, Any] | None) -> dict[str, str]:
    """Render *repo* with the framework in *source*; path to content hash, no volatiles."""
    _write_repo(repo, block)
    for derived in ("build", "var"):
        shutil.rmtree(repo / derived, ignore_errors=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source / "src"), str(source / "packages/osprey-connectors/src")]
    )
    done = subprocess.run(
        [sys.executable, "-c", RENDER, str(source), str(repo)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
    build = repo / "build"
    snapshot = {}
    for path in sorted(build.rglob("*")):
        relative = path.relative_to(build).as_posix()
        if path.is_file() and not relative.startswith(SOURCE_ZONE_COPIES):
            content = path.read_bytes()
            if path.name == MANIFEST_NAME:
                content = _stable_manifest(content)
            snapshot[relative] = hashlib.sha256(content).hexdigest()
    return snapshot


def _differing(left: dict[str, str], right: dict[str, str]) -> list[str]:
    return sorted(path for path in left.keys() | right.keys() if left.get(path) != right.get(path))


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("identity") / "repo"
    path.mkdir()
    copy_bundle_data(path / "data", "hello_world")
    return path


@pytest.fixture(scope="module")
def no_block(repo: Path) -> dict[str, str]:
    return _render(ROOT, repo, None)


@pytest.fixture(scope="module")
def baseline(repo: Path) -> Iterator[tuple[dict[str, str], Path]]:
    """The baseline render, and the worktree it came from for reading its files."""
    base = merge_base(ROOT)
    if _has_path(base, TRIGGER):
        pytest.skip("the watchdog has landed on origin/main; the block comparison guards it")
    worktree = repo.parent / "baseline"
    _git("worktree", "add", "--detach", "--quiet", str(worktree), base)
    try:
        _apply_port(base, worktree)
        yield _render(worktree, repo, None), worktree
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)], cwd=ROOT, capture_output=True
        )


def test_a_disabled_block_renders_what_no_block_renders(
    repo: Path, no_block: dict[str, str]
) -> None:
    disabled = _render(
        ROOT,
        repo,
        {
            "idle-decision": {**DECISION, "enabled": False},
            "idle-agent": {**AGENT, "enabled": False},
        },
    )
    assert _differing(no_block, disabled) == []


def test_the_comparison_sees_an_enabled_watchdog(repo: Path, no_block: dict[str, str]) -> None:
    enabled = _render(ROOT, repo, {"live": {**DECISION, "enabled": True}})
    changed = _differing(no_block, enabled)
    assert "config.yml" in changed
    assert f"{BUILT_HOOKS}/osprey_watchdog_trigger.py" in changed
    assert "watchdogs/live/instructions.md" in changed


def test_only_the_allowlisted_hooks_differ_from_the_baseline(
    no_block: dict[str, str], baseline: tuple[dict[str, str], Path]
) -> None:
    assert _differing(baseline[0], no_block) == ALLOWED_PATHS


@pytest.mark.parametrize("name", sorted(ALLOWED_CHANGES))
def test_each_allowlisted_hook_carries_its_change(
    name: str, baseline: tuple[dict[str, str], Path]
) -> None:
    before = (baseline[1] / TEMPLATE_HOOKS / name).read_text(encoding="utf-8")
    after = (ROOT / TEMPLATE_HOOKS / name).read_text(encoding="utf-8")
    only_before, only_after = ALLOWED_CHANGES[name]
    assert only_after in after and only_after not in before
    if only_before is not None:
        assert only_before in before and only_before not in after


def test_a_checkout_without_origin_main_fails_rather_than_skips(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    with pytest.raises(pytest.fail.Exception, match="needs origin/main"):
        merge_base(tmp_path)


def _commit_all(root: Path, message: str) -> str:
    _git("add", "-A", cwd=root)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message, cwd=root)
    return _git("rev-parse", "HEAD", cwd=root).decode().strip()


def test_the_port_commit_is_replayed_onto_the_baseline(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _git("init", "-q", cwd=root)
    (root / "kept.txt").write_text("base\n", encoding="utf-8")
    base = _commit_all(root, "base")
    (root / "ported.txt").write_text("port\n", encoding="utf-8")
    _commit_all(root, "port\n\nPorted-From: worktree-jev-channel-finder c8c1513a26\n")
    (root / "later.txt").write_text("feature\n", encoding="utf-8")
    _commit_all(root, "feature")
    worktree = tmp_path / "worktree"
    _git("worktree", "add", "--detach", "--quiet", str(worktree), base, cwd=root)
    _apply_port(base, worktree, root)
    assert (worktree / "ported.txt").read_text(encoding="utf-8") == "port\n"
    assert not (worktree / "later.txt").exists()


def test_a_baseline_with_no_port_anywhere_fails(tmp_path: Path) -> None:
    _git("init", "-q", cwd=tmp_path)
    (tmp_path / "kept.txt").write_text("base\n", encoding="utf-8")
    base = _commit_all(tmp_path, "base")
    with pytest.raises(pytest.fail.Exception, match="holds no port"):
        _apply_port(base, tmp_path, tmp_path)
