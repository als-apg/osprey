"""``osprey init --revalidate`` moves a deployment's root entries off their retired paths.

No live row moves yet, so each test plants ``moved_from`` declarations on a
substituted layout: the agent-facing rows under ``agent/``, the operator rows
under ``ops/`` and ``data/code_review`` to ``agent/code_review``; the runtime
child's move from ``data/.runtime`` to ``var/runtime`` is the live table's own.
Each repo is a fresh ``hello-world`` deployment with sources at the retired
paths, profile values, a pipeline file and a ``.dockerignore`` naming them, and
stale canopy staging at its root.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey import repo_layout
from osprey.cli.init_cmd import init
from osprey.cli.init_revalidate import _after_moves_view, _plan_root
from osprey.repo_layout import RootChild, RootEntry, check_layout, with_paths
from osprey.repo_retired import RetiredRootRefused, find_retired_root, refuse_retired_layout
from tests._builds import init_project

#: Where each planted row lives now.
NEW_PATHS: dict[str, str] = {
    "rules": "agent/rules",
    "skills": "agent/skills",
    "mcp_servers": "agent/mcp_servers",
    "web-terminal-context": "agent/context",
    "scripts": "ops/scripts",
    "services": "ops/services",
    "ci-extra.yml": "ops/ci-extra.yml",
    "osprey.service": "ops/osprey.service",
}

#: Sources at the retired paths, beside what :func:`_plant` moves back there
#: from ``hello-world``'s live paths (``ci-extra.yml``, ``mcp_servers/``,
#: ``data/code_review/``).
_OLD_SOURCES = {
    "rules/plant.md": "# A rule of this deployment.\n",
    "rules/.keep-me": "",
    "scripts/run.sh": "#!/bin/sh\necho run\n",
    "services/worker/Dockerfile": "FROM python:3.12\n",
    "data/.runtime/certs/key.pem": "key\n",
}

_PIPELINE = "include:\n  - local: ci-extra.yml\n# ci-extra.yml stays as written in a comment\n"
_NEW_PIPELINE = (
    "include:\n  - local: ops/ci-extra.yml\n# ci-extra.yml stays as written in a comment\n"
)
_DOCKERIGNORE = "scripts/\n/services\n*.pyc\n"
_NEW_DOCKERIGNORE = "ops/scripts/\n/ops/services\n*.pyc\n"

#: Profile values naming retired paths: rewritten where the key holds a path,
#: listed to check by hand where it holds a command.
_PROFILE_PATHS = (
    "  code_review:   # the reviewer\n    prompt: ./data/code_review/reviewer.md   # its prompt\n"
)
_LIFECYCLE = "lifecycle:\n  pre_build:\n    - run: sh scripts/run.sh\n      cwd: scripts\n"

#: Stale canopy staging at the root, and a framework wheel that stays.
_CANOPY = {
    "canopy_payload/index.json": "{}\n",
    "canopy-1.0-py3-none-any.whl": "wheel\n",
    "pydmconverter-1.0-py3-none-any.whl": "wheel\n",
}
_FRAMEWORK_WHEEL = "osprey_framework-1.0-py3-none-any.whl"


def _planted_layout() -> tuple[tuple[RootEntry, ...], tuple[RootChild, ...]]:
    old_paths = {row.key: row.path for row in repo_layout.ROOT_ENTRIES}
    rows = tuple(
        replace(row, moved_from=row.moved_from or (old_paths[row.key],))
        if row.key in NEW_PATHS
        else row
        for row in with_paths(NEW_PATHS)
    )
    # The live children: the runtime child already moved from data/.runtime.
    children = repo_layout.ROOT_CHILDREN
    check_layout(rows, children)
    return rows, children


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the developer's git config (signing, templates) out of the rewrite's `git mv`."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve().parent))
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "OSPREY")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "osprey@example.org")


@pytest.fixture(scope="module")
def _template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh hello-world deployment, made once with the live layout."""
    return init_project(tmp_path_factory.mktemp("template"), "hello-world", "plant")


@pytest.fixture
def planted(monkeypatch: pytest.MonkeyPatch) -> None:
    rows, children = _planted_layout()
    monkeypatch.setattr(repo_layout, "ROOT_ENTRIES", rows)
    monkeypatch.setattr(repo_layout, "ROOT_CHILDREN", children)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout


def _commit_all(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _plant(template: Path, parent: Path, *, git: bool) -> Path:
    """A copy of *template* with sources and references at the retired root paths."""
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(shutil.copytree(template, parent / "plant", symlinks=True))
    # What init wrote at a moved entry's new path goes back to its old one.
    for move in repo_layout.retired_moves():
        new, old = root / move.new, root / move.old
        if new.exists() and not old.exists():
            old.parent.mkdir(parents=True, exist_ok=True)
            new.rename(old)
            if not any(new.parent.iterdir()):
                new.parent.rmdir()
    _write(root, {**_OLD_SOURCES, **_CANOPY, _FRAMEWORK_WHEEL: "wheel\n"})
    _write(root, {".gitlab-ci.yml": _PIPELINE, ".dockerignore": _DOCKERIGNORE})
    profile = root / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert "\nconfig:\n" in text
    text = text.replace("\nconfig:\n", f"\nconfig:\n{_PROFILE_PATHS}", 1)
    profile.write_text(f"{text.rstrip()}\n{_LIFECYCLE}", encoding="utf-8")
    if git:
        _git(root, "init", "-q")
        _commit_all(root, "retired root layout")
    return root


def _files(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def _revalidate(root: Path) -> object:
    return CliRunner().invoke(init, [str(root), "--revalidate"], catch_exceptions=False)


def _assert_current(root: Path) -> None:
    for rel, text in {
        "agent/rules/plant.md": _OLD_SOURCES["rules/plant.md"],
        "agent/rules/.keep-me": "",
        "ops/scripts/run.sh": _OLD_SOURCES["scripts/run.sh"],
        "ops/services/worker/Dockerfile": _OLD_SOURCES["services/worker/Dockerfile"],
        "var/runtime/certs/key.pem": "key\n",
        ".gitlab-ci.yml": _NEW_PIPELINE,
        ".dockerignore": _NEW_DOCKERIGNORE,
    }.items():
        assert (root / rel).read_text(encoding="utf-8") == text, rel
    for gone in (
        "rules",
        "scripts",
        "services",
        "mcp_servers",
        "ci-extra.yml",
        "data/code_review",
        "data/.runtime",
    ):
        assert not (root / gone).exists(), gone
    assert (root / "ops" / "ci-extra.yml").is_file()
    assert (root / "agent" / "code_review" / "reviewer.md").is_file()
    assert (root / "agent" / "mcp_servers" / "example_server" / "server.py").is_file()
    assert find_retired_root(root).empty


@pytest.mark.usefixtures("planted")
def test_tracked_moves_keep_their_history(_template: Path, tmp_path: Path) -> None:
    root = _plant(_template, tmp_path, git=True)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    _assert_current(root)
    staged = _git(root, "diff", "--cached", "--name-status", "-M")
    for old, new in (
        ("rules/plant.md", "agent/rules/plant.md"),
        ("ci-extra.yml", "ops/ci-extra.yml"),
        ("data/code_review/reviewer.md", "agent/code_review/reviewer.md"),
        ("mcp_servers/example_server/server.py", "agent/mcp_servers/example_server/server.py"),
        ("services/worker/Dockerfile", "ops/services/worker/Dockerfile"),
    ):
        assert f"\t{old}\t{new}\n" in staged, staged
    _git(root, "commit", "-q", "-am", "current root layout")
    log = _git(root, "log", "--follow", "--oneline", "--", "agent/rules/plant.md")
    assert len(log.splitlines()) == 2, log


@pytest.mark.usefixtures("planted")
def test_the_diff_reports_each_move_and_edit(_template: Path, tmp_path: Path) -> None:
    root = _plant(_template, tmp_path, git=True)

    output = _revalidate(root).output

    assert "rename rules/plant.md => agent/rules/plant.md" in output
    assert "rename ci-extra.yml => ops/ci-extra.yml" in output
    assert "rename data/code_review/reviewer.md => agent/code_review/reviewer.md" in output
    assert "rename data/.runtime/certs/key.pem => var/runtime/certs/key.pem" in output
    assert "+    prompt: ./agent/code_review/reviewer.md   # its prompt" in output
    assert "+      cwd: ops/scripts" in output
    assert "+  - local: ops/ci-extra.yml" in output
    assert "+ops/scripts/" in output
    assert "--- a/canopy_payload/index.json" in output


@pytest.mark.usefixtures("planted")
def test_line_edits_keep_spelling_and_comments(_template: Path, tmp_path: Path) -> None:
    root = _plant(_template, tmp_path, git=False)

    _revalidate(root)

    text = (root / "profile.yml").read_text(encoding="utf-8")
    assert "  code_review:   # the reviewer\n" in text
    assert "    prompt: ./agent/code_review/reviewer.md   # its prompt\n" in text
    assert "    - run: sh scripts/run.sh\n      cwd: ops/scripts\n" in text
    assert "data/code_review" not in text


@pytest.mark.usefixtures("planted")
def test_untracked_tree_is_renamed(_template: Path, tmp_path: Path) -> None:
    root = _plant(_template, tmp_path, git=False)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    _assert_current(root)


@pytest.mark.usefixtures("planted")
def test_moves_merge_into_an_existing_successor(_template: Path, tmp_path: Path) -> None:
    root = _plant(_template, tmp_path, git=True)
    _write(root, {"agent/rules/current.md": "# Already moved.\n"})
    _commit_all(root, "a current rule")

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (root / "agent" / "rules").iterdir()) == [
        ".keep-me",
        "current.md",
        "plant.md",
    ]
    assert not (root / "rules").exists()
    assert find_retired_root(root).empty


@pytest.mark.usefixtures("planted")
def test_a_clash_refuses_before_anything_changes(_template: Path, tmp_path: Path) -> None:
    root = _plant(_template, tmp_path, git=True)
    _write(root, {"agent/rules/plant.md": "# Kept here too.\n", "ops/ci-extra.yml": "{}\n"})
    before = _files(root)

    result = CliRunner().invoke(init, [str(root), "--revalidate"])

    assert result.exit_code != 0
    for path in ("rules/plant.md", "agent/rules/plant.md", "ci-extra.yml", "ops/ci-extra.yml"):
        assert path in result.output, path
    assert _files(root) == before


@pytest.mark.usefixtures("planted")
def test_host_steps_and_what_to_check_are_printed(_template: Path, tmp_path: Path) -> None:
    root = _plant(_template, tmp_path, git=True)

    output = _revalidate(root).output

    steps = output[output.index("Left for you to do by hand") :]
    assert "# Host steps, once the files moved:" in steps
    assert "osprey build" in steps
    assert "# Check by hand; the paths inside these are not rewritten:" in steps
    for item in ("ops/ci-extra.yml", "ops/services/worker/Dockerfile", "ops/scripts/"):
        assert item in steps, item
    assert "lifecycle.pre_build.*.run: sh scripts/run.sh" in steps


@pytest.mark.usefixtures("planted")
def test_stale_canopy_staging_is_removed(_template: Path, tmp_path: Path) -> None:
    root = _plant(_template, tmp_path, git=True)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    for gone in _CANOPY:
        assert not (root / gone).exists(), gone
    assert not (root / "canopy_payload").exists()
    assert (root / _FRAMEWORK_WHEEL).is_file()


@pytest.mark.usefixtures("planted")
def test_a_current_tree_has_nothing_to_do(_template: Path, tmp_path: Path) -> None:
    root = Path(shutil.copytree(_template, tmp_path / "plant"))
    before = _files(root)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert "nothing to rewrite" in result.output
    assert _files(root) == before


@pytest.mark.usefixtures("planted")
def test_a_second_run_is_a_no_op(_template: Path, tmp_path: Path) -> None:
    root = _plant(_template, tmp_path, git=True)
    _revalidate(root)
    before = _files(root)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert "nothing to rewrite" in result.output
    assert "Host steps" not in result.output
    assert _files(root) == before


@pytest.mark.usefixtures("planted")
def test_a_nested_repository_is_left_for_the_operator(_template: Path, tmp_path: Path) -> None:
    root = _plant(_template, tmp_path, git=True)
    scripts = root / "scripts"
    _git(scripts, "init", "-q")
    _commit_all(scripts, "the operator's scripts")
    scripts_before = _files(scripts)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert _files(scripts) == scripts_before
    assert not (root / "ops" / "scripts").exists()
    output = result.output
    steps = output[output.index("Left for you to do by hand") :]
    assert f"cd {root.resolve()}" in steps
    assert "mv scripts ops/scripts" in steps
    assert (root / "agent" / "rules" / "plant.md").is_file()


@pytest.mark.usefixtures("planted")
def test_the_printed_block_and_revalidate_end_in_the_same_tree(
    _template: Path, tmp_path: Path
) -> None:
    by_hand = _plant(_template, tmp_path / "by-hand", git=True)
    rewritten = _plant(_template, tmp_path / "rewritten", git=True)
    with pytest.raises(RetiredRootRefused) as refused:
        refuse_retired_layout(by_hand)

    subprocess.run(
        ["sh", "-e", "-c", refused.value.format_message()],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # The block moves the files and says to run the command again; revalidate
    # then has no move left, only the passes that are not about the layout
    # (the plant's unselected rule is named by its path).
    after_block = _revalidate(by_hand)
    result = _revalidate(rewritten)

    assert after_block.exit_code == 0, after_block.output
    assert result.exit_code == 0, result.output
    assert _files(by_hand) == _files(rewritten)
    assert _git(by_hand, "ls-files") == _git(rewritten, "ls-files")
    assert _git(by_hand, "diff", "--cached", "--name-status", "-M") == _git(
        rewritten, "diff", "--cached", "--name-status", "-M"
    )


@pytest.mark.usefixtures("planted")
def test_a_moved_directory_reads_through_the_plan(_template: Path, tmp_path: Path) -> None:
    root = _plant(_template, tmp_path, git=False)
    _write(root, {"agent/rules/current.md": "# Already moved.\n"})

    plan = _plan_root(root)
    with _after_moves_view(root, plan, ["agent/rules", "ops/ci-extra.yml"]) as view:
        listed = sorted(p.relative_to(view).as_posix() for p in view.rglob("*") if p.is_file())
        texts = {rel: (view / rel).read_text(encoding="utf-8") for rel in listed}

    assert listed == [
        "agent/rules/.keep-me",
        "agent/rules/current.md",
        "agent/rules/plant.md",
        "ops/ci-extra.yml",
    ]
    assert texts["agent/rules/plant.md"] == _OLD_SOURCES["rules/plant.md"]
    assert (root / "rules" / "plant.md").is_file()
    assert plan.source("agent/rules/plant.md") == "rules/plant.md"
    assert plan.source("agent/rules/current.md") == "agent/rules/current.md"


#: A lane's certificate set as a deployment minted it under the data tree.
_UNMIGRATED_CERTS = {
    f"bluesky_curve/{rel}": f"minted {rel}\n"
    for rel in (
        "bridge/proxy.key_secret",
        "bridge/clients/publisher.key",
        "queueserver/publisher.key_secret",
        "queueserver/proxy.key",
    )
}


@pytest.mark.slow
@pytest.mark.parametrize("migrate", ["printed block", "revalidate"])
def test_a_start_after_the_move_and_a_build_reuses_the_same_key_files(
    _template: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, migrate: str
) -> None:
    """The certificates move with the tree, and the next plain ``osprey up`` keeps them.

    Before the move a start is refused and mints nothing. After it, and a build,
    a start without ``--as-built`` passes the drift gate and provisioning finds
    the set complete under ``var/runtime``, so not one key byte changes.
    """
    from osprey.cli.main import cli
    from osprey.deployment import container_lifecycle
    from osprey.utils.workspace import RUNTIME_DIR_RELPATH
    from tests.fixtures.lifecycle_repo import preserved_environ

    # The live layout, no plant: a fresh deployment whose only retired path is
    # the certificate directory under its data tree.
    root = Path(shutil.copytree(_template, tmp_path / "repo", symlinks=True))
    _write(root, {f"data/.runtime/{rel}": body for rel, body in _UNMIGRATED_CERTS.items()})
    _write(root, {".env": "ANTHROPIC_API_KEY=test-key\n"})
    assert [(move.old, move.new) for move in find_retired_root(root).moves] == [
        ("data/.runtime", "var/runtime")
    ]

    refused = CliRunner().invoke(cli, ["up", "-d", "--repo", str(root)])
    assert refused.exit_code == 1, refused.output
    assert not (root / RUNTIME_DIR_RELPATH).exists()

    if migrate == "revalidate":
        result = _revalidate(root)
        assert result.exit_code == 0, result.output
    else:
        with pytest.raises(RetiredRootRefused) as block:
            refuse_retired_layout(root)
        subprocess.run(
            ["sh", "-e", "-c", block.value.format_message()],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert find_retired_root(root).empty
    with preserved_environ():
        built = CliRunner().invoke(
            cli, ["build", "--repo", str(root), "--skip-deps", "--skip-lifecycle"]
        )
    assert built.exit_code == 0, built.output

    started: list[Path] = []

    def _start(repo_root: Path, **_kwargs: object) -> None:
        # The real start provisions the certificates from the repo it runs in.
        started.append(repo_root)
        container_lifecycle._ensure_bluesky_document_plane_certs({"deployed_services": ["bluesky"]})

    monkeypatch.setattr(container_lifecycle, "up_as_built", _start)
    with preserved_environ():
        result = CliRunner().invoke(cli, ["up", "-d", "--repo", str(root)])
    assert result.exit_code == 0, result.output
    assert started == [root]
    runtime = root / RUNTIME_DIR_RELPATH
    assert {
        path.relative_to(runtime).as_posix(): path.read_text(encoding="utf-8")
        for path in runtime.rglob("*")
        if path.is_file()
    } == _UNMIGRATED_CERTS
