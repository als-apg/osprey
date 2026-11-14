"""A member wheel the framework refuses names the git command that fixes the checkout.

A clone without the member's tags builds it as ``0.0.postN``. The refusal adds
one line for the two checkouts that cause it: a shallow clone must unshallow
(``git fetch --tags`` alone reaches no tag there), a full clone with no tag
must fetch them. A checkout that reaches its tag adds nothing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from osprey.deployment import wheel_build as wb
from osprey.deployment.errors import DevModeUnavailableError

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="builds git checkouts")

REQUIREMENT = "osprey-connectors>=2026.11.11"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "t")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "t@example.invalid")


def _repo(path: Path, *, tag: bool) -> Path:
    path.mkdir()
    _git(path, "init", "-q")
    for n in range(2):
        (path / "f").write_text(str(n))
        _git(path, "add", "f")
        _git(path, "commit", "-q", "-m", str(n))
        if tag and n == 0:
            _git(path, "tag", "osprey-connectors-v2026.11.11")
    return path


def _refusal(member_root: Path, monkeypatch) -> str:
    monkeypatch.setattr(wb, "_wheel_base_requirements", lambda wheel: [REQUIREMENT])
    monkeypatch.setattr(wb, "_describe_glob", lambda root: "osprey-connectors-v*")
    wheel = member_root / "osprey_connectors-0.0.post1-py3-none-any.whl"
    with pytest.raises(DevModeUnavailableError) as raised:
        wb._require_members_satisfy_framework(
            Path("osprey_framework-0.0.post1-py3-none-any.whl"),
            {"osprey-connectors": (member_root, wheel)},
        )
    return str(raised.value) + "\n" + str(raised.value.remedy)


def test_a_shallow_clone_is_told_to_unshallow(tmp_path, monkeypatch):
    origin = _repo(tmp_path / "origin", tag=True)
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{origin}", str(clone)],
        check=True,
        capture_output=True,
    )
    text = _refusal(clone, monkeypatch)
    assert f"git -C {clone} fetch --unshallow --tags" in text
    assert "Fetch the tags:  git fetch --tags" in text, "the existing text is kept"


def test_a_full_clone_without_tags_is_told_to_fetch_them(tmp_path, monkeypatch):
    clone = _repo(tmp_path / "clone", tag=False)
    text = _refusal(clone, monkeypatch)
    assert f"git -C {clone} fetch --tags" in text
    assert "--unshallow" not in text


def test_a_checkout_that_reaches_its_tag_adds_no_line(tmp_path, monkeypatch):
    clone = _repo(tmp_path / "clone", tag=True)
    text = _refusal(clone, monkeypatch)
    assert f"git -C {clone}" not in text


def test_a_path_git_cannot_read_adds_no_line(tmp_path, monkeypatch):
    text = _refusal(tmp_path / "missing", monkeypatch)
    assert "git -C" not in text
