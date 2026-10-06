"""The qmd sidecar image's patch carriage.

The qmd image applies the patch files under ``services/qmd/patches/`` to the
pinned ``@tobilu/qmd`` release, in the order ``OSPREY_QMD_PATCHES`` names them,
and reports the set in its identity file and labels. These checks hold the
three places that name the set — the directory, the ENV list and the reported
version — to one another, run the Dockerfile's own application loop against
stub tools to prove it refuses a set that disagrees with the directory, and
check each patch's shape against the released package's file list.

Whether every hunk lands exactly on the pinned release is proved by the image
build itself (``patch --fuzz=0`` in the RUN): checking it here would need the
published package, which a unit lane does not download.
"""

import os
import pathlib
import re
import shutil
import subprocess

import pytest

import osprey
from tests.deployment._proxy_idiom import run_instructions

QMD_DIR = pathlib.Path(osprey.__file__).parent / "templates" / "services" / "qmd"
DOCKERFILE = QMD_DIR / "Dockerfile"
PATCHES_DIR = QMD_DIR / "patches"

#: The JavaScript files of the published ``@tobilu/qmd@2.5.3`` package
#: (``npm pack``), relative to the package root. A patch may modify only these
#: and may add only a file not among them.
QMD_2_5_3_DIST_JS = frozenset(
    {
        "dist/ast.js",
        "dist/bench/bench.js",
        "dist/bench/score.js",
        "dist/bench/types.js",
        "dist/cli/formatter.js",
        "dist/cli/qmd.js",
        "dist/collections.js",
        "dist/db.js",
        "dist/index.js",
        "dist/llm.js",
        "dist/maintenance.js",
        "dist/mcp/server.js",
        "dist/paths.js",
        "dist/store.js",
    }
)

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _dockerfile() -> str:
    return DOCKERFILE.read_text()


def _env_value(name: str) -> str:
    match = re.search(rf'^\s*(?:ENV\s+)?{name}=("[^"]*"|\S+)', _dockerfile(), re.MULTILINE)
    assert match, f"the qmd Dockerfile sets no {name}"
    return match.group(1).strip('"')


def _listed() -> list[str]:
    return _env_value("OSPREY_QMD_PATCHES").split()


def _patch_files() -> list[str]:
    return sorted(p.name for p in PATCHES_DIR.glob("*.patch"))


def _apply_run() -> str:
    runs = [run for run in run_instructions(_dockerfile()) if "/tmp/qmd-patches" in run]
    assert len(runs) == 1, "expected exactly one RUN applying the qmd patches"
    return runs[0]


def _file_diffs(text: str) -> list[tuple[str, str, list[str]]]:
    """Each file section of a unified diff as (old path, new path, body lines)."""
    lines = text.splitlines()
    starts = [
        i
        for i in range(len(lines) - 1)
        if lines[i].startswith("--- ") and lines[i + 1].startswith("+++ ")
    ]
    sections = []
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        body = lines[i + 2 : end]
        old, new = (lines[i + k][4:].split("\t")[0] for k in (0, 1))
        sections.append((old, new, body))
    return sections


def test_every_patch_file_is_listed_and_every_listed_patch_exists() -> None:
    """The directory and the ENV list name the same set, in file-name order."""
    listed = _listed()
    assert listed, "OSPREY_QMD_PATCHES names no patch"
    assert sorted(listed) == _patch_files(), (
        f"OSPREY_QMD_PATCHES {listed} disagrees with patches/ {_patch_files()}"
    )
    assert listed == sorted(listed), "patches apply in file-name order; list them that way"


def test_the_image_reports_the_patched_build() -> None:
    """Identity file and labels say which release, and which patch set, the image runs."""
    release = re.search(r"^ARG QMD_VERSION=(\S+)$", _dockerfile(), re.MULTILINE)
    assert release
    assert re.fullmatch(
        rf"{re.escape(release.group(1))}\+osprey\.\d+", _env_value("OSPREY_QMD_BUILD")
    )
    text = _dockerfile()
    assert '"OSPREY_QMD_VERSION=${OSPREY_QMD_BUILD}"' in text
    assert 'com.osprey.qmd.version="${OSPREY_QMD_BUILD}"' in text
    assert 'LABEL com.osprey.qmd.patches="${OSPREY_QMD_PATCHES}"' in text


def test_patches_apply_strictly_from_the_package_root() -> None:
    run = _apply_run()
    assert "patch -p1 --forward --batch --fuzz=0" in run
    assert '-d "$qmd_root"' in run and 'qmd_root="$(npm root -g)/@tobilu/qmd"' in run
    assert "apt-get purge -y patch" in run


@pytest.mark.parametrize("name", _patch_files())
def test_each_patch_targets_the_released_package(name: str) -> None:
    """Paths exist in 2.5.3 (or are new), and every hunk's line counts add up.

    A hunk whose header miscounts its lines is rejected by ``patch`` as
    malformed, which would surface only as a failed image build.
    """
    sections = _file_diffs((PATCHES_DIR / name).read_text())
    assert sections, f"{name} holds no file diff"
    for old, new, body in sections:
        if old == "/dev/null":
            assert new.startswith("b/dist/"), new
            assert new[2:] not in QMD_2_5_3_DIST_JS, f"{name} creates {new[2:]}, which 2.5.3 has"
        else:
            assert old.startswith("a/") and new.startswith("b/"), (old, new)
            assert old[2:] == new[2:], f"{name} renames {old} to {new}"
            assert old[2:] in QMD_2_5_3_DIST_JS, f"{name} modifies {old[2:]}, not in 2.5.3"
        hunks = 0
        k = 0
        while k < len(body):
            match = _HUNK.match(body[k])
            k += 1
            if not match:
                continue
            hunks += 1
            want_old = int(match.group(2) if match.group(2) is not None else 1)
            want_new = int(match.group(4) if match.group(4) is not None else 1)
            got_old = got_new = 0
            while k < len(body) and (got_old < want_old or got_new < want_new):
                line = body[k]
                if line.startswith("\\"):
                    k += 1
                    continue
                tag = line[:1] or " "
                assert tag in " +-", f"{name}: unexpected line inside a hunk: {line!r}"
                got_old += tag in " -"
                got_new += tag in " +"
                k += 1
            assert (got_old, got_new) == (want_old, want_new), (
                f"{name}: hunk {match.group(0)} carries {got_old}/{got_new} lines"
            )
        assert hunks, f"{name}: {new} has no hunk"


def test_the_readme_describes_every_patch() -> None:
    readme = (PATCHES_DIR / "README.md").read_text()
    for name in _patch_files():
        assert f"`{name}`" in readme, f"patches/README.md does not describe {name}"


# -- the application loop, run for real against stub tools ----------------------


def _stub_tools(bin_dir: pathlib.Path, qmd_root: pathlib.Path, log: pathlib.Path) -> None:
    stubs = {
        "apt-get": "exit 0",
        "npm": f'[ "$1 $2" = "root -g" ] && echo "{qmd_root.parent.parent}"',
        "qmd": "exit 0",
        # Records the patch it was given (stdin) instead of applying it.
        "patch": f'head -1 >> "{log}"',
    }
    for name, body in stubs.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)


def _run_apply_loop(tmp_path: pathlib.Path, patches: pathlib.Path) -> subprocess.CompletedProcess:
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("needs a POSIX sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qmd_root = tmp_path / "lib" / "node_modules" / "@tobilu" / "qmd"
    qmd_root.mkdir(parents=True)
    log = tmp_path / "applied.log"
    _stub_tools(bin_dir, qmd_root, log)
    body = _apply_run().removeprefix("RUN ").replace("/tmp/qmd-patches", str(patches))
    # The cleanup must not delete the test's copy of the directory.
    body = body.replace(f"rm -rf /var/lib/apt/lists/* {patches}", "true")
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "OSPREY_QMD_PATCHES": " ".join(_listed()),
    }
    return subprocess.run([sh, "-c", body], env=env, capture_output=True, text=True)


def test_the_loop_applies_the_listed_set_in_order(tmp_path: pathlib.Path) -> None:
    patches = tmp_path / "patches"
    shutil.copytree(PATCHES_DIR, patches)
    result = _run_apply_loop(tmp_path, patches)
    assert result.returncode == 0, result.stderr
    applied = [
        line.split()[1] for line in result.stdout.splitlines() if line.startswith("applying ")
    ]
    assert applied == _listed()
    assert len((tmp_path / "applied.log").read_text().splitlines()) == len(_listed())


def test_the_loop_refuses_an_unlisted_patch_file(tmp_path: pathlib.Path) -> None:
    patches = tmp_path / "patches"
    shutil.copytree(PATCHES_DIR, patches)
    (patches / "9999-stray.patch").write_text("--- a/dist/store.js\n+++ b/dist/store.js\n")
    result = _run_apply_loop(tmp_path, patches)
    assert result.returncode != 0
    assert "9999-stray.patch is not listed" in result.stderr


def test_the_loop_refuses_a_listed_patch_that_is_missing(tmp_path: pathlib.Path) -> None:
    patches = tmp_path / "patches"
    shutil.copytree(PATCHES_DIR, patches)
    missing = _listed()[-1]
    (patches / missing).unlink()
    result = _run_apply_loop(tmp_path, patches)
    assert result.returncode != 0
    assert f"patch {missing} is listed but not in patches/" in result.stderr
