"""Staging the CANOPY payload into a deployment repo before ``osprey build``.

``python -m osprey.integrations.canopy.stage`` runs as a ``pre_build`` step with
the repo root as its working directory. It builds nothing: the payload — two
wheels, the viewer (whose ``ir-map.json`` is the widget catalog), Trace, the
staged spec when there is one, and ``STAGED.json`` — is made in the canopy
repo, and the step only checks it and copies it in. These tests
drive :func:`main` in a temporary repo root, and once through
:func:`_run_lifecycle_phase` exactly as ``build_cmd`` does, so the module is
also proven runnable under ``"{python}" -m``.

No CANOPY import is needed: the step runs under the build interpreter, where
CANOPY is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests._canopy_contract import OTHER_COMMIT

from osprey.cli.build_lifecycle import _run_lifecycle_phase
from osprey.cli.build_profile_schema import LifecycleStep
from osprey.cli.profile_conventions import unknown_root_entries
from osprey.integrations.canopy import LIBRARY_SUBPATH, PROVEN_COMMITS, library, stage
from osprey_connectors.errors import BuildProfileError

CANOPY_WHEEL = "canopy-0.4.0-py3-none-any.whl"
PYDM_WHEEL = "pydmconverter-1.2.0-py3-none-any.whl"


CANOPY_SHA = PROVEN_COMMITS["canopy"]
PYDM_SHA = PROVEN_COMMITS["PyDMConverter"]


def _staged(
    canopy_sha: str = CANOPY_SHA, spec: str | None = None, pydm_sha: str = PYDM_SHA
) -> dict:
    """``STAGED.json`` as ``scripts/build-payload.sh`` writes it: one flat object.

    The two commits default to the proven ones, the only pair a build stages.

    ``spec`` is the staged spec's commit; ``spec-digest`` is derived from it
    here, so two specs differ in both keys. ``None`` records no spec.
    """
    return {
        "canopy": canopy_sha,
        "runtime": "a" * 40,
        "sdk": "b" * 40,
        "pydm": pydm_sha,
        "trace": None,
        "canopy_version": "0.4.0",
        "pydm_version": "1.2.0",
        "spec": spec,
        "spec-digest": None if spec is None else f"sha256:{spec[:8] * 8}",
    }


def make_payload(
    root: Path,
    *,
    canopy_sha: str = CANOPY_SHA,
    pydm_sha: str = PYDM_SHA,
    canopy_wheel: str = CANOPY_WHEEL,
    pydm_wheel: str = PYDM_WHEEL,
    marker: str = "one",
    spec: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / canopy_wheel).write_bytes(f"canopy wheel {marker}".encode())
    (root / pydm_wheel).write_bytes(f"pydm wheel {marker}".encode())
    (root / "viewer").mkdir()
    (root / "viewer" / "index.html").write_text(f"<html>{marker}</html>", encoding="utf-8")
    (root / "viewer" / "ir-map.json").write_text("{}", encoding="utf-8")
    (root / "trace").mkdir()
    if spec is not None:
        schema = root / "spec" / "ir" / "screen-ir-1.1.schema.json"
        schema.parent.mkdir(parents=True)
        schema.write_text(json.dumps({"title": f"spec {spec}"}), encoding="utf-8")
        (root / "spec" / "SPEC-SOURCE").write_text(f"canopy-spec {spec}\n", encoding="utf-8")
    (root / "STAGED.json").write_text(
        json.dumps(_staged(canopy_sha, spec, pydm_sha), indent=2), encoding="utf-8"
    )
    return root


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty repo root as the working directory, with the staging variables unset."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.delenv(stage.PAYLOAD_ENV, raising=False)
    monkeypatch.delenv(stage.CHECK_ENV, raising=False)
    return root


@pytest.fixture
def payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo: Path) -> Path:
    assert repo.is_dir()  # the repo fixture first: it unsets what this one sets
    path = make_payload(tmp_path / "payload")
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(path))
    return path


def _root_wheels(repo: Path) -> list[str]:
    return sorted(p.name for p in repo.iterdir() if p.name.lower().endswith(".whl"))


def _tree(root: Path) -> dict[str, bytes | None]:
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def _identity(repo: Path) -> dict[str, tuple[int, int]]:
    """Inode and mtime of everything staged: any rewrite changes one of them."""
    paths = [repo / stage.PAYLOAD_DIRNAME, *(repo / stage.PAYLOAD_DIRNAME).rglob("*")]
    paths += [repo / name for name in _root_wheels(repo)]
    return {str(p): (p.stat().st_ino, p.stat().st_mtime_ns) for p in paths}


# ---------------------------------------------------------------------------
# Where the payload directory comes from
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("payload")
def test_a_payload_named_in_the_environment_is_staged(repo: Path) -> None:
    assert stage.main([]) == 0

    assert _root_wheels(repo) == [CANOPY_WHEEL, PYDM_WHEEL]
    assert (repo / CANOPY_WHEEL).read_bytes() == b"canopy wheel one"
    staged = repo / stage.PAYLOAD_DIRNAME
    assert json.loads((staged / "STAGED.json").read_text(encoding="utf-8")) == _staged()
    assert (staged / "viewer" / "index.html").read_text(encoding="utf-8") == "<html>one</html>"
    assert (staged / "viewer" / "ir-map.json").is_file()
    assert (staged / "trace").is_dir()
    assert not (staged / "widgets").exists()
    assert not (staged / "spec").exists()


@pytest.mark.usefixtures("payload")
def test_the_wheels_are_not_copied_under_the_payload_directory(repo: Path) -> None:
    assert stage.main([]) == 0
    assert not list((repo / stage.PAYLOAD_DIRNAME).rglob("*.whl"))


def test_the_payload_is_read_from_env_shared_when_the_process_leaves_it_unset(
    repo: Path, tmp_path: Path
) -> None:
    path = make_payload(tmp_path / "from-shared")
    (repo / ".env.shared").write_text(f"{stage.PAYLOAD_ENV}={path}\n", encoding="utf-8")

    assert stage.main([]) == 0
    assert (repo / stage.PAYLOAD_DIRNAME / "STAGED.json").is_file()


def test_dot_env_wins_over_env_shared(repo: Path, tmp_path: Path) -> None:
    shared = make_payload(tmp_path / "shared", marker="shared")
    local = make_payload(tmp_path / "local", marker="local")
    (repo / ".env.shared").write_text(f"{stage.PAYLOAD_ENV}={shared}\n", encoding="utf-8")
    (repo / ".env").write_text(f'{stage.PAYLOAD_ENV}="{local}"\n', encoding="utf-8")

    assert stage.main([]) == 0
    assert (repo / CANOPY_WHEEL).read_bytes() == b"canopy wheel local"
    index = repo / stage.PAYLOAD_DIRNAME / "viewer" / "index.html"
    assert index.read_text(encoding="utf-8") == "<html>local</html>"


def test_the_process_environment_wins_over_the_env_files(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from_file = make_payload(tmp_path / "file", marker="file")
    from_env = make_payload(tmp_path / "env", marker="env")
    (repo / ".env").write_text(f"{stage.PAYLOAD_ENV}={from_file}\n", encoding="utf-8")
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(from_env))

    assert stage.main([]) == 0
    assert (repo / CANOPY_WHEEL).read_bytes() == b"canopy wheel env"
    index = repo / stage.PAYLOAD_DIRNAME / "viewer" / "index.html"
    assert index.read_text(encoding="utf-8") == "<html>env</html>"


def test_a_relative_payload_path_is_taken_from_the_repo_root(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_payload(repo / ".payload")
    monkeypatch.setenv(stage.PAYLOAD_ENV, ".payload")

    assert stage.main([]) == 0
    assert (repo / stage.PAYLOAD_DIRNAME / "STAGED.json").is_file()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _assert_nothing_staged(repo: Path) -> None:
    assert _root_wheels(repo) == []
    assert not (repo / stage.PAYLOAD_DIRNAME).exists()


def test_an_unset_payload_stages_nothing_and_passes(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert stage.main([]) == 0
    assert (
        capsys.readouterr().out
        == f"canopy payload: nothing staged ({stage.PAYLOAD_ENV} is not set)\n"
    )
    _assert_nothing_staged(repo)


@pytest.mark.parametrize(
    ("payload", "check", "off"),
    [("", "", True), ("", "1", True), ("/some/payload", "", False), ("", "0", False)],
)
def test_the_panel_is_off_only_when_no_payload_is_named_and_staging_is_on(
    repo: Path, monkeypatch: pytest.MonkeyPatch, payload: str, check: str, off: bool
) -> None:
    monkeypatch.setenv(stage.PAYLOAD_ENV, payload)
    monkeypatch.setenv(stage.CHECK_ENV, check)
    assert stage.panel_off(repo) is off


def test_a_payload_named_in_the_env_file_keeps_the_panel(repo: Path) -> None:
    (repo / ".env").write_text(f"{stage.PAYLOAD_ENV}=/some/payload\n", encoding="utf-8")
    assert stage.panel_off(repo) is False


def test_an_empty_payload_value_counts_as_unset(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty shell value does not hide the one the env file names."""
    path = make_payload(tmp_path / "from-file")
    monkeypatch.setenv(stage.PAYLOAD_ENV, "")
    (repo / ".env").write_text(f"{stage.PAYLOAD_ENV}={path}\n", encoding="utf-8")

    assert stage.main([]) == 0
    assert (repo / stage.PAYLOAD_DIRNAME / "STAGED.json").is_file()


def test_a_missing_payload_directory_is_refused(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nowhere"
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(missing))

    assert stage.main([]) != 0
    assert str(missing) in capsys.readouterr().err
    _assert_nothing_staged(repo)


def test_a_directory_without_staged_json_is_refused(
    repo: Path, payload: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``STAGED.json`` is written last: a directory without it is a half-built payload."""
    (payload / "STAGED.json").unlink()

    assert stage.main([]) != 0
    assert "STAGED.json" in capsys.readouterr().err
    _assert_nothing_staged(repo)


def test_an_unparseable_staged_json_is_refused(repo: Path, payload: Path) -> None:
    (payload / "STAGED.json").write_text("{not json", encoding="utf-8")

    assert stage.main([]) != 0
    _assert_nothing_staged(repo)


@pytest.mark.parametrize("missing", [CANOPY_WHEEL, PYDM_WHEEL])
def test_a_payload_missing_a_wheel_is_refused(
    repo: Path, payload: Path, missing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    (payload / missing).unlink()

    assert stage.main([]) != 0
    assert missing.split("-")[0] in capsys.readouterr().err
    _assert_nothing_staged(repo)


def test_a_payload_with_two_wheels_of_one_package_is_refused(repo: Path, payload: Path) -> None:
    (payload / "canopy-0.3.0-py3-none-any.whl").write_bytes(b"older")

    assert stage.main([]) != 0
    _assert_nothing_staged(repo)


def test_a_payload_from_another_canopy_is_refused(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_payload(tmp_path / "other-canopy", canopy_sha=OTHER_COMMIT)
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(source))

    assert stage.main([]) == 1
    err = capsys.readouterr().err
    assert str(source) in err
    assert f"canopy {OTHER_COMMIT[:12]};" in err
    assert f"proven against {CANOPY_SHA[:12]}:" in err
    assert "build the payload at that commit" in err
    _assert_nothing_staged(repo)


def test_a_payload_from_another_converter_is_refused(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_payload(tmp_path / "other-converter", pydm_sha=OTHER_COMMIT)
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(source))

    assert stage.main([]) == 1
    err = capsys.readouterr().err
    assert str(source) in err
    assert f"PyDMConverter {OTHER_COMMIT[:12]};" in err
    assert f"proven against {PYDM_SHA[:12]}:" in err
    _assert_nothing_staged(repo)


@pytest.mark.parametrize(
    "held",
    [
        pytest.param(..., id="absent"),
        pytest.param(None, id="null"),
        pytest.param("", id="empty"),
        pytest.param(17, id="number"),
    ],
)
def test_a_payload_naming_no_canopy_commit_is_refused(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    held: object,
) -> None:
    source = make_payload(tmp_path / "unnamed-canopy")
    record = _staged()
    if held is ...:
        del record["canopy"]
    else:
        record["canopy"] = held
    (source / "STAGED.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(source))

    assert stage.main([]) == 1
    err = capsys.readouterr().err
    assert "names no canopy commit" in err
    assert f"proven against {CANOPY_SHA[:12]}:" in err
    _assert_nothing_staged(repo)


def test_a_payload_at_a_short_form_of_the_proven_commit_is_refused(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record holds the whole commit: a prefix of the proven one is another value."""
    source = make_payload(tmp_path / "short", canopy_sha=CANOPY_SHA[:12])
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(source))

    assert stage.main([]) == 1
    _assert_nothing_staged(repo)


@pytest.mark.usefixtures("payload")
def test_a_refusal_leaves_the_previous_staging_in_place(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert stage.main([]) == 0
    before = _identity(repo)

    monkeypatch.setenv(stage.PAYLOAD_ENV, str(tmp_path / "nowhere"))
    assert stage.main([]) != 0
    assert _identity(repo) == before


# ---------------------------------------------------------------------------
# The opt-out
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("payload")
def test_check_zero_exits_cleanly_without_staging(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(stage.CHECK_ENV, "0")

    assert stage.main([]) == 0
    _assert_nothing_staged(repo)


def test_check_zero_needs_no_payload_at_all(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(stage.CHECK_ENV, "0")
    assert stage.main([]) == 0
    _assert_nothing_staged(repo)


def test_check_zero_is_read_from_the_env_files_too(repo: Path) -> None:
    (repo / ".env").write_text(f"{stage.CHECK_ENV}=0\n", encoding="utf-8")
    assert stage.main([]) == 0
    _assert_nothing_staged(repo)


@pytest.mark.usefixtures("payload")
def test_any_other_check_value_still_stages(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(stage.CHECK_ENV, "1")
    assert stage.main([]) == 0
    assert _root_wheels(repo) == [CANOPY_WHEEL, PYDM_WHEEL]


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("payload")
def test_an_unchanged_payload_rewrites_nothing(repo: Path) -> None:
    assert stage.main([]) == 0
    before = _identity(repo)

    assert stage.main([]) == 0
    assert _identity(repo) == before


def test_a_changed_payload_replaces_the_staged_directory_whole(
    repo: Path, payload: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (payload / "viewer" / "gone-in-the-next-build.js").write_text("old", encoding="utf-8")
    assert stage.main([]) == 0
    assert (repo / stage.PAYLOAD_DIRNAME / "viewer" / "gone-in-the-next-build.js").is_file()

    newer = make_payload(tmp_path / "newer", marker="two", spec=SPEC_A)
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(newer))
    assert stage.main([]) == 0

    staged = repo / stage.PAYLOAD_DIRNAME
    expected = {k: v for k, v in _tree(newer).items() if not k.endswith(".whl")}
    assert _tree(staged) == expected
    assert (repo / CANOPY_WHEEL).read_bytes() == b"canopy wheel two"
    assert (repo / PYDM_WHEEL).read_bytes() == b"pydm wheel two"


@pytest.mark.usefixtures("payload")
def test_a_new_version_removes_the_wheels_of_the_old_one(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert stage.main([]) == 0

    newer = make_payload(
        tmp_path / "newer",
        canopy_wheel="canopy-0.5.0-py3-none-any.whl",
        pydm_wheel="pydmconverter-1.3.0-py3-none-any.whl",
    )
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(newer))
    assert stage.main([]) == 0

    assert _root_wheels(repo) == [
        "canopy-0.5.0-py3-none-any.whl",
        "pydmconverter-1.3.0-py3-none-any.whl",
    ]


@pytest.mark.usefixtures("payload")
def test_stale_wheels_are_removed_whatever_their_case(repo: Path) -> None:
    (repo / "PyDMConverter-0.9.0-py3-none-any.whl").write_bytes(b"stale")
    (repo / "Canopy-0.1.0-py3-none-any.WHL").write_bytes(b"stale")

    assert stage.main([]) == 0
    assert _root_wheels(repo) == [CANOPY_WHEEL, PYDM_WHEEL]


@pytest.mark.usefixtures("payload")
def test_other_wheels_in_the_repo_root_are_left_alone(repo: Path) -> None:
    """``osprey up --dev`` stages the framework's own wheels in the same directory."""
    others = [
        "osprey_framework-2026.9.0-py3-none-any.whl",
        "osprey_connectors-2026.9.0-py3-none-any.whl",
        "canopy_extras-1.0-py3-none-any.whl",
    ]
    for name in others:
        (repo / name).write_bytes(b"not ours")

    assert stage.main([]) == 0
    assert _root_wheels(repo) == sorted([CANOPY_WHEEL, PYDM_WHEEL, *others])
    assert all((repo / name).read_bytes() == b"not ours" for name in others)


@pytest.mark.usefixtures("payload")
def test_a_wheel_deleted_from_the_repo_root_is_staged_again(repo: Path) -> None:
    """``STAGED.json`` alone does not prove the staging is whole: the wheels sit beside it."""
    assert stage.main([]) == 0
    (repo / PYDM_WHEEL).unlink()

    assert stage.main([]) == 0
    assert _root_wheels(repo) == [CANOPY_WHEEL, PYDM_WHEEL]


@pytest.mark.usefixtures("payload")
def test_leftovers_of_an_interrupted_run_are_cleared(repo: Path) -> None:
    leftover = repo / stage.SCRATCH_DIRNAME
    leftover.mkdir()
    (leftover / "half").write_text("half a copy", encoding="utf-8")

    assert stage.main([]) == 0
    assert not leftover.exists()
    assert not (repo / stage.PAYLOAD_DIRNAME / "half").exists()


@pytest.mark.usefixtures("payload")
def test_staging_leaves_nothing_the_root_warning_would_flag(repo: Path) -> None:
    assert stage.main([]) == 0
    assert unknown_root_entries(repo) == []


# ---------------------------------------------------------------------------
# The screen library is checked first
# ---------------------------------------------------------------------------


def make_library(repo: Path, *, report: bool = True) -> Path:
    """A screen library of one screen in ``repo``, with a report written for it or none."""
    root = repo / LIBRARY_SUBPATH
    (root / "screens" / "sr").mkdir(parents=True)
    (root / "screens" / "sr" / "overview.screen.json").write_bytes(b"{}\n")
    for path in [repo / "data", repo / "data" / "canopy", root, *root.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    if report:
        write_report(root)
    return root


def write_report(root: Path) -> None:
    """The part of CANOPY's report the check reads, for ``root`` as it stands."""
    judged = library.judge(library.scan(root))
    report = {
        "generator": {"schema": "canopy-library-report/1", "canopy_version": "0.4.0"},
        "files": {
            "digest": judged.digest,
            "list": [{"path": path, "sha256": sha} for path, sha in judged.files],
        },
    }
    (root / library.REPORT_JSON).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / library.REPORT_JSON).chmod(0o644)


def test_a_repo_without_a_library_passes_the_check(repo: Path) -> None:
    stage.check_library(repo)


@pytest.mark.usefixtures("payload")
def test_a_library_with_a_current_report_is_staged_over(repo: Path) -> None:
    make_library(repo)
    assert stage.main([]) == 0
    assert _root_wheels(repo) == [CANOPY_WHEEL, PYDM_WHEEL]


@pytest.mark.usefixtures("payload")
def test_a_stale_library_is_refused_before_anything_is_staged(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_library(repo)
    (root / "screens" / "sr" / "new.screen.json").write_bytes(b"{}\n")
    before = _tree(repo)

    assert stage.main([]) == 1

    assert _tree(repo) == before
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert all(line.startswith("canopy library: ") for line in lines)
    assert lines[0] == (
        "canopy library: data/canopy/library does not match its write-binding report"
    )
    assert lines[1].startswith("canopy library: digest mismatch: ")
    assert lines[2] == (
        "canopy library: added: screens/sr/new.screen.json (in the library, not in the report)"
    )
    assert lines[3].endswith(f"python -m canopy_mcp.report {root} --served-at library --write")
    assert len(lines) == 4


@pytest.mark.usefixtures("payload")
def test_a_library_with_no_report_is_refused(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_library(repo, report=False)
    assert stage.main([]) == 1
    err = capsys.readouterr().err
    assert "no report: the library holds 1 files and no LIBRARY-REPORT.json" in err
    assert "python -m canopy_mcp.report" in err
    assert not (repo / stage.PAYLOAD_DIRNAME).exists()


def test_check_zero_does_not_skip_the_library_check(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The switch skips staging; the library is baked into the image either way."""
    monkeypatch.setenv(stage.CHECK_ENV, "0")
    root = make_library(repo)
    assert stage.main([]) == 0
    assert "skipped" in capsys.readouterr().out

    (root / "screens" / "sr" / "overview.screen.json").write_bytes(b'{"edited": true}\n')
    assert stage.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "canopy library: changed: screens/sr/overview.screen.json" in captured.err


def test_the_library_is_checked_before_the_payload_is_looked_for(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No payload named and a stale library: the library is the refusal."""
    make_library(repo, report=False)
    assert stage.main([]) == 1
    err = capsys.readouterr().err
    assert "canopy library: " in err
    assert stage.PAYLOAD_ENV not in err


@pytest.mark.usefixtures("payload")
def test_a_mode_refusal_prints_the_chmod_command(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_library(repo)
    (root / "screens" / "sr" / "overview.screen.json").chmod(0o600)
    assert stage.main([]) == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[1:] == [
        "canopy library: mode: screens/sr/overview.screen.json: not readable by others (mode 0600)",
        f"canopy library: fix the modes with: chmod -R o+rX,o-w {root}",
    ]


@pytest.mark.usefixtures("payload")
def test_run_raises_the_refusal_with_its_lines(repo: Path) -> None:
    make_library(repo, report=False)
    with pytest.raises(stage.StaleLibrary) as excinfo:
        stage.run(repo)
    assert isinstance(excinfo.value, stage.StageError)
    assert excinfo.value.lines == library.library_is_fresh(repo / LIBRARY_SUBPATH).lines


@pytest.mark.usefixtures("payload")
def test_a_library_ds_store_does_not_stop_the_build(repo: Path) -> None:
    root = make_library(repo)
    (root / "screens" / ".DS_Store").write_bytes(b"finder")
    assert stage.main([]) == 0


# ---------------------------------------------------------------------------
# As a pre_build step
# ---------------------------------------------------------------------------

_STEP = LifecycleStep(
    name="canopy-payload", run='"{python}" -m osprey.integrations.canopy.stage', timeout=30
)


def test_the_step_stages_when_run_as_build_cmd_runs_it(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repo root as working directory, an empty stage directory as ``project_path``.

    The runner overlays the *stage* directory's env chain, so the repo's own
    ``.env`` reaches the step only because the step reads it itself.
    """
    path = make_payload(tmp_path / "payload")
    (repo / ".env").write_text(f"{stage.PAYLOAD_ENV}={path}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # the step must use the cwd the runner gives it, not ours
    empty_stage = tmp_path / "stage"
    empty_stage.mkdir()

    _run_lifecycle_phase("pre_build", [_STEP], repo, empty_stage, venv=None, repo_root=repo)

    assert _root_wheels(repo) == [CANOPY_WHEEL, PYDM_WHEEL]
    assert (repo / stage.PAYLOAD_DIRNAME / "STAGED.json").is_file()


def test_the_step_stops_the_build_when_the_payload_is_missing(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(tmp_path / "nowhere"))
    empty_stage = tmp_path / "stage"
    empty_stage.mkdir()

    with pytest.raises(BuildProfileError) as excinfo:
        _run_lifecycle_phase("pre_build", [_STEP], repo, empty_stage, venv=None, repo_root=repo)
    assert "canopy-payload" in str(excinfo.value)


def test_the_step_stops_the_build_on_a_stale_library_though_staging_is_skipped(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(stage.CHECK_ENV, "0")
    root = make_library(repo)
    (root / "screens" / "sr" / "overview.screen.json").unlink()
    empty_stage = tmp_path / "stage"
    empty_stage.mkdir()

    with pytest.raises(BuildProfileError) as excinfo:
        _run_lifecycle_phase("pre_build", [_STEP], repo, empty_stage, venv=None, repo_root=repo)
    message = str(excinfo.value)
    assert "canopy-payload" in message
    assert "missing: screens/sr/overview.screen.json" in message
    assert "canopy_mcp.report" in message


# ---------------------------------------------------------------------------
# The staged spec: copied with the payload, and part of what "current" means
# ---------------------------------------------------------------------------

SPEC_A = "5" * 40
SPEC_B = "6" * 40


def test_a_payload_with_a_spec_stages_it_beside_the_viewer(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_payload(tmp_path / "with-spec", spec=SPEC_A)
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(source))

    assert stage.main([]) == 0

    staged = repo / stage.PAYLOAD_DIRNAME
    assert _tree(staged / "spec") == _tree(source / "spec")
    assert (staged / "spec" / "SPEC-SOURCE").read_text(
        encoding="utf-8"
    ) == f"canopy-spec {SPEC_A}\n"
    assert json.loads((staged / "STAGED.json").read_text(encoding="utf-8"))["spec"] == SPEC_A


@pytest.mark.parametrize(
    "key", [pytest.param("spec", id="spec"), pytest.param("spec-digest", id="spec-digest")]
)
def test_a_changed_spec_record_is_not_current(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """A payload rebuilt with another spec — every repo SHA unchanged — is staged anew."""
    first = make_payload(tmp_path / "first", spec=SPEC_A)
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(first))
    assert stage.main([]) == 0
    assert stage.is_current(repo, stage.read_payload(first))

    second = make_payload(tmp_path / "second", spec=SPEC_A)
    record = json.loads((second / "STAGED.json").read_text(encoding="utf-8"))
    record[key] = _staged(spec=SPEC_B)[key]
    (second / "STAGED.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    payload = stage.read_payload(second)

    assert not stage.is_current(repo, payload)
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(second))
    assert stage.main([]) == 0
    staged = json.loads((repo / stage.PAYLOAD_DIRNAME / "STAGED.json").read_text(encoding="utf-8"))
    assert staged[key] == record[key]


def test_a_spec_added_to_an_otherwise_equal_payload_is_staged(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interim payload (no spec) and the same payload with FR17's spec differ only there."""
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(make_payload(tmp_path / "interim")))
    assert stage.main([]) == 0
    assert not (repo / stage.PAYLOAD_DIRNAME / "spec").exists()

    with_spec = make_payload(tmp_path / "final", spec=SPEC_A)
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(with_spec))
    assert stage.main([]) == 0
    assert (repo / stage.PAYLOAD_DIRNAME / "spec" / "SPEC-SOURCE").is_file()


def test_a_staged_copy_that_lost_its_spec_is_staged_again(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_payload(tmp_path / "with-spec", spec=SPEC_A)
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(source))
    assert stage.main([]) == 0
    staged_spec = repo / stage.PAYLOAD_DIRNAME / "spec"
    for path in sorted(staged_spec.rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    staged_spec.rmdir()

    assert not stage.is_current(repo, stage.read_payload(source))
    assert stage.main([]) == 0
    assert _tree(staged_spec) == _tree(source / "spec")


def test_a_payload_naming_a_spec_it_does_not_hold_is_refused(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_payload(tmp_path / "no-spec-dir")
    (source / "STAGED.json").write_text(json.dumps(_staged(spec=SPEC_A)), encoding="utf-8")
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(source))

    assert stage.main([]) == 1
    assert f"names spec {SPEC_A}" in capsys.readouterr().err
    assert not (repo / stage.PAYLOAD_DIRNAME).exists()


def test_a_spec_dir_its_staged_json_does_not_name_is_refused(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_payload(tmp_path / "unnamed-spec", spec=SPEC_A)
    (source / "STAGED.json").write_text(json.dumps(_staged()), encoding="utf-8")
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(source))

    assert stage.main([]) == 1
    assert "holds spec/ but" in capsys.readouterr().err
    assert not (repo / stage.PAYLOAD_DIRNAME).exists()


def test_a_half_named_spec_is_refused(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_payload(tmp_path / "half", spec=SPEC_A)
    record = _staged(spec=SPEC_A)
    record["spec-digest"] = None
    (source / "STAGED.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(source))

    assert stage.main([]) == 1
    assert "both spec and spec-digest or with neither" in capsys.readouterr().err
    assert not (repo / stage.PAYLOAD_DIRNAME).exists()


def test_a_payload_recorded_before_the_spec_keys_existed_still_stages(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An s4-era ``STAGED.json`` has neither key and no ``spec/``: no spec, staged as before."""
    source = make_payload(tmp_path / "older")
    record = _staged()
    del record["spec"], record["spec-digest"]
    (source / "STAGED.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv(stage.PAYLOAD_ENV, str(source))

    assert stage.main([]) == 0
    assert not stage.read_payload(source).has_spec
    assert not (repo / stage.PAYLOAD_DIRNAME / "spec").exists()
