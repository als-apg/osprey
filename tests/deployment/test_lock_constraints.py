"""A build from an osprey source checkout installs third-party packages at the lock's versions.

``--dev`` stages the checkout's ``uv.lock``, exported as a constraints file,
into every build context, and the file's presence is what puts the image
recipes on the lock-driven install (the recipes' side is pinned in
``test_service_dockerfiles`` and ``test_dockerfile_constraints``). These tests
pin the staging side: the export itself, its refusal of a missing or stale
lock, the byte-identical copy every context receives, the extras manifest an
extras-installing image is handed, and the cleanup that keeps a stale copy
from switching a later release build onto the lock.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import pytest

from osprey.deployment import compose_generator
from osprey.deployment import wheel_build as wb
from osprey.deployment.errors import DevModeUnavailableError
from osprey.deployment.lock_constraints import (
    LOCK_CONSTRAINTS_FILENAME,
    LockExportError,
    export_lock_constraints,
    reset_export_cache,
)
from osprey.deployment.members import WORKSPACE_MEMBERS

REPO = Path(__file__).resolve().parents[2]

needs_uv = pytest.mark.skipif(shutil.which("uv") is None, reason="exports with the host's uv")


@pytest.fixture(autouse=True)
def _fresh_export_memo():
    reset_export_cache()
    yield
    reset_export_cache()


# ── The export ───────────────────────────────────────────────────────────────


@needs_uv
def test_the_repos_lock_exports_as_pins_without_the_workspace():
    text = export_lock_constraints(REPO)
    lines = text.splitlines()
    assert lines, "the export is empty"
    assert not [line for line in lines if line.startswith("#") or line.startswith(" ")], (
        "the export carries a header or annotations; any byte of it that moves rebuilds the "
        "deps layer"
    )
    names = {line.split("==", 1)[0] for line in lines}
    for member in ("osprey-framework", *WORKSPACE_MEMBERS):
        assert member not in names, f"{member} is pinned; the staged wheel would be refused"
    # The marker-split torch pin the optimizer image resolves from the CPU index.
    assert any(line.startswith("torch==") and "+cpu" in line and ";" in line for line in lines)


@needs_uv
def test_the_export_is_deterministic():
    first = export_lock_constraints(REPO)
    reset_export_cache()
    assert export_lock_constraints(REPO) == first


def _tiny_project(root: Path, version: str = "1.0") -> Path:
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "p"\nversion = "{version}"\nrequires-python = ">=3.11"\n'
        "dependencies = []\n"
    )
    return root


def test_a_checkout_without_a_lock_is_refused(tmp_path):
    with pytest.raises(LockExportError) as exc:
        export_lock_constraints(_tiny_project(tmp_path / "p"))
    assert "uv.lock" in exc.value.reason
    assert "uv lock" in exc.value.remedy


@needs_uv
def test_a_stale_lock_is_refused_rather_than_re_resolved(tmp_path):
    """``--locked``: a lock that no longer matches pyproject.toml is an error,
    not a cue to resolve afresh — a fresh resolve is what the file prevents."""
    import subprocess

    project = _tiny_project(tmp_path / "p")
    env = {**os.environ, "UV_OFFLINE": "1"}
    subprocess.run(
        ["uv", "lock", "--quiet", "--project", str(project)], check=True, env=env, timeout=120
    )
    pyproject = project / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace('"1.0"', '"1.1"'))

    with pytest.raises(LockExportError) as exc:
        export_lock_constraints(project)
    assert "uv lock" in exc.value.remedy
    assert "out of date" in exc.value.remedy


def test_no_uv_is_refused(tmp_path, monkeypatch):
    project = _tiny_project(tmp_path / "p")
    (project / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr("osprey.deployment.lock_constraints.find_uv", lambda: None)

    with pytest.raises(LockExportError, match="uv is not installed"):
        export_lock_constraints(project)


# ── Staging ──────────────────────────────────────────────────────────────────


def _wheel(directory: Path, name: str, requires: list[str], extras: list[str] = ()) -> Path:
    stem = f"{name.replace('-', '_')}-0.0.0"
    wheel = directory / f"{stem}-py3-none-any.whl"
    metadata = (
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.0.0\n"
        + "".join(f"Provides-Extra: {e}\n" for e in extras)
        + "".join(f"Requires-Dist: {line}\n" for line in requires)
    )
    with zipfile.ZipFile(wheel, "w") as whl:
        whl.writestr(f"{stem}.dist-info/METADATA", metadata)
    return wheel


#: The framework's own spellings of an extra-gated requirement, as hatchling
#: writes them, plus a member requirement gated on the same extra.
FRAMEWORK_REQUIRES = [
    "aiohttp>=3.14",
    "aioca>=1.7; extra == 'virtual-accelerator'",
    "lume-pva-apg[ca,pva]==0.1.6; (sys_platform == 'linux' and (platform_machine == "
    "'x86_64' or platform_machine == 'aarch64')) and extra == 'virtual-accelerator'",
    "osprey-optimizer>=2026.6.2; extra == 'virtual-accelerator'",
    "azure-servicebus>=7.11; extra == 'teams'",
]

FAKE_EXPORT = "aiohttp==3.14.3\naioca==1.8 ; python_version >= '3.11'\n"


@pytest.fixture
def staged_checkout(tmp_path, monkeypatch):
    """An editable checkout whose wheel builds are stand-ins and whose lock exports as FAKE_EXPORT."""
    import osprey

    checkout = tmp_path / "checkout"
    (checkout / "src" / "osprey").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("[project]\nname='osprey-framework'\n")
    (checkout / "uv.lock").write_text("version = 1\n")
    for member in WORKSPACE_MEMBERS:
        (checkout / "packages" / member).mkdir(parents=True)
        (checkout / "packages" / member / "pyproject.toml").write_text(
            f"[project]\nname='{member}'\n"
        )
    monkeypatch.setattr(osprey, "__file__", str(checkout / "src" / "osprey" / "__init__.py"))

    cache = tmp_path / "cache"
    cache.mkdir()

    def _fake_build(root):
        if Path(root) == checkout:
            return _wheel(
                cache, "osprey-framework", FRAMEWORK_REQUIRES, ["virtual-accelerator", "teams"]
            )
        return _wheel(cache, Path(root).name, [])

    exports = []

    def _export(root):
        exports.append(root)
        return FAKE_EXPORT

    monkeypatch.setattr(wb, "_build_dev_wheel_cached", _fake_build)
    monkeypatch.setattr(wb, "_require_members_satisfy_framework", lambda *a: None)
    monkeypatch.setattr(wb, "export_lock_constraints", _export)
    return checkout, exports


def test_every_context_receives_the_same_lock_bytes(staged_checkout, tmp_path):
    checkout, exports = staged_checkout
    contexts = [tmp_path / name for name in ("service", "project", "render")]
    for ctx in contexts:
        ctx.mkdir()
        assert wb._copy_local_framework_for_override(str(ctx)) is True
    assert {(ctx / LOCK_CONSTRAINTS_FILENAME).read_bytes() for ctx in contexts} == {
        FAKE_EXPORT.encode()
    }
    assert exports and all(root == checkout for root in exports), (
        "the lock exported is not the checkout root's"
    )


@pytest.mark.usefixtures("staged_checkout")
def test_an_extras_image_is_handed_the_extras_own_requirements(tmp_path):
    """Gates removed, other markers kept, members dropped, sorted, newline-terminated."""
    ctx = tmp_path / "va"
    ctx.mkdir()
    assert wb._copy_local_framework_for_override(str(ctx), extras=("virtual-accelerator",))
    assert (ctx / wb.EXTRA_REQUIREMENTS_FILENAME).read_text() == (
        "aioca>=1.7\n"
        'lume-pva-apg[ca,pva]==0.1.6; sys_platform == "linux" and '
        '(platform_machine == "x86_64" or platform_machine == "aarch64")\n'
    )


@pytest.mark.usefixtures("staged_checkout")
def test_a_context_without_extras_gets_no_extras_manifest(tmp_path):
    ctx = tmp_path / "plain"
    ctx.mkdir()
    assert wb._copy_local_framework_for_override(str(ctx))
    assert not (ctx / wb.EXTRA_REQUIREMENTS_FILENAME).exists()


@pytest.mark.usefixtures("staged_checkout")
def test_an_extra_the_wheel_does_not_declare_fails_staging(tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    with pytest.raises(DevModeUnavailableError, match="no-such-extra"):
        wb._copy_local_framework_for_override(str(ctx), extras=("no-such-extra",))
    assert not list(ctx.iterdir()), "a failed staging left files behind"


@pytest.mark.usefixtures("staged_checkout")
def test_a_lock_that_cannot_be_exported_unstages_everything(tmp_path, monkeypatch):
    def _stale(_root):
        raise LockExportError("uv could not export the checkout's uv.lock", "run `uv lock`")

    monkeypatch.setattr(wb, "export_lock_constraints", _stale)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    with pytest.raises(DevModeUnavailableError) as exc:
        wb._copy_local_framework_for_override(str(ctx))
    assert "uv lock" in exc.value.remedy
    assert not list(ctx.iterdir()), "a context with wheels but no lock is half-staged"


@pytest.mark.usefixtures("staged_checkout")
def test_the_cleanup_snapshot_covers_every_staged_file(tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "Dockerfile").write_text("FROM scratch\n")
    before = wb._staged_dev_artifact_paths(str(ctx))
    wb._copy_local_framework_for_override(str(ctx), extras=("teams",))
    added = wb._staged_dev_artifact_paths(str(ctx)) - before
    assert {p.name for p in added} == {p.name for p in ctx.iterdir()} - {"Dockerfile"}


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ('extra == "x"', None),
        ('(python_version >= "3.11") and extra == "x"', 'python_version >= "3.11"'),
    ],
)
def test_the_two_hatchling_gate_shapes_come_apart(marker, expected):
    assert wb._strip_extra_clause(marker, "x") == expected


@pytest.mark.parametrize(
    "marker",
    ['extra == "x" or python_version < "3.11"', 'python_version >= "3.11" and extra == "x"'],
)
def test_any_other_gate_shape_is_refused(marker):
    with pytest.raises(ValueError):
        wb._strip_extra_clause(marker, "x")


# ── The switch is the file ───────────────────────────────────────────────────


def test_a_release_build_clears_a_stale_lock_from_its_context(tmp_path):
    """A context that outlived an earlier --dev run must not keep its lock: the
    file alone would put this release build on the lock-driven install."""
    ctx = tmp_path / "event_dispatcher"
    ctx.mkdir()
    (ctx / "Dockerfile").write_text("FROM scratch\n")
    for name in (
        LOCK_CONSTRAINTS_FILENAME,
        wb.LOCAL_REQUIREMENTS_FILENAME,
        wb.EXTRA_REQUIREMENTS_FILENAME,
        "osprey_framework-0.0.0-py3-none-any.whl",
    ):
        (ctx / name).write_text("stale\n")

    assert compose_generator._stage_dev_wheel_for_context(str(ctx), dev_mode=False) is False
    assert sorted(p.name for p in ctx.iterdir()) == ["Dockerfile"]


def test_the_extras_mapping_names_extras_the_framework_declares():
    import tomllib

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    optimizer = tomllib.loads((REPO / "packages/osprey-optimizer/pyproject.toml").read_text())
    for context, extras in compose_generator._DEV_WHEEL_EXTRAS.items():
        declared = (optimizer if context in compose_generator._DEV_WHEEL_MEMBERS else pyproject)[
            "project"
        ]["optional-dependencies"]
        for extra in extras:
            assert extra in declared, f"{context}: {extra!r} is not an extra its wheel declares"
