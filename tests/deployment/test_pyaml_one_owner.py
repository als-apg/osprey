"""The `pyaml` import package has exactly one owning distribution in every image.

Two pyAML distributions (upstream ``accelerator-middle-layer`` and the staging
``accelerator-middle-layer-apg``) write the same ``pyaml/`` directory, and no
resolver or ``pip check`` notices the pair. The check module reads installed
metadata only; these tests drive it over fixture site-packages directories, then
hold every image that installs the local-requirements manifest to calling it.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

import osprey
from osprey.deployment import pyaml_owner

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMPLATES_DIR = pathlib.Path(osprey.__file__).parent / "templates"
CHECK_COMMAND = "python -m osprey.deployment.pyaml_owner"


def _dist(
    site: pathlib.Path,
    name: str,
    files: list[str] | None,
    top_level: str | None = None,
) -> None:
    """Write one installed distribution's metadata into ``site``.

    ``files`` becomes the RECORD (``None`` writes no RECORD at all, the shape of
    a record-less install); ``top_level`` becomes ``top_level.txt`` when given.
    """
    info = site / f"{name.replace('-', '_')}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
    if files is not None:
        # The record lists files that exist: `Distribution.files` drops entries
        # missing from disk, exactly as it would in a real environment.
        for f in files:
            (site / f).parent.mkdir(parents=True, exist_ok=True)
            (site / f).touch()
        record = [f"{f},," for f in files] + [f"{info.name}/METADATA,,", f"{info.name}/RECORD,,"]
        (info / "RECORD").write_text("\n".join(record) + "\n")
    if top_level is not None:
        (info / "top_level.txt").write_text(top_level)


def _staging(site: pathlib.Path) -> None:
    _dist(site, "accelerator-middle-layer-apg", ["pyaml/__init__.py", "pyaml/lattice/element.py"])


def _upstream(site: pathlib.Path) -> None:
    _dist(site, "accelerator_middle_layer", ["pyaml/__init__.py", "pyaml/arrays/array.py"])


def _neighbours(site: pathlib.Path) -> None:
    """Distributions whose names or files merely resemble pyAML's."""
    _dist(site, "pyaml-cs-osprey", ["pyaml_cs_osprey/__init__.py"])
    _dist(site, "osprey-framework", ["osprey/__init__.py", "osprey/deployment/pyaml_owner.py"])
    _dist(site, "pyamlish", ["pyamlish.py"])


def test_one_owner_passes(tmp_path, capsys):
    _staging(tmp_path)
    _neighbours(tmp_path)
    assert pyaml_owner.owners(path=[str(tmp_path)]) == ["accelerator-middle-layer-apg"]
    assert pyaml_owner.main(path=[str(tmp_path)]) == 0
    assert "accelerator-middle-layer-apg" in capsys.readouterr().out


def test_two_owners_fail_and_name_both(tmp_path, capsys):
    _staging(tmp_path)
    _upstream(tmp_path)
    _neighbours(tmp_path)
    assert pyaml_owner.owners(path=[str(tmp_path)]) == [
        "accelerator-middle-layer",
        "accelerator-middle-layer-apg",
    ]
    assert pyaml_owner.main(path=[str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "accelerator-middle-layer," in err and "accelerator-middle-layer-apg" in err


def test_two_owners_across_path_entries_fail(tmp_path):
    """A second owner on another sys.path entry still shadows or is shadowed."""
    first, second = tmp_path / "a", tmp_path / "b"
    _staging(first)
    _upstream(second)
    assert pyaml_owner.main(path=[str(first), str(second)]) == 1


def test_the_yaml_printer_named_pyaml_is_a_second_owner(tmp_path):
    """PyPI's unrelated `pyaml` YAML printer also installs `pyaml/`."""
    _staging(tmp_path)
    _dist(tmp_path, "pyaml", ["pyaml/__init__.py", "pyaml/cli.py"])
    assert pyaml_owner.main(path=[str(tmp_path)]) == 1


def test_a_single_module_pyaml_py_owns_the_name(tmp_path):
    _staging(tmp_path)
    _dist(tmp_path, "pyaml-single", ["pyaml.py"])
    assert pyaml_owner.owners(path=[str(tmp_path)]) == [
        "accelerator-middle-layer-apg",
        "pyaml-single",
    ]


def test_a_record_less_install_falls_back_to_top_level(tmp_path):
    _staging(tmp_path)
    _dist(tmp_path, "legacy-pyaml", None, top_level="pyaml\n")
    _dist(tmp_path, "legacy-other", None, top_level="other\n")
    assert pyaml_owner.owners(path=[str(tmp_path)]) == [
        "accelerator-middle-layer-apg",
        "legacy-pyaml",
    ]


def test_no_owner_fails(tmp_path, capsys):
    _neighbours(tmp_path)
    assert pyaml_owner.main(path=[str(tmp_path)]) == 1
    assert "no installed distribution owns" in capsys.readouterr().err


def test_the_module_runs_as_a_script_and_exits_nonzero_on_two_owners(tmp_path):
    """The images call it with `python -m`; the exit status is what fails the build."""
    _staging(tmp_path)
    _upstream(tmp_path)
    probe = (
        "import sys; from osprey.deployment import pyaml_owner; "
        f"sys.exit(pyaml_owner.main(path=[{str(tmp_path)!r}]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 1, result.stderr
    assert "2 owning distributions" in result.stderr


def test_this_environment_has_one_owner():
    """The development environment resolves the same single owner the images must."""
    assert pyaml_owner.owners() == ["accelerator-middle-layer-apg"]


# ── every image that installs the local-requirements manifest calls the check ──

MANIFEST = "osprey-local-requirements.txt"
MANIFEST_IMAGES = sorted(
    p
    for p in [*TEMPLATES_DIR.rglob("Dockerfile"), *TEMPLATES_DIR.rglob("Dockerfile.j2")]
    if f"uv pip install -r /tmp/deps-ctx/{MANIFEST}" in p.read_text()
)
SOURCE_IMAGES = [REPO_ROOT / "docker" / "virtual-accelerator" / "Containerfile"]


def test_the_manifest_images_are_discovered():
    names = {str(p.relative_to(TEMPLATES_DIR)) for p in MANIFEST_IMAGES}
    assert {
        "project/Dockerfile.j2",
        "services/virtual_accelerator/Dockerfile",
        "services/bluesky/Dockerfile",
    } <= names


def _wheel_layer(text: str) -> str:
    """The RUN that follows the `*.wh[l]` COPY — the layer that runs on every dev build."""
    match = re.search(r"^COPY [^\n]*\*\.wh\[l\][^\n]*\n(RUN .*?)(?:\n\n|\Z)", text, re.M | re.S)
    assert match, "no wheel layer found"
    return match.group(1)


@pytest.mark.parametrize(
    "image", MANIFEST_IMAGES, ids=[str(p.relative_to(TEMPLATES_DIR)) for p in MANIFEST_IMAGES]
)
def test_manifest_image_asserts_one_owner_after_the_overlay(image):
    """The manifest is staged only beside dev wheels, so the wheel layer runs
    whenever it was installed — and it is the first layer where the checking
    module is guaranteed present (a released primer predates it)."""
    wheel = _wheel_layer(image.read_text())
    assert f"&& {CHECK_COMMAND}" in wheel
    assert wheel.index("--force-reinstall") < wheel.index(CHECK_COMMAND)


@pytest.mark.parametrize("image", SOURCE_IMAGES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_source_built_image_asserts_one_owner(image):
    text = image.read_text()
    assert f"&& {CHECK_COMMAND}" in text
    assert text.index(".[virtual-accelerator]") < text.index(CHECK_COMMAND)


def test_no_image_uninstalls_pyaml():
    """One assertion, no repair machinery: nothing uninstalls a pyAML owner."""
    for image in [*MANIFEST_IMAGES, *SOURCE_IMAGES]:
        assert not re.search(r"uninstall[^\n]*accelerator-middle-layer", image.read_text()), image
