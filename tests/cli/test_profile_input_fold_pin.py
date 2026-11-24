"""Pin of what the profile-input fold sees for each bundled preset.

Each preset is materialized with ``osprey init`` into a directory with a fixed name and
an explicit project name, so nothing recorded depends on the temporary path. The fold
is then read off the materialized ``profile.yml``: every input's key and label in fold
order, one digest per folded input, the profile hash and the preset hash. The result is
compared whole with ``golden/profile_input_fold.json``.

A failure means a deployment of that preset hashes, drifts or is named differently than
before: a moved convention source, a renamed fold label, a reordered input or changed
preset material. A later change to the golden is a declared delta: the change that makes
it names the delta. Regenerate, in the same change as the code that moved it, then read
the diff::

    PYTHONPATH=src:.:packages/osprey-connectors/tests ./.venv/bin/python \\
        tests/cli/test_profile_input_fold_pin.py
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import osprey
import osprey.cli.build_profile_emit
from osprey.cli.build_profile_document import _read_profile_document
from osprey.cli.build_profile_merge import (
    compute_preset_hash,
    compute_profile_hash,
    profile_inputs,
    resolve_profile_document,
)
from osprey.cli.main import cli

_GOLDEN = Path(__file__).parent / "golden" / "profile_input_fold.json"

PRESETS = ("hello-world", "demo")

#: The directory and project name every materialization uses.
_REPO_NAME = "fold-pin"

#: Presets whose materialized data tree carries ``data/README.md``.
_DATA_README_PRESETS = frozenset({"demo"})

#: What the materialized ``data/README.md`` is overwritten with before the fold is read.
#: The README is template prose that is expected to be rewritten, and the whole data
#: tree folds into one digest, so its bytes would otherwise move the ``data/`` digest
#: and both hashes on every wording change. Fixing them keeps this pin about the fold's
#: structure and the rest of the material; the README is checked for presence instead.
_FIXED_DATA_README = b"# data\n"

#: The OSPREY version every materialization is emitted under. Emitted files carry an
#: ``emitted by OSPREY <version>`` header line, and the tier files fold byte for byte,
#: so the running version (which moves with every commit and with the tree's git state)
#: would otherwise move their digests and the profile hash.
_FIXED_OSPREY_VERSION = "0.0.0+fold-pin"

#: Every place the materialization reads the running version from: the package global,
#: and the emitter's own binding of it, taken at import.
_VERSION_READERS = (osprey, osprey.cli.build_profile_emit)


def _materialize(parent: Path, preset: str) -> Path:
    repo = parent / _REPO_NAME
    result = CliRunner().invoke(
        cli,
        ["init", str(repo), "--preset", preset, "--no-git", "--set", f"project_name={_REPO_NAME}"],
    )
    assert result.exit_code == 0, result.output
    return repo


def _fold(repo: Path, preset: str) -> dict[str, Any]:
    """The fold of *repo*'s profile as the golden records it."""
    profile = repo / "profile.yml"
    document = resolve_profile_document(_read_profile_document(profile), profile, warn=False)
    inputs = []
    for entry in profile_inputs(document):
        digest = None
        if entry.label is not None:
            folded = hashlib.sha256()
            entry.fold(folded)
            digest = folded.hexdigest()
        inputs.append({"key": entry.key, "label": entry.label, "digest": digest})
    return {
        "inputs": inputs,
        "profile_hash": compute_profile_hash(profile),
        "preset_hash": compute_preset_hash(preset),
    }


def _observe(parent: Path, preset: str) -> tuple[dict[str, Any], bool]:
    """Materialize *preset* under *parent*; return its fold and whether the README existed."""
    with pytest.MonkeyPatch.context() as patch:
        for module in _VERSION_READERS:
            patch.setattr(module, "__version__", _FIXED_OSPREY_VERSION)
        repo = _materialize(parent, preset)
    readme = repo / "data" / "README.md"
    existed = readme.is_file()
    if preset in _DATA_README_PRESETS and existed:
        readme.write_bytes(_FIXED_DATA_README)
    return _fold(repo, preset), existed


@pytest.fixture(scope="module")
def observed(tmp_path_factory) -> dict[str, list[tuple[dict[str, Any], bool]]]:
    """Each preset materialized twice, under two different parents."""
    return {
        preset: [
            _observe(tmp_path_factory.mktemp(f"fold-{preset}-{n}"), preset) for n in ("a", "b")
        ]
        for preset in PRESETS
    }


@pytest.mark.parametrize("preset", PRESETS)
def test_fold_does_not_depend_on_the_parent_directory(observed, preset: str) -> None:
    (first, _), (second, _) = observed[preset]
    assert first == second


@pytest.mark.parametrize("preset", PRESETS)
def test_fold_equals_the_golden(observed, preset: str) -> None:
    (fold, _), _ = observed[preset]
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert fold == golden[preset]


@pytest.mark.parametrize("preset", sorted(_DATA_README_PRESETS))
def test_materialization_writes_the_data_readme(observed, preset: str) -> None:
    assert all(existed for _, existed in observed[preset])


def _regenerate() -> None:
    """Overwrite the golden from fresh materializations."""
    golden = {}
    for preset in PRESETS:
        with tempfile.TemporaryDirectory() as parent:
            golden[preset], _ = _observe(Path(parent), preset)
    _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    _GOLDEN.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {_GOLDEN}")


if __name__ == "__main__":
    _regenerate()
