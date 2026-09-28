"""The facility schema modules and the NARAD seeds they start from.

The seeds are vendored byte for byte, so their digests are pinned here and in
the seeds README; a copy that drifted from the recorded upstream commit fails
before anything is built on it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS_DIR = REPO_ROOT / "scripts" / "facility_schema" / "seeds"

#: The upstream commit the seeds were copied from.
SEED_COMMIT = "02f64d5"

#: sha256 of each vendored seed at :data:`SEED_COMMIT`.
SEED_DIGESTS = {
    "canonical_ingest.yaml": "2efa863505615c555449d575343534463eda7ed23fe296760eeeed78776e9a91",
    "facility_bindings.yaml": "252208465d48ef582b031fc61be2371754eab6dec10582c933e42b54e53c6849",
    "concept_vocabulary.yaml": "48b1dc6cf7b96721706e9b0bdfc286a6224e59f6fa123d2275ffda86dee14b64",
    "shared_semantics.yaml": "f1836e33402f1ffb63d7518ef966374beb2581780cfbf7f3a3b57710717f6629",
}


@pytest.mark.parametrize("seed", sorted(SEED_DIGESTS))
def test_vendored_seed_matches_its_pinned_digest(seed: str) -> None:
    digest = hashlib.sha256((SEEDS_DIR / seed).read_bytes()).hexdigest()
    assert digest == SEED_DIGESTS[seed]


def test_seeds_dir_holds_exactly_the_four_seeds_and_the_readme() -> None:
    names = sorted(path.name for path in SEEDS_DIR.iterdir())
    assert names == sorted([*SEED_DIGESTS, "README.md"])


def test_readme_records_the_commit_and_every_digest() -> None:
    readme = (SEEDS_DIR / "README.md").read_text(encoding="utf-8")
    assert f"commit `{SEED_COMMIT}`" in readme
    for seed, digest in SEED_DIGESTS.items():
        assert f"| `{seed}` | `{digest}` |" in readme
