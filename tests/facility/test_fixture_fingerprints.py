"""Each MML fixture tree's build against its frozen fingerprint.

A tree is imported into a clean directory, the limits records its build stops
on are widened to the edge the stop names, and the facility file is built in
memory. The fingerprint of that build is one digest per record — class, device,
channel, group, model, wiring record, limits record — and one per imported
deck, keyed by the record's id. A digest covers the record as the facility file
states it, without its provenance, with every float held to twelve significant
digits.

``tests/facility/golden/fingerprint_<tree>.json`` holds the frozen fingerprint.
A change to the importer, the decks or the seeded files that moves a digest
rewrites the golden with the command its ``_reproduce`` line names.

The synthetic tree has no fingerprint: its build stops by design.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("at")

from osprey.facility.build import build_facility
from tests.facility.test_cf_view_parity import GOLDEN_DIR
from tests.facility.test_mml_layer_seed_once import (
    BUILT,
    TREES,
    WIDENED,
    _import,
    _widen,
)

REPRODUCE = "uv run python -m tests.facility.test_fixture_fingerprints --write"
DIGEST_OF = (
    "each record as compact sorted-key UTF-8 JSON, provenance dropped, "
    "floats at 12 significant digits; first 16 hex digits of the sha256"
)

#: The record kinds of the facility file a fingerprint holds, with the key
#: each record is identified by.
RECORD_KINDS: dict[str, str] = {
    "classes": "class",
    "devices": "id",
    "channels": "id",
    "groups": "id",
}


def _golden_path(tree: str) -> Path:
    return GOLDEN_DIR / f"fingerprint_{tree}.json"


def _canonical(value: Any) -> Any:
    """``value`` without provenance and with every float at twelve digits."""
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items() if key != "provenance"}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.12g}")
    return value


def _digest(record: Any) -> str:
    compact = json.dumps(
        _canonical(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()[:16]


def _digests(records: list[dict[str, Any]], key: str) -> dict[str, str]:
    digests = {str(record[key]): _digest(record) for record in records}
    assert len(digests) == len(records), f"records share a {key}"
    return dict(sorted(digests.items()))


def fingerprint(document: dict[str, Any], facility: Path) -> dict[str, Any]:
    """The fingerprint of a built facility file and the decks it names.

    Args:
        document: The facility file ``build_facility`` returned.
        facility: The ``data/facility`` directory it was built from.

    Returns:
        The identity, the record counts, the channel roles and one digest per
        record of every kind.
    """
    records = {kind: _digests(document[kind], key) for kind, key in RECORD_KINDS.items()}
    models = [{k: v for k, v in model.items() if k != "wiring"} for model in document["models"]]
    records["models"] = _digests(models, "name")
    records["wiring"] = _digests(
        [record for model in document["models"] for record in model.get("wiring", [])], "id"
    )
    records["limits"] = _digests(document["limits"]["records"], "address")
    records["decks"] = {
        model["name"]: _digest(json.loads((facility / model["deck"]).read_text(encoding="utf-8")))
        for model in sorted(document["models"], key=lambda model: model["name"])
        if "deck" in model
    }
    roles = Counter(channel["role"] for channel in document["channels"])
    return {
        "identity": _canonical(document["identity"]),
        "counts": {kind: len(digests) for kind, digests in records.items()},
        "roles": dict(sorted(roles.items())),
        "records": records,
    }


def build_fingerprint(tree: str, root: Path) -> dict[str, Any]:
    """Import ``tree`` under ``root``, widen its named bands, build and fingerprint it."""
    facility = _import(root, tree)
    _widen(facility, WIDENED[tree])
    document = build_facility(facility, project_name="demo")
    return fingerprint(document, facility)


def _sources(tree: str) -> list[str]:
    fixture = f"tests/fixtures/mml/{tree}"
    return [f"{fixture}/imported/mml/mapping.yaml"] + [
        f"{fixture}/{stem}.ao.json" for stem in TREES[tree]
    ]


def _golden(tree: str, built: dict[str, Any]) -> dict[str, Any]:
    return {
        "_reproduce": REPRODUCE,
        "_sources": _sources(tree),
        "_digest_of": DIGEST_OF,
        **built,
    }


def _moved(frozen: dict[str, str], built: dict[str, str]) -> list[str]:
    """One line per id the build adds, drops or states differently."""
    lines = [f"+ {key}" for key in built.keys() - frozen.keys()]
    lines += [f"- {key}" for key in frozen.keys() - built.keys()]
    lines += [f"~ {key}" for key in built.keys() & frozen.keys() if built[key] != frozen[key]]
    return sorted(lines, key=lambda line: line[2:])


@pytest.fixture(scope="module", params=BUILT)
def tree(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    return name


@pytest.fixture(scope="module")
def built(tree: str, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return build_fingerprint(tree, tmp_path_factory.mktemp(tree))


@pytest.fixture(scope="module")
def frozen(tree: str) -> dict[str, Any]:
    golden: dict[str, Any] = json.loads(_golden_path(tree).read_text(encoding="utf-8"))
    return golden


def test_every_built_tree_and_no_other_has_a_fingerprint() -> None:
    names = sorted(path.name for path in GOLDEN_DIR.glob("fingerprint_*.json"))
    assert names == sorted(_golden_path(tree).name for tree in BUILT)
    assert "synthetic" in TREES and "synthetic" not in BUILT


def test_the_golden_names_its_sources(tree: str, frozen: dict[str, Any]) -> None:
    assert frozen["_reproduce"] == REPRODUCE
    assert frozen["_sources"] == _sources(tree)
    assert frozen["_digest_of"] == DIGEST_OF


def test_the_golden_counts_its_own_records(frozen: dict[str, Any]) -> None:
    records = frozen["records"]
    assert frozen["counts"] == {kind: len(digests) for kind, digests in records.items()}
    assert all(list(digests) == sorted(digests) for digests in records.values())
    assert sum(frozen["roles"].values()) == frozen["counts"]["channels"]
    assert set(records["decks"]) <= set(records["models"])
    assert records["channels"] and records["wiring"] and records["limits"] and records["decks"]


def test_the_build_holds_every_frozen_record_unchanged(
    tree: str, built: dict[str, Any], frozen: dict[str, Any]
) -> None:
    assert list(built["records"]) == list(frozen["records"])
    moved = {
        kind: lines
        for kind in built["records"]
        if (lines := _moved(frozen["records"][kind], built["records"][kind]))
    }
    assert not moved, (
        f"the {tree} build moved off {_golden_path(tree).name} "
        f"(+ added, - dropped, ~ changed); re-baseline with `{REPRODUCE}`: {moved}"
    )


def test_the_build_holds_the_frozen_identity_and_counts(
    built: dict[str, Any], frozen: dict[str, Any]
) -> None:
    assert built["identity"] == frozen["identity"]
    assert built["counts"] == frozen["counts"]
    assert built["roles"] == frozen["roles"]


def _write() -> None:
    for name in BUILT:
        with tempfile.TemporaryDirectory() as root:
            golden = _golden(name, build_fingerprint(name, Path(root)))
        path = _golden_path(name)
        path.write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit(f"usage: {REPRODUCE}")
    _write()
