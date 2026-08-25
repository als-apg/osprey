"""Tests for the ontology compiler's deterministic JSON renderer.

The rendered text is committed to the repository, so these tests pin the exact
byte-level decisions: key order, indentation, the trailing newline, and the
``_generated`` header naming only the source file name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from osprey.services.facility_knowledge.ontology_compiler.render import (
    GENERATED_HEADER,
    render_json,
)

SOURCE = Path("x/demo_ontology.yaml")

EXPECTED_HEADER = (
    "Generated from demo_ontology.yaml by `osprey knowledge compile-ontology`. Do not edit."
)


@pytest.fixture
def payload() -> dict[str, object]:
    """A realistic compiled payload, with keys deliberately out of order."""
    return {
        "root": "AcceleratorDevice",
        "family_to_class": {
            "DIPOLE": "Dipole",
            "BPM": "BeamPositionMonitor",
            "CAVITY": "AcceleratingCavity",
        },
        "classes": {
            "Dipole": {"parent": "Magnet", "altLabels": ["bend", "bending magnet"]},
            "AcceleratorDevice": {"parent": None, "altLabels": []},
            "AcceleratingCavity": {"parent": "RFDevice", "altLabels": ["cavity"]},
            "Magnet": {"parent": "AcceleratorDevice", "altLabels": []},
            "RFDevice": {"parent": "AcceleratorDevice", "altLabels": []},
            "BeamPositionMonitor": {"parent": "AcceleratorDevice", "altLabels": ["bpm"]},
        },
    }


def test_render_is_byte_identical_across_calls(payload: dict[str, object]) -> None:
    """Rendering the same payload twice produces the same bytes."""
    first = render_json(payload, SOURCE)
    second = render_json(payload, SOURCE)

    assert first == second


def test_render_is_stable_under_input_key_order(payload: dict[str, object]) -> None:
    """Reordering the input mapping does not change the rendered bytes."""

    def flipped(key: str) -> dict[str, object]:
        block = payload[key]
        assert isinstance(block, dict)
        return dict(reversed(list(block.items())))

    reordered: dict[str, object] = {
        "classes": flipped("classes"),
        "family_to_class": flipped("family_to_class"),
        "root": payload["root"],
    }

    assert render_json(reordered, SOURCE) == render_json(payload, SOURCE)


def test_generated_header_names_only_the_source_file_name(payload: dict[str, object]) -> None:
    """The header carries ``Path.name``, not the directory it lives in."""
    assert GENERATED_HEADER.format(name=SOURCE.name) == EXPECTED_HEADER

    document = json.loads(render_json(payload, SOURCE))

    assert document["_generated"] == EXPECTED_HEADER
    assert "x/" not in document["_generated"]


def test_header_ignores_the_directory_part_of_the_source_path(
    payload: dict[str, object],
) -> None:
    """Two checkouts at different absolute paths render the same header."""
    here = render_json(payload, Path("/one/checkout/demo_ontology.yaml"))
    there = render_json(payload, Path("/another/place/entirely/demo_ontology.yaml"))

    assert here == there


def test_top_level_key_order(payload: dict[str, object]) -> None:
    """Top-level keys appear as ``_generated, classes, family_to_class, root``."""
    keys = re.findall(r'^  "([^"]+)":', render_json(payload, SOURCE), flags=re.MULTILINE)

    assert keys == ["_generated", "classes", "family_to_class", "root"]


def test_class_keys_are_sorted(payload: dict[str, object]) -> None:
    """Nested mappings sort too — ``AcceleratingCavity`` precedes ``AcceleratorDevice``."""
    document = json.loads(render_json(payload, SOURCE))
    class_names = list(document["classes"])

    assert class_names == sorted(class_names)
    assert class_names.index("AcceleratingCavity") < class_names.index("AcceleratorDevice")


def test_output_ends_with_exactly_one_newline(payload: dict[str, object]) -> None:
    """The artifact ends with a single trailing LF, as POSIX text files do."""
    rendered = render_json(payload, SOURCE)

    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")
    assert rendered.rstrip("\n") + "\n" == rendered


def test_indent_is_two_spaces(payload: dict[str, object]) -> None:
    """Nesting level one is indented by two spaces, level two by four."""
    rendered = render_json(payload, SOURCE)

    assert '\n  "root": "AcceleratorDevice"' in rendered
    assert '\n    "Dipole": {' in rendered


def test_round_trips_the_payload_with_the_header_added(payload: dict[str, object]) -> None:
    """``json.loads`` returns the payload verbatim plus ``_generated``."""
    document = json.loads(render_json(payload, SOURCE))

    assert document == {"_generated": EXPECTED_HEADER, **payload}


def test_payload_without_root_renders_without_the_key(payload: dict[str, object]) -> None:
    """``root`` is optional upstream, so its absence must not synthesise a key."""
    del payload["root"]

    rendered = render_json(payload, SOURCE)
    document = json.loads(rendered)

    assert "root" not in document
    assert '"root"' not in rendered
    assert list(document) == ["_generated", "classes", "family_to_class"]


def test_render_does_not_mutate_the_payload(payload: dict[str, object]) -> None:
    """The renderer is pure — the caller's mapping is untouched."""
    before = json.dumps(payload, sort_keys=True)

    render_json(payload, SOURCE)

    assert json.dumps(payload, sort_keys=True) == before
    assert "_generated" not in payload


def test_non_ascii_characters_stay_literal() -> None:
    """``ensure_ascii=False`` keeps a synonym readable instead of escaping it."""
    rendered = render_json(
        {"classes": {"Undulator": {"parent": None, "altLabels": ["ondulateur à aimants"]}}},
        SOURCE,
    )

    assert "ondulateur à aimants" in rendered
    assert "\\u00e0" not in rendered


def test_the_stamped_header_replaces_a_generated_key_in_the_payload() -> None:
    """A rendered artifact always carries the true provenance header.

    Compiled payloads carry only ``root``, ``family_to_class`` and ``classes``,
    so this never fires in practice — the test pins the precedence so a stale
    header can never survive a re-render.
    """
    document = json.loads(render_json({"_generated": "stale", "root": "AcceleratorDevice"}, SOURCE))

    assert document["_generated"] == EXPECTED_HEADER
