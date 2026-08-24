"""Coverage for the two source-only readers of a plan's ``PLAN_METADATA`` —
read off disk with `ast`, never by importing the module.

`extract_plan_metadata_from_source` reads and validates the whole block, for
build-time tooling that must refuse anything it cannot read. Two properties are
proven for it here: every shipped plan is readable this way, and every failure
mode surfaces as one `PlanMetadataError`.

`extract_plan_name_from_source` reads only the ``name`` key, for the
source-lookup scan, whose question is which file declares a name rather than
whether that file's whole block is valid. It answers as permissively as an
import does and never raises. The divergence between the two is deliberate and
pinned below.

Pure stdlib + pydantic — no bluesky import, so this runs in the fast
import-clean lane alongside the rest of `plan_metadata.py`'s tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.services.bluesky_bridge.plan_metadata import (
    PlanMetadata,
    PlanMetadataError,
    extract_plan_metadata_from_source,
    extract_plan_name_from_source,
)

_PLANS_CORE = Path(__file__).resolve().parents[3] / "src/osprey/services/bluesky_bridge/plans_core"

_SHIPPED_PLAN_FILES = sorted(p for p in _PLANS_CORE.glob("*.py") if p.name != "__init__.py")


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "a_plan.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_shipped_plan_files_are_discoverable() -> None:
    """Guard the parametrization below against silently matching nothing."""
    assert len(_SHIPPED_PLAN_FILES) >= 3


@pytest.mark.parametrize("path", _SHIPPED_PLAN_FILES, ids=lambda p: p.stem)
def test_every_shipped_plan_yields_valid_metadata(path: Path) -> None:
    metadata = extract_plan_metadata_from_source(path)

    assert isinstance(metadata, PlanMetadata)
    assert metadata.name == path.stem
    assert metadata.description.strip()


@pytest.mark.parametrize("path", _SHIPPED_PLAN_FILES, ids=lambda p: p.stem)
def test_parenthesized_implicit_concatenation_is_read_whole(path: Path) -> None:
    """Every shipped plan writes its ``description`` as adjacent string
    literals inside parentheses; the extractor must return the joined text,
    not just the first fragment."""
    source = path.read_text(encoding="utf-8")
    metadata = extract_plan_metadata_from_source(path)

    assert '"\n' in source.split("PLAN_METADATA", 1)[1].split("}", 1)[0], (
        "test premise: this plan no longer uses multi-line implicit concatenation"
    )
    assert len(metadata.description) > 60


def test_module_is_not_executed(tmp_path: Path) -> None:
    """A plan file whose import-time code would blow up (or write a file) is
    still read cleanly — nothing in it runs."""
    canary = tmp_path / "canary.txt"
    path = _write(
        tmp_path,
        f"import pathlib\n"
        f"pathlib.Path({str(canary)!r}).write_text('executed')\n"
        f"raise SystemExit('this module must never run')\n"
        f'PLAN_METADATA = {{"name": "quiet", "description": "Reads only.", "writes": False}}\n',
    )

    metadata = extract_plan_metadata_from_source(path)

    assert metadata.name == "quiet"
    assert not canary.exists()


def test_missing_plan_metadata_raises_typed_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "def a_plan(devices, params):\n    yield from ()\n")

    with pytest.raises(PlanMetadataError, match="no module-level literal PLAN_METADATA"):
        extract_plan_metadata_from_source(path)


def test_missing_required_key_raises_typed_error_naming_the_key(tmp_path: Path) -> None:
    path = _write(tmp_path, 'PLAN_METADATA = {"name": "half", "description": "No writes key."}\n')

    with pytest.raises(PlanMetadataError, match="writes") as excinfo:
        extract_plan_metadata_from_source(path)

    assert "unknown key(s)" not in str(excinfo.value)


def test_unknown_key_raises_typed_error_naming_the_key(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "PLAN_METADATA = {\n"
        '    "name": "extra",\n'
        '    "description": "Declares a key outside the contract.",\n'
        '    "writes": False,\n'
        '    "category": "accelerator",\n'
        "}\n",
    )

    with pytest.raises(PlanMetadataError, match="unknown key\\(s\\): category"):
        extract_plan_metadata_from_source(path)


def test_wrong_typed_field_raises_typed_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'PLAN_METADATA = {"name": "bad", "description": "Bad.", "writes": "not-a-bool"}\n',
    )

    with pytest.raises(PlanMetadataError, match="writes"):
        extract_plan_metadata_from_source(path)


def test_non_dict_plan_metadata_raises_typed_error(tmp_path: Path) -> None:
    path = _write(tmp_path, 'PLAN_METADATA = ["not", "a", "dict"]\n')

    with pytest.raises(PlanMetadataError, match="no module-level literal PLAN_METADATA"):
        extract_plan_metadata_from_source(path)


def test_computed_plan_metadata_is_rejected_rather_than_run(tmp_path: Path) -> None:
    """A dict whose values are computed at import time cannot be read without
    running the module, so it is refused — not partially guessed at."""
    path = _write(
        tmp_path,
        "import os\n"
        'PLAN_METADATA = {"name": os.environ["PLAN_NAME"], "description": "d", "writes": False}\n',
    )

    with pytest.raises(PlanMetadataError, match="not a literal dict"):
        extract_plan_metadata_from_source(path)


def test_plan_metadata_nested_in_a_function_is_ignored(tmp_path: Path) -> None:
    """Only a module-level binding is part of the plan contract."""
    path = _write(
        tmp_path,
        "def build():\n"
        '    PLAN_METADATA = {"name": "nested", "description": "d", "writes": False}\n'
        "    return PLAN_METADATA\n",
    )

    with pytest.raises(PlanMetadataError, match="no module-level literal PLAN_METADATA"):
        extract_plan_metadata_from_source(path)


def test_last_module_level_assignment_wins(tmp_path: Path) -> None:
    """The node read must match what the module would actually bind."""
    path = _write(
        tmp_path,
        'PLAN_METADATA = {"name": "first", "description": "Superseded.", "writes": False}\n'
        'PLAN_METADATA = {"name": "second", "description": "The live one.", "writes": True}\n',
    )

    assert extract_plan_metadata_from_source(path).name == "second"


def test_unparseable_file_raises_typed_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "def broken(:\n")

    with pytest.raises(PlanMetadataError, match="could not read plan source"):
        extract_plan_metadata_from_source(path)


def test_missing_file_raises_typed_error(tmp_path: Path) -> None:
    with pytest.raises(PlanMetadataError, match="could not read plan source"):
        extract_plan_metadata_from_source(tmp_path / "nope.py")


def test_error_messages_name_the_offending_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "x = 1\n")

    with pytest.raises(PlanMetadataError, match=str(path)):
        extract_plan_metadata_from_source(path)


def test_unhashable_key_raises_the_typed_error_not_a_type_error(tmp_path: Path) -> None:
    """`ast.literal_eval` raises `TypeError`, not `ValueError`, on an unhashable
    key — so it needs its own clause to keep the one-error-type contract.

    Callers catch `PlanMetadataError` and nothing else; a `TypeError` escaping
    here would abort whatever scan is walking plan files rather than skipping
    the one bad file.
    """
    path = _write(tmp_path, "PLAN_METADATA = {[1]: 2}\n")

    with pytest.raises(PlanMetadataError, match="not a literal dict"):
        extract_plan_metadata_from_source(path)


# ---------------------------------------------------------------------------
# `extract_plan_name_from_source` — the identity half. Same file, same reader
# for the `PLAN_METADATA` node, but only the `name` key is read and nothing is
# raised: it answers "which file declares this name?" for callers that must be
# as permissive as an import.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _SHIPPED_PLAN_FILES, ids=lambda p: p.stem)
def test_every_shipped_plan_yields_its_name(path: Path) -> None:
    assert extract_plan_name_from_source(path) == extract_plan_metadata_from_source(path).name


def test_a_non_literal_sibling_value_does_not_hide_the_name(tmp_path: Path) -> None:
    """The regression this function exists for.

    A plan whose ``description`` is a module-level constant imports fine and
    registers, so it is a launchable plan whose source someone can ask for —
    but the whole-block reader cannot evaluate it. The name-only reader must
    still identify it, exactly as an import would.
    """
    path = _write(
        tmp_path,
        '_DESC = "Reads an orbit."\n'
        'PLAN_METADATA = {"name": "facility_plan", "description": _DESC, "writes": False}\n',
    )

    assert extract_plan_name_from_source(path) == "facility_plan"
    # ...and the strict reader still refuses it, unchanged. The two readers
    # answer different questions; only the identity one was ever meant to be
    # this permissive.
    with pytest.raises(PlanMetadataError, match="not a literal dict"):
        extract_plan_metadata_from_source(path)


def test_a_contract_failing_block_still_yields_its_name(tmp_path: Path) -> None:
    """Missing and surplus keys are the strict reader's business, not this one's."""
    path = _write(
        tmp_path,
        'PLAN_METADATA = {"name": "facility_plan", "provenance": "facility"}\n',
    )

    assert extract_plan_name_from_source(path) == "facility_plan"


def test_the_last_module_level_binding_names_the_plan(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'PLAN_METADATA = {"name": "first", "description": "Superseded.", "writes": False}\n'
        'PLAN_METADATA = {"name": "second", "description": "The live one.", "writes": True}\n',
    )

    assert extract_plan_name_from_source(path) == "second"


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("no_metadata", "x = 1\n"),
        ("unparseable", "def broken(:\n"),
        ("computed_dict", "PLAN_METADATA = dict(name='computed')\n"),
        ("non_literal_name", "_NAME = 'n'\nPLAN_METADATA = {'name': _NAME}\n"),
        ("non_string_name", "PLAN_METADATA = {'name': 7}\n"),
        ("no_name_key", "PLAN_METADATA = {'description': 'd', 'writes': False}\n"),
        ("annotated_assignment", "PLAN_METADATA: dict = {'name': 'annotated'}\n"),
        ("unhashable_key", "PLAN_METADATA = {[1]: 2}\n"),
        ("nested_in_a_function", "def f():\n    PLAN_METADATA = {'name': 'nested'}\n"),
        ("spread_only", "PLAN_METADATA = {**_OTHER}\n"),
        ("spread_after_name", "PLAN_METADATA = {'name': 'static', **_EXTRA}\n"),
        ("duplicate_name_last_unreadable", "PLAN_METADATA = {'name': 'a', 'name': _X}\n"),
        ("non_utf8_bytes_name", "PLAN_METADATA = {'name': b'\\xff'}\n"),
    ],
)
def test_an_unidentifiable_file_returns_none(label: str, body: str, tmp_path: Path) -> None:
    """No exception, ever: an unidentifiable file is a skip for every caller.

    ``annotated_assignment`` is a standing limit shared with the strict reader
    — such a plan still imports and registers, and only its source lookup
    misses — and is left as it is rather than widened here.

    ``spread_after_name`` and ``duplicate_name_last_unreadable`` are the cases
    where a name IS written literally but is not the name an import binds: the
    spread (or the later duplicate key) can replace it with anything. Reporting
    the superseded literal would serve a file under a name it does not have and
    404 the plan it really is, so both are unknowable rather than answered.
    ``non_utf8_bytes_name`` is `None` because pydantic refuses to decode it, so
    no such plan ever registers either.
    """
    assert extract_plan_name_from_source(_write(tmp_path, body)) is None, label


def test_a_spread_before_the_name_key_does_not_hide_it(tmp_path: Path) -> None:
    """A ``**spread`` only makes what comes BEFORE it unknowable.

    A literal ``name`` after the spread is the last word in the dict display,
    so it is what an import binds and what this reader reports.
    """
    path = _write(tmp_path, "PLAN_METADATA = {**_EXTRA, 'name': 'real_name'}\n")

    assert extract_plan_name_from_source(path) == "real_name"


def test_a_bytes_name_reads_as_the_name_an_import_binds(tmp_path: Path) -> None:
    """`PlanMetadata.name` is a plain ``str`` field, and pydantic's default lax
    mode fills it from ``bytes`` by decoding — so a file declaring
    ``b"byte_name"`` really does register as ``byte_name``.

    Nobody writes a bytes plan name, but the two readers disagreeing about one
    is the 404-at-the-approval-gate failure this reader exists to prevent, with
    the permissive reader ending up the stricter of the two. Both answer the
    same here.
    """
    path = _write(
        tmp_path,
        'PLAN_METADATA = {"name": b"byte_name", "description": "d", "writes": False}\n',
    )

    assert extract_plan_name_from_source(path) == "byte_name"
    assert extract_plan_metadata_from_source(path).name == "byte_name"


def test_a_missing_file_returns_none(tmp_path: Path) -> None:
    assert extract_plan_name_from_source(tmp_path / "nope.py") is None


def test_the_name_reader_does_not_execute_the_module(tmp_path: Path) -> None:
    """Same guarantee as the strict reader: the file is parsed, never run."""
    canary = tmp_path / "canary.txt"
    path = _write(
        tmp_path,
        "from pathlib import Path\n"
        f"Path({str(canary)!r}).write_text('executed')\n"
        'PLAN_METADATA = {"name": "canary_plan", "description": "d", "writes": False}\n',
    )

    assert extract_plan_name_from_source(path) == "canary_plan"
    assert not canary.exists()
