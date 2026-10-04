"""An app template's ``shared_data.yml``: files its data/ tree takes from another one.

The point of the declaration is that the source holds one copy of each shared
file, so the reader has to refuse the two ways that could be undone: a template
shipping its own copy of a file it also declares, and a declaration naming
something that is not there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.scaffold_pull import list_pullable_paths, plan_pull
from osprey.cli.templates.shared_data import SHARED_DATA_FILENAME, shared_data_files


def _apps(tmp_path: Path) -> Path:
    apps = tmp_path / "apps"
    donor = apps / "donor" / "data"
    (donor / "scenarios" / "a" / "plots").mkdir(parents=True)
    (donor / "scenarios" / "a" / "logbook.json").write_text("[]")
    (donor / "scenarios" / "a" / "scenario.json").write_text("{}")
    (donor / "scenarios" / "a" / "plots" / "p.png").write_bytes(b"png")
    (donor / "corpus.ttl").write_text("ttl")
    (apps / "taker").mkdir()
    return apps


def _declare(apps: Path, text: str) -> Path:
    taker = apps / "taker"
    (taker / SHARED_DATA_FILENAME).write_text(text)
    return taker


_DECLARATION = """
- from: donor
  source: corpus.ttl
  target: corpus.ttl
- from: donor
  source: scenarios
  target: seed
  include: ["*/logbook.json", "*/plots/*"]
"""


def test_no_declaration_shares_nothing(tmp_path):
    assert shared_data_files(_apps(tmp_path) / "taker") == {}


def test_files_and_filtered_directories_land_where_declared(tmp_path):
    apps = _apps(tmp_path)
    taker = _declare(apps, _DECLARATION)

    files = shared_data_files(taker)

    donor = apps / "donor" / "data"
    assert files == {
        "corpus.ttl": donor / "corpus.ttl",
        "seed/a/logbook.json": donor / "scenarios" / "a" / "logbook.json",
        "seed/a/plots/p.png": donor / "scenarios" / "a" / "plots" / "p.png",
    }


def test_a_file_the_template_also_ships_is_refused(tmp_path):
    apps = _apps(tmp_path)
    taker = _declare(apps, _DECLARATION)
    (taker / "data").mkdir()
    (taker / "data" / "corpus.ttl").write_text("a second copy")

    with pytest.raises(ValueError, match=r"\['corpus.ttl'\] are shipped by this template too"):
        shared_data_files(taker)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- from: donor\n  source: missing.ttl\n  target: x", "does not exist"),
        ("- from: donor\n  source: corpus.ttl", "needs 'from', 'source' and 'target'"),
        ("- from: donor\n  source: corpus.ttl\n  target: x\n  rename: y", "unknown keys"),
        ("from: donor", "must be a list"),
    ],
)
def test_a_malformed_declaration_is_refused(tmp_path, text, message):
    taker = _declare(_apps(tmp_path), text)
    with pytest.raises(ValueError, match=message):
        shared_data_files(taker)


def test_scaffold_pull_lists_and_copies_shared_files_like_its_own(tmp_path):
    taker = _declare(_apps(tmp_path), _DECLARATION)
    repo = tmp_path / "repo"
    repo.mkdir()

    listing = list_pullable_paths(taker)
    actions = plan_pull(taker, repo, "data/seed", force=False, with_content=False)

    assert SHARED_DATA_FILENAME not in listing
    assert {"data/", "data/seed/", "data/seed/a/", "data/seed/a/plots/"} <= set(listing)
    assert "data/corpus.ttl" in listing
    assert {a.target.relative_to(repo).as_posix(): a.action for a in actions} == {
        "data/seed/a/logbook.json": "written",
        "data/seed/a/plots/p.png": "written",
    }
    assert all(a.source.is_file() for a in actions)


def test_the_standalone_template_offers_the_control_assistant_narrative_and_corpus():
    import osprey

    apps = Path(osprey.__file__).parent / "templates" / "apps"
    listing = list_pullable_paths(apps / "ariel_standalone")

    assert "data/demo_machine.ttl" in listing
    narratives = sorted(entry for entry in listing if entry.endswith("logbook.json"))
    scenarios = apps / "control_assistant" / "data" / "simulation" / "scenarios"
    expected = sorted(
        f"data/logbook_seed/{path.parent.name}/logbook.json"
        for path in scenarios.glob("*/logbook.json")
    )
    assert narratives == expected
    assert not any(entry.endswith("scenario.json") for entry in listing), "narrative only"
