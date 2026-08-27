"""The file-backed paradigms' terminology tables are rendered, not written.

Three partials — ``_terminology/{hierarchical,in_context,middle_layer}.md.j2``
— used to tell the channel-finder subagent what a facility's devices are
called, in hard-coded rows: ``DCCT``, ``BCM``, ``HCM``, ``VCM``, ``QF``, ``QD``,
``TC``, ``VGC``, ``CCG``. Four of those tokens existed in no shipped channel
database, which is the whole problem in one sentence: a framework prompt cannot
know a facility's vocabulary, and a wrong device token returns no rows and no
error, so nothing ever reported the drift.

Under the 2026-08-27 ruling those rows have exactly one source — the
deployment's own compiled ontology, named by ``facility.ontology`` and read
into the render context as ``facility_vocabulary``. This file holds the ruling
to the three file-backed paradigms:

* the rows a project renders are the ones its ontology declares, and they are
  derived here the same way the render derives them rather than spelled out;
* a project that declares no ontology gets an honest sentence saying so, and no
  device token at all — never a quiet fallback to the demo machine's words;
* the ``.j2`` sources carry none of the tokens, so the only route into a
  rendered prompt is the render context; and
* a declared ontology that will not load stops the build and names the key,
  because a vocabulary that was promised and then dropped is the one outcome
  worse than having none.

The guards run against the terminology section alone. The rest of
``channel-finder.md.j2`` carries its own paradigm prose, which is a separate
arm of the same epic; slicing keeps this file's failures about this file's
subject.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.cli.templates.claude_code import _facility_vocabulary
from osprey.cli.templates.manager import TemplateManager
from osprey.errors import BuildProfileError

#: The paradigms whose terminology table is a partial in the template tree. The
#: ``graph`` paradigm derives its vocabulary from the seeded store instead, and
#: is guarded by ``tests/cli/test_channel_finder_graph_tools.py``.
FILE_BACKED_MODES = ("hierarchical", "in_context", "middle_layer")

_TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/claude_code/claude/agents/_terminology"
)

#: Every device token the three partials used to spell for themselves. None may
#: appear as literal template text, and none may reach a render that declares no
#: ontology.
FORBIDDEN_TOKENS = ("DCCT", "BCM", "BPM", "HCM", "VCM", "QF", "QD", "TC", "VGC", "CCG")

#: The sentence a render with no declared ontology must carry instead of rows.
#: Pinned past the key name, because the half that matters is the promise about
#: what the reader is looking at: routing guidance, and no device vocabulary.
NO_ONTOLOGY_LINE = (
    "No facility ontology is declared (`facility.ontology`), so the table below\n"
    "carries only navigation guidance and no device vocabulary"
)

#: The key the preset declares, as it is written in the rendered ``config.yml``.
DECLARED_KEY_LINE = "  ontology: data/facility_ontology.json"

_SECTION_HEADING = "## Channel Database Terminology"


def _terminology_section(project_dir: Path) -> str:
    """The rendered terminology section, without the rest of the agent file."""
    text = (project_dir / ".claude" / "agents" / "channel-finder.md").read_text(encoding="utf-8")
    start = text.find(_SECTION_HEADING)
    assert start != -1, "the rendered channel finder has no terminology section"
    rest = text[start + len(_SECTION_HEADING) :]
    end = re.search(r"(?m)^## ", rest)
    assert end, "the terminology section is no longer followed by a heading; re-derive the slice"
    return _SECTION_HEADING + rest[: end.start()]


def _forbidden_hits(text: str) -> list[str]:
    """Which of the retired device tokens appear in *text*, word-bounded.

    Word boundaries keep ``TC`` from matching inside ``MATCH`` and ``QF`` from
    matching inside a longer family token, so a hit is a real device token
    rather than a substring of ordinary prose.
    """
    return [token for token in FORBIDDEN_TOKENS if re.search(rf"\b{token}\b", text)]


def _project(tmp_path: Path, name: str, mode: str) -> tuple[TemplateManager, Path]:
    """A control-assistant project in *mode*, rendered the way the CLI renders one."""
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": mode, "deploy_services": True},
    )
    return manager, project_dir


def _rewrite_config(project_dir: Path, replacement: str) -> None:
    """Replace the preset's ``facility.ontology`` line, asserting it was there."""
    config = project_dir / "config.yml"
    text = config.read_text(encoding="utf-8")
    assert DECLARED_KEY_LINE in text, (
        "the control_assistant preset no longer declares facility.ontology; "
        "this file's without-a-key cases have nothing to remove"
    )
    config.write_text(text.replace(DECLARED_KEY_LINE, replacement), encoding="utf-8")


def _declared_vocabulary(project_dir: Path) -> list[dict]:
    """The rows the render itself would derive from this project's ontology."""
    rows = _facility_vocabulary(
        {"facility": {"ontology": "data/facility_ontology.json"}}, project_dir
    )
    assert rows, "the preset's ontology declares no vocabulary at all"
    return rows


# ---------------------------------------------------------------------------
# With an ontology: the rows are the ontology's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_every_declared_synonym_and_family_reaches_the_table(tmp_path, mode):
    """The table's content is the deployment's ontology, class by class.

    Derived rather than spelled: the expectation is read from the same loader
    the render reads, so an ontology that gains a class gains a row here too
    and this test cannot drift from the table it guards.
    """
    _manager, project_dir = _project(tmp_path, f"cf-vocab-{mode}", mode)
    section = _terminology_section(project_dir)

    for row in _declared_vocabulary(project_dir):
        # in_context has no token to search for when a class maps to no family,
        # so those classes are deliberately left out of that paradigm's table.
        if mode == "in_context" and not row["families"]:
            continue
        for synonym in row["synonyms"]:
            assert f'"{synonym}"' in section, (
                f"{mode}: synonym {synonym!r} of class {row['class_name']} is missing"
            )
        for family in row["families"]:
            assert f"`{family}`" in section, (
                f"{mode}: family token {family} of class {row['class_name']} is missing"
            )


@pytest.mark.parametrize("mode", ("hierarchical", "middle_layer"))
def test_a_class_with_no_family_is_marked_non_navigable(tmp_path, mode):
    """An umbrella class is not styled as somewhere the agent can navigate to.

    Five classes in the demo ontology carry synonyms but no family token
    (Corrector, Instrumentation, Magnet, RadioFrequency, Vacuum). Rendering
    them into the navigation column as if they were a device or family name
    invites a lookup the database cannot answer, so the row says what they are
    instead. in_context is excluded: it has no token to search for, so those
    classes get no row at all there.
    """
    _manager, project_dir = _project(tmp_path, f"cf-umbrella-{mode}", mode)
    section = _terminology_section(project_dir)

    umbrellas = [row for row in _declared_vocabulary(project_dir) if not row["families"]]
    assert umbrellas, "the preset's ontology no longer has a class without a family"
    for row in umbrellas:
        assert f"Umbrella class {row['class_name']}" in section
        assert "narrow to a specific" in section
    assert f"Family: class {umbrellas[0]['class_name']}" not in section
    assert f"Device: class {umbrellas[0]['class_name']}" not in section


@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_the_table_names_the_key_it_was_rendered_from(tmp_path, mode):
    """An operator reading the prompt is told where the vocabulary came from."""
    _manager, project_dir = _project(tmp_path, f"cf-provenance-{mode}", mode)
    section = _terminology_section(project_dir)

    assert "`facility.ontology`" in section
    assert NO_ONTOLOGY_LINE not in section


@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_the_paradigm_routing_rows_survive(tmp_path, mode):
    """Deleting the device rows did not take the paradigm's own guidance with it."""
    _manager, project_dir = _project(tmp_path, f"cf-routing-{mode}", mode)
    section = _terminology_section(project_dir)

    assert '| "readback" / "monitor" |' in section
    assert '| "setpoint" / "control" |' in section


# ---------------------------------------------------------------------------
# Without an ontology: an honest sentence, and no borrowed vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_no_declared_ontology_renders_the_honest_line(tmp_path, mode):
    """With the key gone the table says so, and names no device at all."""
    manager, project_dir = _project(tmp_path, f"cf-silent-{mode}", mode)
    _rewrite_config(project_dir, "  # ontology: data/facility_ontology.json")
    manager.regenerate_claude_code(project_dir)

    section = _terminology_section(project_dir)
    assert NO_ONTOLOGY_LINE in section
    assert "`facility.ontology`" in section, "the honest line names the key to set"
    assert _forbidden_hits(section) == [], (
        f"{mode}: a render with no declared ontology still names device tokens — "
        "the demo machine's vocabulary has leaked back into the prompt"
    )
    # The paradigm's own routing guidance is not vocabulary, and stays.
    assert '| "readback" / "monitor" |' in section


# ---------------------------------------------------------------------------
# The sources, and the failure mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_the_partial_source_spells_no_device_token(mode):
    """Tokens may only arrive through the render context, never as template text.

    This is the ruling's enforcement point for the file-backed paradigms: as
    long as the sources are clean, every device word in a rendered prompt is one
    the deployment's own ontology put there.
    """
    source = (_TEMPLATE_ROOT / f"{mode}.md.j2").read_text(encoding="utf-8")
    assert _forbidden_hits(source) == [], (
        f"_terminology/{mode}.md.j2 spells a device token itself. The vocabulary "
        "has one source — facility.ontology, through `facility_vocabulary`."
    )


@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_a_declared_ontology_that_is_not_there_stops_the_build(tmp_path, mode):
    """A broken path is a named error, never a silent skip or a demo fallback."""
    manager, project_dir = _project(tmp_path, f"cf-broken-{mode}", mode)
    _rewrite_config(project_dir, "  ontology: data/no_such_ontology.json")

    with pytest.raises(BuildProfileError) as caught:
        manager.regenerate_claude_code(project_dir)

    message = str(caught.value)
    assert "facility.ontology" in message, "the error must name the key to fix"
    assert "no_such_ontology.json" in message, "the error must name the path it tried"
    # The population this stops hardest is a build profile with its own `data:`
    # tree: the key is rendered from the app template, so "remove it" is not
    # actionable unless the message names the overlay that can.
    assert "`config:`" in message and "bare `facility.ontology:`" in message, (
        "the refusal must name the profile overlay that removes a template-rendered key"
    )


def test_a_scalar_facility_block_is_not_a_traceback(tmp_path):
    """A malformed ``facility:`` falls through to "no ontology", not AttributeError.

    ``facility: "Example Research Facility"`` is a plausible slip, because a
    top-level ``facility_name`` fallback exists and invites the conflation. The
    block goes through the same ``as_dict`` guard every other facility reader in
    the tree uses, so this reader — the one that turns a bad block into a build
    stop — cannot be the one that raises an unhandled type error.
    """
    assert _facility_vocabulary({"facility": "Example Research Facility"}, tmp_path) is None
    assert _facility_vocabulary({"facility": ["not", "a", "mapping"]}, tmp_path) is None


def test_a_home_relative_ontology_path_is_expanded(tmp_path, monkeypatch):
    """``facility.ontology: ~/x.json`` reads from the home directory, like every other path key.

    Every other path in ``config.yml`` goes through ``expanduser()`` before it
    is resolved; a reader that skipped that step would join ``~`` onto the
    project root and stop the build over a file that is right there.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _, project_dir = _project(tmp_path, "cf-home", FILE_BACKED_MODES[0])
    (home / "x.json").write_bytes((project_dir / "data" / "facility_ontology.json").read_bytes())

    rows = _facility_vocabulary({"facility": {"ontology": "~/x.json"}}, tmp_path / "elsewhere")

    assert rows == _declared_vocabulary(project_dir)
