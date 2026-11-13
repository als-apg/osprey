"""The OKF bundle page's examples, and the directory trees five pages draw, are what the code reads and writes.

Each block is found by page, language and a distinctive substring (never by
line number), then handed to the code that owns it:

* ``how-to/facility-knowledge/okf-bundle.rst``: the concept document's
  frontmatter passes ``OKFDocument.validate("authoring")``; the bundle link
  resolves to a concept of the example bundle, and the ``facility:`` links
  parse to the targets they name and resolve in the example facility; the
  ``regenerate_indexes`` snippet runs and leaves a bundle ``check_indexes``
  passes; the "conventional starting shape" tree names the index file the
  regenerator writes and the example bundle's directories.
* ``how-to/build-profiles.rst``: every path of the ``build/`` tree is on disk
  after a real ``osprey build`` of the demo preset, and every
  facility output that build's main render writes is on the tree.
* ``how-to/deploy-project/compose-templates.rst``: the ``services/`` tree names
  the compose templates the build copies into ``build/services/``.
* ``how-to/describe-your-facility.rst``: the ``data/facility/`` tree is the one
  ``osprey init --preset demo`` writes, in the names the facility
  build reads.
* ``how-to/deploy-project/project-image.rst`` and
  ``reference/configuration/profile.rst``: the ``my-facility/`` trees, built on
  disk, are planned by the convention copier as the pages say (the mirror's
  ``Dockerfile`` onto the render root, the server package to
  ``build/_mcp_servers/``), and the server layout is the one the hello-world
  scaffold ships.

A page text that differs from the code is a finding, recorded in
:data:`_STALE` with the owning code and the fix. An entry no open PR fixes
must still describe its page, so the test fails once the page is fixed and
the entry goes too; an entry an open PR fixes (``fixed_by``) passes in either
state, so the test passes before and after that PR, and is deleted once it
merges.
"""

from __future__ import annotations

import re
import shutil
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, NamedTuple

import pytest
import yaml

from osprey.cli.profile_conventions import (
    PROJECT_MIRROR_DIR,
    convention_for,
    plan_convention_copies,
    validate_convention_sources,
)
from osprey.deployment.compose_generator import SERVICES_DIR, TEMPLATE_FILENAME
from osprey.facility import sources
from osprey.facility.knowledge_links import KNOWLEDGE_DIR
from osprey.facility.records import FacilityRecords
from osprey.facility.references import LINK_SCHEME, find_facility_links
from osprey.facility.settings import VOCABULARY_SOURCE
from osprey.facility.views import VIEWS
from osprey.facility.views.benchmarks import QUERY_SOURCES
from osprey.services.facility_knowledge.okf import index as okf_index
from osprey.services.facility_knowledge.okf.bundle import OKFBundle
from osprey.services.facility_knowledge.okf.document import OKFDocument, OKFDocumentError
from tests._builds import BuiltProject
from tests._by_path import load_module

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs" / "source"
_SCANNER = _REPO_ROOT / "scripts" / "docs" / "snippet_coverage.py"
_TEMPLATES = _REPO_ROOT / "src" / "osprey" / "templates"
_EXAMPLE_FACILITY = _TEMPLATES / "facilities" / "example"
_EXAMPLE_BUNDLE = _EXAMPLE_FACILITY / KNOWLEDGE_DIR
_HELLO_WORLD_SERVER = _TEMPLATES / "apps" / "hello_world" / "mcp_servers" / "example_server"

OKF_PAGE = "how-to/facility-knowledge/okf-bundle.rst"
BUILD_PROFILES_PAGE = "how-to/build-profiles.rst"
COMPOSE_PAGE = "how-to/deploy-project/compose-templates.rst"
FACILITY_PAGE = "how-to/describe-your-facility.rst"
PROJECT_IMAGE_PAGE = "how-to/deploy-project/project-image.rst"
PROFILE_PAGE = "reference/configuration/profile.rst"


class Gap(NamedTuple):
    """One way a page differs from the code.

    Attributes:
        kind: ``wrong`` when the needle is page text the code contradicts,
            ``missing`` when the needle is a path the page's tree leaves out.
        fixed_by: The open PR whose page edit fixes it, or ``None``.
        fixed_text: The page text once fixed, compared whitespace-folded: a
            tree line's column alignment is free.
        note: What the code says.
    """

    kind: Literal["wrong", "missing"]
    fixed_by: str | None
    fixed_text: str
    note: str


#: Page text that differs from the code: (page, needle) -> the gap.
_STALE: dict[tuple[str, str], Gap] = {}


@lru_cache(maxsize=1)
def _scanner() -> ModuleType:
    """The snippet-coverage scanner, loaded by path; registered for its dataclasses."""
    return load_module("snippet_coverage", _SCANNER, register=True)


def _page_text(page: str) -> str:
    return (_DOCS / page).read_text(encoding="utf-8")


def _block(page: str, language: str, needle: str) -> str:
    """The text of the one block on *page* in *language* that holds *needle*."""
    found = [
        block
        for block in _scanner().scan_page(_page_text(page), page, _DOCS)
        if block.language == language and needle in block.text
    ]
    assert len(found) == 1, f"{page}: {len(found)} {language} blocks hold {needle!r}"
    return str(found[0].text)


def _folded(text: str) -> str:
    return " ".join(text.split())


def _is_stale(page: str, needle: str) -> bool:
    """Whether a :data:`_STALE` entry still describes *page*, rather than its fixed text.

    A ``wrong`` entry's page holds exactly one of the needle and the fixed
    text; a ``missing`` entry's page is stale until it holds the fixed text.

    Raises:
        AssertionError: The page holds neither (or, for ``wrong``, both): the
            entry describes neither state and must be rewritten.
    """
    gap = _STALE[(page, needle)]
    text = _folded(_page_text(page))
    fixed = _folded(gap.fixed_text) in text
    if gap.kind == "missing":
        return not fixed
    old = _folded(needle) in text
    assert old != fixed, f"{page}: {needle!r} matches neither the stale nor the fixed text"
    return old


def _stale(page: str, kind: str) -> set[str]:
    """The needles of *page*'s entries of *kind* that still describe it."""
    return {
        needle
        for (where, needle), gap in _STALE.items()
        if where == page and gap.kind == kind and _is_stale(where, needle)
    }


def test_every_stale_entry_describes_its_page_stale_or_fixed() -> None:
    """Each entry's page holds its stale text or its fixed text; one no open PR fixes, the stale."""
    fixed = [key for key in _STALE if not _is_stale(*key) and _STALE[key].fixed_by is None]
    assert fixed == [], f"page fixed; drop the _STALE entry: {fixed}"


# -- the tree helper ------------------------------------------------------------

_TREE_GLYPHS = " │├└─"
#: Where an entry's name ends: an annotation arrow, a ``#`` comment, or a
#: description column two spaces away.
_NAME_END = re.compile(r"\s*←|\s+#|\s{2,}")


def tree_paths(text: str) -> tuple[str, list[str]]:
    """The root of a ``text`` tree block and every entry under it, as relative paths.

    Both drawings are read: ``├──``/``└──`` connectors and plain indentation.
    What follows a name is dropped: a ``←`` annotation, a ``#`` comment, or a
    description two or more spaces away. A line of glyphs alone (a ``│``
    spacer) is skipped. A directory keeps its trailing ``/``,
    and an entry's parent is the nearest directory above it drawn further left.

    Args:
        text: The block's text; its first line is the root.

    Returns:
        The root line's name and each entry's path relative to the root, in
        page order.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    root = _NAME_END.split(lines[0].strip(), maxsplit=1)[0]
    stack: list[tuple[int, str]] = []
    paths: list[str] = []
    for line in lines[1:]:
        if not line.strip(_TREE_GLYPHS):
            continue  # a glyph-only spacer line (``│``) draws no entry
        column = len(line) - len(line.lstrip(_TREE_GLYPHS))
        name = _NAME_END.split(line[column:], maxsplit=1)[0]
        while stack and stack[-1][0] >= column:
            stack.pop()
        path = (stack[-1][1] if stack else "") + name
        paths.append(path)
        if name.endswith("/"):
            stack.append((column, path))
    return root, paths


def test_tree_helper_reads_both_drawings() -> None:
    connectors = (
        "root/\n"
        "├── a.md            ← an annotation\n"
        "├── sub/            # a comment\n"
        "│   ├── b.md\n"
        "│   └── deep/\n"
        "│       └── c.md  a description\n"
        "│\n"
        "└── d/\n"
    )
    indented = (
        "root/\n"
        "  a.md            an annotation\n"
        "  sub/            # a comment\n"
        "    b.md\n"
        "    deep/\n"
        "      c.md\n"
        "  d/\n"
    )
    expected = ("root/", ["a.md", "sub/", "sub/b.md", "sub/deep/", "sub/deep/c.md", "d/"])
    assert tree_paths(connectors) == expected
    assert tree_paths(indented) == expected


def _materialize(root: Path, paths: list[str]) -> None:
    """Create each tree path under *root*: a directory for ``/``, else an empty file."""
    for rel in paths:
        target = root / rel
        if rel.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()


def _on_disk(root: Path, rel: str) -> bool:
    """Whether a tree path, ``<placeholder>`` segments read as wildcards, is under *root*."""
    pattern = re.sub(r"<[^>]+>", "*", rel.rstrip("/"))
    matches = list(root.glob(pattern))
    return any(path.is_dir() for path in matches) if rel.endswith("/") else bool(matches)


def _listed(rel: str, paths: list[str]) -> bool:
    """Whether a file path is a tree entry, or lies under one of the tree's directories.

    ``fnmatch``'s ``*`` crosses ``/``: a ``<placeholder>`` matches any depth, and
    a drawn directory covers every file below it.
    """
    for entry in paths:
        pattern = re.sub(r"<[^>]+>", "*", entry)
        if fnmatchcase(rel, pattern) or (entry.endswith("/") and fnmatchcase(rel, pattern + "*")):
            return True
    return False


def test_a_planted_tree_entry_is_not_on_disk(tmp_path: Path) -> None:
    """Negative control: an entry the tree draws but nothing wrote is caught."""
    _materialize(tmp_path, ["a.md", "sub/"])
    _root, paths = tree_paths("root/\n  a.md\n  sub/\n  planted.md\n")
    assert [rel for rel in paths if not _on_disk(tmp_path, rel)] == ["planted.md"]


# -- how-to/facility-knowledge/okf-bundle.rst -----------------------------------


def _concept_block() -> str:
    return _block(OKF_PAGE, "markdown", "type: Subsystem")


def test_concept_document_passes_authoring_validation() -> None:
    document = OKFDocument.parse(_concept_block())
    document.validate("authoring")
    assert document.frontmatter["tags"] == ["magnets", "power-supplies", "PSS"]
    assert document.body.startswith("# Power Supply System")


def test_concept_type_is_one_the_example_bundle_uses() -> None:
    used = {
        OKFDocument.parse(path.read_text(encoding="utf-8")).frontmatter.get("type")
        for path in _EXAMPLE_BUNDLE.rglob("*.md")
    }
    assert OKFDocument.parse(_concept_block()).frontmatter["type"] in used


def test_concept_without_a_type_fails_validation() -> None:
    """Negative control: the same frontmatter less ``type`` is refused at both levels."""
    text = _concept_block().replace("type: Subsystem\n", "")
    document = OKFDocument.parse(text)
    for level in ("consuming", "authoring"):
        with pytest.raises(OKFDocumentError):
            document.validate(level)


def test_bundle_link_resolves_to_an_example_concept() -> None:
    text = _block(OKF_PAGE, "markdown", "/subsystems/pss.md")
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
    assert links == [("Personnel Safety System", "/subsystems/pss.md")]
    concept_id = links[0][1].removesuffix(".md")
    bundle = OKFBundle(_EXAMPLE_BUNDLE)
    assert bundle.resolve_concept_path(concept_id) == (_EXAMPLE_BUNDLE / "subsystems/pss.md")
    assert bundle.read_concept(concept_id).frontmatter["title"].startswith(links[0][0])


def _facility_links_block() -> str:
    return _block(OKF_PAGE, "markdown", f"]({LINK_SCHEME}:")


def test_facility_links_parse_to_the_targets_they_name() -> None:
    found = find_facility_links(_facility_links_block())
    assert [(link.label, link.target) for link in found] == [
        ("the X position", "SR:DIAG:BPM:01:POSITION:X"),
        ("BPM 01", "SR/BPM01"),
        ("the gauge", "SECTOR 1 GAUGE"),
    ]


def test_facility_links_resolve_in_the_example_facility(
    built_demo: BuiltProject,
) -> None:
    """A channel id and a device id resolve; the made-up spaced address is the dangling kind."""
    records = FacilityRecords(built_demo.facility)
    found = find_facility_links(_facility_links_block(), records)
    assert [link.kinds for link in found] == [("channel",), ("device",), ()]


def test_a_link_with_another_scheme_is_no_facility_link() -> None:
    """Negative control: the same links under an unknown scheme are not read."""
    text = _facility_links_block().replace(f"{LINK_SCHEME}:", "elsewhere:")
    assert find_facility_links(text) == []


def test_regenerate_snippet_writes_the_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snippet, run from a project holding the example bundle, leaves it as the CLI would."""
    snippet = _block(OKF_PAGE, "python", "regenerate_indexes(")
    bundle = tmp_path / "data" / "facility" / KNOWLEDGE_DIR
    shutil.copytree(_EXAMPLE_BUNDLE, bundle)
    for index in bundle.rglob(okf_index._INDEX_FILE):
        index.unlink()
    assert okf_index.check_indexes(bundle) != []

    monkeypatch.chdir(tmp_path)
    exec(compile(snippet, OKF_PAGE, "exec"), {})

    written = sorted(bundle.rglob(okf_index._INDEX_FILE))
    expected = [d / okf_index._INDEX_FILE for d in okf_index._directories_to_index(bundle)]
    assert written == sorted(expected)
    assert okf_index.check_indexes(bundle) == []
    first = {path: path.read_bytes() for path in written}
    exec(compile(snippet, OKF_PAGE, "exec"), {})
    assert {path: path.read_bytes() for path in written} == first, "not idempotent"
    root_index = (bundle / okf_index._INDEX_FILE).read_text(encoding="utf-8")
    assert root_index.startswith("---\nokf_version:")
    assert not any(
        path.read_text(encoding="utf-8").startswith("---")
        for path in written
        if path.parent != bundle
    )


def test_starting_shape_tree_names_the_index_file_and_the_example_directories() -> None:
    root, paths = tree_paths(_block(OKF_PAGE, "text", "← bundle root index"))
    assert root == f"data/facility/{KNOWLEDGE_DIR}/"
    assert {Path(rel).name for rel in paths if rel.endswith("index.md")} == {okf_index._INDEX_FILE}
    assert "index.md" in paths, "the bundle root holds an index"
    top_dirs = {rel.rstrip("/") for rel in paths if rel.endswith("/") and rel.count("/") == 1}
    assert top_dirs == {path.name for path in _EXAMPLE_BUNDLE.iterdir() if path.is_dir()}


# -- how-to/build-profiles.rst --------------------------------------------------


def _build_tree() -> list[str]:
    root, paths = tree_paths(_block(BUILD_PROFILES_PAGE, "text", "facility.json"))
    assert root == "build/"
    return paths


def _main_render_files(built: BuiltProject) -> set[str]:
    """The facility outputs of the deployment's own render, the first one a build runs."""
    main = built.outputs[0]
    assert main.render_dir.name == built.repo.name
    return set(main.files)


def _rendered_config(built: BuiltProject) -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(
        (built.build_dir / "config.yml").read_text(encoding="utf-8")
    )
    return document


def test_build_tree_paths_are_on_disk(built_demo: BuiltProject) -> None:
    """Each path is in ``build/``; the one absent is the view graph mode does not write."""
    build_dir = built_demo.build_dir
    block = _block(BUILD_PROFILES_PAGE, "text", "facility.json")
    absent = [rel for rel in _build_tree() if not _on_disk(build_dir, rel)]
    assert absent == ["data/channel_finder/<mode>.json"]

    mode = _rendered_config(built_demo)["channel_finder"]["pipeline_mode"]
    assert mode == "graph"
    assert "graph" not in {view.name for view in VIEWS if view.path == "channel_finder"}
    (line,) = [entry for entry in block.splitlines() if "data/channel_finder/" in entry]
    assert "in every mode but graph" in line


def test_every_facility_output_is_on_the_build_tree(built_demo: BuiltProject) -> None:
    paths = _build_tree()
    unlisted = {rel for rel in _main_render_files(built_demo) if not _listed(rel, paths)}
    assert unlisted == _stale(BUILD_PROFILES_PAGE, "missing")
    for view in VIEWS:
        if view.path != ".":
            assert any(rel.startswith(f"data/{view.path}/") for rel in paths), view.name


def test_a_planted_build_output_is_caught() -> None:
    """Negative control: an output the tree leaves out is unlisted; one it draws is listed."""
    paths = _build_tree()
    assert _listed("data/simulator/seeds.json", paths)
    assert not _listed("data/planted.json", paths)


# -- how-to/deploy-project/compose-templates.rst --------------------------------


def test_compose_tree_names_the_templates_the_build_copies(
    built_demo: BuiltProject,
) -> None:
    root, paths = tree_paths(_block(COMPOSE_PAGE, "text", "# Required: shared network"))
    assert root == f"{SERVICES_DIR}/"
    assert paths == ["docker-compose.yml.j2", "postgresql/", "postgresql/docker-compose.yml.j2"]
    assert {Path(rel).name for rel in paths if not rel.endswith("/")} == {TEMPLATE_FILENAME}
    for base in (_TEMPLATES / SERVICES_DIR, built_demo.build_dir / SERVICES_DIR):
        assert [rel for rel in paths if not _on_disk(base, rel)] == [], base
    root_template = (_TEMPLATES / SERVICES_DIR / TEMPLATE_FILENAME).read_text(encoding="utf-8")
    assert re.search(r"^networks:\n(?:[ #].*\n)*?  osprey-network:", root_template, re.M)


# -- how-to/describe-your-facility.rst ------------------------------------------


def _facility_tree() -> list[str]:
    root, paths = tree_paths(_block(FACILITY_PAGE, "text", "identity.yaml"))
    assert root == "data/facility/"
    return paths


def test_facility_tree_is_what_init_writes(built_demo: BuiltProject) -> None:
    """Every entry but those the page says the preset leaves out is written, and nothing else."""
    facility = built_demo.facility_dir
    paths = _facility_tree()
    not_shipped = [rel for rel in paths if not _on_disk(facility, rel)]
    assert not_shipped == [
        "signals.yaml",
        "policy.yaml",
        "fixes.yaml",
        "imported/<layer>/",
    ]
    sentence = " ".join(_page_text(FACILITY_PAGE).split())
    assert (
        "ships all of them except ``signals.yaml``, ``policy.yaml``, ``fixes.yaml``"
        " and ``imported/``"
    ) in sentence

    drawn_top = {rel.split("/")[0] for rel in paths}
    on_disk_top = {path.name for path in facility.iterdir()}
    assert {name + "/" for name in on_disk_top - drawn_top} == _stale(FACILITY_PAGE, "missing")
    shipped_records = {
        Path(rel).name
        for rel in paths
        if rel.startswith("records/") and rel != "records/" and rel not in not_shipped
    }
    assert shipped_records == {path.name for path in (facility / "records").iterdir()}


def test_facility_tree_spells_the_names_the_build_reads() -> None:
    paths = _facility_tree()
    top_files = {rel for rel in paths if "/" not in rel}
    assert top_files == set(sources._TOP_FILES)
    assert {
        Path(rel).name for rel in paths if rel.startswith("records/") and rel != "records/"
    } == set(sources.RECORD_FILES)
    assert f"{KNOWLEDGE_DIR}/" in paths
    assert "imported/<layer>/" in paths
    # The tree draws these two directories; the build reads them from here.
    assert VOCABULARY_SOURCE.startswith("data/facility/ariel/")
    assert QUERY_SOURCES.startswith("benchmarks/")


# -- the my-facility/ trees: project-image.rst and profile.rst ------------------


def _profile_tree(page: str, needle: str) -> list[str]:
    root, paths = tree_paths(_block(page, "text", needle))
    assert root == "my-facility/"
    return paths


def test_project_mirror_dockerfile_is_copied_onto_the_render_root(tmp_path: Path) -> None:
    paths = _profile_tree(PROJECT_IMAGE_PAGE, "copied verbatim onto the render")
    assert paths == [f"{PROJECT_MIRROR_DIR}/", f"{PROJECT_MIRROR_DIR}/Dockerfile"]
    _materialize(tmp_path, paths)
    validate_convention_sources(tmp_path)
    copies = [(c.category, c.destination, c.is_directory) for c in plan_convention_copies(tmp_path)]
    assert copies == [(PROJECT_MIRROR_DIR, "Dockerfile", False)]


def test_mcp_server_tree_is_the_scaffold_layout_and_lands_in_build(tmp_path: Path) -> None:
    paths = _profile_tree(PROFILE_PAGE, "my_server/")
    convention = convention_for("mcp_servers")
    assert convention is not None
    assert paths[:2] == [f"{convention.source}/", f"{convention.source}/my_server/"]
    package = f"{convention.source}/my_server/"
    files = {Path(rel).name for rel in paths if rel.startswith(package) and rel != package}
    shipped = {path.name for path in _HELLO_WORLD_SERVER.iterdir() if path.suffix == ".py"}
    assert files == shipped
    assert "build/_mcp_servers/" in " ".join(_page_text(PROFILE_PAGE).split())
    _materialize(tmp_path, paths)
    copies = [(c.category, c.destination, c.is_directory) for c in plan_convention_copies(tmp_path)]
    assert copies == [(convention.source, f"{convention.destination}/my_server", True)]
    assert convention.destination == "_mcp_servers"
