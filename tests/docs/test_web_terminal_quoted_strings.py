"""The web-terminal pages quote strings, names and files that live in the code.

A reader copies these literally: a refusal line to search a log for, an
environment variable a cell reads, a ``postMessage`` type a panel listens for,
a stylesheet a panel links, a theme name, a lint code, a figure status, or a
config stanza the preset is said to carry. Each one is defined once in the
source. This module holds every quoted copy equal to its definition, so a
rename in either place fails here instead of in front of an operator.
"""

from __future__ import annotations

import re
import textwrap
import typing
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_presets import _load_preset_raw
from osprey.interfaces.design_system.theme_config import load_theme_registry
from osprey.interfaces.lattice_dashboard.schema import FigureStatus

_REPO = Path(__file__).resolve().parents[2]
_PAGES = _REPO / "docs" / "source" / "how-to" / "web-terminal"
_SRC = _REPO / "src" / "osprey"

_NOTEBOOKS = _PAGES / "notebooks.rst"
_PANELS = _PAGES / "panels.rst"
_THEMING = _PAGES / "theming.rst"
_LATTICE = _PAGES / "lattice-dashboard.rst"
_CANOPY = _PAGES / "canopy.rst"
_MULTI_USER = _PAGES / "multi-user" / "index.rst"
_LOGIN = _PAGES / "multi-user" / "login.rst"

_OPENER = re.compile(r"^(?P<indent> *)\.\. code-block:: (?P<lang>\w+)\s*$")

# The root-absolute prefixes a panel links, and the directories the terminal
# mounts them from (``osprey.interfaces._app_setup``: ``/static/fonts`` is
# ``shared_fonts``, ``/design-system`` is ``design_system/static``).
_ASSET_MOUNTS = {
    "/static/fonts/": _SRC / "interfaces" / "shared_fonts",
    "/design-system/": _SRC / "interfaces" / "design_system" / "static",
}


def _blocks(page: Path, lang: str) -> list[str]:
    """The dedented bodies of every ``.. code-block:: <lang>`` on *page*, in order."""
    lines = page.read_text(encoding="utf-8").splitlines()
    bodies: list[str] = []
    i = 0
    while i < len(lines):
        match = _OPENER.match(lines[i])
        i += 1
        if match is None:
            continue
        indent = len(match["indent"])
        while i < len(lines) and lines[i].strip().startswith(":"):
            i += 1
        body: list[str] = []
        while i < len(lines) and (
            not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent
        ):
            body.append(lines[i])
            i += 1
        if match["lang"] == lang:
            bodies.append(textwrap.dedent("\n".join(body)).strip("\n"))
    return bodies


def _block_with(page: Path, lang: str, needle: str) -> str:
    """The one ``lang`` block on *page* that contains *needle*."""
    found = [body for body in _blocks(page, lang) if needle in body]
    assert len(found) == 1, f"{page.name}: expected one {lang} block holding {needle!r}"
    return found[0]


def _is_subset(part: Any, whole: Any) -> bool:
    """Whether *part* states nothing that *whole* does not (lists match item by item)."""
    if isinstance(part, dict):
        return isinstance(whole, dict) and all(
            key in whole and _is_subset(value, whole[key]) for key, value in part.items()
        )
    if isinstance(part, list):
        return (
            isinstance(whole, list)
            and len(part) == len(whole)
            and all(_is_subset(a, b) for a, b in zip(part, whole, strict=True))
        )
    return bool(part == whole)


def _preset() -> dict[str, Any]:
    raw, _ = _load_preset_raw("control-assistant")
    return raw


def test_the_notebook_refusal_lines_are_the_kernel_hints() -> None:
    """The two re-run lines and the client-import line are the kernel's own hints."""
    from osprey.jupyter_kernel import (
        HINT_RAW_CLIENT_IMPORT,
        HINT_TARGET_CHANGED,
        HINT_WRITES_OFF,
    )

    rerun = _block_with(_NOTEBOOKS, "text", "Re-run the cell.").splitlines()
    client = _block_with(_NOTEBOOKS, "text", "Client libraries")

    assert rerun == [HINT_TARGET_CHANGED, HINT_WRITES_OFF]
    assert client == HINT_RAW_CLIENT_IMPORT


def test_the_switch_in_progress_line_is_the_runtime_message() -> None:
    """The quoted switch-in-progress line is what the runtime raises for pid 5150."""
    from osprey.runtime import SwitchInProgressError

    quoted = _block_with(_NOTEBOOKS, "text", "switch_in_progress:")

    assert quoted == str(SwitchInProgressError("switch_in_progress:5150"))


def test_the_environment_a_cell_reads_is_what_the_kernel_stamps() -> None:
    """The three variables a cell reads back are the ones the kernel sets."""
    from osprey.mcp_server.python_executor import executor

    block = _block_with(_NOTEBOOKS, "python", "os.environ")
    named = re.findall(r'os\.environ\["([A-Z_]+)"\]', block)

    assert named == [
        executor.ENV_CONTROL_TARGET,
        executor.ENV_CONTROL_TARGET_GENERATION,
        executor.ENV_LAUNCH_POSTURE,
    ]


def test_the_runtime_calls_a_cell_imports_are_exported() -> None:
    """Every name the notebook example imports from ``osprey.runtime`` is public."""
    import osprey.runtime

    block = _block_with(_NOTEBOOKS, "python", "from osprey.runtime import")
    imported = re.search(r"^from osprey\.runtime import (.+)$", block, re.MULTILINE)
    assert imported is not None
    names = {name.strip() for name in imported.group(1).split(",")}

    assert names <= set(osprey.runtime.__all__)


def _js_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((_SRC / "interfaces").rglob("*.js"))
    )


def test_the_panel_message_types_are_the_terminal_s() -> None:
    """Every ``postMessage`` type panels.rst names is a string the terminal's JS uses."""
    text = _PANELS.read_text(encoding="utf-8")
    blocks = "\n".join(_blocks(_PANELS, "javascript"))
    types = set(re.findall(r"type: '([a-z-]+)'", blocks))
    types |= set(re.findall(r"``(osprey-[a-z-]+-exit)``", text))
    js = _js_sources()

    assert types >= {"osprey-theme-change", "canopy-host-focus-exit", "osprey-focus-exit"}
    missing = sorted(t for t in types if f"'{t}'" not in js and f'"{t}"' not in js)
    assert missing == []


def test_the_stylesheets_a_panel_links_exist() -> None:
    """Every root-absolute ``<link href>`` on panels.rst is a file the terminal serves."""
    hrefs = re.findall(r'href="(/[^"]+)"', "\n".join(_blocks(_PANELS, "html")))
    assert hrefs, "panels.rst links no stylesheet; the blocks moved"

    for href in hrefs:
        prefix = next((p for p in _ASSET_MOUNTS if href.startswith(p)), None)
        assert prefix is not None, f"{href} is under no mount the terminal serves"
        assert (_ASSET_MOUNTS[prefix] / href.removeprefix(prefix)).is_file(), href


def test_the_eight_theme_names_are_the_shipped_themes() -> None:
    """The panel page's list of eight names is the set of theme files."""
    entries, _ = load_theme_registry()
    ids = {entry.id for entry in entries}
    paragraph = re.search(
        r"\*\*The theme is one of eight names\.\*\*(.+?)\n\n",
        _PANELS.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert paragraph is not None
    listed = set(re.findall(r"``([a-z-]+)``", paragraph.group(1)))
    files = {path.stem for path in (_SRC / "interfaces/design_system/tokens/themes").glob("*.json")}

    assert listed == ids == files


def test_the_theme_families_are_the_shipped_families() -> None:
    """The four families theming.rst lists are the registry's families."""
    _, defaults = load_theme_registry()
    section = _THEMING.read_text(encoding="utf-8").split("Choosing a theme", 1)[1]
    families = set(re.findall(r"^- \*\*([a-z-]+)\*\* —", section, re.MULTILINE))

    assert families == set(defaults)


@pytest.mark.parametrize("page", [_PANELS, _THEMING], ids=lambda p: p.name)
def test_every_theme_value_a_page_writes_is_a_theme_or_family(page: Path) -> None:
    """A theme in a URL, a message, a ``data-theme`` or a config block resolves."""
    entries, defaults = load_theme_registry()
    known = {entry.id for entry in entries} | set(defaults)
    text = page.read_text(encoding="utf-8")
    values = set(re.findall(r"[?&]theme=([a-z-]+)", text))
    values |= set(re.findall(r"data-theme=\"([a-z-]+)\"", text))
    values |= set(re.findall(r"theme: '([a-z-]+)'", text))
    values |= set(re.findall(r"^\s*theme: ([a-z-]+)", "\n".join(_blocks(page, "yaml")), re.M))
    assert values, f"{page.name} writes no theme value; the patterns moved"

    assert sorted(values - known) == []


def _lint_codes() -> set[str]:
    """The codes the web-terminals lint reports.

    Its own findings are the ``code="web_terminals.<code>"`` sites in
    ``lint.py``; the roster checks it reports are the code strings in
    ``roster.py``.
    """
    pkg = _SRC / "deployment" / "web_terminals"
    lint = (pkg / "lint.py").read_text(encoding="utf-8")
    roster = (pkg / "roster.py").read_text(encoding="utf-8")
    return set(re.findall(r'code="(web_terminals\.[a-z_]+)"', lint)) | set(
        re.findall(r'"(web_terminals\.[a-z_]+)"', roster)
    )


@pytest.mark.parametrize("page", [_LOGIN, _MULTI_USER], ids=lambda p: p.name)
def test_every_lint_code_a_page_names_is_one_the_lint_emits(page: Path) -> None:
    """A quoted ``web_terminals.<code>`` is a code the web-terminals lint reports.

    Config keys on these pages are spelled from ``modules.``, so a literal that
    starts at ``web_terminals.`` is a finding code.
    """
    text = page.read_text(encoding="utf-8")
    codes = set(re.findall(r"``(web_terminals\.[a-z_]+)``", text))
    assert codes, f"{page.name} names no lint code; the pattern moved"

    assert sorted(codes - _lint_codes()) == []


def test_the_lattice_statuses_are_the_figure_statuses() -> None:
    """The five statuses the lattice page lists are ``FigureStatus``, spelled the same."""
    section = _LATTICE.read_text(encoding="utf-8").split("When a figure is current", 1)[1]
    listed = re.findall(r"^- ``([a-z_ ]+)`` —", section, re.MULTILINE)

    assert listed == list(typing.get_args(FigureStatus))


def test_the_canopy_stanza_is_what_the_preset_carries() -> None:
    """The profile stanza the CANOPY page adds is already in the control-assistant preset."""
    stanza = yaml.safe_load(_block_with(_CANOPY, "yaml", "web_panels:"))
    preset = _preset()

    assert set(stanza["web_panels"]) <= set(preset["web_panels"])
    assert set(stanza["skills"]) <= set(preset["skills"])
    assert stanza["lifecycle"] == preset["lifecycle"]


def test_the_multi_user_config_block_is_an_excerpt_of_the_preset() -> None:
    """Every key the switching-it-on block states, the preset sets to the same value."""
    block = yaml.safe_load(_block_with(_MULTI_USER, "yaml", "image_source: local"))
    preset = _preset()["config"]["modules.web_terminals"]

    assert _is_subset(block["modules"]["web_terminals"], preset)


def test_the_subset_check_catches_a_value_the_preset_does_not_set() -> None:
    """The excerpt check fails on a changed value and on a key the preset lacks."""
    preset = {"auth": {"method": "password"}, "users": [{"name": "alice"}]}

    assert _is_subset({"auth": {"method": "password"}}, preset)
    assert not _is_subset({"auth": {"method": "oidc"}}, preset)
    assert not _is_subset({"auth": {"method": "password", "port": 1}}, preset)
    assert not _is_subset({"users": [{"name": "alice"}, {"name": "bob"}]}, preset)


_NESTED_MODULES = re.compile(r"^modules:\n +web_terminals:", re.MULTILINE)


@pytest.mark.parametrize("page", [_LOGIN, _THEMING, _MULTI_USER], ids=lambda p: p.name)
def test_profile_snippets_spell_modules_web_terminals_as_one_key(page: Path) -> None:
    """No profile snippet nests ``web_terminals:`` under ``modules:``.

    Under ``config:`` in ``profile.yml`` a nested ``modules:`` replaces the whole
    rendered ``modules`` subtree. The two multi-user blocks that show the built
    ``config.yml`` are nested on purpose and are exempt by content: the
    "Switching it on" excerpt and the registry worked example.
    """
    exempt = ("# osprey up builds the images itself", "url: registry.example.org/")
    nested = [
        body.splitlines()[0:3]
        for body in _blocks(page, "yaml")
        if _NESTED_MODULES.search(body) and not any(mark in body for mark in exempt)
    ]

    assert nested == []
