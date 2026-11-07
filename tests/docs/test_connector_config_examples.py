"""The connector pages' YAML examples are configuration OSPREY actually reads.

Every YAML block on the connector pages is a file under
``docs/source/how-to/control-systems/examples/`` that the page includes. Each
file is checked three ways:

* every key it spells is a key the code reads: declared in
  ``src/osprey/profiles/config_key_manifest.yml``, the ledger
  ``scripts/check_config_keys.py`` holds against the source tree, and not one
  of the spellings that ledger records as deleted. A misspelled key raises
  nothing anywhere else: the build and the connectors ignore a key they do not
  read, so this is the check that catches it;
* the build's own checks of the two sections pass: no block under
  ``archiver:`` that no archiver reads, no incomplete ``limits_checking``
  block;
* the factory takes it: a control-system section builds its connector, and the
  EPICS Archiver Appliance's settings connect (that connector reads them
  without contacting the appliance).

A key an example shows commented out, as an option to turn on, is checked as
if it were turned on.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors
from osprey_connectors.types import archiver_settings_errors, incomplete_limits_blocks

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "docs/source/how-to/control-systems/examples"
MANIFEST = REPO_ROOT / "src/osprey/profiles/config_key_manifest.yml"

YAML_EXAMPLES = sorted(EXAMPLES.glob("*.yml"))

#: A commented-out key line, ``# auth:`` or ``#   token_env: X``: an option the
#: example shows switched off. The key must start right after ``# `` and its
#: indentation, so prose comments ("# EPICS uses one ...") never match.
_COMMENTED_KEY = re.compile(r"^(?P<indent>\s*)# (?P<line>\s*[A-Za-z_][\w.-]*:(\s.*)?)$")


def _with_options_on(text: str) -> str:
    """*text* with every commented-out key line turned back on."""
    return "\n".join(
        match.group("indent") + match.group("line")
        if (match := _COMMENTED_KEY.match(line))
        else line
        for line in text.splitlines()
    )


def _documents(path: Path) -> dict[str, Any]:
    """The example as shown, and with its commented-out options on, by label."""
    text = path.read_text()
    shown = yaml.safe_load(text)
    assert isinstance(shown, dict), f"{path.name} is not a YAML mapping"
    on = yaml.safe_load(_with_options_on(text))
    return {"as shown": shown, "options on": on}


def _leaves(node: Any, prefix: str = "") -> Iterator[str]:
    """The dotted path of every leaf under *node*; a list is one leaf."""
    if isinstance(node, Mapping) and node:
        for key, value in node.items():
            yield from _leaves(value, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix


def _declared_spelling(path: str, document: Mapping[str, Any]) -> str | None:
    """The manifest's spelling of *path*, or ``None`` for a key no manifest can list.

    ``archiver.settings`` and a short-named archiver's own-name block are one
    block (the factory reads either), and the manifest records the keys under
    the own-name spelling. The settings of a connector selected by dotted module
    path are that connector's own, so the manifest cannot list them.
    """
    if not path.startswith("archiver.settings."):
        return path
    archiver_type = str((document.get("archiver") or {}).get("type", ""))
    if "." in archiver_type:
        return None
    return f"archiver.{archiver_type}." + path.removeprefix("archiver.settings.")


@pytest.fixture(scope="module")
def manifest() -> tuple[frozenset[str], frozenset[str]]:
    """The declared keys and the deleted spellings."""
    document = yaml.safe_load(MANIFEST.read_text())
    deleted = {entry if isinstance(entry, str) else entry["path"] for entry in document["deleted"]}
    return frozenset(document["keys"]), frozenset(deleted)


def test_every_yaml_block_on_the_pages_is_an_example_file():
    """No connector page keeps a YAML block inline, where no test would read it."""
    pages = [
        REPO_ROOT / "docs/source/how-to/control-systems/use-connectors.rst",
        REPO_ROOT / "docs/source/how-to/control-systems/use-connectors-standalone.rst",
        REPO_ROOT / "docs/source/reference/contracts/connectors.rst",
    ]
    inline = [
        f"{page.relative_to(REPO_ROOT)}:{number}"
        for page in pages
        for number, line in enumerate(page.read_text().splitlines(), start=1)
        if line.strip() == ".. code-block:: yaml"
    ]
    assert inline == []


def test_every_example_file_is_on_a_page():
    """An example no page includes would be tested and never read."""
    pages = (REPO_ROOT / "docs/source").rglob("*.rst")
    included = " ".join(page.read_text() for page in pages)
    missing = [path.name for path in YAML_EXAMPLES if f"examples/{path.name}" not in included]
    assert missing == []


@pytest.mark.parametrize("path", YAML_EXAMPLES, ids=lambda path: path.name)
def test_every_key_is_one_the_code_reads(path, manifest):
    declared, deleted = manifest
    problems = []
    for label, document in _documents(path).items():
        for leaf in _leaves(document):
            spelling = _declared_spelling(leaf, document)
            if spelling is None:
                continue
            if spelling in deleted:
                problems.append(f"{label}: {leaf} is a deleted spelling")
            elif spelling not in declared:
                problems.append(f"{label}: {leaf} is not a key the code reads")
    assert problems == []


@pytest.mark.parametrize("path", YAML_EXAMPLES, ids=lambda path: path.name)
def test_the_builds_own_checks_pass(path):
    for label, document in _documents(path).items():
        archiver = document.get("archiver")
        if archiver is not None:
            assert archiver_settings_errors(archiver) == [], label
        control_system = document.get("control_system")
        if control_system is not None:
            assert incomplete_limits_blocks(control_system) == [], label


def _with_section(section: str) -> list[Path]:
    return [
        path
        for path in YAML_EXAMPLES
        if "type" in (yaml.safe_load(path.read_text()).get(section) or {})
    ]


@pytest.mark.parametrize("path", _with_section("control_system"), ids=lambda path: path.name)
def test_the_factory_builds_the_control_system_connector(path):
    register_builtin_connectors()
    for label, document in _documents(path).items():
        section = document["control_system"]
        connector, type_config = ConnectorFactory.build_control_system_connector(section)
        assert connector is not None, label
        block = (section.get("connector") or {}).get(section["type"])
        if block is not None:
            assert type_config == block, label


def test_the_archiver_appliance_settings_connect():
    """The one archiver whose ``connect()`` reads its settings without the server."""
    register_builtin_connectors()

    async def connect(section: Mapping[str, Any]) -> None:
        archiver = await ConnectorFactory.create_archiver_connector(dict(section))
        await archiver.disconnect()

    document = _documents(EXAMPLES / "archiver-epics.yml")["as shown"]
    asyncio.run(connect(document["archiver"]))
