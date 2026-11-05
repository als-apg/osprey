"""Every YAML block in the docs spells keys the code reads.

A reader copies a ``.. code-block:: yaml`` into ``config.yml`` or a build
profile, and a key the code does not read raises nothing: the build and the
services ignore it, so the setting silently does nothing. This check reads
every such block under ``docs/source`` and holds each key against the ledger
the code is already held to, ``src/osprey/profiles/config_key_manifest.yml``
(enforced against the source tree by ``scripts/check_config_keys.py``). It is
the page-wide form of the connector pages' own check,
``test_connector_config_examples``, and reuses its key walk.

A block is read one of three ways, by its root keys:

* **config form** — every root is a top-level section of the manifest
  (``ariel:``, ``modules:``). Every leaf must be a declared key, or lie under a
  declared data map, and must not be a spelling the manifest records as
  deleted.
* **profile form** — a root is a build-profile key the loader accepts and not a
  config section (``name:``, ``dispatch:``), or the block has a ``config:``
  block. Every root must be one the loader accepts, the ``config:`` block is
  checked as config form, and the profile blocks the loader closes
  (``mcp_servers:`` entries, ``services:`` entries, ``bluesky:``, ...) may only
  spell the keys the loader accepts there.
* **either** — every root is both (``services:``, ``bluesky:``,
  ``agent_record:``). The block passes if it is valid in either reading.

A ``health:`` block is also handed to ``parse_health_config``, which refuses
what the manifest cannot see: a category's ``checks:`` are the facility's own
data, so leaves under ``health.categories.<name>`` are checked by the parser,
not the ledger.

A block that is not configuration at all (a facility file, ``providers.yml``,
a trigger file, a compose template, an elided list) is listed in
:data:`_NOT_CONFIG` with the file it belongs to; a key the manifest cannot
declare is listed in :data:`_UNDECLARED` with its reason. Both are keyed by
page and content rather than by line number, so an edit elsewhere on a page
does not move an entry, and both are held empty of stale entries.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_record.config import _KNOWN_AGENT_RECORD_KEYS
from osprey.cli.build_profile_archiver import _KNOWN_VA_ARCHIVER_KEYS
from osprey.cli.build_profile_deploy import _KNOWN_DEPLOY_KEYS
from osprey.cli.build_profile_load import (
    _KNOWN_BLUESKY_KEYS,
    _KNOWN_DISPATCH_KEYS,
    _KNOWN_ENVIRONMENT_KEYS,
    _KNOWN_MCP_SERVER_KEYS,
    _KNOWN_OPTIMIZER_KEYS,
    _KNOWN_PROFILE_KEYS,
    _KNOWN_PROVENANCE_KEYS,
    _KNOWN_VA_KEYS,
)
from osprey.cli.build_profile_schema import ServiceDef
from osprey.health.config import parse_health_config
from osprey_connectors.errors import ConfigurationError
from tests.docs.test_connector_config_examples import _declared_spelling, _leaves

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/source"
MANIFEST = REPO_ROOT / "src/osprey/profiles/config_key_manifest.yml"

#: A YAML code block. Other directives (``literalinclude``) name files that
#: their own tests read.
_OPENER = re.compile(r"^(\s*)\.\. code-block:: yaml\s*$")

#: A ``<placeholder>`` the reader fills in. Replaced with a plain word before
#: parsing, so ``port: <dispatcher_port>`` reads as a value.
_PLACEHOLDER = re.compile(r"<[^<>\n]+>")

#: Profile blocks whose entries the loader closes: block → keys one entry may
#: spell.
_ENTRY_KEYS: dict[str, frozenset[str]] = {
    "mcp_servers": _KNOWN_MCP_SERVER_KEYS,
    "services": frozenset(f.name for f in fields(ServiceDef)),
}

#: Profile blocks the loader closes: block → keys it may spell.
_BLOCK_KEYS: dict[str, frozenset[str]] = {
    "agent_record": _KNOWN_AGENT_RECORD_KEYS,
    "bluesky": _KNOWN_BLUESKY_KEYS,
    "deploy": _KNOWN_DEPLOY_KEYS,
    "dispatch": _KNOWN_DISPATCH_KEYS,
    "environment": _KNOWN_ENVIRONMENT_KEYS,
    "optimizer": _KNOWN_OPTIMIZER_KEYS,
    "provenance": _KNOWN_PROVENANCE_KEYS,
    "va_archiver": _KNOWN_VA_ARCHIVER_KEYS,
    "virtual_accelerator": _KNOWN_VA_KEYS,
}

#: Config paths whose children are named by the facility and checked by a
#: parser rather than by the ledger: ``health.categories.<name>.*``.
_PARSED_ELSEWHERE = ("health.categories.",)

#: A line that elides the rest of a list or mapping: ``...``.
_ELISION = re.compile(r"^\s*\.\.\.\s*(#.*)?$", re.MULTILINE)

_FACILITY = "a facility file under data/facility/, read by the facility loader"
_PROVIDERS = "providers.yml, the provider catalog; its entries are data under api.providers"
_TRIGGERS = "triggers.yml, read by the event-dispatch trigger loader"

#: Blocks that are not ``config.yml`` or a build profile, keyed by
#: ``(page, first line of the block)``, with the file they belong to.
_NOT_CONFIG: dict[tuple[str, str], str] = {
    ("getting-started/hello-world-tutorial.rst", "records:"): _FACILITY,
    ("getting-started/hello-world-tutorial.rst", "- address: SR:MAG:CORR:01:CURRENT:SP"): _FACILITY,
    ("how-to/agent-interfaces/event-dispatch.rst", "dispatcher:"): _TRIGGERS,
    ("how-to/agent-interfaces/event-dispatch.rst", "triggers:"): _TRIGGERS,
    ("how-to/ariel/picture-search.rst", "providers:"): _PROVIDERS,
    (
        "how-to/ariel/search-modes.rst",
        "concepts:",
    ): "an ARIEL concept file, read by the query expander",
    ("how-to/bluesky/write-plans.rst", "# build/data/bluesky_devices.yml"): (
        "build/data/bluesky_devices.yml, the device file the queue server mounts"
    ),
    ("how-to/control-systems/use-virtual-accelerator.rst", "- name: <model>"): _FACILITY,
    ("how-to/deploy-a-facility.rst", "code: ca"): _FACILITY,
    (
        "how-to/deploy-project/compose-templates.rst",
        "# services/postgresql/docker-compose.yml.j2",
    ): ("a Jinja compose template, not YAML until rendered"),
    ("how-to/deploy-project/networking.rst", "# services/site_api/docker-compose.yml.j2"): (
        "a Jinja compose template"
    ),
    ("how-to/describe-your-facility.rst", "code: ca"): _FACILITY,
    ("how-to/describe-your-facility.rst", "- id: SR/SECT1"): _FACILITY,
    ("how-to/describe-your-facility.rst", "- id: SR:MAG:QF:01:CURRENT:SP"): _FACILITY,
    ("how-to/describe-your-facility.rst", "- class: Undulator"): _FACILITY,
    ("how-to/describe-your-facility.rst", "schema: osprey.facility.fixes/1"): _FACILITY,
    ("how-to/describe-your-facility.rst", "records:"): _FACILITY,
    ("how-to/describe-your-facility.rst", "SR:BEAM:CURRENT:"): _FACILITY,
    ("how-to/describe-your-facility.rst", "INJ:SEPTUM:PULSE:"): _FACILITY,
    ("how-to/describe-your-facility.rst", "SR:DIAG:DCCT:01:CURRENT:RB:"): _FACILITY,
    ("how-to/describe-your-facility.rst", "- name: LINE"): _FACILITY,
    (
        "how-to/describe-your-facility.rst",
        "kinds: [orm, dispersion, trm, crm, chromaticity_monitor]",
    ): (_FACILITY),
    ("how-to/import-mml-export.rst", "families:"): "an MML import mapping file",
    ("how-to/import-mml-export.rst", "models:"): "an MML import mapping file",
    ("how-to/llm-providers/configure-providers.rst", "providers:"): _PROVIDERS,
    ("how-to/llm-providers/run-open-models.rst", "providers:"): _PROVIDERS,
    ("how-to/run-scenarios.rst", "still:"): "a scenario file's channel selection",
    ("reference/configuration/profile.rst", "providers:"): _PROVIDERS,
}

#: Blocks that show the inside of a config section, keyed by
#: ``(page, first line of the block)``, with the section they sit under.
_FRAGMENTS: dict[tuple[str, str], str] = {
    ("how-to/web-terminal/multi-user/login.rst", "oidc:"): "modules.web_terminals.auth",
    ("how-to/web-terminal/multi-user/login.rst", "shared:"): "modules.web_terminals",
    ("how-to/web-terminal/multi-user/login.rst", "landing:"): "modules.web_terminals",
}

#: Keys a block spells that no manifest entry can declare, keyed by
#: ``(page, dotted key)``, each with its reason.
_EXAMPLE_NAME = "an example name the reader picks; the manifest cannot list a reader's own names"
_UNDECLARED: dict[tuple[str, str], str] = {
    ("how-to/control-systems/phoebus-bridge.rst", "claude_code.servers.phoebus2.extends"): (
        _EXAMPLE_NAME
    ),
    (
        "how-to/control-systems/phoebus-bridge.rst",
        "claude_code.servers.phoebus2.env.PHOEBUS_BRIDGE_URL",
    ): _EXAMPLE_NAME,
    ("how-to/web-terminal/panels.rst", "web.panels.pvinfo.url"): _EXAMPLE_NAME,
    ("how-to/web-terminal/panels.rst", "web.panels.pvinfo.path"): _EXAMPLE_NAME,
    ("how-to/web-terminal/panels.rst", "web.panels.pvinfo.rewrite_prefixes"): _EXAMPLE_NAME,
    (
        "reference/configuration/profile.rst",
        "control_system.connector.mypkg.SiteConnector.limits_checking.enabled",
    ): "a connector selected by dotted module path reads its own settings",
    (
        "reference/configuration/profile.rst",
        "control_system.connector.mypkg.SiteConnector.limits_checking.mode",
    ): "a connector selected by dotted module path reads its own settings",
}


@dataclass(frozen=True)
class Block:
    """One YAML code block on a page."""

    page: str
    line: int
    text: str

    @property
    def where(self) -> str:
        return f"{self.page}:{self.line}"

    @property
    def first(self) -> str:
        """The block's first non-blank line, stripped: its stable name."""
        return next(line.strip() for line in self.text.splitlines() if line.strip())


def _blocks(path: Path, root: Path = DOCS) -> Iterator[Block]:
    """Every YAML code block on *path*, dedented, named relative to *root*."""
    lines = path.read_text(encoding="utf-8").splitlines()
    page = str(path.relative_to(root))
    index = 0
    while index < len(lines):
        opener = _OPENER.match(lines[index])
        index += 1
        if opener is None:
            continue
        base = len(opener.group(1))
        # Directive options and blank lines precede the content.
        while index < len(lines) and (
            not lines[index].strip() or lines[index].strip().startswith(":")
        ):
            index += 1
        start = index
        body: list[str] = []
        while index < len(lines):
            text = lines[index]
            if text.strip() and len(text) - len(text.lstrip()) <= base:
                break
            body.append(text)
            index += 1
        text = textwrap.dedent("\n".join(body)).strip("\n")
        if text:
            yield Block(page, start + 1, text)


@lru_cache(maxsize=1)
def _all_blocks() -> tuple[Block, ...]:
    return tuple(block for path in sorted(DOCS.rglob("*.rst")) for block in _blocks(path))


@lru_cache(maxsize=1)
def _manifest() -> tuple[frozenset[str], frozenset[str], tuple[str, ...]]:
    """The declared keys, the deleted spellings and the data-map prefixes."""
    document = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    keys = document["keys"]
    deleted = {entry if isinstance(entry, str) else entry["path"] for entry in document["deleted"]}
    data_maps = tuple(
        key for key, spec in keys.items() if isinstance(spec, dict) and spec.get("data-map")
    )
    return frozenset(keys), frozenset(deleted), data_maps


def _sections() -> frozenset[str]:
    """The top-level ``config.yml`` sections: the manifest's first segments."""
    return frozenset(key.split(".")[0] for key in _manifest()[0])


def _parse(block: Block) -> Any:
    """The block as YAML, placeholders filled and elisions dropped."""
    text = _ELISION.sub("", _PLACEHOLDER.sub("placeholder", block.text))
    document = yaml.safe_load(text)
    parent = _FRAGMENTS.get((block.page, block.first))
    if parent is not None:
        for key in reversed(parent.split(".")):
            document = {key: document}
    return document


def _config_problems(page: str, document: Mapping[str, Any]) -> list[str]:
    """Every leaf of a config-form *document* the code does not read."""
    declared, deleted, data_maps = _manifest()
    problems = []
    for leaf in _leaves(document):
        spelling = _declared_spelling(leaf, document)
        if spelling is None or (page, leaf) in _UNDECLARED:
            continue
        if spelling in deleted:
            problems.append(f"{leaf} is a deleted spelling")
        elif spelling in declared:
            continue
        elif any(spelling.startswith(f"{prefix}.") for prefix in data_maps):
            continue
        elif any(spelling.startswith(prefix) for prefix in _PARSED_ELSEWHERE):
            continue
        else:
            problems.append(f"{leaf} is not a key the code reads")
    health = document.get("health")
    if health is not None:
        try:
            parse_health_config(health)
        except ConfigurationError as error:
            problems.append(f"parse_health_config refuses the health block: {error}")
    return problems


def _profile_problems(page: str, document: Mapping[str, Any]) -> list[str]:
    """Every key of a profile-form *document* the loader does not accept."""
    problems = [
        f"{root} is not a build-profile key" for root in document if root not in _KNOWN_PROFILE_KEYS
    ]
    for block, accepted in _ENTRY_KEYS.items():
        for name, entry in (document.get(block) or {}).items():
            if not isinstance(entry, Mapping):
                continue
            problems += [
                f"{block}.{name}.{key} is not a key one {block} entry accepts"
                for key in entry
                if key not in accepted
            ]
    for block, accepted in _BLOCK_KEYS.items():
        value = document.get(block)
        if isinstance(value, Mapping):
            problems += [
                f"{block}.{key} is not a key the {block} block accepts"
                for key in value
                if key not in accepted
            ]
    config = document.get("config")
    if isinstance(config, Mapping):
        problems += [f"config: {problem}" for problem in _config_problems(page, config)]
    return problems


def _problems(block: Block) -> list[str]:
    """Why *block* is not configuration the code reads; empty when it is."""
    try:
        document = _parse(block)
    except yaml.YAMLError as error:
        return [f"does not parse: {str(error).splitlines()[0]}"]
    if not isinstance(document, Mapping) or not document:
        return [f"is a {type(document).__name__}, not a config mapping"]
    roots = {str(root) for root in document}
    sections = _sections()
    # A dotted root (``web.panels.events.url:``) is a line of a ``config:`` block.
    in_config = all(root.split(".")[0] in sections for root in roots)
    in_profile = roots <= _KNOWN_PROFILE_KEYS
    if "config" in roots or (in_profile and not in_config):
        return _profile_problems(block.page, document)
    if in_config and not in_profile:
        return _config_problems(block.page, document)
    if in_config and in_profile:
        as_config = _config_problems(block.page, document)
        as_profile = _profile_problems(block.page, document)
        if not as_config or not as_profile:
            return []
        return [f"as config: {problem}" for problem in as_config] + [
            f"as profile: {problem}" for problem in as_profile
        ]
    unknown = sorted(
        root for root in roots - _KNOWN_PROFILE_KEYS if root.split(".")[0] not in sections
    )
    return [f"root keys {unknown} are neither config sections nor build-profile keys"]


def _checked() -> list[Block]:
    return [block for block in _all_blocks() if (block.page, block.first) not in _NOT_CONFIG]


@pytest.mark.parametrize("block", _checked(), ids=lambda block: block.where)
def test_every_yaml_block_spells_keys_the_code_reads(block: Block) -> None:
    assert _problems(block) == [], block.where


def test_the_pages_have_yaml_blocks() -> None:
    """A scanner that found nothing would pass every page."""
    assert len(_all_blocks()) > 100
    assert len(_checked()) > 100


def test_every_not_config_and_fragment_entry_names_a_block() -> None:
    present = {(block.page, block.first) for block in _all_blocks()}
    assert sorted(set(_NOT_CONFIG) - present) == []
    assert sorted(set(_FRAGMENTS) - present) == []


def test_every_undeclared_entry_is_still_spelled() -> None:
    """An entry whose key left its page, or that the manifest now declares, goes."""
    declared = _manifest()[0]
    spelled: set[tuple[str, str]] = set()
    for block in _checked():
        try:
            document = _parse(block)
        except yaml.YAMLError:
            continue
        if not isinstance(document, Mapping):
            continue
        config = document.get("config")
        for source in (document, config if isinstance(config, Mapping) else {}):
            spelled |= {(block.page, leaf) for leaf in _leaves(source)}
    stale = sorted(entry for entry in _UNDECLARED if entry not in spelled or entry[1] in declared)
    assert stale == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ariel:\n  ingestion:\n    adapter: als_logbook\n", []),
        (
            "ariel:\n  ingestion:\n    adaptr: x\n",
            ["ariel.ingestion.adaptr is not a key the code reads"],
        ),
        ("config:\n  web.theme: dark\n", []),
        ("config:\n  web.thme: dark\n", ["config: web.thme is not a key the code reads"]),
        (
            "dispatch:\n  netwrok: host\n",
            ["dispatch.netwrok is not a key the dispatch block accepts"],
        ),
        ("services:\n  qmd:\n    models_dir: /srv\n", []),
        ("services:\n  ariel_sync:\n    template: osprey.ariel_sync\n", []),
        ("health:\n  categories:\n    host:\n      checks: []\n", []),
        (
            "contrl_system:\n  type: epics\n",
            ["root keys ['contrl_system'] are neither config sections nor build-profile keys"],
        ),
    ],
)
def test_the_check_catches_a_misspelled_key(text: str, expected: list[str]) -> None:
    """Negative controls: each reading flags a misspelling and passes the real key."""
    assert _problems(Block("fake.rst", 1, text)) == expected


def test_a_deleted_spelling_is_refused() -> None:
    deleted = sorted(_manifest()[1])[0]
    text = f"config:\n  {deleted}: 1\n"
    assert _problems(Block("fake.rst", 1, text)) == [f"config: {deleted} is a deleted spelling"]


def test_a_core_health_category_with_checks_is_refused() -> None:
    """``parse_health_config`` sees what the ledger cannot."""
    text = "health:\n  categories:\n    providers:\n      checks: []\n"
    [problem] = _problems(Block("fake.rst", 1, text))
    assert "cannot redefine built-in category" in problem


def test_the_block_scanner_reads_nested_blocks(tmp_path: Path) -> None:
    """A block inside a tab or note is read, dedented, at its first content line."""
    page = tmp_path / "page.rst"
    page.write_text(
        "Intro.\n\n"
        ".. tab-item:: One\n\n"
        "   .. code-block:: yaml\n"
        "      :caption: profile.yml\n\n"
        "      bluesky:\n"
        "        plan_dir: plans/\n\n"
        "   Prose after the block.\n",
        encoding="utf-8",
    )
    assert list(_blocks(page, tmp_path)) == [Block("page.rst", 8, "bluesky:\n  plan_dir: plans/")]
