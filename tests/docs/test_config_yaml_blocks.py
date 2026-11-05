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
  (``mcp_servers:``, ``services:`` and ``watchdogs:`` entries, ``bluesky:``,
  ``dispatch:``, ...) may only spell the keys the loader accepts there. Only
  that first level is checked: the closed sets below it
  (``bluesky.external``, an mcp server's ``hooks``, ``agent_record``'s
  ClickHouse block, ``deploy.registry``) and the free-form
  ``services.<name>.config`` are not.
* **either** — every root is both (``services:``, ``bluesky:``,
  ``agent_record:``). The block passes if it is valid in either reading.
  ``watchdogs:`` is read as a profile block only, since the manifest holds it
  as a data map.

A list is one leaf, so the keys inside list items (a roster entry, a health
check) are not checked against the ledger.

A ``health:`` block is also handed to ``parse_health_config``, which refuses
what the manifest cannot see: a category's ``checks:`` are the facility's own
data, so leaves under ``health.categories.<name>`` are checked by the parser,
not the ledger.

A block that belongs to another file is recognised by its content and
skipped (:func:`_other_file`): a Jinja template (it carries a ``{{ name`` or ``{%``);
a list, which no config file or profile is; a mapping keyed by channel
address whose entries are seeds, as ``seeds.yaml`` is; or a mapping whose
roots are all root keys of one other file and not all config or profile keys.
Those root sets come from the files' own schemas and loaders (the facility
schema ``facility/schema/core.yaml``, the MML mapping loader, the bluesky
device file, ``providers.yml``, the trigger loader), not from the pages. A
block that shows the inside of one config section (``oidc:``, ``shared:``) is
checked under that section (:func:`_fragment_parent`). A block no rule places is listed in :data:`_NOT_CONFIG`; a key the manifest
cannot declare is listed in :data:`_UNDECLARED` with its reason. Both are keyed
by page and content (a block's ``:caption:`` or first line) rather than by
line number, and both are held empty of stale entries.
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
from osprey.facility.layers.mml.mapping import _TOP_OPTIONAL as _MML_TOP_OPTIONAL
from osprey.facility.layers.mml.mapping import _TOP_REQUIRED as _MML_TOP_REQUIRED
from osprey.health.config import parse_health_config
from osprey.profiles.providers import _PROVIDERS_KEY
from osprey.services.bluesky_bridge.devices._specs_from_file import (
    ACCEPTED_KEYS as _BLUESKY_DEVICE_KEYS,
)
from osprey.services.watchdog.config import _AGENT_KEYS as _WATCHDOG_AGENT_KEYS
from osprey.services.watchdog.config import _COMMON_KEYS as _WATCHDOG_COMMON_KEYS
from osprey.services.watchdog.config import _DECISION_KEYS as _WATCHDOG_DECISION_KEYS
from osprey_connectors.errors import ConfigurationError
from tests.docs.test_connector_config_examples import _declared_spelling, _leaves

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/source"
MANIFEST = REPO_ROOT / "src/osprey/profiles/config_key_manifest.yml"
_FACILITY_SCHEMA = REPO_ROOT / "src/osprey/facility/schema/core.yaml"
_EXAMPLE_VOCABULARY = REPO_ROOT / "src/osprey/templates/facilities/example/ariel/vocabulary.yml"

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
    "watchdogs": _WATCHDOG_COMMON_KEYS | _WATCHDOG_DECISION_KEYS | _WATCHDOG_AGENT_KEYS,
}

#: Roots read as profile blocks even where the manifest also names them. The
#: manifest's ``watchdogs`` is a data map, so its config reading checks nothing;
#: the profile loader is what closes a watchdog entry's keys.
_PROFILE_ROOTS = frozenset({"watchdogs"})

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

#: Jinja markup: a block carrying it is a template (``docker-compose.yml.j2``).
#: An expression opens on a name (``{{ services.x }}``, ``{{project_root}}``),
#: so a ``str.format`` escape inside a prompt (``{{"keywords": ...}}``) is not one.
_JINJA = re.compile(r"\{\{-?\s*[A-Za-z_]|\{%")

#: A line that elides the rest of a list or mapping: ``...``.
_ELISION = re.compile(r"^\s*\.\.\.\s*(#.*)?$", re.MULTILINE)

#: Blocks no content rule can place, keyed by ``(page, block name)`` (see
#: :attr:`Block.name`), with the file they belong to.
_NOT_CONFIG: dict[tuple[str, str], str] = {
    ("how-to/deploy-project/networking.rst", "services/site_api/docker-compose.yml.j2"): (
        "a compose template with no Jinja markup, rendered by the build"
    ),
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
    caption: str = ""

    @property
    def where(self) -> str:
        return f"{self.page}:{self.line}"

    @property
    def first(self) -> str:
        """The block's first non-blank line, stripped."""
        return next(line.strip() for line in self.text.splitlines() if line.strip())

    @property
    def name(self) -> str:
        """The block's stable name: its ``:caption:``, else its first line, less a ``# ``.

        A file-name comment (``# services/x.yml``) and the same name moved into a
        ``:caption:`` give one name, so either spelling keys the same entry.
        """
        return (self.caption or self.first).removeprefix("#").strip()

    @property
    def is_template(self) -> bool:
        """A Jinja template is not YAML until rendered, and is not config."""
        return bool(_JINJA.search(self.text))


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
        caption = ""
        while index < len(lines) and (
            not lines[index].strip() or lines[index].strip().startswith(":")
        ):
            if lines[index].strip().startswith(":caption:"):
                caption = lines[index].strip().removeprefix(":caption:").strip()
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
            yield Block(page, start + 1, text, caption)


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
    return yaml.safe_load(text)


def _fragment_parent(document: Mapping[str, Any]) -> str | None:
    """The one config section *document* is the inside of, or ``None``.

    A block like ``oidc: {issuer: ...}`` shows keys below a section rather than
    from the top. It belongs under a declared path ``P`` when every root ``r`` is
    a declared key (or a declared key's parent) ``P.r``; with exactly one such
    ``P`` the block is checked as if written there, and a misspelled root, which
    names no declared ``P.r``, has no parent and fails.
    """
    roots = {str(root) for root in document}
    if any(root.split(".")[0] in _sections() | _KNOWN_PROFILE_KEYS for root in roots):
        return None
    parents: set[str] | None = None
    for root in roots:
        found = set()
        for key in _manifest()[0]:
            parts = key.split(".")
            found |= {".".join(parts[:i]) for i in range(1, len(parts)) if parts[i] == root}
        parents = found if parents is None else parents & found
    if parents is None or len(parents) != 1:
        return None
    return parents.pop()


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


@lru_cache(maxsize=1)
def _other_file_roots() -> tuple[tuple[str, frozenset[str]], ...]:
    """``(file, root keys)`` for each non-config file the docs show, from its own schema."""
    schema = yaml.safe_load(_FACILITY_SCHEMA.read_text(encoding="utf-8"))["classes"]

    def attributes(name: str) -> frozenset[str]:
        return frozenset(schema[name]["attributes"])

    vocabulary = yaml.safe_load(_EXAMPLE_VOCABULARY.read_text(encoding="utf-8"))
    return (
        ("data/facility/identity.yaml", attributes("Identity")),
        ("data/facility/limits.yaml", attributes("Limits")),
        ("data/facility/measurement/<model>.yaml", attributes("Measurement")),
        ("a scenario under data/facility/scenarios/", attributes("Scenario")),
        # Read by facility/combine.py: the header and the list of fixes.
        ("data/facility/fixes.yaml", frozenset({"schema", "fixes"})),
        ("the MML import mapping", _MML_TOP_REQUIRED | _MML_TOP_OPTIONAL),
        ("build/data/bluesky_devices.yml", frozenset(_BLUESKY_DEVICE_KEYS)),
        ("an ARIEL vocabulary file", frozenset(vocabulary)),
        ("providers.yml", frozenset({_PROVIDERS_KEY})),
        # load_triggers reads exactly these two keys.
        ("triggers.yml", frozenset({"dispatcher", "triggers"})),
    )


def _other_file(document: Any) -> str | None:
    """The file *document* belongs to when it is not config or a profile, else ``None``."""
    if isinstance(document, list):
        return "a list: a facility record file, which no config file or profile is"
    if not isinstance(document, Mapping) or not document:
        return None
    roots = {str(root) for root in document}
    if all(":" in root for root in roots):
        seed_keys = frozenset(
            yaml.safe_load(_FACILITY_SCHEMA.read_text(encoding="utf-8"))["classes"]["Seed"][
                "attributes"
            ]
        )
        if all(isinstance(seed, Mapping) and set(seed) <= seed_keys for seed in document.values()):
            return "data/facility/seeds.yaml: seeds keyed by channel address"
        return None
    known = _sections() | _KNOWN_PROFILE_KEYS
    if all(root.split(".")[0] in known for root in roots):
        return None
    for name, file_roots in _other_file_roots():
        if roots <= file_roots:
            return name
    return None


def _problems(block: Block) -> list[str]:
    """Why *block* is not configuration the code reads; empty when it is."""
    try:
        document = _parse(block)
    except yaml.YAMLError as error:
        return [f"does not parse: {str(error).splitlines()[0]}"]
    if not isinstance(document, Mapping) or not document:
        return [f"is a {type(document).__name__}, not a config mapping"]
    parent = _fragment_parent(document)
    if parent is not None:
        for key in reversed(parent.split(".")):
            document = {key: document}
    roots = {str(root) for root in document}
    sections = _sections()
    # A dotted root (``web.panels.events.url:``) is a line of a ``config:`` block.
    in_config = all(root.split(".")[0] in sections for root in roots)
    in_profile = roots <= _KNOWN_PROFILE_KEYS
    if "config" in roots or (in_profile and not in_config) or roots & _PROFILE_ROOTS:
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
    return [
        block
        for block in _all_blocks()
        if (block.page, block.name) not in _NOT_CONFIG
        and not block.is_template
        and _other_file(_parsed_or_none(block)) is None
    ]


def _parsed_or_none(block: Block) -> Any:
    try:
        return _parse(block)
    except yaml.YAMLError:
        return None


@pytest.mark.parametrize("block", _checked(), ids=lambda block: block.where)
def test_every_yaml_block_spells_keys_the_code_reads(block: Block) -> None:
    assert _problems(block) == [], block.where


def test_the_pages_have_yaml_blocks() -> None:
    """A scanner that found nothing would pass every page."""
    assert len(_all_blocks()) > 100
    assert len(_checked()) > 100


def test_every_not_config_entry_names_a_block() -> None:
    present = {(block.page, block.name) for block in _all_blocks()}
    assert sorted(set(_NOT_CONFIG) - present) == []


@pytest.mark.parametrize(
    ("text", "parent"),
    [
        ("oidc:\n  issuer: https://sso\n", "modules.web_terminals.auth"),
        ("shared:\n  - logbook\n", "modules.web_terminals"),
        ("landing:\n  groups: []\n", "modules.web_terminals"),
        ("oidcc:\n  issuer: https://sso\n", None),
    ],
)
def test_a_fragment_is_checked_under_its_one_section(text: str, parent: str | None) -> None:
    assert _fragment_parent(_parse(Block("fake.rst", 1, text))) == parent


def test_a_misspelled_key_in_a_fragment_fails() -> None:
    [problem] = _problems(Block("fake.rst", 1, "oidc:\n  isuer: https://sso\n"))
    assert problem == "modules.web_terminals.auth.oidc.isuer is not a key the code reads"


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
            "watchdogs:\n  guard:\n    enabeld: true\n",
            ["watchdogs.guard.enabeld is not a key one watchdogs entry accepts"],
        ),
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
    assert list(_blocks(page, tmp_path)) == [
        Block("page.rst", 8, "bluesky:\n  plan_dir: plans/", "profile.yml")
    ]


def test_a_caption_and_a_file_name_comment_name_a_block_alike() -> None:
    commented = Block("p.rst", 1, "# services/x/docker-compose.yml\nservices: {}")
    captioned = Block("p.rst", 1, "services: {}", caption="services/x/docker-compose.yml")
    assert commented.name == captioned.name == "services/x/docker-compose.yml"


def test_a_jinja_template_is_not_checked() -> None:
    template = Block("p.rst", 1, "services:\n  db:\n    image: {{ services.db.image }}\n")
    assert template.is_template
    assert not Block("p.rst", 1, "web:\n  theme: dark\n").is_template
    prompt = 'ariel:\n  prompt_template: |\n    {{"keywords": []}}\n'
    assert not Block("p.rst", 1, prompt).is_template


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("code: ca\nname: Demo\n", "data/facility/identity.yaml"),
        ("records:\n  - address: X\n", "data/facility/limits.yaml"),
        ("SR:BEAM:CURRENT:\n  nominal: 400\n", "data/facility/seeds.yaml"),
        ("triggers:\n  - name: t\n", "triggers.yml"),
        ("providers:\n  local: {}\n", "providers.yml"),
        ("- id: SR\n", "a list"),
        ("SR:BEAM:CURRENT:\n  nominl: 400\n", None),
        ("name: demo\nconfig:\n  web.theme: dark\n", None),
        ("contrl_system:\n  type: epics\n", None),
        ("name: Demo\n", None),
    ],
)
def test_another_files_block_is_told_by_its_content(text: str, expected: str | None) -> None:
    """Facility, provider and trigger files are recognised by their own schemas' keys."""
    found = _other_file(_parse(Block("fake.rst", 1, text)))
    if expected is None:
        assert found is None
    else:
        assert found is not None and found.startswith(expected)


@pytest.mark.parametrize(
    "text",
    [
        "name: demo\nconfig:\n  ariel.ingestion.adaptr: x\n",
        "name: demo\nextends: control-assistant\nconfig:\n  modules.web_terminals.tls.enabeld: true\n",
        "config:\n  health:\n    titel: Beamline\n",
    ],
)
def test_a_misspelled_key_in_a_profile_config_block_still_fails(text: str) -> None:
    """No content rule takes a profile with a ``config:`` block out of the check."""
    block = Block("fake.rst", 1, text)
    assert _other_file(_parse(block)) is None
    [problem] = _problems(block)
    assert problem.startswith("config: ") and "is not a key the code reads" in problem
