"""The facility-knowledge pages hold to the code they describe.

Three pages state facts a reader acts on, and each fact is owned by code:

* ``how-to/facility-knowledge/use-facility-graph.rst`` shows the Cypher that
  initialises neosemantics on an external store. Its ``n10s.graphconfig.init``
  settings and its uniqueness constraint are the ones the graph seeder issues
  (``N10S_GRAPH_CONFIG``, ``_CONSTRAINT_CYPHER``), so a store a reader prepares
  by hand holds the graph OSPREY's queries are written against.
* ``reference/contracts/facility-graph.rst`` numbers the ``graph`` server's
  tools, and ``use-facility-graph.rst`` names them in prose; both equal the
  tools the server registers, and "four" on both pages is that count.
* ``reference/contracts/facility-graph.rst`` names places, the ``PARTOF`` and
  ``LOCATEDIN`` relationships the graph view writes (``P_PART_OF``,
  ``P_LOCATED_IN``, uppercased by neosemantics), and quotes
  ``RECORD_NAME_MATCH_RULE`` verbatim; no page under ``docs/source`` still
  names the panel's retired derived ``inPlace`` relation.
* ``how-to/facility-knowledge/facility-rules.rst`` names a ``rules:`` entry
  and ``osprey scaffold`` targets, each of which is a build-artifact catalog
  name.

The view sizes ``use-facility-graph.rst`` states are pinned in
``tests/facility/golden/demo_census.json``, which
``tests/facility/test_demo_census.py`` holds against the page; the last test
here checks that those page entries agree with the census' own counts.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "source"
GRAPH_PAGE = DOCS / "how-to" / "facility-knowledge" / "use-facility-graph.rst"
CONTRACT_PAGE = DOCS / "reference" / "contracts" / "facility-graph.rst"
RULES_PAGE = DOCS / "how-to" / "facility-knowledge" / "facility-rules.rst"
CENSUS = REPO_ROOT / "tests" / "facility" / "golden" / "demo_census.json"

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _code_blocks(text: str, language: str) -> list[str]:
    """The dedented body of every ``.. code-block:: <language>`` in ``text``."""
    blocks = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != f".. code-block:: {language}":
            continue
        indent = len(line) - len(line.lstrip())
        body: list[str] = []
        for following in lines[index + 1 :]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            body.append(following)
        blocks.append(textwrap.dedent("\n".join(body)).strip("\n"))
    return blocks


def _section(text: str, heading: str) -> str:
    """The body under ``heading`` (a ``===`` title) up to the next such title."""
    match = re.search(rf"^{re.escape(heading)}\n=+\n", text, re.M)
    assert match, f"no section {heading!r}"
    rest = text[match.end() :]
    following = re.search(r"^\S.*\n=+\n", rest, re.M)
    return rest[: following.start()] if following else rest


# --- the external-store Cypher ---------------------------------------------------


def _cypher_value(literal: str) -> Any:
    literal = literal.strip()
    if literal in {"true", "false"}:
        return literal == "true"
    if literal.startswith("["):
        return [item.strip().strip("'") for item in literal[1:-1].split(",") if item.strip()]
    if literal.startswith("'") and literal.endswith("'"):
        return literal[1:-1]
    raise ValueError(f"unexpected Cypher literal {literal!r}")


def graphconfig_init_params(cypher: str) -> dict[str, Any]:
    """The map literal the block passes to ``n10s.graphconfig.init``."""
    match = re.search(r"n10s\.graphconfig\.init\(\{(.*?)\}\)", cypher, re.S)
    assert match, "no n10s.graphconfig.init({...}) call in the block"
    pairs = re.findall(r"(\w+):\s*(\[[^\]]*\]|'[^']*'|true|false)", match.group(1))
    return {key: _cypher_value(value) for key, value in pairs}


def constraint_statement(cypher: str) -> str:
    """The block's ``CREATE CONSTRAINT`` statement, on one line, without its ``;``."""
    match = re.search(r"CREATE CONSTRAINT.*?;", cypher, re.S)
    assert match, "no CREATE CONSTRAINT statement in the block"
    return " ".join(match.group(0).rstrip(";").split())


def _page_cypher() -> str:
    """The external-store setup block: the one Cypher block that initialises n10s."""
    blocks = _code_blocks(GRAPH_PAGE.read_text(encoding="utf-8"), "cypher")
    (block,) = [block for block in blocks if "n10s.graphconfig.init" in block]
    return block


def test_the_page_initialises_n10s_with_the_seeder_settings() -> None:
    from osprey.services.graph_store import N10S_GRAPH_CONFIG

    assert graphconfig_init_params(_page_cypher()) == N10S_GRAPH_CONFIG


def test_the_page_creates_the_constraint_the_seeder_creates() -> None:
    from osprey.services.graph_store.seeder import _CONSTRAINT_CYPHER

    assert constraint_statement(_page_cypher()) == _CONSTRAINT_CYPHER


def test_a_mutated_cypher_block_is_caught() -> None:
    from osprey.services.graph_store import N10S_GRAPH_CONFIG
    from osprey.services.graph_store.seeder import _CONSTRAINT_CYPHER

    cypher = _page_cypher()
    assert graphconfig_init_params(cypher.replace("'ARRAY'", "'OVERWRITE'")) != N10S_GRAPH_CONFIG
    assert graphconfig_init_params(cypher.replace("keepLangTag", "keepLang")) != N10S_GRAPH_CONFIG
    assert constraint_statement(cypher.replace("r.uri", "r.id")) != _CONSTRAINT_CYPHER


# --- the graph server's tools ----------------------------------------------------


def _registered_graph_tools() -> set[str]:
    """Every tool the ``graph`` server registers, read as ``create_server()`` leaves it.

    The tool modules ``create_server()`` imports are read from its own source and
    imported directly, so no config or store is touched.
    """
    from osprey.mcp_server.graph import server, tools

    (function,) = (
        node
        for node in ast.parse(inspect.getsource(server)).body
        if isinstance(node, ast.FunctionDef) and node.name == "create_server"
    )
    modules = [
        alias.name
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom) and node.module == tools.__name__
        for alias in node.names
    ]
    assert modules, "create_server() imports no tool module"
    for module in modules:
        importlib.import_module(f"{tools.__name__}.{module}")
    listed = asyncio.run(server.mcp.list_tools(run_middleware=False))
    return {tool.name for tool in listed}


_NUMBERED_TOOL = re.compile(r"^\d+\. ``([a-z_]+)`` —", re.M)


def numbered_tools(page: str) -> list[str]:
    return _NUMBERED_TOOL.findall(_section(page, "The Four Tools"))


def prose_tools(page: str) -> set[str]:
    """The tool names the "Leaving the Graph Tools Out" section's prose uses."""
    section = _section(page, "Leaving the Graph Tools Out")
    prose = "\n".join(_strip_code_blocks(section))
    return set(re.findall(r"``([a-z_]+)``", prose))


def _strip_code_blocks(text: str) -> list[str]:
    kept, skipping = [], False
    for line in text.splitlines():
        if line.startswith(".. code-block::"):
            skipping = True
            continue
        if skipping and line.strip() and not line.startswith(" "):
            skipping = False
        if not skipping:
            kept.append(line)
    return kept


def tool_counts(page: str) -> list[int]:
    """Every "<number> tools" / "<number> ``graph`` MCP tools" count on ``page``."""
    words = "|".join(_NUMBER_WORDS)
    pattern = re.compile(rf"\b({words})\s+(?:``graph``\s+MCP\s+)?tools\b", re.I)
    return [_NUMBER_WORDS[word.lower()] for word in pattern.findall(" ".join(page.split()))]


def test_the_contract_numbers_exactly_the_registered_tools() -> None:
    listed = numbered_tools(CONTRACT_PAGE.read_text(encoding="utf-8"))
    registered = _registered_graph_tools()
    assert len(listed) == len(set(listed)), listed
    assert set(listed) == registered, (
        f"missing from the page: {sorted(registered - set(listed))}; "
        f"on the page but not registered: {sorted(set(listed) - registered)}"
    )


def test_the_how_to_names_exactly_the_registered_tools() -> None:
    named = prose_tools(GRAPH_PAGE.read_text(encoding="utf-8"))
    assert named == _registered_graph_tools()


def test_get_schema_named_in_the_query_gate_is_a_graph_tool() -> None:
    section = _section(GRAPH_PAGE.read_text(encoding="utf-8"), "Read-Only by Construction")
    assert "``get_schema`` itself uses" in section
    assert "get_schema" in _registered_graph_tools()


def test_search_channels_is_the_channel_finder_s_tool_not_the_graph_server_s() -> None:
    """The page sends address lookups to ``search_channels``, which only the graph pipeline has."""
    from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE

    assert "``search_channels`` tool" in GRAPH_PAGE.read_text(encoding="utf-8")
    assert "search_channels" in CHANNEL_FINDER_TOOLS_BY_PIPELINE["graph"]
    assert "search_channels" not in _registered_graph_tools()


@pytest.mark.parametrize("page", [GRAPH_PAGE, CONTRACT_PAGE], ids=lambda path: path.name)
def test_every_stated_tool_count_is_the_registered_count(page: Path) -> None:
    counts = tool_counts(page.read_text(encoding="utf-8"))
    assert counts, f"{page.name} states no tool count"
    assert set(counts) == {len(_registered_graph_tools())}, counts


def test_a_renamed_or_miscounted_tool_is_caught() -> None:
    contract = CONTRACT_PAGE.read_text(encoding="utf-8")
    renamed = contract.replace("``read_cypher`` —", "``run_cypher`` —")
    assert set(numbered_tools(renamed)) != _registered_graph_tools()
    how_to = GRAPH_PAGE.read_text(encoding="utf-8")
    assert prose_tools(how_to.replace("``example_queries``", "``examples``")) != (
        _registered_graph_tools()
    )
    assert tool_counts("the same five tools, and the four ``graph`` MCP tools") == [5, 4]


# --- places, their relations and record names ------------------------------------


def _contract_holds_places(page: str) -> list[str]:
    """What the contract page fails to state about places and record names."""
    from osprey.facility.views.graph import P_LOCATED_IN, P_PART_OF
    from osprey.services.graph_store.prompt_snapshot import RECORD_NAME_MATCH_RULE

    holds = _section(page, "What the Graph Holds")
    wanted = {
        "a Places entry": "* **Places** —",
        "PARTOF": f"``{P_PART_OF.upper()}``",
        "LOCATEDIN": f"``{P_LOCATED_IN.upper()}``",
        "the record-name rule": RECORD_NAME_MATCH_RULE,
    }
    return [name for name, text in wanted.items() if text not in holds]


def test_the_contract_names_places_their_relations_and_the_record_name_rule() -> None:
    assert _contract_holds_places(CONTRACT_PAGE.read_text(encoding="utf-8")) == []


def test_the_kinds_count_is_the_kinds_listed() -> None:
    holds = _section(CONTRACT_PAGE.read_text(encoding="utf-8"), "What the Graph Holds")
    (word,) = re.findall(r"\b(\w+) kinds of thing", " ".join(holds.split()))
    kinds = re.findall(r"^\* \*\*[A-Z][\w ]+\*\* —", holds, re.M)
    assert _NUMBER_WORDS[word.lower()] == len(kinds), (word, kinds)


def _in_place_mentions() -> list[str]:
    return [
        path.relative_to(DOCS).as_posix()
        for path in sorted(DOCS.rglob("*"))
        if path.suffix in {".rst", ".md", ".txt", ".py", ".svg"}
        and "inPlace" in path.read_text(encoding="utf-8", errors="replace")
    ]


def test_no_page_names_the_retired_derived_in_place_relation() -> None:
    assert "is marked *derived*" not in GRAPH_PAGE.read_text(encoding="utf-8")
    assert _in_place_mentions() == []


def test_a_dropped_relation_rule_or_stale_in_place_is_caught() -> None:
    from osprey.services.graph_store.prompt_snapshot import RECORD_NAME_MATCH_RULE

    page = CONTRACT_PAGE.read_text(encoding="utf-8")
    assert _contract_holds_places(page.replace("``LOCATEDIN``", "``LOCATED_IN``")) == ["LOCATEDIN"]
    assert _contract_holds_places(page.replace(RECORD_NAME_MATCH_RULE, "")) == [
        "the record-name rule"
    ]
    assert "inPlace" not in "ordinalInPlace"


# --- facility-rules names --------------------------------------------------------


def _catalog_names() -> set[str]:
    from osprey.agent_runner.build_artifacts.catalog import BuildArtifactCatalog

    return {artifact.canonical_name for artifact in BuildArtifactCatalog.default().all_artifacts()}


def _rule_names(page: str) -> list[str]:
    from osprey.cli.templates.manifest import _MANIFEST_CATEGORY_PREFIX

    names = []
    for block in _code_blocks(page, "yaml"):
        selection = yaml.safe_load(block)
        names += [f"{_MANIFEST_CATEGORY_PREFIX['rules']}{rule}" for rule in selection["rules"]]
    return names


def _scaffold_targets(page: str) -> list[str]:
    """Every artifact an ``osprey scaffold <verb> <name>`` on the page names."""
    flat = " ".join(page.replace("``", "").split())
    return re.findall(r"osprey scaffold (?:claim|unclaim|diff) ([a-z][\w/-]*)", flat)


def test_the_rules_list_names_catalog_rules() -> None:
    names = _rule_names(RULES_PAGE.read_text(encoding="utf-8"))
    assert names == ["rules/test-ioc-safety"]
    assert set(names) <= _catalog_names()


def test_every_scaffold_target_is_a_catalog_name() -> None:
    page = RULES_PAGE.read_text(encoding="utf-8")
    targets = _scaffold_targets(page)
    assert {"rules/safety", "output-styles/control-operator"} <= set(targets)
    assert set(targets) <= _catalog_names(), sorted(set(targets) - _catalog_names())
    assert "(``claude-md``)" in page
    assert "claude-md" in _catalog_names()


def test_a_misspelled_rule_or_target_is_caught() -> None:
    page = RULES_PAGE.read_text(encoding="utf-8")
    assert not set(_rule_names(page.replace("- test-ioc-safety", "- test-ioc"))) <= (
        _catalog_names()
    )
    assert not set(_scaffold_targets("osprey scaffold claim rules/saftey")) <= _catalog_names()


# --- the census entries for the graph page -------------------------------------


def test_the_page_census_agrees_with_the_census_counts() -> None:
    """The view sizes pinned on the page are the census' own counts, not copies of a guess."""
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    facility = census["facility"]
    page = GRAPH_PAGE.relative_to(REPO_ROOT).as_posix()
    stated = {e["pin"]: e["value"] for e in census["entries"] if e["file"] == page}
    by_pin = {(e["file"], e["pin"]): e["value"] for e in census["entries"]}
    middle_layer = by_pin[
        ("tests/facility/test_cf_index_middle_layer.py", "text:assert counts == (8297, {value})")
    ]
    in_context = by_pin[
        (
            "tests/facility/test_cf_index_in_context.py",
            "text:569 + len(IN_CONTEXT_ADDITIONS) == {value}",
        )
    ]
    assert stated == {
        "text:Graph view (``data/graph/facility.ttl``)\n     - {value}": facility["channels"],
        "text:written from the facility file: {value} devices": facility["devices"],
        "text:the {value} setpoint channels are": facility["setpoints"],
        "text:hierarchical and graph builds\n     - {value}": facility["channels"],
        "text:middle-layer build\n     - {value}": middle_layer,
        "text:in-context index\n     - {value}": in_context,
    }
