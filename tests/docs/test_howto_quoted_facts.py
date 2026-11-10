"""Facts the how-to pages quote from the code are the code's own.

Each test pins one sentence, number, route, key or example on a how-to page to
the source that defines it, so the page cannot drift from the code without a
failing test here:

* the code-review profile example is a profile the loader and the build's
  code-review check accept;
* the virtual accelerator's archive store is named at its layout port, and the
  retired-``mock`` refusal is quoted word for word;
* the channel addresses the channel-finder page names exist in the preset;
* the ``mcp_servers:`` keys the add-server page names are keys the loader reads;
* every route in the Bluesky bridge route table is a route the bridge serves;
* the mya client floor, the Phoebus ``curl`` hint and the ``osprey query
  --json`` keys match their sources.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from osprey.agent_runner.write_tools import _custom_server_side_effect_tools
from osprey.cli.build_posture_check import CODE_REVIEW_HOOKS, check_code_review
from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_load import load_profile
from osprey.cli.build_profile_schema import McpServerDef
from osprey.port_layout import default_port
from osprey.services.code_review.settings import DEFAULT_PROMPT
from osprey_connectors.archiver._mya_archiver_connector import _REQUIRED_CLIENT
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.types import retired_type_message

REPO_ROOT = Path(__file__).resolve().parents[2]
HOW_TO = REPO_ROOT / "docs/source/how-to"
CODE_REVIEW_PAGE = HOW_TO / "control-systems/code-review.rst"
CODE_REVIEW_EXAMPLE = HOW_TO / "control-systems/examples/profiles/code-review.yml"
VA_PAGE = HOW_TO / "control-systems/use-virtual-accelerator.rst"
CONNECTORS_PAGE = HOW_TO / "control-systems/use-connectors.rst"
PHOEBUS_PAGE = HOW_TO / "control-systems/phoebus-bridge.rst"
CHANNEL_FINDER_PAGE = HOW_TO / "use-channel-finder.rst"
ADD_SERVER_PAGE = HOW_TO / "agent-interfaces/add-mcp-server.rst"
CLI_AGENT_PAGE = HOW_TO / "agent-interfaces/cli-agent.rst"
QUEUE_PAGE = HOW_TO / "bluesky/queue.rst"
PRESET_CHANNELS = REPO_ROOT / "src/osprey/templates/facilities/example/records/channels.yaml"
PHOEBUS_TOOLS = REPO_ROOT / "src/osprey/mcp_server/phoebus/tools/bridge_tools.py"
QUERY_KEYSET = REPO_ROOT / "tests/cli/data/json_keysets/query.json"

_HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")


def _code_block(page: Path, language: str, after: str) -> str:
    """The body of the first ``code-block:: <language>`` on *page* after the text *after*."""
    text = page.read_text()
    start = text.index(after)
    lines = text[start:].splitlines()
    directive = next(
        i for i, line in enumerate(lines) if line.strip() == f".. code-block:: {language}"
    )
    indent = len(lines[directive]) - len(lines[directive].lstrip())
    body: list[str] = []
    for line in lines[directive + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    return textwrap.dedent("\n".join(body)).strip()


# ---------------------------------------------------------------------------
# code-review.rst
# ---------------------------------------------------------------------------


@pytest.fixture
def code_review_repo(tmp_path: Path) -> Path:
    """A profile root holding the page's example, a data tree and the default prompt."""
    (tmp_path / "data").mkdir()
    prompt = tmp_path / DEFAULT_PROMPT
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text("Review this.\n", "utf-8")
    example = yaml.safe_load(CODE_REVIEW_EXAMPLE.read_text())
    profile = {"name": "code-review-example", "data": "data", **example}
    (tmp_path / "profile.yml").write_text(yaml.safe_dump(profile), "utf-8")
    return tmp_path


def test_the_code_review_page_shows_the_example_file():
    assert "literalinclude:: examples/profiles/code-review.yml" in CODE_REVIEW_PAGE.read_text()


def test_the_code_review_example_loads_and_passes_the_build_check(code_review_repo: Path):
    profile = load_profile(code_review_repo / "profile.yml")
    # The example spells dotted overrides; the build check reads the tree they address,
    # and passes silently when no ``code_review`` section is there at all.
    config = _expand_dotted(profile.config)
    assert "code_review" in config
    assert config["code_review"]["enabled"] is True
    assert set(CODE_REVIEW_HOOKS) <= set(profile.hooks)
    check_code_review(config, profile.hooks, code_review_repo)


def test_the_code_review_section_is_refused_at_the_top_level(code_review_repo: Path):
    """The guard bites: the shape the page used to show is the loader's refusal."""
    path = code_review_repo / "profile.yml"
    profile = yaml.safe_load(path.read_text())
    overrides = profile.pop("config")
    profile["code_review"] = {
        key.removeprefix("code_review."): value for key, value in overrides.items()
    }
    path.write_text(yaml.safe_dump(profile), "utf-8")
    with pytest.raises(BuildProfileError, match="Unknown profile key.*'code_review'"):
        load_profile(path)


# ---------------------------------------------------------------------------
# use-virtual-accelerator.rst
# ---------------------------------------------------------------------------


def test_the_archive_store_is_named_at_its_layout_port():
    text = " ".join(VA_PAGE.read_text().split())
    named = re.findall(r"published on host port (\d+)", text)
    assert named == [str(default_port("mongo"))]


def test_the_retired_mock_refusal_is_quoted_word_for_word():
    quoted = _code_block(VA_PAGE, "text", "is no longer a control-system type")
    assert quoted == retired_type_message("mock")


# ---------------------------------------------------------------------------
# use-channel-finder.rst
# ---------------------------------------------------------------------------


def test_the_channel_finder_example_addresses_exist_in_the_preset():
    intro = CHANNEL_FINDER_PAGE.read_text().split(".. seealso::", 1)[0]
    addresses = re.findall(r"``([A-Z]+(?::[A-Z0-9_]+)+)``", intro)
    assert addresses, "the page names no example address"
    channels = PRESET_CHANNELS.read_text()
    missing = [address for address in addresses if f"- id: {address}\n" not in channels]
    assert missing == []


# ---------------------------------------------------------------------------
# add-mcp-server.rst
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["hooks", "writes"])
def test_the_server_keys_the_page_names_are_keys_the_loader_reads(key: str):
    assert f"``{key}:``" in ADD_SERVER_PAGE.read_text()
    assert key in {field.name for field in fields(McpServerDef)}


def test_a_read_only_run_blocks_the_tools_a_server_lists_under_writes(tmp_path: Path):
    """cli-agent.rst: a custom server's ``writes:`` tools are in the read-only block list."""
    assert "lists under ``writes:``" in " ".join(CLI_AGENT_PAGE.read_text().split())
    config = {
        "claude_code": {
            "servers": {
                "my-server": {
                    "command": "my-server",
                    "permissions": {"allow": ["read_data", "set_value"]},
                    "writes": ["set_value"],
                }
            }
        }
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config), "utf-8")
    blocked = _custom_server_side_effect_tools(tmp_path)
    assert "mcp__my-server__set_value" in blocked
    assert "mcp__my-server__read_data" not in blocked


# ---------------------------------------------------------------------------
# bluesky/queue.rst
# ---------------------------------------------------------------------------


def _table_routes() -> list[tuple[str, str]]:
    """(method, path) for every route the queue page's HTTP API table names."""
    table = _code_block(QUEUE_PAGE, "text", "The panels and the agent both talk to")
    routes: list[tuple[str, str]] = []
    for line in table.splitlines():
        words = line.split()
        if words and words[0] in _HTTP_METHODS:
            routes.append((words[0], words[1]))
        # A description that names a sub-resource of /runs/<id> names a GET.
        routes.extend(("GET", path) for path in re.findall(r"(?<!\S)(/runs/<id>/\w+)", line))
    return routes


def _normalized(path: str) -> str:
    return re.sub(r"<[^>]+>|\{[^}]+\}", "{}", path)


def test_every_route_in_the_queue_table_is_one_the_bridge_serves():
    from osprey.services.bluesky_bridge.app import app

    served = {
        (method.upper(), _normalized(path))
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    }
    routes = _table_routes()
    assert ("POST", "/queue/items/<uid>/move") in routes
    assert ("GET", "/runs/<id>/export") in routes
    unserved = [f"{m} {p}" for m, p in routes if (m, _normalized(p)) not in served]
    assert unserved == []


# ---------------------------------------------------------------------------
# Quoted values on other pages
# ---------------------------------------------------------------------------


def test_the_mya_client_floor_is_the_connectors_floor():
    floors = re.findall(r"jlab-archiver-client>=([\d.]+)", CONNECTORS_PAGE.read_text())
    assert floors == [_REQUIRED_CLIENT]


def test_the_phoebus_curl_line_is_the_servers_hint():
    command = _code_block(PHOEBUS_PAGE, "bash", "the server's own error message").strip()
    hint = re.search(r"e\.g\. (curl [^\s\"]+?)\.?\"", PHOEBUS_TOOLS.read_text())
    assert hint is not None
    assert command == hint.group(1)


def test_the_query_json_example_has_the_recorded_keys():
    example = json.loads(_code_block(CLI_AGENT_PAGE, "json", "Output format:"))
    paths = list(example)
    paths.extend(f"tool_traces[].{key}" for key in example["tool_traces"][0])
    assert sorted(paths) == json.loads(QUERY_KEYSET.read_text())
