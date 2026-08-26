"""Tests for the shared write-gate helpers in ``osprey_hook_log``.

The write gates ask two separate questions and these helpers answer one each.
``is_write_tool(tool_name, write_tools)`` asks whether a tool is *covered* by
the generated ``write_tools`` list, matching an entry exactly or, when the entry
ends in ``.*``, on the prefix before it — the spelling ``settings.json``
PreToolUse matchers use for a self-gated MCP server, carried verbatim into
``hook_config.json``. ``is_write_call(tool_name, tool_input, short_name)`` asks
whether the call in hand actually writes, which only the python server's
``execute`` can answer with "no": a ``readonly`` execution_mode — or a missing
one, the server's default — is not a write. It matches on the SHORT name, so an
``extends`` clone of that server keeps the carve-out. Both degrade rather than
raise on malformed input.

``write_tools()`` and ``short_tool_name(tool_name, prefixes)`` are the other
half: the list the gates read, with the fail-closed fallback applied in ONE
place, and the prefix strip that turns a full tool name into the short name a
per-tool rule is written against. Both live here because more than one hook asks
them, and two hooks answering "which tools are writes" or "what is this tool
called" differently is one of them gating a call the other waves through.
"""

import json

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_hook_log as hook_log

# ---------------------------------------------------------------------------
# is_write_call
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execute_readonly_is_not_a_write_call():
    """A readonly python execution writes nothing."""
    assert hook_log.is_write_call("mcp__python__execute", {"execution_mode": "readonly"}) is False


@pytest.mark.unit
def test_execute_readwrite_is_a_write_call():
    """A non-readonly execution_mode is a write."""
    assert hook_log.is_write_call("mcp__python__execute", {"execution_mode": "readwrite"}) is True


@pytest.mark.unit
def test_execute_without_mode_is_not_a_write_call():
    """A missing execution_mode reads as readonly, matching the server default."""
    assert hook_log.is_write_call("mcp__python__execute", {"code": "print(1)"}) is False


@pytest.mark.unit
def test_non_execute_tool_is_always_a_write_call():
    """Every other tool writes whenever it is called, whatever its arguments."""
    assert hook_log.is_write_call("mcp__controls__channel_write", {}) is True
    assert (
        hook_log.is_write_call("mcp__controls__channel_write", {"execution_mode": "readonly"})
        is True
    )


@pytest.mark.unit
@pytest.mark.parametrize("tool_input", [None, "readonly", ["execution_mode"], 7])
def test_execute_with_non_mapping_input_is_not_a_write_call(tool_input):
    """A tool_input that is not a mapping is read as an empty one, so: readonly."""
    assert hook_log.is_write_call("mcp__python__execute", tool_input) is False


@pytest.mark.unit
def test_a_cloned_python_server_keeps_the_readonly_carve_out():
    """`mcp__pyva__execute` is the same tool as `mcp__python__execute`.

    A deployment may clone the python server through `extends` to point one copy
    at a second target. Reading the carve-out off the full name would make every
    readonly execution on the clone a write, which the gates then refuse.
    """
    assert hook_log.is_write_call("mcp__pyva__execute", {"execution_mode": "readonly"}) is False
    assert hook_log.is_write_call("mcp__pyva__execute", {"execution_mode": "readwrite"}) is True


@pytest.mark.unit
def test_a_supplied_short_name_decides_the_carve_out():
    """A caller that already stripped the prefix hands its answer in.

    Server prefixes are generated per render and can carry `__` themselves,
    which the bare `mcp__<server>__<tool>` split would mis-strip. The hooks
    resolve the short name against their own prefix list and pass it.
    """
    assert (
        hook_log.is_write_call(
            "mcp__osprey__python__execute", {"execution_mode": "readonly"}, "execute"
        )
        is False
    )
    assert hook_log.is_write_call("mcp__python__execute", {}, "channel_write") is True


# ---------------------------------------------------------------------------
# is_write_tool
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_exact_entry_matches():
    """An entry equal to the tool name covers it."""
    write_tools = ["mcp__controls__channel_write", "mcp__python__execute"]
    assert hook_log.is_write_tool("mcp__controls__channel_write", write_tools) is True


@pytest.mark.unit
def test_exact_entry_does_not_match_another_tool():
    """An exact entry covers nothing but itself."""
    write_tools = ["mcp__controls__channel_write"]
    assert hook_log.is_write_tool("mcp__controls__channel_read", write_tools) is False


@pytest.mark.unit
def test_wildcard_entry_matches_on_prefix():
    """``mcp__myserver__.*`` covers every tool on that server."""
    write_tools = ["mcp__myserver__.*"]
    assert hook_log.is_write_tool("mcp__myserver__set_current", write_tools) is True


@pytest.mark.unit
def test_wildcard_entry_does_not_match_a_different_prefix():
    """The prefix has to match; a neighbouring server is not covered."""
    write_tools = ["mcp__myserver__.*"]
    assert hook_log.is_write_tool("mcp__otherserver__set_current", write_tools) is False


@pytest.mark.unit
def test_wildcard_entry_matches_the_bare_prefix_itself():
    """``foo.*`` covers ``foo``: the rule is startswith, and ``.*`` matches empty."""
    assert hook_log.is_write_tool("foo", ["foo.*"]) is True


@pytest.mark.unit
def test_non_string_entries_are_ignored():
    """A malformed entry is skipped, and the valid entries still match."""
    write_tools = [None, 42, {"tool": "mcp__python__execute"}, "mcp__python__execute"]
    assert hook_log.is_write_tool("mcp__python__execute", write_tools) is True
    assert hook_log.is_write_tool("mcp__controls__channel_write", write_tools) is False


@pytest.mark.unit
@pytest.mark.parametrize("write_tools", [[], None])
def test_empty_write_tools_covers_nothing(write_tools):
    """With no entries there is nothing to match."""
    assert hook_log.is_write_tool("mcp__python__execute", write_tools) is False


# ---------------------------------------------------------------------------
# write_tools
# ---------------------------------------------------------------------------


@pytest.fixture
def hook_config_file(tmp_path, monkeypatch):
    """Point ``load_hook_config`` at a hook_config.json this test writes."""

    def _write(payload):
        path = tmp_path / "hook_config.json"
        path.write_text(json.dumps(payload))
        monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(path))
        return path

    return _write


@pytest.mark.unit
def test_write_tools_returns_the_generated_list(hook_config_file):
    """A rendered deployment's own matchers, including a self-gated server."""
    hook_config_file({"write_tools": ["mcp__controls__channel_write", "mcp__bluesky__.*"]})

    assert hook_log.write_tools() == ["mcp__controls__channel_write", "mcp__bluesky__.*"]


@pytest.mark.unit
def test_write_tools_falls_back_when_the_key_is_absent(hook_config_file):
    """A hook_config with no ``write_tools`` key gets the framework two."""
    hook_config_file({"server_prefixes": ["mcp__controls__"]})

    assert hook_log.write_tools() == [
        "mcp__controls__channel_write",
        "mcp__python__execute",
    ]


@pytest.mark.unit
def test_write_tools_falls_back_when_the_file_is_missing(tmp_path, monkeypatch):
    """No hook_config at all is fail-closed, not "gate nothing"."""
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(tmp_path / "absent.json"))

    assert hook_log.write_tools() == hook_log.FALLBACK_WRITE_TOOLS


@pytest.mark.unit
def test_an_explicitly_empty_list_is_taken_at_its_word(hook_config_file):
    """A generated empty list gates nothing — the renderer lint owns that case.

    ``_lint_write_tools_are_gated`` refuses to emit a render whose write tools
    reach no gate, so an empty list here is a deployment that has none rather
    than a file this could not read. Substituting the fallback would gate tools
    the deployment does not have.
    """
    hook_config_file({"write_tools": []})

    assert hook_log.write_tools() == []


# ---------------------------------------------------------------------------
# short_tool_name
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_short_tool_name_strips_a_matching_prefix():
    """The ordinary case: a server prefix comes off and the tool name is left."""
    assert (
        hook_log.short_tool_name("mcp__controls__channel_write", ["mcp__controls__"])
        == "channel_write"
    )


@pytest.mark.unit
def test_the_longest_matching_prefix_wins():
    """One server prefix can be a prefix of another, and the shorter one lies.

    Stripping ``mcp__bluesky__`` off a ``mcp__bluesky_va__`` tool would leave
    ``va__queue_start``, which matches no per-tool rule anyone wrote.
    """
    prefixes = ["mcp__bluesky__", "mcp__bluesky_va__"]

    assert hook_log.short_tool_name("mcp__bluesky_va__queue_start", prefixes) == "queue_start"


@pytest.mark.unit
def test_an_unlisted_server_falls_back_to_the_mcp_shape():
    """A tool from a server no prefix list carries is still an MCP tool name."""
    assert hook_log.short_tool_name("mcp__other__do_thing", ["mcp__controls__"]) == "do_thing"


@pytest.mark.unit
def test_a_tool_name_in_no_mcp_shape_is_its_own_short_name():
    """A built-in tool has no server to strip."""
    assert hook_log.short_tool_name("Bash", ["mcp__controls__"]) == "Bash"


@pytest.mark.unit
@pytest.mark.parametrize("prefixes", [[], None, [None, 42]])
def test_short_tool_name_survives_a_useless_prefix_list(prefixes):
    """No usable prefixes leaves the MCP-shape fallback to answer."""
    assert hook_log.short_tool_name("mcp__controls__channel_write", prefixes) == "channel_write"
