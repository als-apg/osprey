"""Tests for :func:`_parse_profile` — the raw YAML dict to ``BuildProfile`` contract.

Pins the fail-fast branches of the per-block parse loops: a malformed block is
never silently skipped or coerced, it raises
:class:`~osprey.errors.BuildProfileError` naming the offending block (and, for
the keyed ``mcp_servers:``/``services:`` loops, the offending entry) so a
profile author reads which key to fix out of the message itself. Covered here:
an ``mcp_servers:`` entry that is not a mapping, an ``mcp_servers:`` entry that
declares neither transport (``command`` nor ``url``), both halves of the
``services:`` loop (the non-mapping raise and the ``ServiceDef`` it builds from
a valid mapping), the non-mapping guards on the optional ``bluesky:`` and
``virtual_accelerator:`` blocks, and the ``virtual_accelerator:`` block's closed
key set and ``live_standin`` typing.

Complements the block-scoped suites that exercise the *accepted* shapes:
``test_bluesky_service_registration.py`` for ``bluesky:`` field defaults and
``excluded_plans`` typing, ``test_build_profile.py`` for ``bluesky_web:``,
and ``test_profile_schema.py`` for ``dispatch:``.
"""

from __future__ import annotations

import pytest

from osprey.cli.build_profile import ServiceDef, _parse_profile
from osprey.errors import BuildProfileError

# ── mcp_servers: ─────────────────────────────────────────────────────────────


def test_mcp_server_entry_must_be_a_mapping() -> None:
    """A non-mapping mcp_servers entry raises, naming the server."""
    with pytest.raises(BuildProfileError, match="MCP server 'broken' must be a mapping"):
        _parse_profile({"name": "x", "mcp_servers": {"broken": "not-a-mapping"}})


def test_mcp_server_entry_list_value_must_be_a_mapping() -> None:
    """A list-valued mcp_servers entry is rejected the same way a string one is."""
    with pytest.raises(BuildProfileError, match="MCP server 'listy' must be a mapping"):
        _parse_profile({"name": "x", "mcp_servers": {"listy": ["osprey", "serve"]}})


def test_mcp_server_without_command_or_url_raises() -> None:
    """An mcp_servers entry declaring no transport at all raises."""
    with pytest.raises(
        BuildProfileError, match="MCP server 'bare' must have either 'command' or 'url'"
    ):
        _parse_profile({"name": "x", "mcp_servers": {"bare": {}}})


def test_mcp_server_with_only_permissions_raises() -> None:
    """Permissions alone are not a transport — the no-command-no-url raise still fires."""
    raw = {"name": "x", "mcp_servers": {"perms-only": {"permissions": {"allow": ["Read"]}}}}
    with pytest.raises(
        BuildProfileError, match="MCP server 'perms-only' must have either 'command' or 'url'"
    ):
        _parse_profile(raw)


def test_mcp_server_with_empty_command_and_url_raises() -> None:
    """Explicit empty ``command``/``url`` strings are falsy and raise like omitting them."""
    with pytest.raises(
        BuildProfileError, match="MCP server 'empty' must have either 'command' or 'url'"
    ):
        _parse_profile({"name": "x", "mcp_servers": {"empty": {"command": "", "url": ""}}})


def test_mcp_server_transport_defaults_to_http() -> None:
    """A URL server without a transport key parses as streamable-HTTP."""
    profile = _parse_profile({"name": "x", "mcp_servers": {"api": {"url": "http://host:9000/mcp"}}})
    assert profile.mcp_servers["api"].transport == "http"


def test_mcp_server_transport_sse_parses() -> None:
    """An explicit transport: sse with a url is accepted and carried through."""
    profile = _parse_profile(
        {
            "name": "x",
            "mcp_servers": {"legacy": {"transport": "sse", "url": "http://host:9000/sse"}},
        }
    )
    assert profile.mcp_servers["legacy"].transport == "sse"


def test_mcp_server_invalid_transport_raises() -> None:
    """A transport outside http/sse raises, naming the bad value."""
    with pytest.raises(
        BuildProfileError, match="MCP server 'typo' transport must be 'http' or 'sse'"
    ):
        _parse_profile(
            {
                "name": "x",
                "mcp_servers": {"typo": {"transport": "ssse", "url": "http://host:9000/mcp"}},
            }
        )


def test_mcp_server_transport_with_command_raises() -> None:
    """Stdio servers have no transport choice — declaring one is rejected."""
    with pytest.raises(
        BuildProfileError, match="MCP server 'local' declares 'transport' with 'command'"
    ):
        _parse_profile(
            {
                "name": "x",
                "mcp_servers": {"local": {"transport": "http", "command": "uvx"}},
            }
        )


def test_mcp_server_sse_requires_explicit_url() -> None:
    """transport: sse with only a port raises — the derived URL is an /mcp endpoint."""
    with pytest.raises(
        BuildProfileError, match="MCP server 'legacy' transport 'sse' requires an explicit 'url'"
    ):
        _parse_profile({"name": "x", "mcp_servers": {"legacy": {"transport": "sse", "port": 9000}}})


# ── services: ────────────────────────────────────────────────────────────────


def test_service_entry_must_be_a_mapping() -> None:
    """A non-mapping services entry raises, naming the service."""
    with pytest.raises(BuildProfileError, match="Service 'postgresql' must be a mapping"):
        _parse_profile({"name": "x", "services": {"postgresql": "not-a-mapping"}})


def test_service_entry_builds_a_servicedef() -> None:
    """A valid services mapping becomes a ServiceDef carrying template and config."""
    raw = {
        "name": "x",
        "services": {
            "postgresql": {
                "template": "services/postgresql",
                "config": {"port": 5432, "database": "ariel"},
            }
        },
    }
    profile = _parse_profile(raw)
    assert list(profile.services) == ["postgresql"]
    service = profile.services["postgresql"]
    assert isinstance(service, ServiceDef)
    assert service.template == "services/postgresql"
    assert service.config == {"port": 5432, "database": "ariel"}


def test_service_entry_defaults_when_empty_mapping() -> None:
    """An empty services mapping parses to an empty template and an empty config."""
    profile = _parse_profile({"name": "x", "services": {"bare": {}}})
    assert profile.services["bare"] == ServiceDef(template="", config={})


def test_multiple_service_entries_are_all_parsed() -> None:
    """Every services entry is kept, keyed by its profile name."""
    raw = {
        "name": "x",
        "services": {
            "postgresql": {"template": "services/postgresql"},
            "openobserve": {"template": "services/openobserve", "config": {"port": 5080}},
        },
    }
    profile = _parse_profile(raw)
    assert sorted(profile.services) == ["openobserve", "postgresql"]
    assert profile.services["openobserve"].config == {"port": 5080}


def test_no_services_block_parses_to_empty_dict() -> None:
    """A profile without a services block gets an empty services mapping."""
    assert _parse_profile({"name": "x"}).services == {}


# ── bluesky: / virtual_accelerator: ──────────────────────────────────────────


def test_bluesky_not_a_mapping_raises() -> None:
    """A non-mapping 'bluesky' block raises during parsing."""
    with pytest.raises(BuildProfileError, match="Profile 'bluesky' must be a mapping"):
        _parse_profile({"name": "x", "bluesky": "not-a-mapping"})


def test_virtual_accelerator_not_a_mapping_raises() -> None:
    """A non-mapping 'virtual_accelerator' block raises during parsing."""
    with pytest.raises(BuildProfileError, match="Profile 'virtual_accelerator' must be a mapping"):
        _parse_profile({"name": "x", "virtual_accelerator": "not-a-mapping"})


def test_virtual_accelerator_unknown_key_raises_with_suggestion() -> None:
    """A misspelled VA subkey is named, corrected and rejected rather than dropped."""
    with pytest.raises(
        BuildProfileError,
        match=r"Unknown virtual_accelerator key\(s\): 'live_standing' "
        r"\(did you mean 'live_standin'\?\)",
    ):
        _parse_profile({"name": "x", "virtual_accelerator": {"live_standing": 5074}})


def test_virtual_accelerator_live_standin_parses() -> None:
    """A live_standin port lands on the parsed VAConfig."""
    profile = _parse_profile({"name": "x", "virtual_accelerator": {"live_standin": 5074}})
    assert profile.virtual_accelerator is not None
    assert profile.virtual_accelerator.live_standin == 5074


def test_virtual_accelerator_without_live_standin_parses_to_none() -> None:
    """Omitting live_standin leaves the deployment with no stand-in lane."""
    profile = _parse_profile({"name": "x", "virtual_accelerator": {"port": 5064}})
    assert profile.virtual_accelerator is not None
    assert profile.virtual_accelerator.live_standin is None


def test_virtual_accelerator_live_standin_must_be_an_int() -> None:
    """A non-integer live_standin raises rather than reaching port validation."""
    with pytest.raises(
        BuildProfileError,
        match="virtual_accelerator.live_standin must be a Channel Access port number "
        r"\(got '5074'\)",
    ):
        _parse_profile({"name": "x", "virtual_accelerator": {"live_standin": "5074"}})


def test_virtual_accelerator_live_standin_rejects_a_boolean() -> None:
    """`live_standin: true` names no port, so it is not a port number either."""
    with pytest.raises(
        BuildProfileError,
        match="virtual_accelerator.live_standin must be a Channel Access port number "
        r"\(got True\)",
    ):
        _parse_profile({"name": "x", "virtual_accelerator": {"live_standin": True}})
