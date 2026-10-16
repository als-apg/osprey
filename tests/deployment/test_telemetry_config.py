"""The telemetry store and gateway as the config states them.

Three things are pinned here: every key resolves to its default when unset and
refuses an unusable value by name; the two memory formulas compute what their
terms say; and the topology rule gives the address that is right from where a
client actually runs — proven by dialling it through a canned transport, since
an address is only correct if a request sent to it arrives.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from osprey.deployment import telemetry_config as tc
from osprey.deployment.host_ports import _SERVICE_LAYOUT_SLOTS, derive_reader_origin_bindings
from osprey.port_layout import LAYOUT, SLOTS_BY_NAME, default_port

MIB = 1024 * 1024

DEPLOYED = {"deployed_services": ["telemetry_store", "telemetry_gateway"]}


def _config(**overrides: Any) -> dict[str, Any]:
    return {**DEPLOYED, **overrides}


# ── Slots ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("slot", "offset", "tier", "key"),
    [
        ("reader", 2, "gateway", "modules.web_terminals.reader_port"),
        ("history", 3, "gateway", None),
        ("telemetry_gateway", 51, "services", "services.telemetry_gateway.port"),
        ("telemetry_metrics", 52, "services", None),
        ("telemetry_store", 804, "stores", "services.telemetry_store.port"),
    ],
)
def test_every_telemetry_slot_is_in_the_layout(
    slot: str, offset: int, tier: str, key: str | None
) -> None:
    entry = SLOTS_BY_NAME[slot]
    assert (entry.offset, entry.tier, entry.config_key, entry.per_index) == (
        offset,
        tier,
        key,
        False,
    )


def test_offset_fifty_is_unassigned_and_no_slot_carries_the_retired_name() -> None:
    assert all(entry.offset != 50 for entry in LAYOUT)
    assert "openobserve" not in SLOTS_BY_NAME


def test_the_port_preflight_knows_every_published_telemetry_slot() -> None:
    """Each slot a compose service publishes is named by that service's entry."""
    assert _SERVICE_LAYOUT_SLOTS[tc.STORE_SERVICE] == ("telemetry_store",)
    assert _SERVICE_LAYOUT_SLOTS[tc.GATEWAY_SERVICE] == ("telemetry_gateway", "telemetry_metrics")
    assert _SERVICE_LAYOUT_SLOTS[tc.HISTORY_SERVICE] == ("history",)


def test_the_readers_origin_port_is_derived_for_the_preflight() -> None:
    """nginx is host-networked, so the ``reader`` slot is derived, not parsed."""
    config = _config(modules={"web_terminals": {"enabled": True}})
    (binding,) = derive_reader_origin_bindings(config)
    assert (binding.service, binding.host_port, binding.host_network) == (
        "nginx",
        default_port("reader"),
        True,
    )
    assert binding.remedy == "deployment.port_base"

    config["modules"]["web_terminals"]["reader_port"] = 8443
    (moved,) = derive_reader_origin_bindings(config)
    assert (moved.host_port, moved.remedy) == (8443, "modules.web_terminals.reader_port")


@pytest.mark.parametrize(
    "config",
    [
        {"modules": {"web_terminals": {"enabled": True}}},
        _config(modules={"web_terminals": {"enabled": False}}),
        _config(),
    ],
    ids=["no-store", "terminals-off", "no-terminals"],
)
def test_no_readers_origin_port_without_both_halves(config: dict[str, Any]) -> None:
    assert derive_reader_origin_bindings(config) == []


# ── Keys and defaults ────────────────────────────────────────────────────────


def test_every_port_defaults_to_its_slot_at_the_deployments_base() -> None:
    config = _config(deployment={"port_base": 20000})
    assert tc.store_port(config) == 20804
    assert tc.gateway_port(config) == 20051
    assert tc.metrics_port(config) == 20052
    assert tc.reader_port(config) == 20002
    assert tc.history_port(config) == 20003


def test_a_stated_port_wins_over_the_layout() -> None:
    config = _config(
        services={"telemetry_store": {"port": 9000}, "telemetry_gateway": {"port": 4400}},
        modules={"web_terminals": {"reader_port": 8443}},
    )
    assert (tc.store_port(config), tc.gateway_port(config), tc.reader_port(config)) == (
        9000,
        4400,
        8443,
    )


@pytest.mark.parametrize("value", ["9000", 0, 70000, True, 1.5])
def test_an_unusable_port_is_refused_naming_the_key(value: Any) -> None:
    config = _config(services={"telemetry_store": {"port": value}})
    with pytest.raises(ValueError, match=r"services\.telemetry_store\.port must be an integer"):
        tc.store_port(config)


def test_limits_default_and_reader_slots_is_two() -> None:
    limits = tc.store_limits(_config())
    assert limits == dict(tc.DEFAULT_LIMITS)
    assert limits["reader_slots"] == 2
    assert set(limits) == {
        "result_rows",
        "result_bytes",
        "seconds",
        "concurrent",
        "reader_slots",
        "per_minute",
        "rows_read",
        "bytes_read",
        "query_memory",
    }


def test_a_stated_limit_replaces_only_itself() -> None:
    config = _config(services={"telemetry_store": {"limits": {"reader_slots": 5}}})
    assert tc.store_limits(config) == {**tc.DEFAULT_LIMITS, "reader_slots": 5}


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"reader_slot": 3}, "has no key reader_slot"),
        ({"seconds": 0}, r"limits\.seconds must be a positive integer"),
        ({"seconds": "30"}, r"limits\.seconds must be a positive integer"),
        ([], "limits must be a mapping"),
    ],
)
def test_an_unusable_limit_is_refused_naming_it(limits: Any, message: str) -> None:
    config = _config(services={"telemetry_store": {"limits": limits}})
    with pytest.raises(ValueError, match=message):
        tc.store_limits(config)


def test_retention_defaults_to_fourteen_days() -> None:
    assert tc.retention_days(_config()) == 14
    assert tc.retention_days(_config(services={"telemetry_store": {"retention_days": 30}})) == 30


def test_history_concurrency_is_a_constant_of_two() -> None:
    assert tc.HISTORY_CONCURRENCY == 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2g", 2 * 1024 * MIB),
        ("2GiB", 2 * 1024 * MIB),
        ("512m", 512 * MIB),
        ("1.5gb", 1536 * MIB),
        ("64k", 65536),
        (1048576, MIB),
        ("1048576", MIB),
    ],
)
def test_memory_is_read_as_compose_reads_it(value: Any, expected: int) -> None:
    assert tc.parse_memory(value, "k") == expected


@pytest.mark.parametrize("value", ["", "2 gigs", "g", 0, -1, True, None, "0m"])
def test_an_unusable_memory_value_is_refused_naming_the_key(value: Any) -> None:
    with pytest.raises(ValueError, match=r"services\.x\.memory must be a memory size"):
        tc.parse_memory(value, "services.x.memory")


@pytest.mark.parametrize(
    ("name", "reserved"),
    [
        ("own", True),
        ("default", True),
        ("osprey_admin", True),
        ("ingest", True),
        ("archive", True),
        ("history", True),
        ("readers", True),
        ("views", True),
        ("views_own", True),
        ("reader_limits", True),
        ("view_limits", True),
        ("history_limits", True),
        ("reader_quota", True),
        ("view_quota", True),
        ("view_alice", True),
        ("view__all", True),
        ("viewer", False),
        ("carol", False),
        ("operator", False),
    ],
)
def test_the_stores_own_account_names_are_reserved(name: str, reserved: bool) -> None:
    assert tc.is_reserved_account_name(name) is reserved


@pytest.mark.parametrize(
    ("web_terminals", "deployed"),
    [
        ({"enabled": True, "auth": {"method": "password"}}, True),
        ({"enabled": True, "auth": {"method": "oidc"}}, True),
        ({"enabled": True, "auth": {"method": "token"}}, False),
        ({"enabled": True, "auth": {"method": "none"}}, False),
        ({"enabled": True}, False),
        ({"enabled": False, "auth": {"method": "password"}}, False),
    ],
)
def test_the_history_page_exists_with_a_store_and_a_sign_in(
    web_terminals: dict[str, Any], deployed: bool
) -> None:
    config = _config(modules={"web_terminals": web_terminals})
    assert tc.history_page_deployed(config) is deployed
    assert tc.history_page_deployed({"modules": {"web_terminals": web_terminals}}) is False


# ── Memory formulas ──────────────────────────────────────────────────────────


def test_the_store_formula_at_the_defaults() -> None:
    """(2 reader slots + 2 page queries + 2 spare) × 200 MB + 437 MiB, under 0.9 × 2 GiB."""
    demand = tc.store_memory_demand(_config())
    assert demand == 6 * 200_000_000 + 437 * MIB
    assert round(demand / 1e6) == 1658
    ceiling = tc.STORE_MEMORY_RATIO * tc.store_memory_bytes(_config())
    assert round(ceiling / 1e6) == 1933
    assert demand <= ceiling


def test_the_store_formula_counts_every_term() -> None:
    config = _config(
        services={"telemetry_store": {"limits": {"reader_slots": 7, "query_memory": 1000}}}
    )
    assert tc.store_memory_demand(config) == (7 + 2 + 2) * 1000 + 437 * MIB


def test_the_gateway_formula_at_the_defaults() -> None:
    """256 MiB + 8 × 20 MiB + 2 × 100 × 1 MiB × R."""
    assert tc.gateway_max_request_body_size(_config()) == 20 * MIB
    assert tc.gateway_max_log_size(_config()) == MIB
    assert tc.gateway_memory_floor(_config(), 1) == (256 + 160 + 200) * MIB
    assert tc.gateway_memory_floor(_config(), 5) == (256 + 160 + 1000) * MIB
    assert tc.gateway_memory_bytes(_config(), 5) == tc.gateway_memory_floor(_config(), 5)


def test_the_gateway_formula_follows_the_keys_that_size_a_record() -> None:
    config = _config(
        claude_code={"telemetry": {"content_max_length": 1_000_000}},
        audit={"tool_call": {"max_inline_bytes": 4 * MIB}},
    )
    assert tc.gateway_max_request_body_size(config) == 16 * 1_000_000 * 3
    # 2 × 4 MiB + 256 KiB, rounded up to a whole mebibyte.
    assert tc.gateway_max_log_size(config) == 9 * MIB
    assert tc.gateway_memory_floor(config, 3) == (
        256 * MIB + 8 * 48_000_000 + 2 * 100 * 9 * MIB * 3
    )


def test_a_stated_gateway_memory_is_taken_as_stated() -> None:
    config = _config(services={"telemetry_gateway": {"memory": "512m"}})
    assert tc.gateway_memory_bytes(config, 5) == 512 * MIB


# ── Topology ─────────────────────────────────────────────────────────────────


def _dial(url: str, path: str = "/") -> httpx.URL:
    """Send one request to *url* through a canned transport; return where it went."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, text="Ok.\n")

    with httpx.Client(transport=httpx.MockTransport(handler), base_url=url) as client:
        assert client.get(path).status_code == 200
    (reached,) = seen
    return reached


PINNED = _config(deployment={"bind_address": "10.0.0.5"})
WILDCARD = _config(deployment={"bind_address": "0.0.0.0"})


@pytest.mark.parametrize("config", [_config(), PINNED, WILDCARD], ids=["default", "pinned", "any"])
def test_a_bridge_container_reaches_the_gateway_by_service_name(config: dict[str, Any]) -> None:
    reached = _dial(tc.gateway_url(config, bridge=True), "/v1/logs")
    assert (reached.host, reached.port, reached.path) == ("telemetry_gateway", 4318, "/v1/logs")


@pytest.mark.parametrize(
    ("config", "host"),
    [(_config(), "127.0.0.1"), (WILDCARD, "127.0.0.1"), (PINNED, "10.0.0.5")],
    ids=["default", "any", "pinned"],
)
def test_the_host_namespace_reaches_the_gateway_where_it_publishes(
    config: dict[str, Any], host: str
) -> None:
    reached = _dial(tc.gateway_url(config, bridge=False), "/v1/logs")
    assert (reached.host, reached.port) == (host, default_port("telemetry_gateway"))


def test_the_published_gateway_port_follows_its_key() -> None:
    config = {**PINNED, "services": {"telemetry_gateway": {"port": 4400}}}
    assert tc.gateway_address(config, bridge=False) == ("10.0.0.5", 4400)
    assert tc.gateway_address(config, bridge=True) == ("telemetry_gateway", 4318)


def test_an_ipv6_bind_is_bracketed_in_the_url() -> None:
    config = _config(deployment={"bind_address": "::"})
    assert (
        tc.gateway_url(config, bridge=False) == f"http://[::1]:{default_port('telemetry_gateway')}"
    )


@pytest.mark.parametrize("config", [_config(), PINNED, WILDCARD], ids=["default", "pinned", "any"])
def test_a_bridge_container_reaches_the_store_by_service_name(config: dict[str, Any]) -> None:
    reached = _dial(tc.store_url(config, bridge=True), "/ping")
    assert (reached.host, reached.port, reached.path) == ("telemetry_store", 8123, "/ping")


@pytest.mark.parametrize("config", [_config(), PINNED, WILDCARD], ids=["default", "pinned", "any"])
def test_the_host_namespace_reaches_the_store_on_loopback_whatever_the_bind(
    config: dict[str, Any],
) -> None:
    """The store publishes on loopback unconditionally; the bind address is not its."""
    reached = _dial(tc.store_url(config, bridge=False), "/ping")
    assert (reached.host, reached.port) == ("127.0.0.1", default_port("telemetry_store"))


def test_a_process_carrying_no_host_is_a_host_launch() -> None:
    assert tc.runtime_gateway_url(PINNED, {}) == tc.gateway_url(PINNED, bridge=False)
    assert tc.runtime_gateway_url(PINNED, {tc.GATEWAY_HOST_ENV: " "}) == tc.gateway_url(
        PINNED, bridge=False
    )


def test_a_container_dials_what_it_was_rendered_with() -> None:
    environ = {tc.GATEWAY_HOST_ENV: "telemetry_gateway", tc.GATEWAY_PORT_ENV: "4318"}
    reached = _dial(tc.runtime_gateway_url(PINNED, environ), "/v1/logs")
    assert (reached.host, reached.port) == ("telemetry_gateway", 4318)
    assert tc.runtime_gateway_url(PINNED, {tc.GATEWAY_HOST_ENV: "gw"}) == "http://gw:4318"


@pytest.mark.parametrize("port", ["http", "0", "70000", "-1"])
def test_a_rendered_port_that_is_not_one_is_refused(port: str) -> None:
    environ = {tc.GATEWAY_HOST_ENV: "telemetry_gateway", tc.GATEWAY_PORT_ENV: port}
    with pytest.raises(ValueError, match="OSPREY_OTEL_GATEWAY_PORT must be an integer port"):
        tc.runtime_gateway_url(PINNED, environ)


# --- the query page ----------------------------------------------------------


def test_the_default_store_image_has_a_query_page_hash() -> None:
    """The image a deployment gets unasked is one whose page is served."""
    tag = tc.DEFAULT_STORE_IMAGE.rsplit(":", 1)[1]
    assert tag in tc.PLAY_SCRIPT_SHA256
    assert tc.store_image_tag({}) == tag
    assert tc.play_script_sha256({}) == tc.PLAY_SCRIPT_SHA256[tag]


def test_the_hash_of_the_pinned_image_is_the_one_read_off_it() -> None:
    """Pinned to the tag: bumping the default image has to bring its own entry."""
    assert tc.DEFAULT_STORE_IMAGE == "clickhouse/clickhouse-server:26.8.18.2"
    assert tc.PLAY_SCRIPT_SHA256 == {
        "26.8.18.2": "85Yv/siUrCiWuCu6drIALnoviB3sLNDOC8OHSaT95/Y=",
    }


@pytest.mark.parametrize("digest", sorted(tc.PLAY_SCRIPT_SHA256.values()))
def test_every_entry_is_a_base64_sha256(digest: str) -> None:
    import base64

    assert len(base64.b64decode(digest, validate=True)) == 32


@pytest.mark.parametrize(
    ("image", "tag"),
    [
        ("clickhouse/clickhouse-server:26.8.18.2", "26.8.18.2"),
        ("registry.example.org:5000/mirror/clickhouse-server:26.8.18.2", "26.8.18.2"),
        ("clickhouse/clickhouse-server:26.8.18.2@sha256:" + "a" * 64, "26.8.18.2"),
        ("  clickhouse/clickhouse-server:26.8.18.2  ", "26.8.18.2"),
        ("clickhouse/clickhouse-server", ""),
        ("registry.example.org:5000/clickhouse-server", ""),
        ("clickhouse/clickhouse-server@sha256:" + "a" * 64, ""),
        ("clickhouse/clickhouse-server:latest", "latest"),
    ],
)
def test_the_tag_is_read_off_any_spelling_of_an_image(image: str, tag: str) -> None:
    config = {"services": {"telemetry_store": {"image": image}}}
    assert tc.store_image_tag(config) == tag
    known = tag in tc.PLAY_SCRIPT_SHA256
    assert (tc.play_script_sha256(config) is not None) is known


@pytest.mark.parametrize("image", ["", "  ", 26, ["clickhouse"]])
def test_an_image_that_is_not_a_reference_is_refused_by_name(image: Any) -> None:
    config = {"services": {"telemetry_store": {"image": image}}}
    with pytest.raises(ValueError, match=r"services\.telemetry_store\.image"):
        tc.store_image(config)


def test_a_reader_link_names_its_own_query_endpoint_and_never_a_key() -> None:
    origin = "https://readers.example.org:8443"
    bare = tc.reader_play_url(origin)
    assert bare == (
        "https://readers.example.org:8443/telemetry/play"
        "?url=https%3A%2F%2Freaders.example.org%3A8443%2Ftelemetry%2F"
    )
    assert tc.reader_play_url(origin, "2nd-shift") == f"{bare}&user=2nd-shift"
    assert "password" not in tc.reader_play_url(origin, "2nd-shift")
    assert bare.startswith(f"{origin}{tc.READER_PLAY_PATH}?")
    assert tc.READER_QUERY_PATH == "/telemetry/"
