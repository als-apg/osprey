"""Load-time checks on the telemetry store, its gateway and its readers.

One test per rule, on the smallest config that draws it, and one on the
smallest that does not: a lint that fires on everything pins nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.cli.build_profile_presets import _load_preset_raw, list_presets
from osprey.deployment import telemetry_config as tc
from osprey.deployment.errors import UnmetPreconditionsError
from osprey.deployment.telemetry_credentials import ensure_telemetry_credentials
from osprey.deployment.telemetry_lint import (
    TelemetryFinding,
    host_identity_advisories,
    lint_telemetry,
    preflight_problems,
    reader_suffix_collision_problems,
    telemetry_errors,
    telemetry_warnings,
)
from osprey.deployment.web_terminals.lint import _nest_dotted, lint_web_terminals
from osprey.deployment.web_terminals.render import (
    _QUEUESERVER_AUDIT_IDENTITIES,
    _RESERVED_AUDIT_IDENTITY_RE,
    _check_roster_audit_identities,
)

MIB = 1024 * 1024


def _config(
    *,
    readers: Any = None,
    store: dict[str, Any] | None = None,
    gateway: dict[str, Any] | None = None,
    web: dict[str, Any] | None = None,
    **top: Any,
) -> dict[str, Any]:
    """A config that deploys the store, with one unremarkable reader by default."""
    block = {"readers": [{"name": "carol"}] if readers is None else readers, **(store or {})}
    config: dict[str, Any] = {
        "deployed_services": ["telemetry_store", "telemetry_gateway"],
        "services": {"telemetry_store": block, "telemetry_gateway": gateway or {}},
        **top,
    }
    if web is not None:
        config["modules"] = {"web_terminals": {"enabled": True, **web}}
    return config


def _codes(config: dict[str, Any], **kwargs: Any) -> list[str]:
    return [finding.code for finding in lint_telemetry(config, **kwargs)]


def _one(config: dict[str, Any], code: str, **kwargs: Any) -> TelemetryFinding:
    (finding,) = [f for f in lint_telemetry(config, **kwargs) if f.code == code]
    return finding


def test_a_sound_config_draws_nothing() -> None:
    assert lint_telemetry(_config()) == []
    assert (
        lint_telemetry(_config(web={"auth": {"method": "password"}, "tls": {"enabled": True}}))
        == []
    )


def test_a_deployment_without_the_store_is_not_asked_about_it() -> None:
    config = _config(readers=[{"name": "own"}, {"name": "own"}], store={"memory": "1m"})
    config["deployed_services"] = []
    assert lint_telemetry(config) == []
    assert preflight_problems(config) == []


# ── Readers ──────────────────────────────────────────────────────────────────


def test_no_reader_is_said_once_and_blocks_nothing() -> None:
    finding = _one(_config(readers=[]), "telemetry.no_reader")
    assert finding.severity == "info"
    assert telemetry_errors(_config(readers=[])) == []
    assert telemetry_warnings(_config(readers=[])) == []


def test_readers_must_be_a_list() -> None:
    assert _one(_config(readers={"carol": {}}), "telemetry.invalid_readers").severity == "error"


@pytest.mark.parametrize(
    "entry", ["carol", {"tables": ["otel_logs"]}, {"name": "Carol"}, {"name": "a b"}, {"name": ""}]
)
def test_a_reader_entry_needs_a_usable_name(entry: Any) -> None:
    assert _one(_config(readers=[entry]), "telemetry.invalid_reader_name").severity == "error"


@pytest.mark.parametrize("name", ["2nd-shift", "jean-luc", "ops_team", "x"])
def test_the_roster_charset_is_a_readers_charset(name: str) -> None:
    assert lint_telemetry(_config(readers=[{"name": name}])) == []


@pytest.mark.parametrize(
    "name", ["own", "default", "ingest", "archive", "history", "readers", "views", "view_alice"]
)
def test_a_reader_may_not_take_one_of_the_stores_own_names(name: str) -> None:
    finding = _one(_config(readers=[{"name": name}]), "telemetry.reserved_reader_name")
    assert finding.severity == "error"
    assert repr(name) in finding.message


def test_a_reader_named_twice_is_refused_once() -> None:
    readers = [{"name": "carol"}, {"name": "dave"}, {"name": "carol", "days_back": 3}]
    finding = _one(_config(readers=readers), "telemetry.duplicate_reader")
    assert finding.severity == "error"
    assert "'carol'" in finding.message and "'dave'" not in finding.message


def test_jean_luc_and_jean_underscore_luc_share_a_suffix() -> None:
    config = _config(readers=[{"name": "jean-luc"}, {"name": "jean_luc"}, {"name": "carol"}])
    finding = _one(config, "telemetry.reader_suffix_collision")
    assert finding.severity == "error"
    assert "'jean-luc' and 'jean_luc'" in finding.message
    assert "JEAN_LUC" in finding.message
    assert reader_suffix_collision_problems(_config()) == []


@pytest.mark.parametrize("days", [0, -1, "7", 1.5, True, None])
def test_days_back_is_a_whole_number_of_days(days: Any) -> None:
    config = _config(readers=[{"name": "carol", "days_back": days}])
    assert _one(config, "telemetry.invalid_days_back").severity == "error"


def test_days_back_cannot_exceed_what_the_store_keeps() -> None:
    config = _config(readers=[{"name": "carol", "days_back": 15}])
    finding = _one(config, "telemetry.invalid_days_back")
    assert "retention_days (14)" in finding.message
    for days in (1, 14):
        assert lint_telemetry(_config(readers=[{"name": "carol", "days_back": days}])) == []
    longer = _config(readers=[{"name": "carol", "days_back": 15}], store={"retention_days": 30})
    assert lint_telemetry(longer) == []


# ── oidc_subject ─────────────────────────────────────────────────────────────

OIDC = {"auth": {"method": "oidc"}, "tls": {"enabled": True}}
PASSWORD = {"auth": {"method": "password"}, "tls": {"enabled": True}}
EXTERNAL = [{"name": "erin", "oidc_subject": "erin@lab.example"}]


def test_an_external_readers_subject_under_oidc_is_fine() -> None:
    assert lint_telemetry(_config(readers=EXTERNAL, web=OIDC)) == []


def test_a_readers_subject_without_oidc_is_a_warning() -> None:
    finding = _one(_config(readers=EXTERNAL, web=PASSWORD), "telemetry.reader_subject_without_oidc")
    assert finding.severity == "warn"
    assert _one(_config(readers=EXTERNAL), "telemetry.reader_subject_without_oidc")


def test_a_roster_named_reader_may_not_also_set_a_subject() -> None:
    web = {**OIDC, "users": [{"name": "erin", "index": 0}, "frank"]}
    finding = _one(_config(readers=EXTERNAL, web=web), "telemetry.roster_reader_subject")
    assert finding.severity == "error"
    bare = [{"name": "frank", "oidc_subject": "f@lab.example"}]
    assert _one(_config(readers=bare, web=web), "telemetry.roster_reader_subject")


@pytest.mark.parametrize("subject", ["", "  ", 7, None])
def test_a_subject_is_a_non_empty_string(subject: Any) -> None:
    config = _config(readers=[{"name": "erin", "oidc_subject": subject}], web=OIDC)
    assert _one(config, "telemetry.invalid_reader_subject").severity == "error"


# ── Memory ───────────────────────────────────────────────────────────────────


def test_the_store_memory_rule_names_every_term_with_its_value() -> None:
    config = _config(store={"memory": "1g"})
    finding = _one(config, "telemetry.store_memory")
    assert finding.severity == "error"
    for term in (
        "2 reader slots [services.telemetry_store.limits.reader_slots]",
        "2 history-page queries",
        "2 for the archive and the schema check",
        "6 queries × 191 MiB [services.telemetry_store.limits.query_memory]",
        "437 MiB idle",
        "= 1,581 MiB",
        "0.9 × 1,024 MiB [services.telemetry_store.memory] = 922 MiB",
    ):
        assert term in finding.message, term


def test_the_store_memory_rule_passes_at_the_boundary() -> None:
    demand = tc.store_memory_demand(_config())
    exact = int(demand / tc.STORE_MEMORY_RATIO) + 1
    assert "telemetry.store_memory" not in _codes(_config(store={"memory": exact}))
    assert "telemetry.store_memory" in _codes(_config(store={"memory": exact - 2}))


def test_more_reader_slots_need_more_memory() -> None:
    assert "telemetry.store_memory" in _codes(_config(store={"limits": {"reader_slots": 6}}))


def test_the_store_memory_rule_passes_on_every_presets_default_config() -> None:
    deploying = []
    for name in list_presets():
        raw, _ = _load_preset_raw(name)
        config = _nest_dotted(raw.get("config") or {})
        if not tc.store_deployed(config):
            continue
        deploying.append(name)
        assert "telemetry.store_memory" not in _codes(config), name
        assert tc.store_memory_demand(config) <= tc.STORE_MEMORY_RATIO * tc.store_memory_bytes(
            config
        ), name
        assert preflight_problems(config) == [], name
        assert telemetry_errors(config) == [], name
    assert {"hello-world", "ariel-standalone", "control-assistant"} <= set(deploying)


def test_the_gateway_memory_rule_names_r_and_every_term() -> None:
    config = _config(gateway={"memory": "512m"})
    finding = _one(config, "telemetry.gateway_memory", receivers=5)
    assert finding.severity == "error"
    for term in (
        "services.telemetry_gateway.memory is 512 MiB",
        "256 MiB + 8 × 20 MiB (the largest request body)",
        "2 × 100 × 1 MiB (the longest ledger line) × R",
        "R = 5",
        "= 1,416 MiB",
    ):
        assert term in finding.message, term


def test_the_gateway_memory_rule_needs_r_and_a_stated_limit() -> None:
    low = _config(gateway={"memory": "512m"})
    assert "telemetry.gateway_memory" not in _codes(low)
    assert "telemetry.gateway_memory" not in _codes(low, receivers=0)
    assert "telemetry.gateway_memory" not in _codes(_config(), receivers=50)
    assert "telemetry.gateway_memory" not in _codes(_config(gateway={"memory": "2g"}), receivers=5)
    assert "telemetry.gateway_memory" in _codes(_config(gateway={"memory": "600m"}), receivers=1)
    assert "telemetry.gateway_memory" not in _codes(
        _config(gateway={"memory": "616m"}), receivers=1
    )


# ── Values and network ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "key"),
    [
        ({"store": {"port": "x"}}, "services.telemetry_store.port"),
        ({"gateway": {"port": 0}}, "services.telemetry_gateway.port"),
        ({"store": {"retention_days": 0}}, "services.telemetry_store.retention_days"),
        ({"store": {"limits": {"seconds": -1}}}, "services.telemetry_store.limits.seconds"),
        ({"store": {"memory": "lots"}}, "services.telemetry_store.memory"),
        ({"gateway": {"memory": "lots"}}, "services.telemetry_gateway.memory"),
        ({"store": {"image": ""}}, "services.telemetry_store.image"),
        ({"web": {"reader_port": "x"}}, "modules.web_terminals.reader_port"),
    ],
)
def test_an_unusable_value_is_an_error_naming_its_key(kwargs: dict[str, Any], key: str) -> None:
    findings = [f for f in lint_telemetry(_config(**kwargs)) if f.code == "telemetry.invalid_value"]
    assert [f.severity for f in findings] == ["error"]
    assert key in findings[0].message


@pytest.mark.parametrize("service", ["telemetry_store", "telemetry_gateway"])
def test_neither_service_may_sit_on_the_host_network(service: str) -> None:
    block = {"network": "host"}
    config = _config(store=block) if service == "telemetry_store" else _config(gateway=block)
    finding = _one(config, "telemetry.host_network")
    assert finding.severity == "error"
    assert f"services.{service}.network" in finding.message


# ── The readers' origin ──────────────────────────────────────────────────────

TLS = {"tls": {"enabled": True}}


def test_an_external_origin_needs_a_reader_origin_beside_it() -> None:
    web = {**TLS, "external_origin": "https://osprey.lab.example"}
    assert _one(_config(web=web), "telemetry.reader_origin_required").severity == "error"
    web["reader_origin"] = "https://osprey.lab.example:8443"
    assert lint_telemetry(_config(web=web)) == []


def test_the_two_origins_share_a_host() -> None:
    web = {
        **TLS,
        "external_origin": "https://osprey.lab.example",
        "reader_origin": "https://history.lab.example",
    }
    finding = _one(_config(web=web), "telemetry.reader_origin_host")
    assert finding.severity == "error"
    assert "'history.lab.example'" in finding.message and "'osprey.lab.example'" in finding.message
    web["reader_origin"] = "https://OSPREY.lab.example:8443"
    assert lint_telemetry(_config(web=web)) == []


@pytest.mark.parametrize(
    "origin",
    ["osprey.lab.example", "https://h/", "https://h/path", "ftp://h", "https://h:port", "", 7],
)
def test_a_reader_origin_is_an_origin_and_nothing_else(origin: Any) -> None:
    web = {**TLS, "reader_origin": origin}
    assert _one(_config(web=web), "telemetry.invalid_reader_origin").severity == "error"


def test_a_reader_origin_alone_is_allowed() -> None:
    web = {**TLS, "reader_origin": "https://osprey.lab.example:8443"}
    assert lint_telemetry(_config(web=web)) == []


def test_reader_keys_in_cleartext_on_a_reachable_origin_draw_a_warning() -> None:
    web = {"reader_origin": "http://osprey.lab.example:10002"}
    finding = _one(_config(web=web), "telemetry.reader_cleartext")
    assert finding.severity == "warn"


@pytest.mark.parametrize(
    "web",
    [
        {"reader_origin": "http://127.0.0.1:10002"},
        {"reader_origin": "http://localhost:10002"},
        {"reader_origin": "https://osprey.lab.example"},
        {"tls": {"enabled": True}, "reader_origin": "http://osprey.lab.example:10002"},
    ],
)
def test_no_cleartext_warning_on_loopback_behind_tls_or_with_tls_on(web: dict[str, Any]) -> None:
    assert "telemetry.reader_cleartext" not in _codes(_config(web=web))


def test_a_derived_loopback_or_underivable_origin_draws_no_cleartext_warning() -> None:
    """With no reader_origin the terminals' own derived origin decides."""
    assert "telemetry.reader_cleartext" not in _codes(_config(web={}))
    loopback = _config(web={}, deploy={"fqdn": "localhost"})
    assert "telemetry.reader_cleartext" not in _codes(loopback)


@pytest.mark.parametrize("bind", ["0.0.0.0", "::", "10.0.0.5"])
@pytest.mark.parametrize(
    ("web", "deploy"),
    [
        ({}, {"fqdn": "localhost"}),
        ({}, None),
        ({"reader_origin": "http://127.0.0.1:10002"}, None),
    ],
)
def test_a_port_published_beyond_this_machine_is_exposed_whatever_its_name(
    bind: str, web: dict[str, Any], deploy: dict[str, Any] | None
) -> None:
    """The bind decides too: a loopback name, or none, over a network bind."""
    top: dict[str, Any] = {"deployment": {"bind_address": bind}}
    if deploy is not None:
        top["deploy"] = deploy
    assert _one(_config(web=web, **top), "telemetry.reader_cleartext").severity == "warn"


def test_an_https_reader_origin_is_not_cleartext_in_any_spelling_or_on_any_bind() -> None:
    web = {"reader_origin": "HTTPS://osprey.lab.example"}
    config = _config(web=web, deployment={"bind_address": "0.0.0.0"})
    assert "telemetry.reader_cleartext" not in _codes(config)


def test_no_reader_means_no_key_to_carry_in_cleartext() -> None:
    web = {"reader_origin": "http://osprey.lab.example:10002"}
    config = _config(readers=[], web=web, deployment={"bind_address": "0.0.0.0"})
    assert "telemetry.reader_cleartext" not in _codes(config)


def test_a_derived_reachable_origin_draws_the_cleartext_warning() -> None:
    config = _config(web={}, deploy={"fqdn": "osprey.lab.example"})
    assert "telemetry.reader_cleartext" in _codes(config)


# ── history ──────────────────────────────────────────────────────────────────


def _roles_web(**extra: Any) -> dict[str, Any]:
    return {
        **TLS,
        "users": [{"name": "alice", "index": 0, "role": "operator", "history": "own"}],
        "authorization": {
            "roles": {
                "operator": {"persona": "operator"},
                "lead": {"persona": "operator", "history": "all"},
            },
        },
        **extra,
    }


def test_history_all_on_a_role_nobody_holds_draws_a_warning() -> None:
    finding = _one(_config(web=_roles_web()), "telemetry.unbound_history_role")
    assert finding.severity == "warn"
    assert "roles.lead" in finding.message


def test_a_roster_bound_role_is_bound() -> None:
    web = _roles_web()
    web["users"].append({"name": "bob", "index": 1, "role": "lead"})
    assert lint_telemetry(_config(web=web)) == []


@pytest.mark.parametrize("wrong", [["lead"], {"lead": 1}, 7, None])
def test_a_wrong_typed_role_or_claim_target_binds_nothing_and_raises_nothing(wrong: Any) -> None:
    """The web-terminal lint names those shapes; this rule must survive them."""
    web = _roles_web()
    web["users"].append({"name": "bob", "index": 1, "role": wrong})
    web["authorization"]["claims"] = {"claim": "groups", "map": {"lab-leads": wrong}}
    assert "telemetry.unbound_history_role" in _codes(_config(web=web))


def test_a_claim_granted_role_is_bound() -> None:
    web = _roles_web()
    web["authorization"]["claims"] = {"claim": "groups", "map": {"lab-leads": "lead"}}
    assert lint_telemetry(_config(web=web)) == []


@pytest.mark.parametrize("value", ["everyone", "All", True, None, ["all"]])
def test_history_is_own_or_all_on_a_user_and_on_a_role(value: Any) -> None:
    web = _roles_web()
    web["users"][0]["history"] = value
    finding = _one(_config(web=web), "telemetry.invalid_history")
    assert finding.severity == "error" and "'alice'" in finding.message

    web = _roles_web()
    web["authorization"]["roles"]["operator"]["history"] = value
    assert "roles.operator" in _one(_config(web=web), "telemetry.invalid_history").message


def test_a_history_value_is_checked_without_the_store_too() -> None:
    web = _roles_web()
    web["users"][0]["history"] = "everyone"
    config = _config(web=web)
    config["deployed_services"] = []
    assert _codes(config) == ["telemetry.invalid_history"]


# ── Host identity ────────────────────────────────────────────────────────────


def test_a_different_runtime_account_draws_a_warning() -> None:
    config = _config(deployment={"host_identity": "builder"})
    finding = _one(config, "telemetry.host_identity_mismatch", runtime_user="operator")
    assert finding.severity == "warn"
    assert "'builder'" in finding.message and "'operator'" in finding.message


def test_the_same_account_an_unasked_command_and_an_unstamped_config_draw_none() -> None:
    config = _config(deployment={"host_identity": "builder"})
    assert lint_telemetry(config, runtime_user="builder") == []
    assert lint_telemetry(config) == []
    assert lint_telemetry(_config(), runtime_user="operator") == []


def test_a_start_says_so_and_goes_on(tmp_path, monkeypatch) -> None:
    from osprey.deployment import container_lifecycle

    said: list[str] = []
    monkeypatch.setattr(container_lifecycle, "_required_env_problems", lambda *a, **k: [])
    monkeypatch.setattr(container_lifecycle, "_unreleased_pin_problem", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_invoking_account", lambda: "operator")
    monkeypatch.setattr(container_lifecycle, "_warn_fact", said.append)
    config = _config(deployment={"host_identity": "builder"})
    # The pass sits below the step that provisions the credential files it reads.
    ensure_telemetry_credentials(tmp_path, config, reader_keys=True)

    container_lifecycle._collect_unmet_preconditions(
        config, tmp_path, {}, dev_mode=False, web_terminals_enabled=False
    )

    assert said == host_identity_advisories(config, "operator")
    assert len(said) == 1 and "'builder'" in said[0]
    assert host_identity_advisories(config, None) == []
    assert host_identity_advisories(config, "builder") == []
    config["deployed_services"] = []
    assert host_identity_advisories(config, "operator") == []


# ── The build and validate surfaces ──────────────────────────────────────────


def test_the_build_gate_carries_the_telemetry_errors_and_warnings() -> None:
    """A profile's dotted ``config:`` block reaches the telemetry rules nested."""
    from osprey.cli.build_profile_deploy import (
        deploy_aware_config_errors,
        deploy_aware_config_warnings,
    )

    config = {
        "deployed_services": ["telemetry_store", "telemetry_gateway"],
        "services.telemetry_store.memory": "1g",
        "services.telemetry_store.readers": [{"name": "erin", "oidc_subject": "e@lab.example"}],
    }
    (error,) = deploy_aware_config_errors(None, config)
    assert "do not fit its memory limit" in error
    (warning,) = deploy_aware_config_warnings(None, config)
    assert "oidc_subject is set" in warning

    sound = {"deployed_services": ["telemetry_store"], "services.telemetry_store.readers": []}
    assert deploy_aware_config_errors(None, sound) == []
    assert deploy_aware_config_warnings(None, sound) == []


# ── Roster names ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "queueserver",
        "ops-queueserver",
        "bluesky-va-queueserver",
        "history",
        "own",
        "view_x",
        "osprey_admin",
        "ingest",
        "archive",
        "readers",
        "views",
        "views_own",
        "default",
    ],
)
def test_a_roster_user_may_not_take_a_reserved_name(name: str) -> None:
    config = {"modules": {"web_terminals": {"enabled": True, "users": [{"name": name}]}}}
    findings = [
        f for f in lint_web_terminals(config) if f.code == "web_terminals.reserved_audit_identity"
    ]
    assert [f.severity for f in findings] == ["error"]
    assert repr(name) in findings[0].message
    with pytest.raises(ValueError, match="collide with a service's own audit identity"):
        _check_roster_audit_identities([{"user": name}])


@pytest.mark.parametrize("name", ["alice", "carol", "operator", "jean-luc", "2nd-shift", "viewer"])
def test_an_ordinary_roster_name_is_not_reserved(name: str) -> None:
    assert _RESERVED_AUDIT_IDENTITY_RE.fullmatch(name) is None
    _check_roster_audit_identities([{"user": name}])


def test_every_lane_queueserver_identity_is_reserved() -> None:
    """Derived from the lane keys, so a new lane's directory is reserved with it."""
    from osprey.bluesky_bridge_connection import LANE_KEYS
    from osprey.deployment.compose_generator import lane_queueserver_audit_identities

    assert len(_QUEUESERVER_AUDIT_IDENTITIES) == len(LANE_KEYS)
    rendered = lane_queueserver_audit_identities({"deployed_services": list(LANE_KEYS)})
    assert sorted(rendered) == sorted(_QUEUESERVER_AUDIT_IDENTITIES)
    for identity in rendered:
        assert _RESERVED_AUDIT_IDENTITY_RE.fullmatch(identity), identity


# ── The start preflight ──────────────────────────────────────────────────────


def test_the_preflight_holds_the_two_start_blocking_rules() -> None:
    config = _config(readers=[{"name": "jean-luc"}, {"name": "jean_luc"}], store={"memory": "1g"})
    problems = preflight_problems(config)
    assert len(problems) == 2
    (suffix_problem, suffix_remedy), (memory_problem, memory_remedy) = problems
    assert "'jean-luc' and 'jean_luc'" in suffix_problem and suffix_remedy == "Rename one of them"
    assert "do not fit its memory limit" in memory_problem
    assert memory_remedy.startswith("Raise services.telemetry_store.memory")
    assert preflight_problems(_config()) == []


def test_osprey_up_refuses_a_start_over_them(tmp_path, monkeypatch) -> None:
    from osprey.deployment import container_lifecycle

    monkeypatch.setattr(container_lifecycle, "_required_env_problems", lambda *a, **k: [])
    monkeypatch.setattr(container_lifecycle, "_unreleased_pin_problem", lambda *a, **k: None)
    config = _config(readers=[{"name": "jean-luc"}, {"name": "jean_luc"}])
    # The pass sits below the step that provisions the credential files it reads.
    ensure_telemetry_credentials(tmp_path, config, reader_keys=True)
    with pytest.raises(UnmetPreconditionsError) as raised:
        container_lifecycle._collect_unmet_preconditions(
            config, tmp_path, {}, dev_mode=False, web_terminals_enabled=False
        )
    assert "'jean-luc' and 'jean_luc'" in str(raised.value.reason)
    assert "JEAN_LUC" in str(raised.value.reason)

    ensure_telemetry_credentials(tmp_path, _config(), reader_keys=True)
    container_lifecycle._collect_unmet_preconditions(
        _config(), tmp_path, {}, dev_mode=False, web_terminals_enabled=False
    )


# ── The lint verb ────────────────────────────────────────────────────────────


def _lint_verb(tmp_path, store: dict[str, Any]):
    """Run ``osprey scaffold web-terminals lint`` on a built repo deploying the store."""
    import yaml
    from click.testing import CliRunner

    from osprey.cli.scaffold_cmd import scaffold
    from tests.cli.test_scaffold_cmd import _web_terminals_repo

    repo = _web_terminals_repo(tmp_path)
    built = repo / "build" / "config.yml"
    config = yaml.safe_load(built.read_text(encoding="utf-8"))
    config["deployed_services"] = ["telemetry_store", "telemetry_gateway"]
    config.setdefault("services", {})["telemetry_store"] = store
    built.write_text(yaml.safe_dump(config), encoding="utf-8")
    return CliRunner().invoke(scaffold, ["web-terminals", "lint", "--repo", str(repo)])


def test_the_lint_verb_says_a_note_and_does_not_fail_on_it(tmp_path) -> None:
    result = _lint_verb(tmp_path, {"readers": []})
    assert result.exit_code == 0, result.output
    # One word each: the lines are rendered through rich, which wraps them.
    assert "nobody" in result.output
    assert "Errors:" not in result.output


def test_the_lint_verb_fails_on_a_telemetry_error(tmp_path) -> None:
    result = _lint_verb(tmp_path, {"readers": [{"name": "own"}]})
    assert result.exit_code != 0, result.output
    assert "Errors:" in result.output
    assert "'own'" in result.output


# --- the query page ----------------------------------------------------------


def test_an_image_with_no_query_page_hash_draws_a_warning_naming_the_way_out() -> None:
    image = "clickhouse/clickhouse-server:99.1.1.1"
    finding = _one(_config(store={"image": image}, web={}), "telemetry.unknown_query_page")
    assert finding.severity == "warn"
    for needle in (
        "services.telemetry_store.image",
        image,
        "PLAY_SCRIPT_SHA256",
        "/telemetry/play",
        "404",
        "26.8.18.2",
    ):
        assert needle in finding.message
    assert finding.message in telemetry_warnings(_config(store={"image": image}, web={}))
    assert not any(
        image in error for error in telemetry_errors(_config(store={"image": image}, web={}))
    )


@pytest.mark.parametrize(
    "store",
    [{}, {"image": tc.DEFAULT_STORE_IMAGE}, {"image": "mirror.example.org:5000/ch:26.8.18.2"}],
)
def test_a_known_image_draws_no_query_page_warning(store: dict[str, Any]) -> None:
    assert "telemetry.unknown_query_page" not in _codes(_config(store=store, web={}))


def test_without_web_terminals_no_query_page_is_served_and_none_is_warned_about() -> None:
    config = _config(store={"image": "clickhouse/clickhouse-server:99.1.1.1"})
    assert "telemetry.unknown_query_page" not in _codes(config)


def test_an_image_that_is_not_a_reference_is_the_value_rules_finding_alone() -> None:
    codes = _codes(_config(store={"image": ""}, web={}))
    assert "telemetry.invalid_value" in codes
    assert "telemetry.unknown_query_page" not in codes
