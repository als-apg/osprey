"""The identity tables: the source-identity map (spool directory, OTLP login,
transcript root; shared-mode dispatch workers collapse to one identity), the
trusted on-behalf list (the auth sidecar and the deployed queueservers), and the D18 credential →
scope table, whose logins never contain ``:``.
"""

from __future__ import annotations

import copy

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record import identities as ids
from osprey.agent_record.config import HISTORY_SERVICE
from osprey.deployment.web_terminals.render import AUTH_SIDECAR_AUDIT_IDENTITY
from tests.deployment.web_terminals.test_golden_render import EXAMPLE_CONFIG


def _config(
    *,
    roster: bool = True,
    workers: int = 0,
    shared: bool = False,
    bridges: tuple[str, ...] = (),
    lanes: bool = False,
    host: str | None = "builder",
    readers: list[dict] | None = None,
) -> dict:
    config: dict = copy.deepcopy(EXAMPLE_CONFIG) if roster else {"project_name": "p", "modules": {}}
    deployed = list(bridges)
    services: dict = {}
    if workers:
        deployed.append("dispatch_worker")
        services["dispatch_worker"] = {
            "worker_count": workers,
            "workspace_mode": "shared" if shared else "isolated",
        }
    if lanes:
        deployed.append("bluesky")
    config["deployed_services"] = deployed
    config["services"] = services
    if host is not None:
        config["deployment"] = {"host_identity": host}
    if readers is not None:
        config["agent_record"] = {"readers": readers}
    return config


# --- fixed names ------------------------------------------------------------------


def test_fixed_identities_match_the_deployment_names():
    assert ids.SIDECAR_IDENTITY == AUTH_SIDECAR_AUDIT_IDENTITY
    assert ids.HISTORY_IDENTITY == HISTORY_SERVICE
    assert dict(ids.BRIDGE_IDENTITIES) == {
        "teams_bridge": "teams-bridge",
        "gchat_bridge": "gchat-bridge",
        "nextcloud_bridge": "nextcloud-bridge",
    }


# --- the source-identity map ----------------------------------------------------------


def test_isolated_workers_keep_their_own_identity():
    config = _config(workers=3)
    assert ids.dispatch_identities(config) == [
        "dispatch-worker-1",
        "dispatch-worker-2",
        "dispatch-worker-3",
    ]
    assert ids.spool_source_identity(config, "dispatch-worker-2") == "dispatch-worker-2"
    assert ids.login_source_identity(config, "dispatch-worker-2") == "dispatch-worker-2"
    assert (
        ids.transcript_source_identity(config, "dispatch_claude_projects_2") == "dispatch-worker-2"
    )


def test_shared_mode_workers_map_to_one_identity():
    config = _config(workers=3, shared=True)
    shared = ids.SHARED_DISPATCH_IDENTITY
    assert ids.dispatch_identities(config) == [shared]
    for i in (1, 2, 3):
        assert ids.spool_source_identity(config, f"dispatch-worker-{i}") == shared
        assert ids.login_source_identity(config, f"dispatch-worker-{i}") == shared
    assert ids.transcript_source_identity(config, "dispatch_claude_projects") == shared
    assert ids.identity_class(config, shared) == c.SOURCE_DISPATCH


def test_other_spool_directories_are_their_own_identity():
    config = _config()
    assert ids.spool_source_identity(config, "alice") == "alice"
    assert ids.spool_source_identity(config, "sidecar") == "sidecar"
    for bad in ("", "..", "a:b", "a/b"):
        with pytest.raises(ValueError):
            ids.spool_source_identity(config, bad)


def test_reader_logins_are_not_sources():
    with pytest.raises(ValueError):
        ids.login_source_identity(_config(), "reader.ops")


def test_transcript_roots_map_terminals_workers_and_host():
    config = _config(workers=2)
    assert ids.transcript_roots(config) == {
        "alice-claude-projects": "alice",
        "bob-claude-projects": "bob",
        "dispatch_claude_projects_1": "dispatch-worker-1",
        "dispatch_claude_projects_2": "dispatch-worker-2",
        "host": "builder",
    }
    for root, identity in ids.transcript_roots(config).items():
        assert ids.transcript_source_identity(config, root) == identity


def test_shared_transcript_root_and_no_host_without_host_identity():
    config = _config(workers=2, shared=True, host=None)
    assert ids.transcript_roots(config) == {
        "alice-claude-projects": "alice",
        "bob-claude-projects": "bob",
        "dispatch_claude_projects": ids.SHARED_DISPATCH_IDENTITY,
    }
    with pytest.raises(ValueError, match="host_identity"):
        ids.transcript_source_identity(config, "host")


@pytest.mark.parametrize(
    "root", ["claude-projects", "-claude-projects", "workspace", "dispatch_claude_projects_0"]
)
def test_unknown_transcript_roots_are_refused(root):
    with pytest.raises(ValueError):
        ids.transcript_source_identity(_config(), root)


def test_identity_classes():
    config = _config(workers=1, bridges=("teams_bridge",))
    assert ids.identity_class(config, "alice") == c.SOURCE_TERMINAL
    assert ids.identity_class(config, "dispatch-worker-1") == c.SOURCE_DISPATCH
    assert ids.identity_class(config, "teams-bridge") == c.SOURCE_BRIDGE
    assert ids.identity_class(config, "sidecar") == c.SOURCE_SIDECAR
    assert ids.identity_class(config, "builder") == c.SOURCE_HOST
    assert ids.identity_class(config, "history") == c.SOURCE_HISTORY
    assert ids.identity_class(config, "operator") == c.SOURCE_OPERATOR
    assert ids.identity_class(config, "reader:ops") == c.SOURCE_READER
    assert ids.identity_class(config, "bluesky-web") == c.SOURCE_SERVICE


def test_allow_list_through_the_identity_map():
    config = _config(workers=2, shared=True, bridges=("gchat_bridge",))
    worker = ids.spool_source_identity(config, "dispatch-worker-2")
    bridge = ids.spool_source_identity(config, "gchat-bridge")
    spool = c.ROUTE_SPOOL
    assert c.kind_allowed("run", route=spool, source_class=ids.identity_class(config, worker))
    assert not c.kind_allowed("run", route=spool, source_class=ids.identity_class(config, bridge))
    assert c.kind_allowed(
        "bridge_turn", route=spool, source_class=ids.identity_class(config, bridge)
    )
    assert not c.kind_allowed(
        "bridge_turn", route=spool, source_class=ids.identity_class(config, "alice")
    )


# --- the trusted on-behalf list ----------------------------------------------------------


@pytest.mark.parametrize("lanes", [False, True])
def test_trusted_on_behalf_is_the_sidecar_and_the_queueservers(lanes):
    config = _config(lanes=lanes)
    trusted = ids.trusted_on_behalf(config)
    assert trusted.sidecar == AUTH_SIDECAR_AUDIT_IDENTITY
    assert trusted.queueservers == (("queueserver",) if lanes else ())


def test_on_behalf_of_kept_for_sidecar_everywhere_and_queueservers_on_preflight_only():
    trusted = ids.trusted_on_behalf(_config(lanes=True))
    assert ids.keep_on_behalf_of(trusted, "sidecar", "http")
    assert ids.keep_on_behalf_of(trusted, "sidecar", None)
    assert ids.keep_on_behalf_of(trusted, "queueserver", "preflight")
    assert not ids.keep_on_behalf_of(trusted, "queueserver", "http")
    assert not ids.keep_on_behalf_of(trusted, "alice", "preflight")
    assert not ids.keep_on_behalf_of(trusted, "dispatch-worker-1", None)


# --- the credential → scope table (D18) --------------------------------------------------


def _full_config(**kw) -> dict:
    return _config(
        workers=2,
        bridges=("teams_bridge", "gchat_bridge", "nextcloud_bridge"),
        readers=[
            {"name": "ops", "kinds": ["audit", "run"], "days_back": 30},
            {"name": "all-kinds"},
        ],
        **kw,
    )


def test_credential_table_names_every_d18_login_with_its_scope():
    table = ids.credential_table(_full_config())
    assert list(table) == [
        "alice",
        "bob",
        "dispatch-worker-1",
        "dispatch-worker-2",
        "teams-bridge",
        "gchat-bridge",
        "nextcloud-bridge",
        "sidecar",
        "builder",
        "history",
        "operator",
        "reader.ops",
        "reader.all-kinds",
    ]
    assert table["alice"].scope == c.Own("alice")
    assert table["builder"].scope == c.Own("builder")
    assert table["dispatch-worker-1"].scope == c.Source("dispatch-worker-1")
    assert table["teams-bridge"].scope == c.Source("teams-bridge")
    assert table["sidecar"].scope == c.Source("sidecar")
    assert table["operator"].scope == c.All()
    assert table["reader.ops"].scope == c.Reader("ops", frozenset({"audit", "run"}), 30)
    assert table["reader.ops"].identity == "reader:ops"
    assert table["reader.all-kinds"].scope == c.Reader("all-kinds")


def test_only_history_may_assert_a_viewer():
    table = ids.credential_table(_full_config())
    asserting = [login for login, grant in table.items() if grant.may_assert_viewer]
    assert asserting == ["history"]
    assert isinstance(table["history"].scope, ids.ViewerAsserted)
    assert ids.VIEWER_HEADER == "X-Osprey-Record-Viewer"


def test_shared_mode_has_one_worker_login():
    table = ids.credential_table(_full_config(shared=True))
    workers = [g for g in table.values() if g.source_class == c.SOURCE_DISPATCH]
    assert [g.login for g in workers] == [ids.SHARED_DISPATCH_IDENTITY]


def test_no_login_contains_a_colon():
    table = ids.credential_table(_full_config())
    for login, grant in table.items():
        assert ":" not in login
        assert grant.login == login
        assert ids.login_for(grant.identity) == login


def test_login_mapping_for_readers():
    assert ids.login_for("reader:ops") == "reader.ops"
    assert ids.login_for("alice") == "alice"
    with pytest.raises(ValueError):
        ids.login_for("reader:")


def test_minimal_deployment_still_has_history_and_operator():
    table = ids.credential_table(_config(roster=False, host=None))
    assert list(table) == ["history", "operator"]


@pytest.mark.parametrize("host", ["operator", "history", "alice"])
def test_a_login_claimed_twice_is_refused_naming_both(host):
    with pytest.raises(ValueError, match="claimed twice") as info:
        ids.credential_table(_config(host=host))
    assert host in str(info.value)


def test_a_reader_with_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown kind"):
        ids.credential_table(_config(readers=[{"name": "ops", "kinds": ["gossip"]}]))


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"deployment": {"host_identity": "  builder "}}, "builder"),
        ({"deployment": {"host_identity": "   "}}, None),
        ({"deployment": {"host_identity": 7}}, None),
        ({"deployment": "builder"}, None),
        ({}, None),
        (None, None),
    ],
    ids=["stripped", "blank", "not-a-string", "deployment-not-a-mapping", "absent", "no-config"],
)
def test_host_identity_is_the_stamped_name_or_none(config, expected):
    assert ids.host_identity(config) == expected
