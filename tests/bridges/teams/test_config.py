"""TeamsBridgeConfig: env parsing, the cloud table, the version fallback, the boot requires."""

import dataclasses
from importlib import metadata

import pytest

from osprey.bridges.core import CoreConfig
from osprey.bridges.teams import TeamsBridgeConfig, require_boot
from osprey.bridges.teams import config as config_module
from osprey.port_layout import default_port

# A fully-configured deployment: the eight values require_startup() insists on.
COMPLETE_ENV = {
    "TEAMS_APP_ID": "11111111-2222-3333-4444-555555555555",
    "TEAMS_APP_SECRET": "app-secret",
    "TEAMS_TENANT_ID": "66666666-7777-8888-9999-000000000000",
    "TEAMS_SERVICEBUS_CONNECTION_STRING": (
        "Endpoint=sb://facility.servicebus.windows.net/;"
        "SharedAccessKeyName=bridge-listen;SharedAccessKey=key"
    ),
    "TEAMS_SERVICEBUS_QUEUE": "osprey-teams-events",
    "DISPATCH_TRIGGER": "teams-question",
    "EVENT_DISPATCHER_TOKEN": "disp-token",
    "DISPATCH_WORKER_TOKEN": "work-token",
}

# What require_boot checks on top of require_startup. Bare in compose, so an
# unset one arrives as "" and no code default ever replaces it.
BOOT_URL_ENV = {
    "DISPATCHER_URL": "http://dispatcher:10010",
    "WORKER_URL": "http://worker:10011",
}

# The installed distribution's version — what from_env falls back to when the
# image carries no APP_VERSION_DISPLAY. Read here rather than hard-coded: the
# point is that the two agree, not what either one says today.
INSTALLED_VERSION = metadata.version("osprey-framework")


def complete(**overrides: str) -> dict[str, str]:
    """A complete env with `overrides` applied (an empty value means "unset")."""
    return {**COMPLETE_ENV, **overrides}


def bootable(**overrides: str) -> dict[str, str]:
    """A complete env that also carries the two URLs require_boot checks."""
    return {**COMPLETE_ENV, **BOOT_URL_ENV, **overrides}


# --- from_env parsing -----------------------------------------------------


def test_from_env_maps_the_teams_specific_fields():
    cfg = TeamsBridgeConfig.from_env(complete(TEAMS_CLOUD="gcchigh"))
    assert cfg.app_id == "11111111-2222-3333-4444-555555555555"
    assert cfg.app_secret == "app-secret"
    assert cfg.tenant_id == "66666666-7777-8888-9999-000000000000"
    assert cfg.servicebus_connection_string == COMPLETE_ENV["TEAMS_SERVICEBUS_CONNECTION_STRING"]
    assert cfg.servicebus_queue == "osprey-teams-events"
    assert cfg.cloud == "gcchigh"


def test_from_env_defaults_every_teams_credential_to_empty_when_unset():
    cfg = TeamsBridgeConfig.from_env({})
    assert cfg.app_id == ""
    assert cfg.app_secret == ""
    assert cfg.tenant_id == ""
    assert cfg.servicebus_connection_string == ""
    assert cfg.servicebus_queue == ""


def test_dataclass_defaults_match_from_env_defaults():
    # A directly-constructed config is just as unconfigured, so nothing can be
    # inherited by accident by a collaborator that builds one itself.
    cfg = TeamsBridgeConfig()
    assert (cfg.app_id, cfg.app_secret, cfg.tenant_id) == ("", "", "")
    assert (cfg.servicebus_connection_string, cfg.servicebus_queue) == ("", "")
    assert cfg.cloud == "commercial"


def test_from_env_reads_os_environ_when_no_mapping_given(monkeypatch):
    for name, value in complete().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TEAMS_CLOUD", "gcchigh")
    cfg = TeamsBridgeConfig.from_env()
    assert cfg.app_id == "11111111-2222-3333-4444-555555555555"
    assert cfg.cloud == "gcchigh"
    assert cfg.core.trigger == "teams-question"


@pytest.mark.parametrize(
    "prefixed",
    ["TEAMS_DISPATCH_TRIGGER", "TEAMS_POLL_BUDGET", "TEAMS_DEDUP_PATH", "TEAMS_VERSION_TAG"],
)
def test_prefixed_spellings_of_shared_names_are_not_read(prefixed):
    # The neutral half plus APP_VERSION_DISPLAY are names shared with every other
    # adapter; a TEAMS_-prefixed spelling must not quietly configure one.
    cfg = TeamsBridgeConfig.from_env({prefixed: "999"})
    assert cfg.core.trigger == ""
    assert cfg.core.poll_budget == 330.0
    assert cfg.core.dedup_path == "/data/dedup.json"
    assert cfg.version_tag == INSTALLED_VERSION


def test_neutral_vars_are_delegated_to_core_config():
    cfg = TeamsBridgeConfig.from_env(
        complete(
            DISPATCHER_URL="http://disp:10010/",
            WORKER_URL="http://work:10011/",
            POLL_INTERVAL="0.5",
            DEDUP_PATH="/data/teams_dedup.json",
            HISTORY_PATH="/data/teams_history.json",
        )
    )
    assert cfg.core.dispatcher_url == "http://disp:10010"
    assert cfg.core.worker_url == "http://work:10011"
    assert cfg.core.event_dispatcher_token == "disp-token"
    assert cfg.core.dispatch_worker_token == "work-token"
    assert cfg.core.trigger == "teams-question"
    assert cfg.core.poll_interval == 0.5
    assert cfg.core.dedup_path == "/data/teams_dedup.json"
    assert cfg.core.history_path == "/data/teams_history.json"


# --- CoreConfig is composed, not subclassed ------------------------------


def test_core_is_a_composed_projection_not_a_base_class():
    cfg = TeamsBridgeConfig.from_env(complete())
    assert isinstance(cfg.core, CoreConfig)
    assert not isinstance(cfg, CoreConfig)
    assert CoreConfig not in type(cfg).__mro__


def test_config_is_immutable():
    # One instance is shared by the Service Bus receive loop, the ChannelOps
    # instance and the drain thread; build variants with dataclasses.replace.
    cfg = TeamsBridgeConfig.from_env(complete())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.app_id = "other"
    assert dataclasses.replace(cfg, app_id="other").app_id == "other"


def test_from_env_applies_no_trigger_default():
    # The teams-question default belongs to the deployment surface (rendered as
    # DISPATCH_TRIGGER in the compose template). If from_env defaulted it, a
    # hand-rolled deployment would silently dispatch to a trigger nobody chose
    # and require_startup could never catch the omission.
    assert TeamsBridgeConfig.from_env({}).core.trigger == ""
    assert TeamsBridgeConfig.from_env(complete(DISPATCH_TRIGGER="")).core.trigger == ""
    assert TeamsBridgeConfig().core.trigger == ""


# --- the cloud table ------------------------------------------------------


@pytest.mark.parametrize(
    ("cloud", "login_host", "token_scope"),
    [
        ("commercial", "login.microsoftonline.com", "https://api.botframework.com/.default"),
        ("gcchigh", "login.microsoftonline.us", "https://api.botframework.us/.default"),
    ],
)
def test_each_cloud_row_selects_its_login_host_and_scope(cloud, login_host, token_scope):
    cfg = TeamsBridgeConfig.from_env(complete(TEAMS_CLOUD=cloud))
    assert cfg.login_host == login_host
    assert cfg.token_scope == token_scope


def test_the_cloud_table_carries_exactly_the_two_supported_clouds():
    assert sorted(config_module.CLOUDS) == ["commercial", "gcchigh"]


def test_an_unset_or_empty_cloud_is_the_commercial_cloud():
    # TEAMS_CLOUD is optional and rendered bare in compose, so an unset one
    # arrives as "" rather than as an absent key. Both spellings of "the
    # operator said nothing" must land on the default rather than raising.
    assert TeamsBridgeConfig.from_env(complete()).cloud == "commercial"
    assert TeamsBridgeConfig.from_env(complete(TEAMS_CLOUD="")).cloud == "commercial"


@pytest.mark.parametrize("raw", ["gov", "GCCHigh", "usgov", "commercial "])
def test_an_unknown_cloud_is_rejected_at_construction(raw):
    # A cloud nobody supports must fail at boot, not at the first token request
    # inside a worker thread: the wrong login host would otherwise surface as an
    # authentication failure that says nothing about the misspelling.
    with pytest.raises(ValueError) as excinfo:
        TeamsBridgeConfig.from_env(complete(TEAMS_CLOUD=raw))
    message = str(excinfo.value)
    assert "TEAMS_CLOUD" in message
    assert "commercial" in message
    assert "gcchigh" in message


def test_an_unknown_cloud_is_rejected_on_a_direct_construction_too():
    # from_env is not the only way in — a collaborator building a variant with
    # dataclasses.replace must hit the same check.
    with pytest.raises(ValueError, match="TEAMS_CLOUD"):
        TeamsBridgeConfig(cloud="gov")
    cfg = TeamsBridgeConfig.from_env(complete())
    with pytest.raises(ValueError, match="TEAMS_CLOUD"):
        dataclasses.replace(cfg, cloud="gov")


# --- version_tag: set once, in from_env -----------------------------------


def test_version_tag_prefers_the_image_build_arg():
    # APP_VERSION_DISPLAY is baked into the image by the build and names the
    # deployed release, which is more precise than the installed wheel's version.
    cfg = TeamsBridgeConfig.from_env(complete(APP_VERSION_DISPLAY="v2026.8.0+abc1234"))
    assert cfg.version_tag == "v2026.8.0+abc1234"


@pytest.mark.parametrize("raw", [None, ""])
def test_version_tag_falls_back_to_the_installed_distribution(raw):
    # Unset *and* empty both mean "the build arg said nothing": compose renders an
    # unset bare ${APP_VERSION_DISPLAY} as "", so only falling back on the absent
    # key would leave every composed deployment with a blank ack suffix.
    env = complete() if raw is None else complete(APP_VERSION_DISPLAY=raw)
    assert TeamsBridgeConfig.from_env(env).version_tag == INSTALLED_VERSION


def test_version_tag_is_empty_when_the_distribution_is_not_installed(monkeypatch):
    # A source checkout run straight from src/ has no installed distribution.
    # That is a plainer ack, never a boot failure.
    def not_installed(_name: str) -> str:
        raise metadata.PackageNotFoundError("osprey-framework")

    monkeypatch.setattr(config_module.metadata, "version", not_installed)
    assert TeamsBridgeConfig.from_env(complete()).version_tag == ""


def test_version_tag_is_resolved_once_in_from_env(monkeypatch):
    # The lookup is a metadata read, not a property: ops appends the tag to every
    # ack, and re-reading the distribution per message would be a per-message cost
    # for a value that cannot change while the process runs.
    calls: list[str] = []

    def counting(name: str) -> str:
        calls.append(name)
        return "v9.9.9"

    monkeypatch.setattr(config_module.metadata, "version", counting)
    cfg = TeamsBridgeConfig.from_env(complete())
    assert cfg.version_tag == "v9.9.9"
    assert cfg.version_tag == "v9.9.9"
    assert calls == ["osprey-framework"]


def test_the_build_arg_short_circuits_the_distribution_lookup(monkeypatch):
    def never(_name: str) -> str:
        raise AssertionError("APP_VERSION_DISPLAY was set; nothing should be looked up")

    monkeypatch.setattr(config_module.metadata, "version", never)
    assert TeamsBridgeConfig.from_env(complete(APP_VERSION_DISPLAY="v1")).version_tag == "v1"


# --- require_startup ------------------------------------------------------


def test_require_startup_passes_on_a_complete_config():
    TeamsBridgeConfig.from_env(complete()).require_startup()  # no raise


@pytest.mark.parametrize("name", sorted(COMPLETE_ENV))
def test_require_startup_names_each_missing_var(name):
    cfg = TeamsBridgeConfig.from_env(complete(**{name: ""}))
    with pytest.raises(ValueError, match=name) as excinfo:
        cfg.require_startup()
    # Only the one that is actually missing is reported.
    others = [other for other in COMPLETE_ENV if other != name]
    assert not [other for other in others if other in str(excinfo.value)]


def test_require_startup_lists_every_missing_var_in_one_raise():
    # A wholly unconfigured process must not need eight restarts to learn what it
    # is missing.
    with pytest.raises(ValueError) as excinfo:
        TeamsBridgeConfig.from_env({}).require_startup()
    message = str(excinfo.value)
    assert message.startswith("missing required config: ")
    for name in COMPLETE_ENV:
        assert name in message


@pytest.mark.parametrize("name", ["TEAMS_CLOUD", "APP_VERSION_DISPLAY"])
def test_require_startup_ignores_the_optional_vars(name):
    # The cloud has a default and the version tag only plainens the ack: neither
    # is a bridge that must refuse to start.
    env = complete(TEAMS_CLOUD="gcchigh", APP_VERSION_DISPLAY="v1")
    env[name] = ""
    TeamsBridgeConfig.from_env(env).require_startup()  # no raise


def test_require_startup_requires_both_halves_of_the_queue_coordinates():
    # A connection string without a queue name, or the reverse, cannot receive
    # anything — and would otherwise fail only inside the receive loop.
    cfg = TeamsBridgeConfig.from_env(
        complete(TEAMS_SERVICEBUS_CONNECTION_STRING="", TEAMS_SERVICEBUS_QUEUE="")
    )
    with pytest.raises(ValueError) as excinfo:
        cfg.require_startup()
    assert "TEAMS_SERVICEBUS_CONNECTION_STRING" in str(excinfo.value)
    assert "TEAMS_SERVICEBUS_QUEUE" in str(excinfo.value)


# --- require_boot: the bare-${VAR} compose trap ---------------------------


def test_require_boot_passes_when_the_urls_are_set():
    require_boot(TeamsBridgeConfig.from_env(bootable()))  # no raise


@pytest.mark.parametrize("name", sorted(BOOT_URL_ENV))
def test_require_boot_rejects_a_url_that_rendered_empty(name):
    # An unset bare ${DISPATCHER_URL} in compose reaches the process as "", not as
    # an absent key, so CoreConfig.from_env's localhost fallback never fires. Left
    # unchecked the bridge would POST to a protocol-less URL forever.
    cfg = TeamsBridgeConfig.from_env(bootable(**{name: ""}))
    with pytest.raises(ValueError, match=name):
        require_boot(cfg)


def test_require_boot_names_env_vars_not_field_names():
    cfg = TeamsBridgeConfig.from_env(bootable(DISPATCHER_URL="", WORKER_URL=""))
    with pytest.raises(ValueError) as excinfo:
        require_boot(cfg)
    message = str(excinfo.value)
    assert "DISPATCHER_URL" in message
    assert "WORKER_URL" in message
    # The engine's own check reports field names; the abort has to be diagnosable
    # from the container log, where only the env spelling exists.
    assert "dispatcher_url" not in message
    assert "worker_url" not in message


def test_require_boot_reports_the_startup_vars_before_the_urls():
    # require_startup runs first, so a wholly unconfigured process is told about
    # its credentials rather than only about two URLs it never set either.
    with pytest.raises(ValueError) as excinfo:
        require_boot(TeamsBridgeConfig.from_env({}))
    assert "TEAMS_APP_ID" in str(excinfo.value)


def test_require_startup_does_not_cover_the_urls():
    # The split is deliberate: require_startup is the credentials check and
    # require_boot adds the compose-trap check on top.
    cfg = TeamsBridgeConfig.from_env(complete(DISPATCHER_URL=""))
    cfg.require_startup()  # no raise
    with pytest.raises(ValueError, match="DISPATCHER_URL"):
        require_boot(cfg)


def test_absent_url_vars_still_fall_back_to_the_code_defaults():
    # Only the *empty* render is the trap. A deployment that never mentions the
    # URLs at all is the local-dev shape CoreConfig's defaults exist for, and
    # require_boot must not break it.
    cfg = TeamsBridgeConfig.from_env(complete())
    assert (cfg.core.dispatcher_url, cfg.core.worker_url) == (
        f"http://localhost:{default_port('dispatcher')}",
        f"http://localhost:{default_port('worker', 1)}",
    )
    require_boot(cfg)  # no raise


# --- no shipped secret ----------------------------------------------------


def test_no_field_default_carries_a_credential():
    # Every credential-shaped field must default to empty: a shipped literal would
    # be one facility's secret compiled into every deployment.
    credentials = {
        "app_id",
        "app_secret",
        "tenant_id",
        "servicebus_connection_string",
        "servicebus_queue",
    }
    assert not [
        f
        for f in dataclasses.fields(TeamsBridgeConfig)
        if f.name in credentials and f.default != ""
    ]
