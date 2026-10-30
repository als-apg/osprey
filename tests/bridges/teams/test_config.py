"""TeamsBridgeConfig: env parsing, the cloud table, the version fallback, the startup require."""

import dataclasses
from importlib import metadata

import pytest

from osprey.bridges.teams import TeamsBridgeConfig
from osprey.bridges.teams import config as config_module

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


# The installed distribution's version — what from_env falls back to when the
# image carries no APP_VERSION_DISPLAY. Read here rather than hard-coded: the
# point is that the two agree, not what either one says today.
INSTALLED_VERSION = metadata.version("osprey-framework")


def complete(**overrides: str) -> dict[str, str]:
    """A complete env with `overrides` applied (an empty value means "unset")."""
    return {**COMPLETE_ENV, **overrides}


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
    assert cfg.mentions is True


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


# --- the cloud table ------------------------------------------------------


@pytest.mark.parametrize(
    ("cloud", "login_host", "token_scope", "graph_host", "graph_scope"),
    [
        (
            "commercial",
            "login.microsoftonline.com",
            "https://api.botframework.com/.default",
            "graph.microsoft.com",
            "https://graph.microsoft.com/.default",
        ),
        (
            "gcchigh",
            "login.microsoftonline.us",
            "https://api.botframework.us/.default",
            "graph.microsoft.us",
            "https://graph.microsoft.us/.default",
        ),
    ],
)
def test_each_cloud_row_selects_its_login_host_and_scope(
    cloud, login_host, token_scope, graph_host, graph_scope
):
    cfg = TeamsBridgeConfig.from_env(complete(TEAMS_CLOUD=cloud))
    assert cfg.login_host == login_host
    assert cfg.token_scope == token_scope
    assert cfg.graph_host == graph_host
    assert cfg.graph_scope == graph_scope


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


@pytest.mark.parametrize(
    "name", ["TEAMS_CLOUD", "APP_VERSION_DISPLAY", "TEAMS_FILES_DRIVE_ID", "TEAMS_FILES_FOLDER"]
)
def test_require_startup_ignores_the_optional_vars(name):
    # The cloud has a default, the version tag only plainens the ack, and without a
    # file library files are named rather than shared: none of them is a bridge that
    # must refuse to start.
    env = complete(
        TEAMS_CLOUD="gcchigh",
        APP_VERSION_DISPLAY="v1",
        TEAMS_FILES_DRIVE_ID="b!drive",
        TEAMS_FILES_FOLDER="osprey",
    )
    env[name] = ""
    if name == "TEAMS_FILES_DRIVE_ID":
        env["TEAMS_FILES_FOLDER"] = ""
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


# --- the file library -----------------------------------------------------------


def test_from_env_maps_the_file_library_pair():
    cfg = TeamsBridgeConfig.from_env(
        complete(TEAMS_FILES_DRIVE_ID="b!abc_DEF-123", TEAMS_FILES_FOLDER="/osprey/answers/")
    )
    assert cfg.files_drive_id == "b!abc_DEF-123"
    assert cfg.files_folder == "osprey/answers"


@pytest.mark.parametrize("env", [{}, {"TEAMS_FILES_DRIVE_ID": "", "TEAMS_FILES_FOLDER": ""}])
def test_the_file_library_is_unset_by_default(env):
    # Compose renders both bare, so an unset pair arrives as "" as well as absent.
    cfg = TeamsBridgeConfig.from_env(complete(**env))
    assert cfg.files_drive_id == ""
    assert cfg.files_folder == ""
    assert cfg.files_folder_segments == ()


def test_a_folder_without_a_library_is_refused():
    with pytest.raises(
        ValueError, match="TEAMS_FILES_FOLDER is set but TEAMS_FILES_DRIVE_ID is not"
    ):
        TeamsBridgeConfig.from_env(complete(TEAMS_FILES_FOLDER="osprey"))


def test_a_folder_is_split_into_segments_without_its_slashes():
    cfg = TeamsBridgeConfig.from_env(
        complete(TEAMS_FILES_DRIVE_ID="b!drive", TEAMS_FILES_FOLDER="/Osprey files//runs/")
    )
    assert cfg.files_folder_segments == ("Osprey files", "runs")


@pytest.mark.parametrize("folder", ["..", ".", "a:b", "a?b", "runs/../other"])
def test_a_folder_segment_that_could_leave_the_library_is_refused(folder):
    with pytest.raises(ValueError, match="TEAMS_FILES_FOLDER"):
        TeamsBridgeConfig.from_env(
            complete(TEAMS_FILES_DRIVE_ID="b!drive", TEAMS_FILES_FOLDER=folder)
        )


@pytest.mark.parametrize("drive_id", ["b!a/b", "b!a?x", "b!a#x", "b!a b", "b!a\tb"])
def test_a_drive_id_that_is_not_one_path_segment_is_refused(drive_id):
    with pytest.raises(ValueError, match="TEAMS_FILES_DRIVE_ID"):
        TeamsBridgeConfig.from_env(complete(TEAMS_FILES_DRIVE_ID=drive_id))
