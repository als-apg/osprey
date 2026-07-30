"""NextcloudBridgeConfig: env parsing, the CoreConfig projection, startup require."""

import dataclasses
import logging

import pytest

from osprey.bridges.core import CoreConfig
from osprey.bridges.nextcloud_talk import NextcloudBridgeConfig

CONFIG_LOGGER = "osprey.bridges.nextcloud_talk.config"

# A fully-configured deployment: the seven values require_startup() insists on.
COMPLETE_ENV = {
    "NEXTCLOUD_BASE_URL": "https://cloud.example.org",
    "NEXTCLOUD_BOT_ACCOUNT": "osprey-bot",
    "NEXTCLOUD_APP_PASSWORD": "app-pw",
    "NEXTCLOUD_ROOMS": "roomA,roomB",
    "DISPATCH_TRIGGER": "nextcloud-question",
    "EVENT_DISPATCHER_TOKEN": "disp-token",
    "DISPATCH_WORKER_TOKEN": "work-token",
}


def complete(**overrides: str) -> dict[str, str]:
    """A complete env with `overrides` applied (an empty value means "unset")."""
    return {**COMPLETE_ENV, **overrides}


# --- from_env parsing -----------------------------------------------------


def test_from_env_maps_nextcloud_fields():
    cfg = NextcloudBridgeConfig.from_env(complete(OFFSETS_PATH="/data/nc_offsets.json"))
    assert cfg.base_url == "https://cloud.example.org"
    assert cfg.bot_account == "osprey-bot"
    assert cfg.app_password == "app-pw"
    assert cfg.rooms == ("roomA", "roomB")
    assert cfg.offsets_path == "/data/nc_offsets.json"


def test_from_env_defaults_when_unset():
    cfg = NextcloudBridgeConfig.from_env({})
    assert cfg.base_url == ""
    assert cfg.bot_account == ""
    assert cfg.app_password == ""
    assert cfg.rooms == ()
    # Same bridge volume as the core stores, and the same naming shape as
    # dedup_path/history_path.
    assert cfg.offsets_path == "/data/offsets.json"


def test_dataclass_defaults_match_from_env_defaults():
    # A directly-constructed config is just as unconfigured, so nothing can be
    # inherited by accident by a collaborator that builds one itself.
    cfg = NextcloudBridgeConfig()
    assert (cfg.base_url, cfg.bot_account, cfg.app_password, cfg.rooms) == ("", "", "", ())
    assert cfg.offsets_path == "/data/offsets.json"


def test_from_env_reads_os_environ_when_no_mapping_given(monkeypatch):
    for name, value in complete().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("NEXTCLOUD_ROOMS", " tokenX , tokenY ")
    cfg = NextcloudBridgeConfig.from_env()
    assert cfg.bot_account == "osprey-bot"
    assert cfg.rooms == ("tokenX", "tokenY")
    assert cfg.core.trigger == "nextcloud-question"


# --- base_url trailing-slash stripping -----------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://cloud.example.org/", "https://cloud.example.org"),
        ("https://cloud.example.org///", "https://cloud.example.org"),
        ("https://cloud.example.org", "https://cloud.example.org"),
        # A subpath install keeps its path, minus the trailing slash: every OCS
        # and DAV path the client builds is appended to this.
        ("https://example.org/nextcloud/", "https://example.org/nextcloud"),
    ],
)
def test_base_url_trailing_slash_is_stripped(raw, expected):
    assert NextcloudBridgeConfig.from_env(complete(NEXTCLOUD_BASE_URL=raw)).base_url == expected


# --- NEXTCLOUD_ROOMS splitting -------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ()),
        ("   ", ()),
        (",", ()),
        (",,,", ()),
        ("solo", ("solo",)),
        ("a,b,c", ("a", "b", "c")),
        # Whitespace around entries: a wrapped compose line or a copy-paste.
        (" a , b ", ("a", "b")),
        ("\ta\n,\tb\n", ("a", "b")),
        # Trailing / leading / doubled separators drop out rather than producing
        # an empty room token that would spawn a thread polling nothing.
        ("a,b,", ("a", "b")),
        (",a,b", ("a", "b")),
        ("a,,b", ("a", "b")),
        ("a, ,b", ("a", "b")),
        # Order is preserved — the room threads are started in written order.
        ("z,a", ("z", "a")),
    ],
)
def test_rooms_splitting(raw, expected):
    assert NextcloudBridgeConfig.from_env(complete(NEXTCLOUD_ROOMS=raw)).rooms == expected


# --- CoreConfig is composed, not subclassed ------------------------------


def test_core_is_a_composed_projection_not_a_base_class():
    cfg = NextcloudBridgeConfig.from_env(complete())
    assert isinstance(cfg.core, CoreConfig)
    assert not isinstance(cfg, CoreConfig)
    assert CoreConfig not in type(cfg).__mro__


def test_neutral_vars_are_delegated_to_core_config():
    cfg = NextcloudBridgeConfig.from_env(
        complete(
            DISPATCHER_URL="http://disp:8010/",
            WORKER_URL="http://work:9190/",
            POLL_INTERVAL="0.5",
            DEDUP_PATH="/data/nc_dedup.json",
            HISTORY_PATH="/data/nc_history.json",
        )
    )
    assert cfg.core.dispatcher_url == "http://disp:8010"
    assert cfg.core.worker_url == "http://work:9190"
    assert cfg.core.event_dispatcher_token == "disp-token"
    assert cfg.core.dispatch_worker_token == "work-token"
    assert cfg.core.trigger == "nextcloud-question"
    assert cfg.core.poll_interval == 0.5
    assert cfg.core.dedup_path == "/data/nc_dedup.json"
    assert cfg.core.history_path == "/data/nc_history.json"


def test_config_is_immutable():
    # One instance is shared by every room thread, the ChannelOps instance, and
    # the drain thread; build variants with dataclasses.replace instead.
    cfg = NextcloudBridgeConfig.from_env(complete())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.base_url = "https://elsewhere.example.org"
    assert dataclasses.replace(cfg, rooms=("other",)).rooms == ("other",)


# --- no trigger default --------------------------------------------------


def test_from_env_applies_no_trigger_default():
    # The nextcloud-question default belongs to the deployment surface (rendered
    # as DISPATCH_TRIGGER in the compose template). If from_env defaulted it, a
    # hand-rolled deployment would silently dispatch to a trigger nobody chose
    # and require_startup could never catch the omission.
    assert NextcloudBridgeConfig.from_env({}).core.trigger == ""
    assert NextcloudBridgeConfig.from_env(complete(DISPATCH_TRIGGER="")).core.trigger == ""
    assert NextcloudBridgeConfig().core.trigger == ""


def test_missing_trigger_is_what_require_startup_catches():
    cfg = NextcloudBridgeConfig.from_env(complete(DISPATCH_TRIGGER=""))
    with pytest.raises(ValueError, match="DISPATCH_TRIGGER"):
        cfg.require_startup()


# --- require_startup -----------------------------------------------------


def test_require_startup_passes_on_a_complete_config():
    NextcloudBridgeConfig.from_env(complete()).require_startup()  # no raise


@pytest.mark.parametrize("name", sorted(COMPLETE_ENV))
def test_require_startup_names_each_missing_var(name):
    cfg = NextcloudBridgeConfig.from_env(complete(**{name: ""}))
    with pytest.raises(ValueError, match=name) as excinfo:
        cfg.require_startup()
    # Only the one that is actually missing is reported.
    others = [other for other in COMPLETE_ENV if other != name]
    assert not [other for other in others if other in str(excinfo.value)]


def test_require_startup_lists_every_missing_var_in_one_raise():
    # A wholly unconfigured process must not need seven restarts to learn what it
    # is missing.
    with pytest.raises(ValueError) as excinfo:
        NextcloudBridgeConfig.from_env({}).require_startup()
    message = str(excinfo.value)
    assert message.startswith("missing required config: ")
    for name in COMPLETE_ENV:
        assert name in message


def test_require_startup_requires_at_least_one_room():
    cfg = NextcloudBridgeConfig.from_env(complete(NEXTCLOUD_ROOMS=" , "))
    with pytest.raises(ValueError, match="NEXTCLOUD_ROOMS"):
        cfg.require_startup()
    # One room is enough.
    NextcloudBridgeConfig.from_env(complete(NEXTCLOUD_ROOMS="only")).require_startup()


def test_require_startup_requires_both_dispatch_tokens():
    cfg = NextcloudBridgeConfig.from_env(
        complete(EVENT_DISPATCHER_TOKEN="", DISPATCH_WORKER_TOKEN="")
    )
    with pytest.raises(ValueError) as excinfo:
        cfg.require_startup()
    assert "EVENT_DISPATCHER_TOKEN" in str(excinfo.value)
    assert "DISPATCH_WORKER_TOKEN" in str(excinfo.value)


# --- non-https warns, never raises ---------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["http://cloud.example.org", "HTTP://cloud.example.org", "cloud.example.org"],
)
def test_non_https_base_url_warns_and_does_not_raise(raw, caplog):
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        cfg = NextcloudBridgeConfig.from_env(complete(NEXTCLOUD_BASE_URL=raw))
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert warnings, "a non-https base URL must warn"
    assert "https" in warnings[0].getMessage()
    # Warned, not rejected: a facility may run against an internal host, and the
    # startup require must still pass.
    cfg.require_startup()


@pytest.mark.parametrize("raw", ["https://cloud.example.org", "HTTPS://cloud.example.org"])
def test_https_base_url_does_not_warn(raw, caplog):
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        NextcloudBridgeConfig.from_env(complete(NEXTCLOUD_BASE_URL=raw))
    assert not [rec for rec in caplog.records if rec.levelno == logging.WARNING]


def test_unset_base_url_does_not_warn_about_https(caplog):
    # An empty base_url is a *missing var*, which require_startup reports; warning
    # about its scheme too would just be noise.
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        NextcloudBridgeConfig.from_env({})
    assert not [rec for rec in caplog.records if rec.levelno == logging.WARNING]
