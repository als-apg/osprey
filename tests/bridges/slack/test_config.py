"""SlackBridgeConfig: env parsing, the version fallback, token redaction, the startup require."""

import dataclasses
from importlib import metadata

import pytest

from osprey.bridges.core import CoreConfig
from osprey.bridges.slack import SlackBridgeConfig, require_boot
from osprey.bridges.slack import config as config_module

# A fully-configured deployment: the five values require_startup() insists on.
# The first key is a credential — the cross-adapter contract suite relies on that.
COMPLETE_ENV = {
    "SLACK_BOT_TOKEN": "xoxb-1111-2222-bot-secret",
    "SLACK_APP_TOKEN": "xapp-1-A111-3333-app-secret",  # gitleaks:allow - a fake test token
    "DISPATCH_TRIGGER": "slack-question",
    "EVENT_DISPATCHER_TOKEN": "disp-token",
    "DISPATCH_WORKER_TOKEN": "work-token",
}

# The installed distribution's version — what from_env falls back to when the
# image carries no APP_VERSION_DISPLAY.
INSTALLED_VERSION = metadata.version("osprey-framework")

REQUIRED = list(COMPLETE_ENV)


def complete(**overrides: str) -> dict[str, str]:
    """A complete env with `overrides` applied (an empty value means "unset")."""
    return {**COMPLETE_ENV, **overrides}


# --- from_env parsing -----------------------------------------------------


def test_from_env_maps_the_slack_specific_fields():
    cfg = SlackBridgeConfig.from_env(complete())
    assert cfg.bot_token == "xoxb-1111-2222-bot-secret"
    assert cfg.app_token == "xapp-1-A111-3333-app-secret"  # gitleaks:allow - a fake test token
    assert cfg.mentions is True


def test_from_env_delegates_neutral_names_unprefixed():
    cfg = SlackBridgeConfig.from_env(
        complete(POLL_BUDGET="400", DISPATCHER_URL="http://d:1/", BRIDGE_TRUST_ENV="true")
    )
    assert isinstance(cfg.core, CoreConfig)
    assert cfg.core.trigger == "slack-question"
    assert cfg.core.event_dispatcher_token == "disp-token"
    assert cfg.core.dispatch_worker_token == "work-token"
    assert cfg.core.poll_budget == 400.0
    assert cfg.core.dispatcher_url == "http://d:1"
    assert cfg.core.trust_env is True


def test_config_composes_core_rather_than_subclassing_it():
    assert CoreConfig not in SlackBridgeConfig.__mro__


def test_from_env_defaults_every_credential_to_empty_and_applies_no_trigger_default():
    cfg = SlackBridgeConfig.from_env({})
    assert cfg.bot_token == ""
    assert cfg.app_token == ""
    assert cfg.core.trigger == ""


def test_dataclass_defaults_match_from_env_defaults():
    cfg = SlackBridgeConfig()
    assert (cfg.bot_token, cfg.app_token, cfg.version_tag) == ("", "", "")
    assert cfg.mentions is True
    assert cfg.core == CoreConfig()


def test_from_env_reads_os_environ_when_no_mapping_given(monkeypatch):
    for name, value in complete().items():
        monkeypatch.setenv(name, value)
    cfg = SlackBridgeConfig.from_env()
    assert cfg.bot_token == COMPLETE_ENV["SLACK_BOT_TOKEN"]
    assert cfg.core.trigger == "slack-question"


@pytest.mark.parametrize(
    "prefixed",
    ["SLACK_DISPATCH_TRIGGER", "SLACK_POLL_BUDGET", "SLACK_DEDUP_PATH", "SLACK_VERSION_TAG"],
)
def test_prefixed_spellings_of_shared_names_are_not_read(prefixed):
    cfg = SlackBridgeConfig.from_env({prefixed: "999"})
    assert cfg.core.trigger == ""
    assert cfg.core.poll_budget == 330.0
    assert cfg.core.dedup_path == "/data/dedup.json"
    assert cfg.version_tag == INSTALLED_VERSION


def test_config_is_frozen():
    cfg = SlackBridgeConfig.from_env(complete())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.bot_token = "other"  # type: ignore[misc]


# --- mentions ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, True), ("", True), ("true", True), ("1", True), ("false", False), ("0", False)],
)
def test_mentions_flag(raw, expected):
    env = complete() if raw is None else complete(SLACK_MENTIONS=raw)
    assert SlackBridgeConfig.from_env(env).mentions is expected


# --- version tag ----------------------------------------------------------


def test_version_tag_prefers_app_version_display():
    cfg = SlackBridgeConfig.from_env(complete(APP_VERSION_DISPLAY="v2026.11.1+abc"))
    assert cfg.version_tag == "v2026.11.1+abc"


@pytest.mark.parametrize("env", [{}, {"APP_VERSION_DISPLAY": ""}])
def test_unset_or_empty_version_falls_back_to_installed_version(env):
    assert SlackBridgeConfig.from_env(env).version_tag == INSTALLED_VERSION


def test_version_tag_is_empty_without_an_installed_distribution(monkeypatch):
    def missing(_name):
        raise metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(config_module.metadata, "version", missing)
    assert SlackBridgeConfig.from_env({}).version_tag == ""


# --- tokens never in repr -------------------------------------------------


def test_tokens_are_never_in_repr():
    cfg = SlackBridgeConfig.from_env(complete())
    text = repr(cfg)
    assert COMPLETE_ENV["SLACK_BOT_TOKEN"] not in text
    assert COMPLETE_ENV["SLACK_APP_TOKEN"] not in text
    assert "bot_token" not in text
    assert "app_token" not in text
    # Fields that are not credentials stay visible for diagnosis.
    assert "mentions=True" in text


# --- require_startup --------------------------------------------------------


def test_a_complete_env_passes_require_startup():
    SlackBridgeConfig.from_env(complete()).require_startup()


@pytest.mark.parametrize("name", REQUIRED)
@pytest.mark.parametrize("how", ["absent", "empty"])
def test_each_missing_required_var_is_named(name, how):
    env = complete()
    if how == "absent":
        del env[name]
    else:
        env[name] = ""
    with pytest.raises(ValueError, match=rf"missing required config: {name}$"):
        SlackBridgeConfig.from_env(env).require_startup()


def test_every_missing_var_is_named_in_one_raise():
    with pytest.raises(ValueError) as excinfo:
        SlackBridgeConfig.from_env({}).require_startup()
    assert str(excinfo.value) == f"missing required config: {', '.join(REQUIRED)}"


@pytest.mark.parametrize("optional", ["SLACK_MENTIONS", "APP_VERSION_DISPLAY"])
def test_optional_vars_are_not_required(optional):
    env = complete(**{optional: ""})
    SlackBridgeConfig.from_env(env).require_startup()


# --- require_boot ---------------------------------------------------------


def test_require_boot_passes_on_a_complete_env():
    require_boot(SlackBridgeConfig.from_env(complete()))


def test_require_boot_runs_require_startup_first():
    with pytest.raises(ValueError, match="SLACK_BOT_TOKEN"):
        require_boot(SlackBridgeConfig.from_env({}))


@pytest.mark.parametrize("url_var", ["DISPATCHER_URL", "WORKER_URL"])
def test_require_boot_names_an_empty_core_url(url_var):
    with pytest.raises(ValueError, match=rf"missing required config: {url_var}$") as excinfo:
        require_boot(SlackBridgeConfig.from_env(complete(**{url_var: ""})))
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_require_boot_names_both_empty_core_urls_in_one_raise():
    env = complete(DISPATCHER_URL="", WORKER_URL="")
    with pytest.raises(ValueError, match=r"missing required config: DISPATCHER_URL, WORKER_URL$"):
        require_boot(SlackBridgeConfig.from_env(env))


def test_absent_core_urls_take_the_localhost_defaults():
    # Absent (not empty) URLs fall back to CoreConfig's defaults, so boot passes.
    require_boot(SlackBridgeConfig.from_env(complete()))
    cfg = SlackBridgeConfig.from_env(complete())
    assert cfg.core.dispatcher_url and cfg.core.worker_url
