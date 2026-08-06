"""GoogleChatBridgeConfig: env parsing, the CoreConfig projection, the boot requires."""

import dataclasses
import logging

import pytest

from osprey.bridges.core import CoreConfig
from osprey.bridges.google_chat import GoogleChatBridgeConfig, require_boot

CONFIG_LOGGER = "osprey.bridges.google_chat.config"

SUBSCRIPTION = "projects/facility-gcp/subscriptions/gchat-osprey-events-sub"

# A fully-configured deployment: the six values require_startup() insists on.
COMPLETE_ENV = {
    "GCHAT_SA_KEY": "/secrets/gchat-sa.json",
    "GCHAT_SUBSCRIPTION": SUBSCRIPTION,
    "GCHAT_APP_ID": "users/1234567890",
    "DISPATCH_TRIGGER": "gchat-question",
    "EVENT_DISPATCHER_TOKEN": "disp-token",
    "DISPATCH_WORKER_TOKEN": "work-token",
}

# What require_boot checks on top of require_startup. Bare in compose, so an
# unset one arrives as "" and no code default ever replaces it.
BOOT_URL_ENV = {
    "DISPATCHER_URL": "http://dispatcher:8010",
    "WORKER_URL": "http://worker:9190",
}


def complete(**overrides: str) -> dict[str, str]:
    """A complete env with `overrides` applied (an empty value means "unset")."""
    return {**COMPLETE_ENV, **overrides}


def bootable(**overrides: str) -> dict[str, str]:
    """A complete env that also carries the two URLs require_boot checks."""
    return {**COMPLETE_ENV, **BOOT_URL_ENV, **overrides}


# --- from_env parsing -----------------------------------------------------


def test_from_env_maps_the_google_specific_fields():
    cfg = GoogleChatBridgeConfig.from_env(
        complete(
            GCS_BUCKET="osprey-artifacts",
            GCS_PROJECT="facility-gcp",
            APP_VERSION_DISPLAY="v2026.7.1+abc1234",
        )
    )
    assert cfg.sa_key == "/secrets/gchat-sa.json"
    assert cfg.subscription == SUBSCRIPTION
    assert cfg.app_id == "users/1234567890"
    assert cfg.gcs_bucket == "osprey-artifacts"
    assert cfg.gcs_project == "facility-gcp"
    assert cfg.version_tag == "v2026.7.1+abc1234"


def test_from_env_defaults_every_google_field_to_empty_when_unset():
    cfg = GoogleChatBridgeConfig.from_env({})
    assert cfg.sa_key == ""
    assert cfg.subscription == ""
    assert cfg.app_id == ""
    assert cfg.gcs_bucket == ""
    assert cfg.gcs_project == ""
    assert cfg.version_tag == ""


def test_dataclass_defaults_match_from_env_defaults():
    # A directly-constructed config is just as unconfigured, so nothing can be
    # inherited by accident by a collaborator that builds one itself.
    cfg = GoogleChatBridgeConfig()
    assert (cfg.sa_key, cfg.subscription, cfg.app_id) == ("", "", "")
    assert (cfg.gcs_bucket, cfg.gcs_project, cfg.version_tag) == ("", "", "")


def test_from_env_reads_os_environ_when_no_mapping_given(monkeypatch):
    for name, value in complete().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("GCS_BUCKET", "osprey-artifacts")
    cfg = GoogleChatBridgeConfig.from_env()
    assert cfg.app_id == "users/1234567890"
    assert cfg.gcs_bucket == "osprey-artifacts"
    assert cfg.core.trigger == "gchat-question"


def test_version_tag_comes_from_the_unprefixed_image_build_arg():
    # APP_VERSION_DISPLAY is baked into the image by the build, not set per
    # channel — a GCHAT_-prefixed spelling of it would never be populated.
    cfg = GoogleChatBridgeConfig.from_env(complete(APP_VERSION_DISPLAY="v2026.8.0"))
    assert cfg.version_tag == "v2026.8.0"
    assert GoogleChatBridgeConfig.from_env(complete(GCHAT_VERSION_TAG="v9")).version_tag == ""


# --- neutral settings carry no GCHAT_ prefix ------------------------------


def test_neutral_vars_are_delegated_to_core_config():
    cfg = GoogleChatBridgeConfig.from_env(
        complete(
            DISPATCHER_URL="http://disp:8010/",
            WORKER_URL="http://work:9190/",
            POLL_INTERVAL="0.5",
            DEDUP_PATH="/data/gchat_dedup.json",
            HISTORY_PATH="/data/gchat_history.json",
        )
    )
    assert cfg.core.dispatcher_url == "http://disp:8010"
    assert cfg.core.worker_url == "http://work:9190"
    assert cfg.core.event_dispatcher_token == "disp-token"
    assert cfg.core.dispatch_worker_token == "work-token"
    assert cfg.core.trigger == "gchat-question"
    assert cfg.core.poll_interval == 0.5
    assert cfg.core.dedup_path == "/data/gchat_dedup.json"
    assert cfg.core.history_path == "/data/gchat_history.json"


@pytest.mark.parametrize(
    ("prefixed", "field_name", "default"),
    [
        ("GCHAT_POLL_INTERVAL", "poll_interval", 2.0),
        ("GCHAT_POLL_BUDGET", "poll_budget", 330.0),
        ("GCHAT_DRAIN_INTERVAL", "drain_interval", 60.0),
        ("GCHAT_RETRY_MIN_AGE", "retry_min_age", 1200.0),
        ("GCHAT_DEDUP_PATH", "dedup_path", "/data/dedup.json"),
        ("GCHAT_HISTORY_PATH", "history_path", "/data/history.json"),
        ("GCHAT_TRIGGER", "trigger", ""),
    ],
)
def test_prefixed_spellings_of_neutral_settings_are_not_read(prefixed, field_name, default):
    # The neutral half is one set of names shared with every other adapter; a
    # GCHAT_-prefixed spelling must not quietly configure it.
    cfg = GoogleChatBridgeConfig.from_env({prefixed: "999"})
    assert getattr(cfg.core, field_name) == default


# --- CoreConfig is composed, not subclassed ------------------------------


def test_core_is_a_composed_projection_not_a_base_class():
    cfg = GoogleChatBridgeConfig.from_env(complete())
    assert isinstance(cfg.core, CoreConfig)
    assert not isinstance(cfg, CoreConfig)
    assert CoreConfig not in type(cfg).__mro__


def test_config_is_immutable():
    # One instance is shared by the subscriber loop, the ChannelOps instance and
    # the drain thread; build variants with dataclasses.replace instead.
    cfg = GoogleChatBridgeConfig.from_env(complete())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.subscription = "projects/other/subscriptions/other-sub"
    assert dataclasses.replace(cfg, app_id="users/999").app_id == "users/999"


# --- no trigger default, no subscription default -------------------------


def test_from_env_applies_no_trigger_default():
    # The gchat-question default belongs to the deployment surface (rendered as
    # DISPATCH_TRIGGER in the compose template). If from_env defaulted it, a
    # hand-rolled deployment would silently dispatch to a trigger nobody chose
    # and require_startup could never catch the omission.
    assert GoogleChatBridgeConfig.from_env({}).core.trigger == ""
    assert GoogleChatBridgeConfig.from_env(complete(DISPATCH_TRIGGER="")).core.trigger == ""
    assert GoogleChatBridgeConfig().core.trigger == ""


def test_from_env_applies_no_subscription_default():
    # A shipped literal names one facility's GCP project. A deployment that
    # forgot GCHAT_SUBSCRIPTION must fail closed, not pull somebody else's queue.
    assert GoogleChatBridgeConfig.from_env({}).subscription == ""
    assert GoogleChatBridgeConfig.from_env(complete(GCHAT_SUBSCRIPTION="")).subscription == ""
    assert GoogleChatBridgeConfig().subscription == ""


def test_no_field_default_or_module_constant_names_a_real_subscription():
    # Belt and braces for the test above: a fully-qualified name embeds one
    # facility's GCP project, so neither a field default nor a module constant may
    # carry one — that is how the shipped literal would come back.
    from osprey.bridges.google_chat import config as config_module

    def is_resource_name(value: object) -> bool:
        return (
            isinstance(value, str) and value.startswith("projects/") and "/subscriptions/" in value
        )

    assert not [
        f for f in dataclasses.fields(GoogleChatBridgeConfig) if is_resource_name(f.default)
    ]
    assert not [
        name
        for name, value in vars(config_module).items()
        if not name.startswith("__") and is_resource_name(value)
    ]


@pytest.mark.parametrize("missing_var", ["DISPATCH_TRIGGER", "GCHAT_SUBSCRIPTION"])
def test_the_undefaulted_values_are_what_require_startup_catches(missing_var):
    cfg = GoogleChatBridgeConfig.from_env(complete(**{missing_var: ""}))
    with pytest.raises(ValueError, match=missing_var):
        cfg.require_startup()


# --- require_startup -----------------------------------------------------


def test_require_startup_passes_on_a_complete_config():
    GoogleChatBridgeConfig.from_env(complete()).require_startup()  # no raise


@pytest.mark.parametrize("name", sorted(COMPLETE_ENV))
def test_require_startup_names_each_missing_var(name):
    cfg = GoogleChatBridgeConfig.from_env(complete(**{name: ""}))
    with pytest.raises(ValueError, match=name) as excinfo:
        cfg.require_startup()
    # Only the one that is actually missing is reported.
    others = [other for other in COMPLETE_ENV if other != name]
    assert not [other for other in others if other in str(excinfo.value)]


def test_require_startup_lists_every_missing_var_in_one_raise():
    # A wholly unconfigured process must not need six restarts to learn what it
    # is missing.
    with pytest.raises(ValueError) as excinfo:
        GoogleChatBridgeConfig.from_env({}).require_startup()
    message = str(excinfo.value)
    assert message.startswith("missing required config: ")
    for name in COMPLETE_ENV:
        assert name in message


def test_require_startup_requires_both_dispatch_tokens():
    cfg = GoogleChatBridgeConfig.from_env(
        complete(EVENT_DISPATCHER_TOKEN="", DISPATCH_WORKER_TOKEN="")
    )
    with pytest.raises(ValueError) as excinfo:
        cfg.require_startup()
    assert "EVENT_DISPATCHER_TOKEN" in str(excinfo.value)
    assert "DISPATCH_WORKER_TOKEN" in str(excinfo.value)


@pytest.mark.parametrize("name", ["GCS_BUCKET", "GCS_PROJECT", "APP_VERSION_DISPLAY"])
def test_require_startup_ignores_the_optional_vars(name):
    # No bucket means text-only delivery and no version tag means a plainer ack:
    # both are degraded service, not a bridge that must refuse to start.
    env = complete(GCS_BUCKET="b", GCS_PROJECT="p", APP_VERSION_DISPLAY="v1")
    env[name] = ""
    GoogleChatBridgeConfig.from_env(env).require_startup()  # no raise


# --- require_boot: the bare-${VAR} compose trap ---------------------------


def test_require_boot_passes_when_the_urls_are_set():
    require_boot(GoogleChatBridgeConfig.from_env(bootable()))  # no raise


@pytest.mark.parametrize("name", sorted(BOOT_URL_ENV))
def test_require_boot_rejects_a_url_that_rendered_empty(name):
    # An unset bare ${DISPATCHER_URL} in compose reaches the process as "", not as
    # an absent key, so CoreConfig.from_env's localhost fallback never fires. Left
    # unchecked the bridge would POST to a protocol-less URL forever.
    cfg = GoogleChatBridgeConfig.from_env(bootable(**{name: ""}))
    with pytest.raises(ValueError, match=name):
        require_boot(cfg)


def test_require_boot_names_env_vars_not_field_names():
    cfg = GoogleChatBridgeConfig.from_env(bootable(DISPATCHER_URL="", WORKER_URL=""))
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
        require_boot(GoogleChatBridgeConfig.from_env({}))
    assert "GCHAT_SA_KEY" in str(excinfo.value)


def test_require_startup_does_not_cover_the_urls():
    # The split is deliberate: require_startup is the credentials check and
    # require_boot adds the compose-trap check on top.
    cfg = GoogleChatBridgeConfig.from_env(complete(DISPATCHER_URL=""))
    cfg.require_startup()  # no raise
    with pytest.raises(ValueError, match="DISPATCHER_URL"):
        require_boot(cfg)


def test_absent_url_vars_still_fall_back_to_the_code_defaults():
    # Only the *empty* render is the trap. A deployment that never mentions the
    # URLs at all is the local-dev shape CoreConfig's defaults exist for, and
    # require_boot must not break it.
    cfg = GoogleChatBridgeConfig.from_env(complete())
    assert (cfg.core.dispatcher_url, cfg.core.worker_url) == (
        "http://localhost:8010",
        "http://localhost:9190",
    )
    require_boot(cfg)  # no raise


# --- a malformed subscription warns, never raises -------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "gchat-osprey-events-sub",
        "projects/facility-gcp/topics/gchat-events",
        "facility-gcp/subscriptions/gchat-events-sub",
    ],
)
def test_a_subscription_that_is_not_fully_qualified_warns_and_does_not_raise(raw, caplog):
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        cfg = GoogleChatBridgeConfig.from_env(complete(GCHAT_SUBSCRIPTION=raw))
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert warnings, "a subscription that is not fully-qualified must warn"
    assert "projects/" in warnings[0].getMessage()
    # Warned, not rejected: the resource name is Google's to validate, and the
    # startup require must still pass.
    cfg.require_startup()


def test_a_fully_qualified_subscription_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        GoogleChatBridgeConfig.from_env(complete())
    assert not [rec for rec in caplog.records if rec.levelno == logging.WARNING]


def test_an_unset_subscription_does_not_warn_about_its_shape(caplog):
    # An empty subscription is a *missing var*, which require_startup reports;
    # warning about its shape too would just be noise.
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        GoogleChatBridgeConfig.from_env({})
    assert not [rec for rec in caplog.records if rec.levelno == logging.WARNING]
