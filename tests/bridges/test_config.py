"""CoreConfig: env mapping, poll-budget guard, and require()."""

import pytest

from osprey.bridges.core.config import CoreConfig


def test_from_env_maps_all_neutral_fields():
    env = {
        "DISPATCHER_URL": "http://disp:8010/",  # trailing slash must be stripped
        "WORKER_URL": "http://work:9190/",
        "EVENT_DISPATCHER_TOKEN": "disp-token",
        "DISPATCH_WORKER_TOKEN": "work-token",
        "DISPATCH_TRIGGER": "email-question",
        "POLL_INTERVAL": "0.5",
        "POLL_BUDGET": "360",
        "DRAIN_INTERVAL": "30",
        "RETRY_MIN_AGE": "600",
        "RETRY_GIVE_UP": "86400",
        "RETRY_LIFETIME_CAP": "259200",
        "GITLAB_URL": "https://gl.example.com/",  # trailing slash must be stripped
        "GITLAB_PROJECT": "team/proj",
        "GITLAB_ISSUES_TOKEN": "gl-token",
        "DEDUP_PATH": "/data/email_dedup.json",
        "HISTORY_PATH": "/data/email_history.json",
    }
    cfg = CoreConfig.from_env(env)
    assert cfg.dispatcher_url == "http://disp:8010"
    assert cfg.worker_url == "http://work:9190"
    assert cfg.event_dispatcher_token == "disp-token"
    assert cfg.dispatch_worker_token == "work-token"
    assert cfg.trigger == "email-question"
    assert cfg.poll_interval == 0.5
    assert cfg.poll_budget == 360.0
    assert cfg.drain_interval == 30.0
    assert cfg.retry_min_age == 600.0
    assert cfg.retry_give_up == 86400.0
    assert cfg.retry_lifetime_cap == 259200.0
    assert cfg.gitlab_url == "https://gl.example.com"
    assert cfg.gitlab_project == "team/proj"
    assert cfg.gitlab_issues_token == "gl-token"
    assert cfg.dedup_path == "/data/email_dedup.json"
    assert cfg.history_path == "/data/email_history.json"


def test_from_env_defaults_when_unset():
    cfg = CoreConfig.from_env({})
    assert cfg.dispatcher_url == "http://localhost:8010"
    assert cfg.worker_url == "http://localhost:9190"
    assert cfg.trigger == ""
    assert cfg.poll_interval == 2.0
    assert cfg.poll_budget == 330.0
    assert cfg.drain_interval == 60.0
    assert cfg.retry_min_age == 1200.0
    assert cfg.retry_give_up == 172800.0
    assert cfg.retry_lifetime_cap == 604800.0
    # No alert host is assumed: an unconfigured deployment files no give-up
    # issues and its health gate skips the alert-issue check entirely.
    assert cfg.gitlab_url == ""
    assert cfg.gitlab_project == ""
    assert cfg.gitlab_issues_token == ""


def test_gitlab_issues_token_is_unprefixed_shared_secret():
    # Read from the neutral GITLAB_ISSUES_TOKEN, same pattern as the dispatch
    # tokens — channel configs do NOT prefix it.
    cfg = CoreConfig.from_env({"GITLAB_ISSUES_TOKEN": "shared"})
    assert cfg.gitlab_issues_token == "shared"


def test_drain_retry_empty_string_falls_back_to_default():
    cfg = CoreConfig.from_env(
        {
            "DRAIN_INTERVAL": "",
            "RETRY_MIN_AGE": "",
            "RETRY_GIVE_UP": "",
            "RETRY_LIFETIME_CAP": "",
        }
    )
    assert cfg.drain_interval == 60.0
    assert cfg.retry_min_age == 1200.0
    assert cfg.retry_give_up == 172800.0
    assert cfg.retry_lifetime_cap == 604800.0


@pytest.mark.parametrize(
    "field", ["drain_interval", "retry_min_age", "retry_give_up", "retry_lifetime_cap"]
)
def test_non_positive_duration_is_rejected(field):
    with pytest.raises(ValueError, match=field):
        CoreConfig(**{field: 0})
    with pytest.raises(ValueError, match=field):
        CoreConfig(**{field: -1.0})


def test_empty_string_numeric_falls_back_to_default():
    # An explicitly-empty env var must not crash float() — it means "unset".
    cfg = CoreConfig.from_env({"POLL_INTERVAL": "", "POLL_BUDGET": ""})
    assert cfg.poll_interval == 2.0
    assert cfg.poll_budget == 330.0


def test_poll_budget_below_worker_cap_is_rejected():
    with pytest.raises(ValueError, match="poll_budget"):
        CoreConfig(poll_budget=299)


def test_require_raises_on_missing_field():
    cfg = CoreConfig(dispatcher_url="http://disp:8010", event_dispatcher_token="")
    with pytest.raises(ValueError, match="event_dispatcher_token"):
        cfg.require("event_dispatcher_token")


def test_require_passes_when_all_present():
    cfg = CoreConfig(event_dispatcher_token="t", dispatch_worker_token="w")
    cfg.require("event_dispatcher_token", "dispatch_worker_token")  # no raise


# --- Alert host is opt-in -------------------------------------------------


def test_alert_host_configured_only_when_env_sets_it():
    # Both halves of the alert config come from the environment; nothing about a
    # particular facility is baked in as a default.
    cfg = CoreConfig.from_env({"GITLAB_URL": "https://gl.example.com/", "GITLAB_PROJECT": "a/b"})
    assert cfg.gitlab_url == "https://gl.example.com"
    assert cfg.gitlab_project == "a/b"


def test_alert_host_default_is_empty_on_the_dataclass_too():
    # Not just from_env(): a directly-constructed config is unconfigured too, so
    # a channel that composes CoreConfig cannot inherit an alert host by accident.
    cfg = CoreConfig()
    assert cfg.gitlab_url == ""
    assert cfg.gitlab_project == ""


# --- poll_budget is validated against the configured worker timeout -------


def test_worker_timeout_defaults_to_worker_cap():
    assert CoreConfig().worker_timeout == 300.0
    assert CoreConfig.from_env({}).worker_timeout == 300.0


def test_worker_timeout_read_from_dispatch_timeout_sec():
    # Same var the worker itself reads, so one setting raises the cap on both
    # halves of the pair.
    cfg = CoreConfig.from_env({"DISPATCH_TIMEOUT_SEC": "600", "POLL_BUDGET": "660"})
    assert cfg.worker_timeout == 600.0
    assert cfg.poll_budget == 660.0


def test_worker_timeout_empty_string_falls_back_to_default():
    cfg = CoreConfig.from_env({"DISPATCH_TIMEOUT_SEC": ""})
    assert cfg.worker_timeout == 300.0


def test_poll_budget_below_raised_worker_timeout_is_rejected():
    # A facility that raises the worker's cap but leaves the default poll budget
    # would abandon runs that are still alive — reject that at config load.
    with pytest.raises(ValueError, match="poll_budget"):
        CoreConfig(poll_budget=330, worker_timeout=600)


def test_poll_budget_below_raised_worker_timeout_is_rejected_from_env():
    with pytest.raises(ValueError, match="poll_budget"):
        CoreConfig.from_env({"DISPATCH_TIMEOUT_SEC": "600"})


def test_poll_budget_at_or_above_worker_timeout_is_accepted():
    assert CoreConfig(poll_budget=600, worker_timeout=600).poll_budget == 600.0
    assert CoreConfig(poll_budget=700, worker_timeout=600).poll_budget == 700.0


def test_poll_budget_below_lowered_worker_timeout_is_still_the_floor():
    # A site that lowers the worker cap lowers the floor with it: 200s of budget
    # is fine against a 120s worker, and still rejected against a 240s one.
    assert CoreConfig(poll_budget=200, worker_timeout=120).poll_budget == 200.0
    with pytest.raises(ValueError, match="poll_budget"):
        CoreConfig(poll_budget=200, worker_timeout=240)


@pytest.mark.parametrize("bad", [0, -1.0])
def test_non_positive_worker_timeout_is_rejected(bad):
    with pytest.raises(ValueError, match="worker_timeout"):
        CoreConfig(worker_timeout=bad)


# --- trust_env is explicit config ----------------------------------------


def test_trust_env_defaults_to_false():
    # Direct connections to the pair: an ambient HTTP(S)_PROXY from a dev shell
    # or CI runner must not mount itself in front of the dispatcher/worker.
    assert CoreConfig().trust_env is False
    assert CoreConfig.from_env({}).trust_env is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " True "])
def test_trust_env_enabled_by_truthy_env(raw):
    assert CoreConfig.from_env({"BRIDGE_TRUST_ENV": raw}).trust_env is True


@pytest.mark.parametrize("raw", ["", "0", "false", "no", "off", "maybe"])
def test_trust_env_stays_false_for_anything_else(raw):
    assert CoreConfig.from_env({"BRIDGE_TRUST_ENV": raw}).trust_env is False


def test_trust_env_settable_directly():
    assert CoreConfig(trust_env=True).trust_env is True
