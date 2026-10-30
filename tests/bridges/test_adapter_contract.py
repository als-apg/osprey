"""The contract every chat adapter's config and ops class keeps with the engine.

Each adapter has its own suite for what is particular to it (its credentials, its wire
format, its client). What is asserted here is the part that must read the same for all
of them: the ``CoreConfig`` projection, the startup and boot requires, the mentions
switch, and the ``ChannelOps`` signatures. One table row per adapter; an adapter that
adds a startup variable adds it to its own suite's ``COMPLETE_ENV``, which this table
reads.
"""

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from osprey.bridges.core import ChannelOps, CoreConfig
from osprey.bridges.google_chat import GoogleChatBridgeConfig
from osprey.bridges.google_chat import require_boot as google_chat_require_boot
from osprey.bridges.google_chat.ops import GoogleChatOps
from osprey.bridges.nextcloud_talk import NextcloudBridgeConfig
from osprey.bridges.nextcloud_talk.ops import NextcloudTalkOps
from osprey.bridges.teams import TeamsBridgeConfig
from osprey.bridges.teams import require_boot as teams_require_boot
from osprey.bridges.teams.ops import TeamsOps
from osprey.port_layout import default_port
from tests.bridges.google_chat.test_config import COMPLETE_ENV as GOOGLE_CHAT_ENV
from tests.bridges.nextcloud.test_config import COMPLETE_ENV as NEXTCLOUD_ENV
from tests.bridges.teams.test_config import COMPLETE_ENV as TEAMS_ENV
from tests.bridges.test_ports import PROTOCOL_MEMBERS


@dataclass(frozen=True)
class Adapter:
    """One chat adapter, as far as the shared contract needs to know it."""

    name: str
    config: type
    #: The fully-configured deployment: every value ``require_startup()`` insists on.
    env: Mapping[str, str]
    mentions_var: str
    ops: type
    #: The adapter's boot check, when it renders bare ``${VAR}`` URLs in compose.
    require_boot: Callable[[Any], None] | None = None

    def complete(self, **overrides: str) -> dict[str, str]:
        """A complete env with `overrides` applied (an empty value means "unset")."""
        return {**self.env, **overrides}

    def bootable(self, **overrides: str) -> dict[str, str]:
        """A complete env that also carries the two URLs ``require_boot`` checks."""
        return {**self.env, **BOOT_URL_ENV, **overrides}


TEAMS = Adapter(
    "teams", TeamsBridgeConfig, TEAMS_ENV, "TEAMS_MENTIONS", TeamsOps, teams_require_boot
)
GOOGLE_CHAT = Adapter(
    "google_chat",
    GoogleChatBridgeConfig,
    GOOGLE_CHAT_ENV,
    "GCHAT_MENTIONS",
    GoogleChatOps,
    google_chat_require_boot,
)
NEXTCLOUD = Adapter(
    "nextcloud", NextcloudBridgeConfig, NEXTCLOUD_ENV, "NEXTCLOUD_MENTIONS", NextcloudTalkOps
)

ADAPTERS = (TEAMS, GOOGLE_CHAT, NEXTCLOUD)
BOOTED = tuple(adapter for adapter in ADAPTERS if adapter.require_boot is not None)

# What require_boot checks on top of require_startup. Bare in compose, so an unset one
# arrives as "" and no code default ever replaces it.
BOOT_URL_ENV = {
    "DISPATCHER_URL": "http://dispatcher:10010",
    "WORKER_URL": "http://worker:10011",
}


def adapter_id(adapter: Adapter) -> str:
    return adapter.name


def per_adapter(adapters: tuple[Adapter, ...], values: Callable[[Adapter], list[Any]]) -> list[Any]:
    """``(adapter, value)`` params for every value each adapter is checked against."""
    return [
        pytest.param(adapter, value, id=f"{adapter.name}-{value}")
        for adapter in adapters
        for value in values(adapter)
    ]


# --- CoreConfig is composed, not subclassed --------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS, ids=adapter_id)
def test_core_is_a_composed_projection_not_a_base_class(adapter):
    cfg = adapter.config.from_env(adapter.complete())
    assert isinstance(cfg.core, CoreConfig)
    assert not isinstance(cfg, CoreConfig)
    assert CoreConfig not in type(cfg).__mro__


@pytest.mark.parametrize("adapter", ADAPTERS, ids=adapter_id)
def test_neutral_vars_are_delegated_to_core_config(adapter):
    cfg = adapter.config.from_env(
        adapter.complete(
            DISPATCHER_URL="http://disp:10010/",
            WORKER_URL="http://work:10011/",
            POLL_INTERVAL="0.5",
            DEDUP_PATH="/data/adapter_dedup.json",
        )
    )
    assert cfg.core.dispatcher_url == "http://disp:10010"
    assert cfg.core.worker_url == "http://work:10011"
    assert cfg.core.event_dispatcher_token == "disp-token"
    assert cfg.core.dispatch_worker_token == "work-token"
    assert cfg.core.trigger == adapter.env["DISPATCH_TRIGGER"]
    assert cfg.core.poll_interval == 0.5
    assert cfg.core.dedup_path == "/data/adapter_dedup.json"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=adapter_id)
def test_config_is_immutable(adapter):
    # One instance is shared by the receive loop or room threads, the ChannelOps
    # instance and the drain thread; build variants with dataclasses.replace instead.
    cfg = adapter.config.from_env(adapter.complete())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.mentions = False
    assert dataclasses.replace(cfg, mentions=False).mentions is False


@pytest.mark.parametrize("adapter", ADAPTERS, ids=adapter_id)
def test_from_env_applies_no_trigger_default(adapter):
    # The trigger default belongs to the deployment surface (rendered as
    # DISPATCH_TRIGGER in the compose template). If from_env defaulted it, a
    # hand-rolled deployment would silently dispatch to a trigger nobody chose and
    # require_startup could never catch the omission.
    assert adapter.config.from_env({}).core.trigger == ""
    assert adapter.config.from_env(adapter.complete(DISPATCH_TRIGGER="")).core.trigger == ""
    assert adapter.config().core.trigger == ""


# --- require_startup ---------------------------------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS, ids=adapter_id)
def test_require_startup_passes_on_a_complete_config(adapter):
    adapter.config.from_env(adapter.complete()).require_startup()  # no raise


@pytest.mark.parametrize(("adapter", "name"), per_adapter(ADAPTERS, lambda a: sorted(a.env)))
def test_require_startup_names_each_missing_var(adapter, name):
    cfg = adapter.config.from_env(adapter.complete(**{name: ""}))
    with pytest.raises(ValueError, match=name) as excinfo:
        cfg.require_startup()
    # Only the one that is actually missing is reported.
    others = [other for other in adapter.env if other != name]
    assert not [other for other in others if other in str(excinfo.value)]


@pytest.mark.parametrize("adapter", ADAPTERS, ids=adapter_id)
def test_require_startup_lists_every_missing_var_in_one_raise(adapter):
    # A wholly unconfigured process must not need one restart per variable to learn
    # what it is missing.
    with pytest.raises(ValueError) as excinfo:
        adapter.config.from_env({}).require_startup()
    message = str(excinfo.value)
    assert message.startswith("missing required config: ")
    for name in adapter.env:
        assert name in message


# --- require_boot: the bare-${VAR} compose trap ------------------------------


@pytest.mark.parametrize("adapter", BOOTED, ids=adapter_id)
def test_require_boot_passes_when_the_urls_are_set(adapter):
    adapter.require_boot(adapter.config.from_env(adapter.bootable()))  # no raise


@pytest.mark.parametrize(("adapter", "name"), per_adapter(BOOTED, lambda a: sorted(BOOT_URL_ENV)))
def test_require_boot_rejects_a_url_that_rendered_empty(adapter, name):
    # An unset bare ${DISPATCHER_URL} in compose reaches the process as "", not as an
    # absent key, so CoreConfig.from_env's localhost fallback never fires. Left
    # unchecked the bridge would POST to a protocol-less URL forever.
    cfg = adapter.config.from_env(adapter.bootable(**{name: ""}))
    with pytest.raises(ValueError, match=name):
        adapter.require_boot(cfg)


@pytest.mark.parametrize("adapter", BOOTED, ids=adapter_id)
def test_require_boot_names_env_vars_not_field_names(adapter):
    cfg = adapter.config.from_env(adapter.bootable(DISPATCHER_URL="", WORKER_URL=""))
    with pytest.raises(ValueError) as excinfo:
        adapter.require_boot(cfg)
    message = str(excinfo.value)
    assert "DISPATCHER_URL" in message
    assert "WORKER_URL" in message
    # The engine's own check reports field names; the abort has to be diagnosable
    # from the container log, where only the env spelling exists.
    assert "dispatcher_url" not in message
    assert "worker_url" not in message


@pytest.mark.parametrize("adapter", BOOTED, ids=adapter_id)
def test_require_boot_reports_the_startup_vars_before_the_urls(adapter):
    # require_startup runs first, so a wholly unconfigured process is told about its
    # credentials rather than only about two URLs it never set either.
    with pytest.raises(ValueError) as excinfo:
        adapter.require_boot(adapter.config.from_env({}))
    first_credential = next(iter(adapter.env))
    assert first_credential in str(excinfo.value)


@pytest.mark.parametrize("adapter", BOOTED, ids=adapter_id)
def test_require_startup_does_not_cover_the_urls(adapter):
    # The split is deliberate: require_startup is the credentials check and
    # require_boot adds the compose-trap check on top.
    cfg = adapter.config.from_env(adapter.complete(DISPATCHER_URL=""))
    cfg.require_startup()  # no raise
    with pytest.raises(ValueError, match="DISPATCHER_URL"):
        adapter.require_boot(cfg)


@pytest.mark.parametrize("adapter", BOOTED, ids=adapter_id)
def test_absent_url_vars_still_fall_back_to_the_code_defaults(adapter):
    # Only the *empty* render is the trap. A deployment that never mentions the URLs
    # at all is the local-dev shape CoreConfig's defaults exist for, and require_boot
    # must not break it.
    cfg = adapter.config.from_env(adapter.complete())
    assert (cfg.core.dispatcher_url, cfg.core.worker_url) == (
        f"http://localhost:{default_port('dispatcher')}",
        f"http://localhost:{default_port('worker', 1)}",
    )
    adapter.require_boot(cfg)  # no raise


# --- mentions ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "env"), per_adapter(ADAPTERS, lambda a: [{}, {a.mentions_var: ""}])
)
def test_mentions_default_on_when_unset_or_empty(adapter, env):
    assert adapter.config.from_env(adapter.complete(**env)).mentions is True


@pytest.mark.parametrize(
    ("adapter", "raw"), per_adapter(ADAPTERS, lambda a: ["false", "0", "no", "off", "ture"])
)
def test_mentions_turn_off_for_a_false_spelling(adapter, raw):
    cfg = adapter.config.from_env(adapter.complete(**{adapter.mentions_var: raw}))
    assert cfg.mentions is False


@pytest.mark.parametrize(
    ("adapter", "raw"),
    [
        *per_adapter(ADAPTERS, lambda a: ["true", "1", "yes", "on"]),
        # Case and surrounding whitespace do not matter.
        pytest.param(NEXTCLOUD, " TRUE ", id="nextcloud- TRUE "),
    ],
)
def test_mentions_stay_on_for_a_true_spelling(adapter, raw):
    cfg = adapter.config.from_env(adapter.complete(**{adapter.mentions_var: raw}))
    assert cfg.mentions is True


@pytest.mark.parametrize("adapter", ADAPTERS, ids=adapter_id)
def test_mentions_are_not_a_startup_requirement(adapter):
    assert adapter.mentions_var not in adapter.env
    adapter.config.from_env(adapter.complete(**{adapter.mentions_var: "false"})).require_startup()
    adapter.config.from_env(adapter.complete()).require_startup()  # no raise


# --- ChannelOps signatures ---------------------------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS, ids=adapter_id)
def test_every_member_signature_matches_the_protocol(adapter):
    """Parameter names, order, and count, member by member.

    ``isinstance`` cannot see any of this, and a mismatch is not a type error the engine
    would notice until it called the member in production — the retry drain's members in
    particular may not run for hours. The canonical member list is imported from the
    core seam's own suite, so a member added to the protocol fails here too.
    """
    for name in PROTOCOL_MEMBERS:
        expected = inspect.signature(getattr(ChannelOps, name))
        actual = inspect.signature(getattr(adapter.ops, name))
        assert list(actual.parameters) == list(expected.parameters), name
