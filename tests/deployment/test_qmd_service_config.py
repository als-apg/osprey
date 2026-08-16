"""Tests for the ``services.qmd`` config schema resolver."""

import pytest

from osprey.deployment.host_ports import _SERVICE_REMEDY_KEYS
from osprey.deployment.qmd_service import (
    DEFAULT_BIND_ADDRESS,
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_PORT,
    PORT_CONFIG_KEY,
    QMDServiceConfig,
    resolve_bind_address,
    resolve_qmd_service_config,
)


class TestAbsentBlock:
    """A deployment without a qmd sidecar resolves to ``None``, not defaults."""

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"services": {}},
            {"services": {"postgresql": {"port_host": 5432}}},
            {"services": None},
            {"services": {"qmd": None}},
        ],
        ids=["none", "empty", "no-qmd", "other-service", "services-null", "qmd-null"],
    )
    def test_absent_resolves_to_none(self, config: dict | None) -> None:
        assert resolve_qmd_service_config(config) is None


class TestDefaults:
    """A present-but-empty block fills in every default."""

    def test_empty_block_gets_defaults(self) -> None:
        resolved = resolve_qmd_service_config({"services": {"qmd": {}}})
        assert resolved == QMDServiceConfig(
            port=DEFAULT_PORT,
            bind_address=DEFAULT_BIND_ADDRESS,
            interval_seconds=DEFAULT_INTERVAL_SECONDS,
        )

    def test_default_port_avoids_qmd_own_daemon_port(self) -> None:
        """8181 names the container-internal daemon; the published port differs."""
        assert DEFAULT_PORT != 8181

    def test_explicit_values_win(self) -> None:
        resolved = resolve_qmd_service_config(
            {
                "deployment": {"bind_address": "0.0.0.0"},
                "services": {"qmd": {"port": 9999, "interval": 300}},
            }
        )
        assert resolved is not None
        assert (resolved.port, resolved.interval_seconds) == (9999, 300)
        assert resolved.bind_address == "0.0.0.0"


class TestBindAddress:
    """``bind_address`` is project-wide, never a per-service key."""

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"deployment": {}},
            {"deployment": None},
            {"deployment": {"bind_address": None}},
            {"deployment": {"bind_address": "   "}},
        ],
        ids=["none", "empty", "no-address", "deployment-null", "address-null", "blank"],
    )
    def test_defaults_to_loopback(self, config: dict | None) -> None:
        assert resolve_bind_address(config) == "127.0.0.1"

    def test_reads_project_wide_key(self) -> None:
        assert resolve_bind_address({"deployment": {"bind_address": " 10.0.0.5 "}}) == "10.0.0.5"

    def test_per_service_bind_address_is_ignored(self) -> None:
        """A hand-written ``services.qmd.bind_address`` must not take effect.

        Honouring it would let the sidecar publish on an interface the rest of
        the stack does not, which is exactly the split the single project-wide
        key exists to prevent.
        """
        resolved = resolve_qmd_service_config(
            {"services": {"qmd": {"bind_address": "0.0.0.0"}}},
        )
        assert resolved is not None
        assert resolved.bind_address == "127.0.0.1"


class TestValidation:
    """A malformed scalar refuses rather than silently defaulting."""

    @pytest.mark.parametrize("bad", [0, -1, "8180", 8180.0, True, [8180]])
    def test_bad_port_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"services\.qmd\.port"):
            resolve_qmd_service_config({"services": {"qmd": {"port": bad}}})

    @pytest.mark.parametrize("bad", [0, -30, "30", 30.0, True])
    def test_bad_interval_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"services\.qmd\.interval"):
            resolve_qmd_service_config({"services": {"qmd": {"interval": bad}}})


class TestBaseUrl:
    """Clients connect over loopback regardless of the publish interface."""

    def test_uses_configured_port(self) -> None:
        assert QMDServiceConfig(port=8180).base_url == "http://127.0.0.1:8180"

    def test_wildcard_publish_still_dials_loopback(self) -> None:
        assert QMDServiceConfig(port=8180, bind_address="0.0.0.0").base_url == (
            "http://127.0.0.1:8180"
        )


def test_port_conflict_preflight_knows_the_sidecar() -> None:
    """The deploy-time port sweep can name the key that moves the qmd port."""
    assert _SERVICE_REMEDY_KEYS["qmd"] == PORT_CONFIG_KEY == "services.qmd.port"
