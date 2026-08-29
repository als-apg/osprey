"""``modules.web_terminals.nginx_port`` is an override, not a requirement.

The one thing every test here is about: nginx's published port is the gateway
slot of the deployment's own port block, and the config key merely moves it. So
a deployment that never mentions the key still has a front door, and every
reader that used to demand the key — the render, the summary card's landing
page, the Rule 11 overlap set, the scaffold's verify probe — must find that
door at the base the deployment actually resolved rather than at the framework
default.

The other half is that "unset" and "wrong" are different answers. An absent key
resolves; a key set to something that is not a port number refuses, naming
itself, because a `nginx_port: "8600"` is an author's intent that cannot be
honoured rather than a config that leans on the layout.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from osprey.deployment.deploy_summary import landing_page_url
from osprey.deployment.web_terminals.lint import lint_web_terminals
from osprey.deployment.web_terminals.ports import resolve_nginx_port
from osprey.deployment.web_terminals.render import (
    deployment_external_origin,
    render_web_terminals,
)
from osprey.port_layout import DEFAULT_PORT_BASE, default_port

pytestmark = pytest.mark.unit


#: A deployment that spells no port at all: the module is on, there is a roster,
#: and every port in it — nginx's included — comes from the layout.
_NO_PORTS_CONFIG: dict[str, Any] = {
    "facility": {"name": "Demo Light Source", "prefix": "dls"},
    "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
    "modules": {
        "web_terminals": {
            "enabled": True,
            "users": ["alice", "bob"],
        }
    },
}


def _config(**overrides: Any) -> dict[str, Any]:
    """A copy of :data:`_NO_PORTS_CONFIG` with ``modules.web_terminals`` extended."""
    config = copy.deepcopy(_NO_PORTS_CONFIG)
    config["modules"]["web_terminals"].update(overrides)
    return config


class TestTheResolver:
    def test_an_unset_key_is_the_gateway_slot_of_the_default_block(self):
        """The common case: no key, and the answer is the layout's, not an error."""
        assert resolve_nginx_port(_NO_PORTS_CONFIG) == default_port("nginx", base=DEFAULT_PORT_BASE)

    def test_an_unset_key_follows_the_base_the_deployment_resolved(self):
        """The whole point of the block: move the base, move the front door.

        20000 rather than 10000 — a deployment on its own block is described in
        its own block, never in terms of the framework default.
        """
        config = _config()
        config["deployment"] = {"port_base": 20000}

        assert resolve_nginx_port(config) == 20000

    def test_a_configured_port_wins_over_the_block(self):
        """An absolute value is exactly that: it moves nginx out of the block."""
        config = _config(nginx_port=8600)
        config["deployment"] = {"port_base": 20000}

        assert resolve_nginx_port(config) == 8600

    def test_a_config_with_no_web_terminals_at_all_still_answers(self):
        """The resolver is a layout lookup, not a module gate — callers own the gate."""
        assert resolve_nginx_port({}) == default_port("nginx", base=DEFAULT_PORT_BASE)
        assert resolve_nginx_port(None) == default_port("nginx", base=DEFAULT_PORT_BASE)

    @pytest.mark.parametrize("value", ["8600", True, 8600.0, [8600]])
    def test_a_value_that_is_not_a_port_is_refused_by_name(self, value):
        """`true` is in here on purpose: bool subclasses int, and it is a typo."""
        with pytest.raises(ValueError, match=r"modules\.web_terminals\.nginx_port"):
            resolve_nginx_port(_config(nginx_port=value))


class TestTheReadersFollow:
    def test_a_config_with_no_nginx_port_renders(self):
        """The refusal this replaces: the render used to demand the key."""
        artifacts = render_web_terminals(_NO_PORTS_CONFIG)

        expected = default_port("nginx", base=DEFAULT_PORT_BASE)
        assert f"listen {expected};" in artifacts["nginx/nginx.conf"]

    def test_the_rendered_origin_carries_the_resolved_port(self):
        """The origin every terminal checks a mutating request against."""
        config = _config()
        config["deployment"] = {"port_base": 20000}

        assert deployment_external_origin(config) == "http://dls-deploy.dls.example.org:20000"

    def test_the_landing_page_url_is_set_without_the_key(self):
        """The summary card's closing call to action still has an address."""
        assert landing_page_url(_NO_PORTS_CONFIG) == (
            f"http://dls-deploy.dls.example.org:{default_port('nginx', base=DEFAULT_PORT_BASE)}"
        )

    def test_the_landing_page_url_declines_a_port_that_is_not_a_port(self):
        """Advisory to the end: a card that cannot name the address names none."""
        assert landing_page_url(_config(nginx_port="8600")) is None

    def test_an_unset_nginx_port_joins_the_rule_11_overlap_set(self):
        """A port the stack binds collides whether or not the config spells it.

        The service is put exactly on the gateway slot, so the only way this
        finding appears is if the overlap set resolved nginx's port rather than
        reading the key.
        """
        config = _config()
        config["services"] = {
            "conflicting": {"port": default_port("nginx", base=DEFAULT_PORT_BASE)}
        }

        findings = lint_web_terminals(config)

        overlaps = [f for f in findings if f.code == "web_terminals.port_overlap"]
        assert len(overlaps) == 1
        assert "web_terminals.nginx_port" in overlaps[0].message
        assert "services.conflicting.port" in overlaps[0].message

    def test_the_overlap_set_follows_a_moved_base(self):
        """20000, not 10000 — the collision set is the ports this deployment binds."""
        config = _config()
        config["deployment"] = {"port_base": 20000}
        config["services"] = {"conflicting": {"port": 20000}}

        overlaps = [f for f in lint_web_terminals(config) if f.code == "web_terminals.port_overlap"]

        assert len(overlaps) == 1
        assert "Port 20000" in overlaps[0].message

    def test_a_port_that_is_not_a_port_does_not_break_the_linter(self):
        """Types are render's refusal; lint keeps to the one thing it is about."""
        config = _config(nginx_port="8600")
        config["services"] = {"conflicting": {"port": 8600}}

        overlaps = [f for f in lint_web_terminals(config) if f.code == "web_terminals.port_overlap"]

        assert overlaps == []
