"""The context the bundled config templates require, stated once.

``templates/project/config.yml.j2`` refuses a render that reaches its panel
block without ``builtin_panels`` or ``selected_web_panels``, and it reads a
handful of other keys the build derives. Five test modules assemble a context
for it by hand, so a key the template starts requiring reds them one at a time
and each is patched separately. The context that satisfies the templates is
stated here, once, and every by-hand renderer spreads it.

``tests/templates/test_framework_config_template.py`` holds it to that: it
renders every name in :data:`CONFIG_TEMPLATES` from this context, so a new
required key trips one case here rather than five modules over five sittings.
"""

from __future__ import annotations

from typing import Any

from osprey.cli.templates.manager import TemplateManager
from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports
from osprey.profiles.web_panels import BUILTIN_PANELS

# The one bundled template that renders an ``execution:`` block. It is derived
# from the profile's ``environment:`` declaration, so it belongs to the
# framework template rather than to any deployment's own ``config:``.
CONFIG_TEMPLATES: tuple[str, ...] = ("project/config.yml.j2",)

CONFIG_TEMPLATE = CONFIG_TEMPLATES[0]

#: A catalog small enough to read in a failure message, with an entry carrying a
#: key the contract does not name — those pass through verbatim.
PROVIDER_CATALOG: dict[str, Any] = {
    "house": {
        "api_key": "${HOUSE_API_KEY}",
        "base_url": "https://gateway.example.org/v1",
        "models": {"haiku": "small", "sonnet": "mid", "opus": "large"},
    },
    "bare": {"base_url": "http://127.0.0.1:8000/v1", "timeout": 30},
}

#: What a hello-world-shaped profile gives the template: no channel-finder
#: agent, no logbook, no web panels — with the panel registry the manager
#: injects on every render and a catalog small enough to read in a failure
#: message. The floor every other case adds to.
MINIMAL_CONFIG_CONTEXT: dict[str, Any] = {
    "project_name": "demo",
    "project_root": "/repos/demo",
    "default_provider": "anthropic",
    "default_model": "haiku",
    "port_base": DEFAULT_PORT_BASE,
    "osprey_ports": layout_ports(DEFAULT_PORT_BASE),
    "provider_catalog": PROVIDER_CATALOG,
    "builtin_panels": sorted(BUILTIN_PANELS),
    "selected_web_panels": [],
    "ariel_server_on": False,
}


def render_config(template_name: str = CONFIG_TEMPLATE, /, **overrides: Any) -> str:
    """The named config template rendered from the minimal context, *overrides* laid over it.

    *template_name* is positional-only so a context key of the same name cannot
    collide with it.
    """
    context = {**MINIMAL_CONFIG_CONTEXT, **overrides}
    return TemplateManager().jinja_env.get_template(template_name).render(**context)
