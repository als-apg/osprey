"""The channel-suggestion keys a generated ``config.yml`` must actually carry.

Channel typeahead is on by default, which only means anything if the keys that
control it are written into every project's config where an operator can see
and change them. The assertion here is therefore on *resolution*, not on text:
the packaged template is rendered and the result handed to ``ConfigBuilder``,
because config.yml is never flattened — a dotted key written into the template
by mistake would still appear in a text search while reading as nothing at all.

The block also has to survive every branch of the template. ``web:`` is full of
conditionals (builtin panels selected or not, panel presets defined or not), so
each render below exercises a different combination and expects the same two
values out of all of them.
"""

from __future__ import annotations

import pytest
import yaml

from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _packaged_template(rel_path: str):
    """Compile a packaged template, addressed from the templates root.

    Rooted at the packaged ``templates/`` directory rather than compiled as a
    bare ``jinja2.Template`` so relative loads resolve the way they do under
    ``osprey build``, and left in the default Undefined mode so ``| default()``
    chains behave as they do in production.
    """
    from importlib import resources

    from jinja2 import Environment, FileSystemLoader

    import osprey

    templates_root = resources.files(osprey).joinpath("templates")
    env = Environment(loader=FileSystemLoader(str(templates_root)), autoescape=False)
    return env.get_template(rel_path)


def _render(project_root: str, **overrides) -> str:
    """Render the control-assistant config with a minimal realistic context.

    Only the variables the template dereferences unconditionally are supplied;
    everything else is left undefined so the ``| default(...)`` and
    ``is defined`` guards run as they do for a project that selected nothing.
    """
    context = {
        # Built by TemplateManager._project_context in a real
        # render; this helper reaches the template directly.
        "port_base": DEFAULT_PORT_BASE,
        "osprey_ports": layout_ports(DEFAULT_PORT_BASE),
        "project_name": "demo",
        "project_root": project_root,
        "builtin_panels": ["ariel", "channel-finder"],
        "selected_web_panels": ["ariel", "channel-finder"],
        "channel_finder_mode": "in_context",
        "default_pipeline": "in_context",
        "default_provider": "anthropic",
        "default_model": "sonnet",
        "enable_in_context": True,
        **overrides,
    }
    return _packaged_template("apps/control_assistant/config.yml.j2").render(**context)


def _resolved(tmp_path, **overrides):
    """Render, write, and load through the production config reader."""
    from osprey_connectors.config import ConfigBuilder

    config_path = tmp_path / "config.yml"
    rendered = _render(str(tmp_path), **overrides)
    config_path.write_text(rendered)

    # A malformed render would surface as a resolution miss below rather than
    # as a parse error, so the YAML is checked on its own first.
    assert yaml.safe_load(rendered) is not None

    return ConfigBuilder(str(config_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChannelSuggestionKeys:
    """The keys resolve from a rendered config, in every template branch."""

    def test_generated_config_turns_channel_suggestions_on(self, tmp_path):
        builder = _resolved(tmp_path)

        assert builder.get("web.channel_suggestions.enabled") is True

    def test_generated_config_caps_the_snapshot_at_fifty_thousand_channels(self, tmp_path):
        builder = _resolved(tmp_path)

        assert builder.get("web.channel_suggestions.max_channels") == 50000

    @pytest.mark.parametrize(
        "branch",
        [
            pytest.param({"selected_web_panels": []}, id="no-builtin-panels"),
            pytest.param({"default_panel": "artifacts"}, id="default-panel-set"),
            pytest.param(
                {"panel_presets": {"Machine setup": ["channel-finder", "artifacts"]}},
                id="panel-presets-defined",
            ),
        ],
    )
    def test_keys_are_written_whatever_else_the_web_section_renders(self, tmp_path, branch):
        builder = _resolved(tmp_path, **branch)

        assert builder.get("web.channel_suggestions.enabled") is True
        assert builder.get("web.channel_suggestions.max_channels") == 50000

    def test_the_keys_are_nested_not_dotted(self, tmp_path):
        """A dotted key would be stored verbatim and read by nothing.

        ``ConfigBuilder.get`` walks the mapping, so a top-level
        ``"web.channel_suggestions.enabled"`` string key resolves to nothing
        while still matching a text search for the dotted spelling.
        """
        rendered = _render(str(tmp_path))
        loaded = yaml.safe_load(rendered)

        assert "channel_suggestions" in loaded["web"]
        assert not any(key.startswith("web.") for key in loaded)
