"""The channel-suggestion keys a generated ``config.yml`` must actually carry.

Channel typeahead is on by default, which only means anything if the keys that
control it are written into every project's config where an operator can see
and change them. Those keys come from the preset's ``config:`` block — the
framework template renders no ``web`` section — so this is a guard on the
preset, and on what the build makes of it.

The assertion is on *resolution*, not on text: the preset's block is expanded
the way a build expands it and handed to ``ConfigBuilder``, because config.yml
is never flattened. Presets author config keys dotted on purpose; a dotted key
that survived the expansion would still match a text search for the dotted
spelling while reading as nothing at all.
"""

from __future__ import annotations

import yaml

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_profile

PRESET = "control-assistant"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expanded() -> dict:
    """The preset's resolved ``config:`` block, dotted keys folded in."""
    profile, _profile_dir = resolve_build_profile(None, PRESET)
    return _expand_dotted(profile.config)


def _resolved(tmp_path):
    """Write the expanded config out and load it through the production reader."""
    from osprey_connectors.config import ConfigBuilder

    config = _expanded()
    config["project_root"] = str(tmp_path)
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    return ConfigBuilder(str(config_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChannelSuggestionKeys:
    """The keys resolve from the config the preset writes."""

    def test_generated_config_turns_channel_suggestions_on(self, tmp_path):
        builder = _resolved(tmp_path)

        assert builder.get("web.channel_suggestions.enabled") is True

    def test_generated_config_caps_the_snapshot_at_fifty_thousand_channels(self, tmp_path):
        builder = _resolved(tmp_path)

        assert builder.get("web.channel_suggestions.max_channels") == 50000

    def test_the_keys_are_nested_not_dotted(self):
        """A dotted key would be stored verbatim and read by nothing.

        ``ConfigBuilder.get`` walks the mapping, so a top-level
        ``"web.channel_suggestions.enabled"`` string key resolves to nothing
        while still matching a text search for the dotted spelling.
        """
        config = _expanded()

        assert "channel_suggestions" in config["web"]
        assert not any(key.startswith("web.") for key in config)
