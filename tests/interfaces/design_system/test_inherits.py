"""Tests for the shared ``$extensions.inherits`` opt-out resolution helpers.

These functions are the single source both the validator and the CSS emitter
consult to decide which mode group supplies a theme stem's tokens, so their
filtering (drop malformed maps for the emitter, expose raw for the validator)
and their fallback ordering are load-bearing contracts.
"""

from __future__ import annotations

from types import SimpleNamespace

from osprey.interfaces.design_system.generator.inherits import (
    observed_modes,
    raw_inherits,
    source_mode,
    string_inherits,
)


def _tree(interface_metadata: dict) -> SimpleNamespace:
    """A stand-in TokenTree exposing only the attribute the helpers read."""
    return SimpleNamespace(interface_metadata=interface_metadata)


class TestRawInherits:
    def test_absent_stem_returns_empty_dict(self):
        assert raw_inherits(_tree({}), "wt-crt") == {}

    def test_absent_inherits_key_returns_empty_dict(self):
        assert raw_inherits(_tree({"wt-crt": {}}), "wt-crt") == {}

    def test_returns_raw_value_verbatim(self):
        meta = {"wt-crt": {"inherits": {"high-contrast-dark": "dark"}}}
        assert raw_inherits(_tree(meta), "wt-crt") == {"high-contrast-dark": "dark"}

    def test_non_object_value_passed_through_uncoerced(self):
        """A malformed (non-dict) inherits value is returned as-is so the
        validator can report it rather than the helper silently swallowing it."""
        meta = {"wt-crt": {"inherits": "not-a-map"}}
        assert raw_inherits(_tree(meta), "wt-crt") == "not-a-map"


class TestStringInherits:
    def test_non_dict_raw_yields_empty(self):
        meta = {"wt-crt": {"inherits": ["not", "a", "map"]}}
        assert string_inherits(_tree(meta), "wt-crt") == {}

    def test_keeps_only_str_to_str_entries(self):
        meta = {
            "wt-crt": {
                "inherits": {
                    "high-contrast-dark": "dark",
                    "high-contrast-light": "light",
                    "bad-value": 123,
                }
            }
        }
        assert string_inherits(_tree(meta), "wt-crt") == {
            "high-contrast-dark": "dark",
            "high-contrast-light": "light",
        }


class TestObservedModes:
    def test_extracts_leading_dot_segment(self):
        tokens = {
            "dark.wt-crt.opacity": 1,
            "light.wt-crt.opacity": 1,
            "dark.wt-crt.glow": 2,
        }
        assert observed_modes(tokens) == {"dark", "light"}

    def test_empty_tokens_yield_empty_set(self):
        assert observed_modes({}) == set()


class TestSourceMode:
    def test_direct_authorship_returns_theme_stem(self):
        tokens = {"high-contrast-dark.wt-crt.x": 1}
        assert source_mode(_tree({}), "wt-crt", tokens, "high-contrast-dark") == (
            "high-contrast-dark"
        )

    def test_valid_opt_out_returns_inherited_base(self):
        meta = {"wt-crt": {"inherits": {"high-contrast-dark": "dark"}}}
        tokens = {"dark.wt-crt.x": 1}
        assert source_mode(_tree(meta), "wt-crt", tokens, "high-contrast-dark") == "dark"

    def test_inherits_base_not_authored_returns_none(self):
        """An inherits entry pointing at a group the interface doesn't author
        contributes nothing — the helper must not fabricate a source."""
        meta = {"wt-crt": {"inherits": {"high-contrast-dark": "dark"}}}
        tokens = {"light.wt-crt.x": 1}
        assert source_mode(_tree(meta), "wt-crt", tokens, "high-contrast-dark") is None

    def test_no_authorship_and_no_inherits_returns_none(self):
        tokens = {"light.wt-crt.x": 1}
        assert source_mode(_tree({}), "wt-crt", tokens, "high-contrast-dark") is None

    def test_direct_authorship_wins_over_inherits(self):
        """When the interface authors the group directly, the inherits map is
        never consulted — the theme stem itself is the source."""
        meta = {"wt-crt": {"inherits": {"high-contrast-dark": "dark"}}}
        tokens = {"high-contrast-dark.wt-crt.x": 1, "dark.wt-crt.x": 2}
        assert source_mode(_tree(meta), "wt-crt", tokens, "high-contrast-dark") == (
            "high-contrast-dark"
        )
