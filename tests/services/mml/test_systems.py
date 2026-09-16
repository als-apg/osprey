"""Tests for system-token resolution and the multi-input merge.

The resolution rules are ordered, so each rule has a case showing it wins over
the rules after it. The merge cases pin the shape every later stage reads:
``{system: {family: body}}`` with ``_exports`` and ``_import_order`` beside it.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from osprey.services.mml.loaders import LoadedInput
from osprey.services.mml.loaders.json_any import load_json
from osprey.services.mml.loaders.mat import load_mat
from osprey.services.mml.systems import (
    EXPORTS_KEY,
    IMPORT_ORDER_KEY,
    input_systems,
    merge_inputs,
    resolve_system,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"

_BPM = {
    "FamilyName": "BPM",
    "DeviceList": [[1, 1], [1, 2]],
    "Status": [True, False],
    "X": {"ChannelNames": ["SR:BPM1:X", "  "], "Handles": [1, 2]},
}
_ZERO = {"FamilyName": "BEND", "DeviceList": [[1, 1]], "Setpoint": {"Units": "Hardware"}}


def _flat(
    ao: dict | None = None,
    *,
    ad: dict | None = None,
    export: dict | None = None,
    name: str = "flat.json",
) -> LoadedInput:
    return LoadedInput(
        ao=ao if ao is not None else {"BPM": _BPM},
        ad=ad,
        export=export,
        system_keyed=False,
        source=Path(name),
    )


def _keyed(ao: dict, *, ad: dict | None = None, export: dict | None = None) -> LoadedInput:
    return LoadedInput(ao=ao, ad=ad, export=export, system_keyed=True, source=Path("keyed.json"))


class TestResolveSystem:
    """The ordered rules of FR1."""

    def test_system_keyed_input_has_no_single_token(self):
        """A system-keyed input resolves to None: its keys are its systems."""
        assert resolve_system(_keyed({"SR": {"BPM": _BPM}}), None) is None

    def test_system_keyed_input_refuses_explicit_system(self):
        """Rule 1: --system on a system-keyed input is a usage error naming the file."""
        with pytest.raises(click.UsageError, match="keyed.json"):
            resolve_system(_keyed({"SR": {"BPM": _BPM}}), "SR")

    def test_explicit_system_wins_over_export_and_ad(self):
        """Rule 2 precedes rule 3."""
        loaded = _flat(export={"submachine": "EXP"}, ad={"SubMachine": "ADS"})
        assert resolve_system(loaded, "LTB") == "LTB"

    def test_export_submachine_wins_over_ad(self):
        """Rule 3 reads _export.submachine before AD.SubMachine."""
        loaded = _flat(export={"submachine": "EXP"}, ad={"SubMachine": "ADS"})
        assert resolve_system(loaded, None) == "EXP"

    def test_ad_submachine_when_export_names_none(self):
        """An _export block without submachine falls through to the AD."""
        loaded = _flat(export={"exporter": "mml_export 1.0.0"}, ad={"SubMachine": "ADS"})
        assert resolve_system(loaded, None) == "ADS"

    def test_blank_submachine_is_not_a_token(self):
        """A blank _export.submachine falls through to AD.SubMachine."""
        loaded = _flat(export={"submachine": "  "}, ad={"SubMachine": "ADS"})
        assert resolve_system(loaded, None) == "ADS"

    def test_no_rule_applies_is_usage_error_naming_file(self):
        """Rule 4: nothing yields a token."""
        with pytest.raises(click.UsageError, match="orphan.json"):
            resolve_system(_flat(ad={"Machine": "Quokka"}, name="orphan.json"), None)

    def test_underscore_token_refused(self):
        """A token starting with '_' would be skipped by every reader."""
        with pytest.raises(click.UsageError):
            resolve_system(_flat(), "_hidden")

    def test_paired_fixture_resolves_from_sibling_ad(self):
        """The exporter's paired output takes AD.SubMachine."""
        loaded = load_json(FIXTURES / "paired" / "quokka.ring.ao.json")
        assert resolve_system(loaded, None) == "RING"

    def test_mat_fixture_resolves_from_its_ad(self):
        """A .mat with an AD variable needs no --system."""
        loaded = load_mat(FIXTURES / "mat" / "quokka_booster.mat")
        assert resolve_system(loaded, None) == "BOOSTER"

    def test_flat_fixture_without_ad_needs_system(self):
        """A flat export with no AD falls to rule 4, or takes --system."""
        loaded = load_json(FIXTURES / "tango" / "export.json")
        with pytest.raises(click.UsageError):
            resolve_system(loaded, None)
        assert resolve_system(loaded, "RING") == "RING"


class TestInputSystems:
    """The systems one input contributes."""

    def test_system_keyed_keys_in_file_order_skipping_underscore(self):
        """File order is kept and '_'-prefixed keys are not systems."""
        loaded = _keyed({"_note": "x", "SR": {"BPM": _BPM}, "BR": {"BPM": _BPM}})
        assert input_systems(loaded, None) == ["SR", "BR"]

    def test_flat_input_contributes_one_token(self):
        assert input_systems(_flat(), "LTB") == ["LTB"]


class TestMergeInputs:
    """The merged AO and AD."""

    def test_dialect_fixture_keeps_file_order_and_description(self):
        """The system-keyed fixture merges RING then BOOST, _description intact."""
        ao, _ad = merge_inputs([(load_json(FIXTURES / "dialect" / "export.json"), None)])
        assert ao[IMPORT_ORDER_KEY] == ["RING", "BOOST"]
        assert isinstance(ao["RING"]["_description"], str)
        assert "BEND" in ao["RING"]

    def test_import_order_is_file_order_then_command_line_order(self):
        """Non-alphabetical order survives across and within inputs."""
        keyed = _keyed({"SR": {"BPM": _BPM}, "BR": {"BPM": _BPM}, "GTL": {"BPM": _BPM}})
        ao, _ad = merge_inputs([(keyed, None), (_flat(name="ltb.json"), "LTB"), (_flat(), "BTS")])
        assert ao[IMPORT_ORDER_KEY] == ["SR", "BR", "GTL", "LTB", "BTS"]
        assert isinstance(ao[IMPORT_ORDER_KEY], list)

    def test_every_body_is_normalised(self):
        """Bodies pass through normalize_family: logicals, blanks, Handles."""
        ao, _ad = merge_inputs([(_flat(), "SR")])
        body = ao["SR"]["BPM"]
        assert body["Status"] == [1, 0]
        assert body["X"]["ChannelNames"] == ["SR:BPM1:X", None]
        assert "Handles" not in body["X"]

    def test_family_without_channel_key_is_kept(self):
        """A zero-channel family is never dropped."""
        ao, _ad = merge_inputs([(_flat({"BPM": _BPM, "BEND": _ZERO}), "SR")])
        assert ao["SR"]["BEND"] == _ZERO

    def test_input_is_not_mutated(self):
        loaded = _flat()
        merge_inputs([(loaded, "SR")])
        assert loaded.ao["BPM"]["Status"] == [True, False]

    def test_duplicate_token_across_inputs_refused(self):
        """Two inputs resolving to one system is a usage error naming both files."""
        with pytest.raises(click.UsageError, match=r"a\.json.*b\.json"):
            merge_inputs([(_flat(name="a.json"), "SR"), (_flat(name="b.json"), "SR")])

    def test_duplicate_between_keyed_and_flat_refused(self):
        keyed = _keyed({"SR": {"BPM": _BPM}})
        with pytest.raises(click.UsageError, match="SR"):
            merge_inputs([(keyed, None), (_flat(), "SR")])

    def test_explicit_system_on_keyed_input_refused(self):
        with pytest.raises(click.UsageError):
            merge_inputs([(_keyed({"SR": {"BPM": _BPM}}), "SR")])

    def test_unresolvable_flat_input_refused(self):
        with pytest.raises(click.UsageError, match="flat.json"):
            merge_inputs([(_flat(), None)])

    def test_exports_stored_per_system(self):
        """Each system gets its input's _export block; systems without one get none."""
        export = {"exporter": "mml_export 1.0.0", "machine": "Quokka"}
        keyed = _keyed({"SR": {"BPM": _BPM}, "BR": {"BPM": _BPM}}, export=export)
        ao, _ad = merge_inputs([(keyed, None), (_flat(), "LTB")])
        assert ao[EXPORTS_KEY] == {"SR": export, "BR": export}

    def test_exports_key_always_present(self):
        ao, _ad = merge_inputs([(_flat(), "SR")])
        assert ao[EXPORTS_KEY] == {}

    def test_ad_keyed_by_system_without_export_block(self):
        """The AD lands under its system, with its own _export block removed."""
        loaded = load_json(FIXTURES / "paired" / "quokka.ring.ao.json")
        ao, ad = merge_inputs([(loaded, None)])
        assert list(ad) == ["RING"]
        assert ad["RING"]["Machine"] == "Quokka"
        assert "_export" not in ad["RING"]
        assert ao[EXPORTS_KEY]["RING"]["exporter"] == "mml_export 1.0.0"
        assert "_export" not in ao["RING"]

    def test_system_keyed_ad_split_per_system(self):
        """A system-keyed AD contributes each system's own entry."""
        keyed = _keyed(
            {"SR": {"BPM": _BPM}, "BR": {"BPM": _BPM}},
            ad={"SR": {"SubMachine": "SR"}, "BR": {"SubMachine": "BR"}},
        )
        _ao, ad = merge_inputs([(keyed, None)])
        assert ad == {"SR": {"SubMachine": "SR"}, "BR": {"SubMachine": "BR"}}

    def test_systems_without_ad_are_absent_from_ad(self):
        _ao, ad = merge_inputs([(_flat(), "SR")])
        assert ad == {}

    def test_mixed_fixture_inputs_merge(self):
        """A paired JSON, a .mat and a flat export with --system merge together."""
        ao, ad = merge_inputs(
            [
                (load_json(FIXTURES / "paired" / "quokka.ring.ao.json"), None),
                (load_mat(FIXTURES / "mat" / "quokka_booster.mat"), None),
                (load_json(FIXTURES / "tango" / "export.json"), "TANGO"),
            ]
        )
        assert ao[IMPORT_ORDER_KEY] == ["RING", "BOOSTER", "TANGO"]
        assert set(ad) == {"RING", "BOOSTER"}
        systems = [key for key in ao if not key.startswith("_")]
        assert systems == ["RING", "BOOSTER", "TANGO"]
