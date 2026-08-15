"""Tests for the strict key check inside a profile's ``dispatch:`` block.

Every field of the block is read with ``.get()``, so before the check an
unknown key was silently dropped: ``netwrok: host`` validated clean, the parser
took the ``bridge`` default, and the deployment came up healthy but unreachable
from the address the facility had asked for. The block is now closed the same
way ``environment:``, ``bluesky:`` and the top level are — unknown keys
hard-error, naming every offender and its closest known spelling.

The valid set is *derived* from :class:`DispatchConfig`'s dataclass fields, the
same declaration ``_parse_profile`` reads the block into, so the check cannot
drift from what the parser actually honors.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_load import (
    _KNOWN_DISPATCH_KEYS,
    _parse_profile,
    load_profile,
)
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.cli.build_profile_schema import DispatchConfig
from osprey.errors import BuildProfileError


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


# ── the known set is derived, not hand-maintained ────────────────────────────


def test_known_keys_are_the_schema_fields() -> None:
    """One source of truth: the dataclass the block is parsed into.

    A hand-kept duplicate list would drift the moment a field is added or
    renamed — accepting a key the parser ignores, or rejecting one it reads.
    """
    assert _KNOWN_DISPATCH_KEYS == {f.name for f in fields(DispatchConfig)}


def test_every_valid_key_is_accepted_and_read() -> None:
    """A block spelling every field parses, and each value reaches the config."""
    spelled = DispatchConfig(
        triggers="facility_triggers.yml",
        worker_count=3,
        workspace_mode="shared",
        max_concurrent_runs=4,
        max_queue_depth=99,
        dispatcher_port=8021,
        worker_port_base=9200,
        timeout_sec=600,
        inactivity_sec=240,
        facility_name="ALS",
        pv_strip_prefix="SR:",
        network="host",
    )
    block = {f.name: getattr(spelled, f.name) for f in fields(DispatchConfig)}

    profile = _parse_profile({"name": "x", "dispatch": block})

    assert profile.dispatch == spelled


def test_empty_block_still_takes_the_defaults() -> None:
    """The check rejects unknown keys, it does not require any."""
    profile = _parse_profile({"name": "x", "dispatch": {}})

    assert profile.dispatch == DispatchConfig(triggers="")


# ── unknown keys are rejected ────────────────────────────────────────────────


def test_the_misspelled_network_axis_is_rejected() -> None:
    """The key this check exists for: a typo that silently changes topology."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "dispatch": {"triggers": "t.yml", "netwrok": "host"}})

    message = str(excinfo.value)
    assert "'netwrok'" in message
    assert "did you mean 'network'?" in message


def test_all_unknown_dispatch_keys_are_named_in_one_error() -> None:
    """Accumulated, not first-wins — one pass fixes the whole block."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile({"name": "x", "dispatch": {"zzz_nonsense": 1, "wrkers": 2, "timeout": 30}})

    message = str(excinfo.value)
    assert "'zzz_nonsense'" in message
    assert "'wrkers'" in message
    assert "'timeout'" in message
    assert "valid keys are:" in message
    for key in _KNOWN_DISPATCH_KEYS:
        assert key in message


def test_error_style_matches_the_top_level_check() -> None:
    """Same exception, same wording — operators see one behavior, not two.

    Pinned against the top-level unknown-key error rather than a literal: if
    that message is ever reworded, this check must move with it.
    """
    with pytest.raises(BuildProfileError) as block_err:
        _parse_profile({"name": "x", "dispatch": {"zzz_nonsense": 1}})
    with pytest.raises(BuildProfileError) as top_err:
        _parse_profile({"name": "x", "zzz_nonsense": 1})

    block = str(block_err.value)
    top = str(top_err.value)
    assert block.startswith("Unknown dispatch key(s): 'zzz_nonsense'.")
    assert top.startswith("Unknown profile key(s): 'zzz_nonsense'.")
    # Everything after the offender list is the identical remedy sentence.
    assert (
        block.split("'zzz_nonsense'.", 1)[1].split("valid keys are:")[0]
        == top.split("'zzz_nonsense'.", 1)[1].split("valid keys are:")[0]
    )


def test_unknown_key_survives_a_file_load(tmp_path: Path) -> None:
    """The check is on the parse path every profile file takes."""
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "dispatch": {"netwrok": "host"}})

    with pytest.raises(BuildProfileError, match="'netwrok'"):
        load_profile(profile)


def test_unknown_key_from_a_set_pair_is_rejected() -> None:
    """``--set`` cannot inject a key the block does not define either."""
    with pytest.raises(BuildProfileError, match="'netwrok'"):
        resolve_build_profile(None, "control-assistant", set_pairs=("dispatch.netwrok=host",))


# ── the shipped presets survive the promotion ────────────────────────────────


def test_the_bundled_dispatch_preset_still_resolves() -> None:
    """The one shipped profile carrying a ``dispatch:`` block parses intact."""
    profile, _dir = resolve_build_profile(None, "control-assistant")

    assert profile.dispatch is not None
    assert profile.dispatch.triggers == "tutorial_triggers.yml"
