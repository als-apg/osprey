"""Tests for the single profile-YAML read point and its alias normalization.

``app_template:`` is the profile-YAML spelling of the app-template selector;
``data_bundle:`` remains an accepted alias (and stays the canonical Python
field name). Every raw profile document is read through
``_read_profile_document``, which normalizes the spelling once per document —
so only one spelling survives a layer and mixed-spelling ``extends:`` /
override chains merge child-wins through the ordinary deep merge.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli import build_profile_presets
from osprey.cli.build_profile import (
    _load_preset_raw,
    _parse_profile,
    compute_profile_hash,
    load_profile,
    resolve_build_profile,
)
from osprey.errors import BuildProfileError


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def fake_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A preset directory the test owns, replacing the bundled one."""
    presets = tmp_path / "presets"
    presets.mkdir()
    monkeypatch.setattr(build_profile_presets, "_presets_dir", lambda: presets)
    return presets


# ---------------------------------------------------------------------------
# Single-document normalization
# ---------------------------------------------------------------------------


def test_app_template_populates_the_canonical_field(tmp_path: Path) -> None:
    """The YAML-surface spelling lands on the ``data_bundle`` field."""
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "app_template": "hello_world"})

    assert load_profile(profile).data_bundle == "hello_world"


def test_data_bundle_alias_still_loads(tmp_path: Path) -> None:
    """Pre-rename profiles keep working — the old spelling is an accepted alias."""
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "data_bundle": "hello_world"})

    assert load_profile(profile).data_bundle == "hello_world"


def test_both_spellings_with_differing_values_error_legibly(tmp_path: Path) -> None:
    """One document cannot name two different bundles."""
    profile = tmp_path / "p.yml"
    profile.write_text(
        "name: p\napp_template: hello_world\ndata_bundle: control_assistant\n",
        encoding="utf-8",
    )

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile(profile)

    message = str(excinfo.value)
    assert "app_template" in message
    assert "data_bundle" in message
    assert "hello_world" in message
    assert "control_assistant" in message
    assert str(profile) in message


def test_both_spellings_with_the_same_value_are_accepted(tmp_path: Path) -> None:
    """Redundant is not contradictory — only differing values are an error."""
    profile = tmp_path / "p.yml"
    profile.write_text(
        "name: p\napp_template: hello_world\ndata_bundle: hello_world\n", encoding="utf-8"
    )

    assert load_profile(profile).data_bundle == "hello_world"


def test_parse_profile_normalizes_a_hand_assembled_dict() -> None:
    """The parser is reached directly by callers that build their own raw dict.

    ``app_template`` is allowlisted, so without normalizing here it would parse
    cleanly and be ignored — the loader default silently winning over what the
    caller asked for.
    """
    profile = _parse_profile({"name": "p", "app_template": "hello_world"})

    assert profile.data_bundle == "hello_world"


def test_preset_read_normalizes(fake_presets: Path) -> None:
    """``_load_preset_raw`` returns the canonical field regardless of spelling."""
    _write_yaml(fake_presets / "spelled.yml", {"name": "spelled", "app_template": "hello_world"})

    raw, _path = _load_preset_raw("spelled")

    assert raw["data_bundle"] == "hello_world"
    assert "app_template" not in raw


# ---------------------------------------------------------------------------
# Layer chains: extends, overrides, --set
# ---------------------------------------------------------------------------


def test_extends_parent_spelling_reaches_the_child(fake_presets: Path, tmp_path: Path) -> None:
    """A child inheriting a parent's ``app_template:`` gets the parent's bundle.

    Regression for the silent-default trap: an unnormalized ``extends:`` parent
    read leaves the child with no recognized key, so it builds the loader
    default instead of what the parent named.
    """
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})
    child = _write_yaml(tmp_path / "child.yml", {"name": "child", "extends": "base"})

    profile, _dir = resolve_build_profile(child, None)

    assert profile.data_bundle == "hello_world"


def test_extends_bundled_preset_keeps_its_bundle(tmp_path: Path) -> None:
    """``extends: hello-world`` against the real bundled preset, whichever spelling it uses."""
    child = _write_yaml(tmp_path / "child.yml", {"name": "child", "extends": "hello-world"})

    profile, _dir = resolve_build_profile(child, None)

    assert profile.data_bundle == "hello_world"


@pytest.mark.parametrize(
    ("parent_key", "child_key"),
    [("app_template", "data_bundle"), ("data_bundle", "app_template")],
)
def test_mixed_spelling_extends_chain_resolves_child_wins(
    fake_presets: Path, tmp_path: Path, parent_key: str, child_key: str
) -> None:
    """Either spelling order collapses to one key per layer, so the child wins."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", parent_key: "control_assistant"})
    child = _write_yaml(
        tmp_path / "child.yml", {"name": "child", "extends": "base", child_key: "hello_world"}
    )

    profile, _dir = resolve_build_profile(child, None)

    assert profile.data_bundle == "hello_world"


def test_override_file_spelling_wins_over_the_base(fake_presets: Path, tmp_path: Path) -> None:
    """A ``-O`` file spelled the new way overrides a base spelled the old way."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "data_bundle": "control_assistant"})
    override = _write_yaml(tmp_path / "o.yml", {"app_template": "hello_world"})

    profile, _dir = resolve_build_profile(None, "base", overrides=(override,))

    assert profile.data_bundle == "hello_world"


def test_set_pair_overrides_the_preset_bundle(fake_presets: Path) -> None:
    """``--set app_template=...`` reaches the field like any other override layer."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "data_bundle": "control_assistant"})

    profile, _dir = resolve_build_profile(None, "base", set_pairs=("app_template=hello_world",))

    assert profile.data_bundle == "hello_world"


def test_set_pairs_with_both_spellings_error(fake_presets: Path) -> None:
    """``--set`` is one authored layer, so it gets the same double-spelling check."""
    _write_yaml(fake_presets / "base.yml", {"name": "base"})

    with pytest.raises(BuildProfileError, match="app_template"):
        resolve_build_profile(
            None, "base", set_pairs=("app_template=hello_world", "data_bundle=control_assistant")
        )


def test_profile_hash_is_blind_to_the_spelling(tmp_path: Path) -> None:
    """Two profiles differing only in spelling hash identically."""
    new_spelling = _write_yaml(tmp_path / "new.yml", {"name": "p", "app_template": "hello_world"})
    old_spelling = _write_yaml(tmp_path / "old.yml", {"name": "p", "data_bundle": "hello_world"})

    assert compute_profile_hash(new_spelling) == compute_profile_hash(old_spelling)


# ---------------------------------------------------------------------------
# Single-read-point guard
# ---------------------------------------------------------------------------


# (module, enclosing function, call) triples allowed to deserialize YAML in the
# profile pipeline. Everything else must route through _read_profile_document,
# or a new read would silently skip alias normalization.
_ALLOWED_YAML_READS = {
    ("build_profile_document.py", "_read_profile_document", "yaml.safe_load"),
    # Scalar `--set` values, not documents — normalized as a layer instead.
    ("build_profile_resolve.py", "_parse_set_pairs", "yaml.safe_load"),
    # The emitter's ruamel round-trip is a comment source; it must keep the
    # YAML-surface spelling rather than normalize it.
    ("build_profile_emit.py", "emit_standalone_profile_yaml", "_yaml.load"),
    # Not a profile document at all: this reopens the RENDERED config.yml to
    # retruth one end-of-line comment the stand-in's override falsified. It
    # reads no key by name, so there is no alias to normalize.
    ("build_profile_standin.py", "rewrite_strict_limits_comment", "_yaml.load"),
}


def _enclosing_function(tree: ast.Module, lineno: int) -> str:
    """Name of the innermost function containing ``lineno``."""
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end and (best is None or node.lineno > best.lineno):
                best = node
    return best.name if best is not None else "<module>"


def _yaml_read_sites(source_path: Path) -> set[tuple[str, str, str]]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    sites: set[tuple[str, str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        func = node.func
        if func.attr not in {"safe_load", "load"} or not isinstance(func.value, ast.Name):
            continue
        site = _enclosing_function(tree, node.lineno)
        sites.add((source_path.name, site, f"{func.value.id}.{func.attr}"))
    return sites


def test_yaml_document_reads_happen_only_in_the_helper() -> None:
    """Every profile-document parse routes through ``_read_profile_document``."""
    pipeline_dir = Path(build_profile_presets.__file__).parent
    modules = sorted(pipeline_dir.glob("build_profile_*.py"))
    assert modules, f"no build_profile_*.py modules found under {pipeline_dir}"

    found: set[tuple[str, str, str]] = set()
    for module in modules:
        found |= _yaml_read_sites(module)

    assert found - _ALLOWED_YAML_READS == set()
    # The helper itself must still be the read point — an empty result would
    # otherwise pass vacuously.
    assert ("build_profile_document.py", "_read_profile_document", "yaml.safe_load") in found
