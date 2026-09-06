"""Tests for the single profile-YAML read point and the preset-side data bundle.

Every raw profile document is parsed through ``_parse_profile_document``, so a
document read can never skip the normalization the pipeline applies to a layer.
``app_template:`` is no longer part of that surface: it is a PRESET-side key
naming the packaged data tree ``osprey init`` copies, consumed by
``_load_preset_raw`` the way ``extends:`` is consumed, and refused outright in a
repo ``profile.yml``. Consuming it is why resolution records the bundled preset
a profile's ``extends:`` chain passed through: that record is what still answers
"which packaged data tree is this from" for a profile carrying no
``provenance:`` block of its own.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli import build_profile_presets
from osprey.cli.build_cmd import _profile_data_bundle
from osprey.cli.build_profile import _load_preset_raw, resolve_build_profile
from osprey.cli.build_profile_presets import (
    DEFAULT_DATA_BUNDLE,
    PRESET_DATA_BUNDLE_KEY,
    preset_data_bundle,
)
from osprey.errors import BuildProfileError


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    """Write a profile document, and the ``data:`` tree a repo profile must name.

    A bundled preset is exempt — it has no profile directory to anchor a tree
    against, and ``osprey init`` is what materializes one — so the key is added
    only for files written outside the preset directory.
    """
    if path.parent.name != "presets":
        body = {**body, "data": "data"}
        (path.parent / "data").mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def fake_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A preset directory the test owns, replacing the bundled one.

    One patch is enough: every reader of the directory goes through
    ``build_profile_presets._presets_dir``, including the ``data:`` exemption
    for a bundled preset, which asks ``is_bundled_preset_dir`` rather than
    comparing against a directory it bound at import time.
    """
    presets = tmp_path / "presets"
    presets.mkdir()
    monkeypatch.setattr(build_profile_presets, "_presets_dir", lambda: presets)
    return presets


# ---------------------------------------------------------------------------
# The bundle is preset-side: consumed on read, never part of the profile
# ---------------------------------------------------------------------------


def test_preset_read_consumes_the_bundle_key(fake_presets: Path) -> None:
    """``_load_preset_raw`` returns a layer with no ``app_template:`` in it.

    Popping at the single read point is what lets the key stay out of the
    profile schema while the shipped presets keep spelling it: every layer the
    merge sees is already free of it, so no resolved profile can carry it.
    """
    _write_yaml(fake_presets / "spelled.yml", {"name": "spelled", "app_template": "hello_world"})

    raw, _path = _load_preset_raw("spelled")

    assert PRESET_DATA_BUNDLE_KEY not in raw
    assert "data_bundle" not in raw


def test_a_preset_naming_a_bundle_resolves_cleanly(fake_presets: Path) -> None:
    """A preset spelling the consumed key is still a valid profile layer."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})

    profile, _dir = resolve_build_profile(None, "base")

    assert profile.name == "base"


def test_an_inherited_bundle_key_is_consumed_too(fake_presets: Path) -> None:
    """The key is consumed on EVERY preset layer, not only the one named.

    Regression for the persona path: the ``control-assistant-*`` presets carry
    no ``app_template:`` of their own and reach the base one through
    ``extends:``. Popping only at the named preset left the inherited key in
    the merged raw, where the unknown-key check refused it — so
    ``osprey init --preset control-assistant`` failed on every persona while
    ``hello-world``, which has no children, still worked.
    """
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})
    _write_yaml(fake_presets / "child.yml", {"name": "child", "extends": "base"})

    profile, _dir = resolve_build_profile(None, "child")

    assert profile.name == "child"


def _presets_extending(base: str) -> list[str]:
    """Every bundled preset whose ``extends:`` names ``base``.

    Derived rather than listed, so a persona preset added later is covered the
    day it ships. Read verbatim (``_read_preset_document``) because the point of
    these presets is the key ``_load_preset_raw`` consumes.
    """
    target = build_profile_presets._normalize_preset_name(base)
    found = []
    for name in build_profile_presets.list_presets():
        raw, _path = build_profile_presets._read_preset_document(name)
        parent = raw.get("extends")
        if (
            isinstance(parent, str)
            and build_profile_presets._normalize_preset_name(parent) == target
        ):
            found.append(name)
    return found


def test_the_persona_presets_are_discoverable() -> None:
    """Guards the parametrization below from passing vacuously."""
    assert _presets_extending("control-assistant")


@pytest.mark.parametrize("preset", _presets_extending("control-assistant"))
def test_every_shipped_persona_preset_resolves(preset: str) -> None:
    """The same regression against the real presets `osprey init` materializes.

    Each of these inherits the base preset's bundle key, and every one of them
    is resolved while a `control-assistant` repo is being initialized.
    """
    profile, _dir = resolve_build_profile(None, preset)

    assert profile.name


def test_preset_data_bundle_reads_the_preset_file(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})

    assert preset_data_bundle("base") == "hello_world"


def test_preset_data_bundle_follows_extends(fake_presets: Path) -> None:
    """A preset that inherits the key names no bundle of its own."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})
    _write_yaml(fake_presets / "child.yml", {"name": "child", "extends": "base"})

    assert preset_data_bundle("child") == "hello_world"


def test_the_nearest_bundle_on_the_chain_wins(fake_presets: Path) -> None:
    """Same answer the deep merge would have given: the child's key wins."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "control_assistant"})
    _write_yaml(
        fake_presets / "child.yml",
        {"name": "child", "extends": "base", "app_template": "hello_world"},
    )

    assert preset_data_bundle("child") == "hello_world"


def test_preset_data_bundle_accepts_either_cli_spelling(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "my-preset.yml", {"name": "mine", "app_template": "hello_world"})

    assert preset_data_bundle("my_preset") == preset_data_bundle("my-preset") == "hello_world"


@pytest.mark.parametrize("name", [None, "", "no-such-preset"])
def test_preset_data_bundle_falls_back_when_there_is_no_preset(
    fake_presets: Path, name: str | None
) -> None:
    """No preset, or one this installation does not ship, is not an error here.

    The bundle only decides which packaged data tree is copied; a caller with
    no preset in hand gets the framework default rather than an exception it
    would have to translate.
    """
    assert preset_data_bundle(name) == DEFAULT_DATA_BUNDLE


def test_a_preset_chain_naming_no_bundle_falls_back(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base"})
    _write_yaml(fake_presets / "child.yml", {"name": "child", "extends": "base"})

    assert preset_data_bundle("child") == DEFAULT_DATA_BUNDLE


def test_a_cyclic_extends_chain_terminates(fake_presets: Path) -> None:
    """The walk is guarded, so a cycle answers instead of hanging.

    ``_resolve_extends`` rejects the cycle properly when the preset is used;
    this walk only has to not be the thing that spins.
    """
    _write_yaml(fake_presets / "a.yml", {"name": "a", "extends": "b"})
    _write_yaml(fake_presets / "b.yml", {"name": "b", "extends": "a"})

    assert preset_data_bundle("a") == DEFAULT_DATA_BUNDLE


def test_every_bundled_preset_names_a_packaged_bundle() -> None:
    """The shipped presets must resolve to a data tree that exists on disk."""
    from osprey.cli.build_profile import list_presets
    from osprey.cli.templates.manager import TemplateManager

    template_root = Path(TemplateManager().template_root)
    for name in build_profile_presets.list_presets():
        bundle = preset_data_bundle(name)
        assert (template_root / "apps" / bundle / "data").is_dir(), f"{name} -> {bundle}"
    assert list_presets()


# ---------------------------------------------------------------------------
# A profile that inherits its bundle instead of recording a preset
# ---------------------------------------------------------------------------


def test_a_preset_build_keeps_its_own_bundle(fake_presets: Path) -> None:
    """``--preset`` mode records the preset it was asked for.

    No bundled preset carries a ``provenance:`` block — nothing materialized
    it — so a build resolved straight from a preset has only this record to
    name its bundle. Without it every ``--preset`` build would read as the
    framework default.
    """
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})

    profile, _dir = resolve_build_profile(None, "base")

    assert profile.provenance is None
    assert profile.inherited_preset == "base"
    assert _profile_data_bundle(profile) == "hello_world"


def test_a_preset_build_inherits_the_bundle_through_extends(fake_presets: Path) -> None:
    """A persona preset names no bundle of its own; the base one answers.

    The recorded name is the preset asked for, and the bundle is resolved from
    there — so recording the nearest preset is enough, whatever depth the key
    sits at.
    """
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})
    _write_yaml(fake_presets / "child.yml", {"name": "child", "extends": "base"})

    profile, _dir = resolve_build_profile(None, "child")

    assert profile.inherited_preset == "child"
    assert _profile_data_bundle(profile) == "hello_world"


def test_extends_bundled_preset_keeps_its_bundle(fake_presets: Path, tmp_path: Path) -> None:
    """A hand-written profile that only ``extends:`` a preset keeps its bundle.

    The bundle used to reach the build as an inherited profile key, so this
    shape got it from the deep merge. The key is preset-side now and consumed
    during resolution, so resolution records the preset it passed through and
    the build reads the bundle back from that — otherwise the profile would
    silently build on the framework default, copying another bundle's
    ``services/`` and ``machine_data/`` trees.
    """
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})
    profile_file = _write_yaml(tmp_path / "profile.yml", {"name": "hand", "extends": "base"})

    profile, _dir = resolve_build_profile(profile_file, None)

    assert profile.provenance is None
    assert profile.inherited_preset == "base"
    assert _profile_data_bundle(profile) == "hello_world"


def test_the_nearest_preset_on_the_chain_is_the_one_recorded(
    fake_presets: Path, tmp_path: Path
) -> None:
    """Nearest-first, the order the deep merge resolved the key by."""
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "control_assistant"})
    _write_yaml(
        fake_presets / "child.yml",
        {"name": "child", "extends": "base", "app_template": "hello_world"},
    )
    profile_file = _write_yaml(tmp_path / "profile.yml", {"name": "hand", "extends": "child"})

    profile, _dir = resolve_build_profile(profile_file, None)

    assert profile.inherited_preset == "child"
    assert _profile_data_bundle(profile) == "hello_world"


def test_a_preset_reached_through_a_local_parent_is_recorded(
    fake_presets: Path, tmp_path: Path
) -> None:
    """The chain is followed through the facility's own files, not just presets.

    An ALS-style profile extends a sibling ``*-base.yml``, and that file is what
    names the preset. The bundle was inherited down the whole chain before, so
    it is recorded from wherever on the chain the preset sits.
    """
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})
    _write_yaml(tmp_path / "site-base.yml", {"name": "site", "extends": "base"})
    profile_file = _write_yaml(
        tmp_path / "profile.yml", {"name": "hand", "extends": "site-base.yml"}
    )

    profile, _dir = resolve_build_profile(profile_file, None)

    assert profile.inherited_preset == "base"
    assert _profile_data_bundle(profile) == "hello_world"


def test_a_recorded_provenance_wins_over_the_extends_chain(
    fake_presets: Path, tmp_path: Path
) -> None:
    """What ``osprey init`` materialized beats what the chain passes through.

    Both records name a preset, and they can disagree: a materialized profile
    keeps the ``extends:`` it was emitted with while its ``provenance:`` names
    what it was actually built from. Provenance is the profile's own statement
    about its origin, so it is asked first.
    """
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})
    _write_yaml(fake_presets / "other.yml", {"name": "other", "app_template": "ariel_standalone"})
    profile_file = _write_yaml(
        tmp_path / "profile.yml",
        {
            "name": "hand",
            "extends": "base",
            "provenance": {"preset": "other", "preset_hash": "unchecked-here"},
        },
    )

    profile, _dir = resolve_build_profile(profile_file, None)

    assert _profile_data_bundle(profile) == "ariel_standalone"


def test_an_unshipped_provenance_preset_falls_through_to_the_chain(
    fake_presets: Path, tmp_path: Path
) -> None:
    """A stale ``provenance.preset`` does not cost the profile its bundle.

    The recorded preset was renamed, removed, or came from another OSPREY, so
    it names no bundle here. The ``extends:`` chain still resolved, and the
    preset it reached is a better answer than the framework default: without
    this the build would copy another bundle's ``services/`` and
    ``machine_data/`` trees for a profile whose chain says exactly which trees
    it wants.
    """
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})
    profile_file = _write_yaml(
        tmp_path / "profile.yml",
        {
            "name": "hand",
            "extends": "base",
            "provenance": {"preset": "gone", "preset_hash": "unchecked-here"},
        },
    )

    profile, _dir = resolve_build_profile(profile_file, None)

    assert profile.inherited_preset == "base"
    assert _profile_data_bundle(profile) == "hello_world"


def test_a_profile_with_no_preset_anywhere_gets_the_default(tmp_path: Path) -> None:
    """Nothing to inherit from is still not an error — it is the default."""
    profile_file = _write_yaml(tmp_path / "profile.yml", {"name": "hand"})

    profile, _dir = resolve_build_profile(profile_file, None)

    assert profile.inherited_preset is None
    assert _profile_data_bundle(profile) == DEFAULT_DATA_BUNDLE


# ---------------------------------------------------------------------------
# The retired profile-side spellings
# ---------------------------------------------------------------------------


def test_a_profile_naming_a_bundle_is_refused(tmp_path: Path) -> None:
    """The one thing a profile may not say about the app template.

    The refusal itself, and the reason it is worded the way it is, are pinned
    in ``test_profile_unknown_keys.py``; asserted here because this file is
    where the preset-side/profile-side split lives.
    """
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "app_template": "hello_world"})

    with pytest.raises(BuildProfileError, match="no longer a profile key"):
        resolve_build_profile(profile, None)


def test_the_old_field_spelling_is_an_ordinary_unknown_key(tmp_path: Path) -> None:
    """The retired ``data_bundle`` key was never a profile spelling.

    It gets the generic message rather than the expand one: nothing OSPREY ever
    emitted wrote that key into a profile, so a file carrying it is a typo, not
    a profile from an older release.
    """
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "data_bundle": "hello_world"})

    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(profile, None)

    assert "'data_bundle'" in str(excinfo.value)
    assert "valid keys are:" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Single-read-point guard
# ---------------------------------------------------------------------------


# (module, enclosing function, call) triples allowed to deserialize YAML in the
# profile pipeline. Everything else must route through _parse_profile_document
# (or _read_profile_document, its file-reading front), or a new read would
# silently skip the pipeline's per-document normalization.
_ALLOWED_YAML_READS = {
    ("build_profile_document.py", "_parse_profile_document", "yaml_loader.safe_load"),
    # Scalar `--set` values, not documents — layered like any other document.
    ("build_profile_resolve.py", "_parse_set_pairs", "yaml.safe_load"),
    # The emitter's ruamel round-trip is a comment source; it reads the preset
    # file as text so the preset's own comments survive into the emission.
    ("build_profile_emit.py", "emit_standalone_profile_yaml", "_yaml.load"),
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
    """Every profile-document parse routes through ``_parse_profile_document``."""
    pipeline_dir = Path(build_profile_presets.__file__).parent
    modules = sorted(pipeline_dir.glob("build_profile_*.py"))
    assert modules, f"no build_profile_*.py modules found under {pipeline_dir}"

    found: set[tuple[str, str, str]] = set()
    for module in modules:
        found |= _yaml_read_sites(module)

    assert found - _ALLOWED_YAML_READS == set()
    # The helper itself must still be the read point — an empty result would
    # otherwise pass vacuously.
    assert (
        "build_profile_document.py",
        "_parse_profile_document",
        "yaml_loader.safe_load",
    ) in found
