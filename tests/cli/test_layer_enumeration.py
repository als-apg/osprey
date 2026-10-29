"""One listing of layer files, shared by the build, the hash and the fingerprint.

The build loop renders one project per layer file; the content hash folds
every layer file; the staleness fingerprint digests every layer file by name.
All three read :func:`~osprey.cli.profile_root.layer_files`, so an edit to any
layer — a tier, an app, or a file in the retired layer directory — moves the hash the
deploy-side check reads, and nothing the build renders goes unfingerprinted.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import get_args

import pytest

from osprey.cli import build_profile_merge, scaffold_layers
from osprey.cli.build_cmd import _layer_files
from osprey.cli.build_profile_merge import compute_profile_hash, layer_fold_label
from osprey.cli.profile_root import ProfileKind, layer_files
from osprey.deployment.staleness import MATERIAL_PREFIX, profile_fingerprint
from osprey.deployment.web_terminals.render_naming import RENDER_KIND_TIER
from osprey.deployment.web_terminals.roster import APP_ENTRY_KIND
from osprey.utils import workspace
from osprey.utils.workspace import (
    APPS_DIRNAME,
    LAYER_DIR_KINDS,
    LAYER_DIRNAMES,
    RETIRED_LAYER_DIRS,
    TIERS_DIRNAME,
    layer_dir_kind,
)
from osprey_connectors.errors import BuildProfileError
from tests._retired_layout import RETIRED_LAYER_DIRNAME


def _repo(tmp_path: Path, *layers: str) -> Path:
    """A tiny repo: ``profile.yml`` plus one file per relative layer path."""
    root = tmp_path / "my-profile"
    root.mkdir(parents=True)
    (root / "profile.yml").write_text("name: my-profile\n")
    for layer in layers:
        file = root / layer
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"name: {file.stem}\n")
    return root.resolve()


# ── The helper ──────────────────────────────────────────────────────────────


def test_layer_dirnames_name_every_layer_directory() -> None:
    assert LAYER_DIRNAMES == (TIERS_DIRNAME, APPS_DIRNAME, *RETIRED_LAYER_DIRS)
    assert LAYER_DIRNAMES == (*LAYER_DIR_KINDS, *RETIRED_LAYER_DIRS)


def test_every_layer_directory_is_read_as_a_known_kind() -> None:
    assert {layer_dir_kind(d) for d in LAYER_DIRNAMES} == set(LAYER_DIR_KINDS.values())
    assert all(successor in LAYER_DIR_KINDS for successor in RETIRED_LAYER_DIRS.values())
    assert layer_dir_kind("profiles") is None


def test_scaffold_layer_kinds_are_the_live_layer_directories() -> None:
    assert set(get_args(scaffold_layers.LayerKind)) == set(LAYER_DIR_KINDS)


def test_render_kinds_are_the_layer_kind_words() -> None:
    assert {RENDER_KIND_TIER, APP_ENTRY_KIND} == set(LAYER_DIR_KINDS.values())


def test_fold_label_refuses_a_path_outside_a_layer_directory(tmp_path: Path) -> None:
    root = _repo(tmp_path, "tiers/readonly.yml", f"{RETIRED_LAYER_DIRNAME}/operator.yml")

    assert layer_fold_label(root / "tiers" / "readonly.yml") == "tier"
    with pytest.raises(KeyError):
        layer_fold_label(root / "profile.yml")


_WINDOW_CLOSED_PROBE = textwrap.dedent(
    """
    import importlib.util, json, logging, sys
    from pathlib import Path

    # Load the layout leaf with its retired map emptied -- the one edit that
    # closes the deprecation window -- before anything that reads it.
    source, retired_map, layer = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    root = layer.parent.parent
    text = Path(source).read_text(encoding="utf-8")
    assert text.count(retired_map) == 1, retired_map
    spec = importlib.util.spec_from_loader("osprey.utils.workspace", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = source
    sys.modules[spec.name] = module
    exec(compile(text.replace(retired_map, "{}"), source, "exec"), module.__dict__)

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)

    from osprey.cli.profile_root import layer_files, resolve_profile_root

    resolved_root, kind = resolve_profile_root(layer)
    print(json.dumps({
        "layers": [str(path) for _, path in layer_files(root)],
        "kind": str(kind),
        "root": str(resolved_root),
        "warnings": [r.getMessage() for r in records if r.levelno >= logging.WARNING],
    }))
    """
)


def test_closing_the_window_is_deleting_the_retired_map_entry(tmp_path: Path) -> None:
    """With the retired map empty, the old folder is no layer directory at all."""
    root = tmp_path / "repo"
    (root / RETIRED_LAYER_DIRNAME).mkdir(parents=True)
    (root / RETIRED_LAYER_DIRNAME / "operator.yml").write_text("name: operator\n")
    (root / "profile.yml").write_text("name: repo\n")
    retired_map = f'{{"{RETIRED_LAYER_DIRNAME}": TIERS_DIRNAME}}'

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _WINDOW_CLOSED_PROBE,
            workspace.__file__,
            retired_map,
            str(root / RETIRED_LAYER_DIRNAME / "operator.yml"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout.strip().splitlines()[-1])

    assert seen["layers"] == []
    assert seen["kind"] == "root"
    assert seen["root"] == str((root / RETIRED_LAYER_DIRNAME).resolve())
    assert seen["warnings"] == []


def test_layer_files_lists_kind_and_path_for_every_layer(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        "tiers/readwrite.yml",
        "tiers/admin.yml",
        "apps/logbook.yml",
        *(f"{retired}/operator.yml" for retired in RETIRED_LAYER_DIRS),
    )

    assert layer_files(root) == [
        (ProfileKind.TIER, root / "tiers" / "admin.yml"),
        (ProfileKind.TIER, root / "tiers" / "readwrite.yml"),
        (ProfileKind.APP, root / "apps" / "logbook.yml"),
        *((ProfileKind.TIER, root / retired / "operator.yml") for retired in RETIRED_LAYER_DIRS),
    ]


def test_layer_files_skips_dotfiles_and_nested_trees(tmp_path: Path) -> None:
    root = _repo(tmp_path, "apps/logbook.yml", "apps/nested/deep.yml")
    (root / "apps" / ".DS_Store").write_text("")

    assert layer_files(root) == [(ProfileKind.APP, root / "apps" / "logbook.yml")]


def test_layer_files_has_no_suffix_filter(tmp_path: Path) -> None:
    root = _repo(tmp_path, "tiers/readonly.yaml", "tiers/notes.txt")

    assert [path.name for _, path in layer_files(root)] == ["notes.txt", "readonly.yaml"]


def test_layer_files_is_empty_without_layer_directories(tmp_path: Path) -> None:
    assert layer_files(_repo(tmp_path)) == []


# ── Build loop and hash fold agree ──────────────────────────────────────────


def _folded_layers(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[Path]:
    """The layer files the content-hash fold read, in fold order."""
    seen: list[Path] = []
    real = build_profile_merge._fold_source_tree

    def spy(digest, label, source, **kwargs):
        if Path(source).parent.name in LAYER_DIRNAMES:
            seen.append(source)
        return real(digest, label, source, **kwargs)

    monkeypatch.setattr(build_profile_merge, "_fold_source_tree", spy)
    assert compute_profile_hash(root / "profile.yml") is not None
    return seen


def test_build_loop_and_hash_fold_see_the_same_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(
        tmp_path,
        "tiers/readonly.yml",
        "tiers/readwrite.yml",
        "apps/logbook.yml",
        "apps/knowledge.yml",
        *(f"{retired}/operator.yml" for retired in RETIRED_LAYER_DIRS),
    )

    built = [path for _, path in _layer_files(root)]

    assert built == _folded_layers(monkeypatch, root)
    assert len(built) == 4 + len(RETIRED_LAYER_DIRS)


def test_build_loop_refuses_a_tier_and_a_render_of_one_name(tmp_path: Path) -> None:
    root = _repo(tmp_path, "tiers/readonly.yml", f"{RETIRED_LAYER_DIRNAME}/readonly.yml")

    with pytest.raises(BuildProfileError, match="share the name 'readonly'"):
        _layer_files(root)


def test_build_loop_refuses_a_symlinked_layer_from_another_repo(tmp_path: Path) -> None:
    other = _repo(tmp_path / "other", "apps/logbook.yml")
    root = _repo(tmp_path / "mine")
    (root / "apps").mkdir()
    (root / "apps" / "logbook.yml").symlink_to(other / "apps" / "logbook.yml")

    with pytest.raises(BuildProfileError, match="not a layer over this repo"):
        _layer_files(root)


# ── An edit to one layer moves the hash and the fingerprint ─────────────────


@pytest.mark.parametrize(
    "layer",
    [
        "apps/logbook.yml",
        "tiers/readonly.yml",
        *(f"{retired}/operator.yml" for retired in RETIRED_LAYER_DIRS),
    ],
)
def test_editing_one_layer_moves_hash_and_fingerprint(tmp_path: Path, layer: str) -> None:
    root = _repo(
        tmp_path,
        "apps/logbook.yml",
        "tiers/readonly.yml",
        *(f"{retired}/operator.yml" for retired in RETIRED_LAYER_DIRS),
    )
    profile = root / "profile.yml"

    before_hash = compute_profile_hash(profile)
    before = profile_fingerprint(profile)
    assert before is not None and before_hash == before.profile_hash

    (root / layer).write_text("name: edited\ndescription: changed\n")

    after = profile_fingerprint(profile)
    assert after is not None
    assert compute_profile_hash(profile) != before_hash
    assert after.profile_hash != before.profile_hash

    key = f"{MATERIAL_PREFIX}{layer}"
    assert key in before.key_digests
    assert after.key_digests[key] != before.key_digests[key]
    # Only the edited file's own digest moved among the layer digests.
    others = [
        k
        for k in before.key_digests
        if k.startswith(MATERIAL_PREFIX) and k != key and "/" in k.removeprefix(MATERIAL_PREFIX)
    ]
    for other in others:
        if other.removeprefix(MATERIAL_PREFIX).split("/", 1)[0] in LAYER_DIRNAMES:
            assert after.key_digests[other] == before.key_digests[other]


def test_fingerprint_digests_one_key_per_layer_file(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        "apps/logbook.yml",
        "tiers/readonly.yml",
        *(f"{retired}/operator.yml" for retired in RETIRED_LAYER_DIRS),
    )

    fingerprint = profile_fingerprint(root / "profile.yml")

    assert fingerprint is not None
    layer_keys = sorted(
        key.removeprefix(MATERIAL_PREFIX)
        for key in fingerprint.key_digests
        if key.removeprefix(MATERIAL_PREFIX).split("/", 1)[0] in LAYER_DIRNAMES
    )
    assert layer_keys == sorted(f"{path.parent.name}/{path.name}" for _, path in layer_files(root))


def test_adding_an_app_moves_the_hash(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    before = compute_profile_hash(root / "profile.yml")

    (root / "apps").mkdir()
    (root / "apps" / "logbook.yml").write_text("name: logbook\n")

    assert compute_profile_hash(root / "profile.yml") != before
