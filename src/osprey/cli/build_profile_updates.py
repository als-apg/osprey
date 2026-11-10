"""Preset updates: what a repo's presets have changed since the repo last took them.

``osprey init`` writes a preset out as ``profile.yml``, its logins as
``tiers/<n>.yml``, the provider catalog as ``providers.yml`` and the trigger
set as ``triggers.yml``. From then on those files are the operator's to edit.
To tell an edit from a change the preset made later, ``osprey init`` also saves
what the presets produced, as plain data, under ``.osprey/preset-base/``: the
*saved copy*.

Three documents answer the question for each covered file:

* the saved copy: what the preset produced when the repo last took it;
* today's output: what the installed preset produces now
  (:func:`~osprey.cli.build_profile_emit.materialized_profile`,
  :func:`~osprey.cli.build_profile_emit.emit_layer_delta_yaml`);
* the operator's file.

The delta between the first two is what the preset changed
(:func:`preset_delta`). The operator's file is never part of it. ``osprey init
--revalidate`` applies that delta where the operator left the preset's value
alone, then rewrites the saved copy to today's output. ``osprey validate`` and
``osprey build`` only say, in one line, that there is something to apply
(:func:`preset_update_note`).

``profile.yml`` and each tier file are keyed by the bundled source their own
``provenance:`` names, so a tier copied under a new name still pairs with the
login it came from. ``providers.yml`` and ``triggers.yml`` are keyed by path. A
copied application is code, not configuration: it is compared by the digest of
the bundled tree, and a change is only ever reported.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from osprey.profiles.providers import (
    PROVIDERS_FILENAME,
    compute_providers_hash,
    packaged_catalog_path,
)
from osprey.utils.workspace import (
    APPS_DIRNAME,
    PRESET_TRACKING_DIRNAME,
    PROFILE_FILENAME,
    TIERS_DIRNAME,
)
from osprey_connectors.errors import BuildProfileError

from .application_material import app_build_output, application_dirs
from .build_profile_document import _parse_profile_document
from .build_profile_merge import _fold_source_tree, compute_preset_hash
from .build_profile_resolve import _dotted_leaves
from .profile_conventions import PROFILE_TRIGGERS_FILENAME

#: Where the saved copy lives, relative to the repo root. The top-level
#: ``.osprey`` is what ``osprey init --force`` holds aside and restores.
PRESET_BASE_DIR = Path(PRESET_TRACKING_DIRNAME) / "preset-base"

#: The saved copy of the copied applications: bundled tree digest by name.
_APPS_BASE = "apps.yml"

_BASE_README = """\
What OSPREY's presets produced for this repo, as plain data, the last time the
repo took them (`osprey init`, or `osprey init --revalidate`). OSPREY compares
it with what the installed presets produce, so it applies only what a preset
changed and leaves your own edits alone. Keep it committed. Do not edit it by
hand: `osprey init --revalidate` rewrites it.
"""

#: Top-level keys a repo writes for itself rather than copies from its preset:
#: display and deployment names, the materialized data tree, a mounted machine
#: repository, the schema floor and the provenance stamp.
_MATERIALIZATION_KEYS: frozenset[str] = frozenset(
    {"name", "project_name", "data", "machine", "provenance", "requires_osprey_version"}
)

#: Top-level blocks whose lists are ordered: the lifecycle step lists.
_ORDERED_ROOTS: frozenset[str] = frozenset({"lifecycle"})


# ─────────────────────────────────────────────────────────────────────────────
# Applications
# ─────────────────────────────────────────────────────────────────────────────

#: The sidecar ``osprey init`` writes inside each application directory it
#: copies from a preset: the preset's name and hash and the digest of the tree
#: as copied. A file of its own because ``app.yml`` refuses unknown keys.
APP_PROVENANCE_FILENAME = ".provenance"


def _app_source_skip(app_dir: Path) -> Callable[[str], bool]:
    """Which files under *app_dir* are not the application's source.

    The files the build writes into it
    (:func:`~osprey.cli.application_material.app_build_output`), the byte-code a
    source checkout leaves beside its Python — the copy drops it too — and the
    sidecar itself.
    """
    build_output = app_build_output(app_dir)

    def skip(rel: str) -> bool:
        return (
            build_output(rel)
            or rel == APP_PROVENANCE_FILENAME
            or "__pycache__" in rel.split("/")
            or rel.endswith((".pyc", ".pyo"))
        )

    return skip


def app_tree_digest(app_dir: Path) -> str:
    """The digest of an application directory's source, as its sidecar records it."""
    digest = hashlib.sha256()
    _fold_source_tree(digest, "app", app_dir, skip=_app_source_skip(app_dir))
    return f"sha256:{digest.hexdigest()}"


def app_provenance_text(app_dir: Path, preset: str, preset_hash: str | None) -> str:
    """The ``.provenance`` sidecar for *app_dir*, copied from *preset*."""
    body = yaml.safe_dump(
        {"preset": preset, "preset_hash": preset_hash, "tree_digest": app_tree_digest(app_dir)},
        sort_keys=False,
    )
    return (
        "# Written by `osprey init`: the preset this application was copied from, and\n"
        "# the digest of the files as copied. `osprey init --revalidate` says when the\n"
        "# preset's own copy of the application has changed since.\n"
        f"{body}"
    )


def _bundled_app_digests(root: Path) -> dict[str, tuple[str, str]]:
    """Each copied application: ``name → (bundled digest today, digest its sidecar records)``.

    Only an application whose ``.provenance`` sidecar names a bundled preset
    that still ships it is listed. An application with no sidecar is the
    operator's own.
    """
    from .build_profile_presets import _load_preset_raw, _normalize_preset_name

    digests: dict[str, tuple[str, str]] = {}
    for app_dir in application_dirs(root):
        sidecar = app_dir / APP_PROVENANCE_FILENAME
        if not sidecar.is_file():
            continue
        recorded = _parse_profile_document(
            sidecar.read_text(encoding="utf-8"),
            f"{app_dir.relative_to(root).as_posix()}/{APP_PROVENANCE_FILENAME}",
        )
        if not isinstance(recorded, Mapping) or not recorded.get("preset"):
            continue
        try:
            preset_path = _load_preset_raw(_normalize_preset_name(str(recorded["preset"])))[1]
        except BuildProfileError:
            continue
        bundled = preset_path.parent / APPS_DIRNAME / app_dir.name
        if bundled.is_dir():
            digests[app_dir.name] = (
                app_tree_digest(bundled),
                str(recorded.get("tree_digest") or ""),
            )
    return digests


# ─────────────────────────────────────────────────────────────────────────────
# The covered files, and today's preset output for each
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CoveredFile:
    """One repo file a preset wrote, paired with what that preset produces today.

    Attributes:
        rel: The repo-relative file.
        key: Where its saved copy lives under :data:`PRESET_BASE_DIR`.
        today: Today's preset output for it, without the keys a repo writes
            for itself.
        recorded_hash: The hash the repo recorded for the file's source.
        installed_hash: The installed source's hash.
    """

    rel: str
    key: str
    today: dict[str, Any]
    recorded_hash: str
    installed_hash: str


def _strip(document: Any) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        return {}
    return {str(k): v for k, v in document.items() if k not in _MATERIALIZATION_KEYS}


def _provenance(document: Any) -> Mapping[str, Any]:
    block = document.get("provenance") if isinstance(document, Mapping) else None
    return block if isinstance(block, Mapping) else {}


def _read(root: Path, rel: str, texts: Mapping[str, str]) -> Any:
    if rel in texts:
        return _parse_profile_document(texts[rel], rel)
    if (root / rel).is_file():
        return _parse_profile_document((root / rel).read_text(encoding="utf-8"), rel)
    return None


def _triggers_source(preset: str) -> Path | None:
    """The bundled trigger file *preset* names under ``dispatch.triggers``, if any."""
    from .build_profile_merge import _resolve_extends
    from .build_profile_presets import _load_preset_raw, resolve_triggers_path

    raw, anchor = _load_preset_raw(preset)
    resolved = _resolve_extends(dict(raw), anchor)
    dispatch = resolved.get("dispatch")
    name = dispatch.get("triggers") if isinstance(dispatch, Mapping) else None
    if not isinstance(name, str) or not name:
        return None
    source = resolve_triggers_path(anchor.parent, name)
    return source.path if source is not None else None


def covered_files(
    root: Path,
    *,
    preset: str | None = None,
    texts: Mapping[str, str] | None = None,
    gone: Iterable[str] = (),
) -> list[CoveredFile]:
    """Every file in *root* a bundled preset wrote, with today's output for it.

    Args:
        root: The repo root.
        preset: The preset ``profile.yml`` pairs with; default: the one its
            ``provenance:`` names.
        texts: Repo-relative files to read from here rather than from disk:
            the rewrite ``osprey init --revalidate`` has planned in memory.
        gone: Repo-relative files that rewrite deletes.

    Returns:
        The covered files, ``profile.yml`` first. A file whose source this
        OSPREY does not bundle is left out.

    Raises:
        BuildProfileError: A file does not parse, or a preset will not emit.
    """
    from .build_profile_emit import emit_layer_delta_yaml, materialized_profile
    from .build_profile_presets import _normalize_preset_name, list_presets

    texts = texts or {}
    profile = _read(root, PROFILE_FILENAME, texts)
    if not isinstance(profile, Mapping):
        return []
    stamp = _provenance(profile)
    recorded = _normalize_preset_name(str(stamp.get("preset") or ""))
    name = _normalize_preset_name(preset) if preset else recorded
    bundled = bool(name) and name in list_presets()
    profile_name = str(profile.get("name") or "")
    covered: list[CoveredFile] = []
    if bundled:
        today = materialized_profile(
            name, project_name=str(profile.get("project_name") or ""), profile_name=profile_name
        )
        covered.append(
            CoveredFile(
                PROFILE_FILENAME,
                f"presets/{name}.yml",
                _strip(today),
                str(stamp.get("preset_hash") or "") if name == recorded else "",
                compute_preset_hash(name) or "",
            )
        )
    on_disk = (p.relative_to(root).as_posix() for p in sorted(root.glob(f"{TIERS_DIRNAME}/*.yml")))
    tiers = {
        rel
        for rel in (*texts, *on_disk)
        if rel.startswith(f"{TIERS_DIRNAME}/") and rel.endswith(".yml") and rel.count("/") == 1
    }
    for rel in sorted(tiers - set(gone)):
        document = _read(root, rel, texts)
        source = _provenance(document).get("preset")
        if not source:
            continue
        stem = _normalize_preset_name(str(source)).rsplit("/", 1)[-1]
        installed = compute_preset_hash(f"{TIERS_DIRNAME}/{stem}")
        if installed is None:
            continue
        emitted = emit_layer_delta_yaml(
            preset_name=f"{TIERS_DIRNAME}/{stem}", profile_name=profile_name, profile_filename=rel
        )
        covered.append(
            CoveredFile(
                rel,
                f"{TIERS_DIRNAME}/{stem}.yml",
                _strip(_parse_profile_document(emitted, rel)),
                str(_provenance(document).get("preset_hash") or ""),
                installed,
            )
        )
    if PROVIDERS_FILENAME in texts or (root / PROVIDERS_FILENAME).is_file():
        packaged = packaged_catalog_path()
        catalog = _parse_profile_document(packaged.read_text(encoding="utf-8"), "packaged")
        covered.append(
            CoveredFile(
                PROVIDERS_FILENAME,
                PROVIDERS_FILENAME,
                dict(catalog) if isinstance(catalog, Mapping) else {},
                str(stamp.get("providers_hash") or ""),
                compute_providers_hash(packaged),
            )
        )
    triggers = _triggers_source(name) if bundled else None
    if triggers is not None and (
        PROFILE_TRIGGERS_FILENAME in texts or (root / PROFILE_TRIGGERS_FILENAME).is_file()
    ):
        document = _parse_profile_document(triggers.read_text(encoding="utf-8"), triggers.name)
        # Copied verbatim and never stamped, so no hash says it moved.
        covered.append(
            CoveredFile(
                PROFILE_TRIGGERS_FILENAME,
                PROFILE_TRIGGERS_FILENAME,
                dict(document) if isinstance(document, Mapping) else {},
                "",
                "",
            )
        )
    return covered


# ─────────────────────────────────────────────────────────────────────────────
# The saved copy
# ─────────────────────────────────────────────────────────────────────────────


def has_saved_copy(root: Path) -> bool:
    """Whether *root* carries a saved copy at all; a repo made before it existed does not."""
    return (root / PRESET_BASE_DIR).is_dir()


def read_saved_copy(root: Path) -> dict[str, Any]:
    """The saved copy, by its path under :data:`PRESET_BASE_DIR`; empty when there is none."""
    base = root / PRESET_BASE_DIR
    saved: dict[str, Any] = {}
    if not base.is_dir():
        return saved
    for path in sorted(base.rglob("*.yml")):
        key = path.relative_to(base).as_posix()
        saved[key] = _parse_profile_document(path.read_text(encoding="utf-8"), key)
    return saved


def saved_copy_texts(root: Path, covered: Iterable[CoveredFile]) -> dict[str, str]:
    """The saved copy recording *covered* and *root*'s copied applications, as files.

    Returns:
        Repo-relative path → text. The whole saved copy: a file it does not
        name is not part of it.
    """
    base = PRESET_BASE_DIR.as_posix()
    texts = {f"{base}/README": _BASE_README}
    for item in covered:
        texts[f"{base}/{item.key}"] = _dump(item.today)
    apps = {name: digest for name, (digest, _copied) in _bundled_app_digests(root).items()}
    if apps:
        texts[f"{base}/{_APPS_BASE}"] = _dump(dict(sorted(apps.items())))
    return texts


def write_saved_copy(root: Path, texts: Mapping[str, str]) -> None:
    """Replace *root*'s saved copy with *texts* (:func:`saved_copy_texts`)."""
    shutil.rmtree(root / PRESET_BASE_DIR, ignore_errors=True)
    for rel, text in texts.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def save_preset_base(root: Path) -> None:
    """Write the saved copy for the repo at *root* as it stands; ``osprey init`` calls it."""
    write_saved_copy(root, saved_copy_texts(root, covered_files(root)))


def _dump(document: Any) -> str:
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=4096)


def starting_point(item: CoveredFile, saved: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """The document *item*'s delta is taken from, and whether it is a one-time fill.

    The saved copy when there is one. A file with none (a repo made before the
    saved copy existed, a tier added since, a preset just switched to) starts
    from today's output while its recorded hash still matches the installed
    source: there is nothing to apply. Otherwise nothing records what the
    preset said when the file was written, so the file is filled once the way
    ``osprey init --revalidate`` always filled it: every ``config:`` leaf
    ``profile.yml`` lacks, and every provider entry and key ``providers.yml``
    lacks.

    Returns:
        ``(document, filled)``: *filled* is true for that one-time fill.
    """
    held = saved.get(item.key)
    if isinstance(held, Mapping):
        return dict(held), False
    if item.recorded_hash == item.installed_hash or item.rel not in (
        PROFILE_FILENAME,
        PROVIDERS_FILENAME,
    ):
        return dict(item.today), False
    if item.rel == PROVIDERS_FILENAME:
        return {}, True
    return {key: value for key, value in item.today.items() if key != "config"}, True


# ─────────────────────────────────────────────────────────────────────────────
# The normalised comparison
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Entry:
    """A path step into a list of mappings: the member whose ``name`` is *name*."""

    name: str


@dataclass(frozen=True)
class Member:
    """A path step into a list of strings: the member *value*."""

    value: str


#: The path step holding an ordered list's member order.
ORDER = "<order>"

#: A flattened path: mapping keys, :class:`Entry` and :class:`Member` steps.
FlatPath = tuple[Any, ...]


def flatten(document: Mapping[str, Any]) -> dict[FlatPath, Any]:
    """*document* as ``path → leaf value``, the one form every comparison uses.

    * ``config:`` keys are flattened to dotted leaves, so a nested mapping, a
      dotted key and any mix of the two read as one key.
    * A list of strings is a set of :class:`Member` steps. A list of mappings
      that all carry a distinct ``name`` is a mapping by :class:`Entry`. Order
      is ignored for both, except below ``lifecycle:``, where the member order
      is a leaf of its own (:data:`ORDER`).
    * Any other list is one value.
    """
    out: dict[FlatPath, Any] = {}
    for key, value in document.items():
        if key == "config" and isinstance(value, Mapping) and value:
            for path, leaf in _dotted_leaves(dict(value)):
                _flatten(leaf, ("config", ".".join(path)), out, ordered=False)
        else:
            _flatten(value, (str(key),), out, ordered=key in _ORDERED_ROOTS)
    return out


def _named(value: list[Any]) -> bool:
    names = [item.get("name") for item in value if isinstance(item, Mapping)]
    return (
        len(names) == len(value)
        and all(isinstance(name, str) for name in names)
        and len(set(names)) == len(names)
    )


def _flatten(value: Any, path: FlatPath, out: dict[FlatPath, Any], *, ordered: bool) -> None:
    if isinstance(value, Mapping):
        if not value:
            out[path] = {}
        for key, item in value.items():
            _flatten(item, (*path, str(key)), out, ordered=ordered)
    elif isinstance(value, list) and value and all(isinstance(i, str) for i in value):
        for item in value:
            out[(*path, Member(item))] = True
        if ordered:
            out[(*path, ORDER)] = tuple(value)
    elif isinstance(value, list) and value and _named(value):
        for item in value:
            entry = (*path, Entry(item["name"]))
            rest = {k: v for k, v in item.items() if k != "name"}
            if rest:
                _flatten(rest, entry, out, ordered=ordered)
            else:
                out[entry] = {}
        if ordered:
            out[(*path, ORDER)] = tuple(item["name"] for item in value)
    else:
        out[path] = _plain(value)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def subtree(flat: Mapping[FlatPath, Any], path: FlatPath) -> dict[FlatPath, Any]:
    """The leaves of *flat* at or below *path*, relative to it."""
    size = len(path)
    return {p[size:]: v for p, v in flat.items() if p[:size] == path}


def _unshadowed(flat: dict[FlatPath, Any], other: Mapping[FlatPath, Any]) -> dict[FlatPath, Any]:
    """*flat* without an empty-container leaf that *other* holds members under."""
    under = prefixes(p[:-1] for p in other if len(p) > 1)
    return {
        path: value for path, value in flat.items() if not (value in ({}, []) and path in under)
    }


def prefixes(flat: Iterable[FlatPath]) -> set[FlatPath]:
    """Every node *flat*'s leaves pass through, the leaves included."""
    return {p[:cut] for p in flat for cut in range(1, len(p) + 1)}


@dataclass(frozen=True)
class Change:
    """One thing a preset changed: the node at *path*, from *old* to *new*.

    Both sides are :func:`subtree` forms, and an empty one is absence: an
    added node has an empty *old*, a removed one an empty *new*.
    """

    path: FlatPath
    old: dict[FlatPath, Any]
    new: dict[FlatPath, Any]

    @property
    def kind(self) -> str:
        """``added``, ``removed`` or ``changed``."""
        if not self.old:
            return "added"
        if not self.new:
            return "removed"
        return "changed"


def render_path(path: FlatPath) -> str:
    """A flattened path as an operator names it: ``config.web.theme``, ``web_panels: ariel``."""
    text = ""
    for step in path:
        if isinstance(step, Entry):
            text += f"[{step.name}]"
        elif isinstance(step, Member):
            text += f": {step.value}"
        elif step == ORDER:
            text += " (order)"
        else:
            text += f".{step}" if text else str(step)
    return text


def preset_delta(saved: Mapping[str, Any], today: Mapping[str, Any]) -> list[Change]:
    """What the preset changed from *saved* to *today*, node by node.

    A node one side lacks entirely is one change at its shallowest absent step:
    a new panel is one added member, a new step one added entry. A leaf both
    sides carry with different values is one change at that leaf.
    """
    before, after = flatten(saved), flatten(today)
    if before == after:
        return []
    # An empty list or mapping on one side and members on the other is the
    # members being added or removed, not a value replaced.
    before, after = _unshadowed(before, after), _unshadowed(after, before)
    had, has = prefixes(before), prefixes(after)
    nodes: list[FlatPath] = []
    for path in (*after, *(p for p in before if p not in after)):
        node = next(
            (
                path[:cut]
                # `config:` is not a node of its own: its keys are the leaves.
                for cut in range(2 if path[0] == "config" else 1, len(path) + 1)
                if path[:cut] not in had or path[:cut] not in has
            ),
            path,
        )
        if node not in nodes:
            nodes.append(node)
    changes = [
        Change(node, subtree(before, node), subtree(after, node))
        for node in nodes
        if subtree(before, node) != subtree(after, node)
    ]
    paths = {change.path for change in changes}
    return [
        change
        for change in changes
        if not any(change.path[:cut] in paths for cut in range(1, len(change.path)))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# What validate and build say
# ─────────────────────────────────────────────────────────────────────────────


def changed_apps(root: Path, saved: Mapping[str, Any]) -> list[str]:
    """The copied applications whose bundled tree changed since the repo last took it.

    Compared with the digest the saved copy records, or, when it records none,
    the one the application's sidecar recorded at copy time. The operator's
    edits to the copy are never part of it.
    """
    held = saved.get(_APPS_BASE)
    recorded = held if isinstance(held, Mapping) else {}
    return [
        name
        for name, (bundled, copied) in sorted(_bundled_app_digests(root).items())
        if bundled != recorded.get(name, copied)
    ]


def preset_update_note(profile_file: Path) -> str | None:
    """The one line ``osprey validate`` and ``osprey build`` print about preset updates.

    Never a refusal: the profile is the operator's to edit, and applying what a
    preset changed is ``osprey init --revalidate``'s job. Silent for a profile
    with no ``provenance:``, which took nothing from a preset.

    Raises:
        BuildProfileError: A covered file does not parse, or a preset will not emit.
    """
    from .build_profile_presets import _normalize_preset_name, list_presets
    from .profile_root import resolve_profile_root

    root, _kind = resolve_profile_root(profile_file)
    preset = _provenance(_read(root, PROFILE_FILENAME, {})).get("preset")
    if not preset:
        return None
    preset = _normalize_preset_name(str(preset))
    if preset not in list_presets():
        return (
            f"provenance names preset {preset!r}, which this OSPREY does not bundle: "
            f"the profile cannot be compared with it"
        )
    if not has_saved_copy(root):
        return "This repo predates preset tracking: run `osprey init --revalidate` to start it."
    saved = read_saved_copy(root)
    moved = changed_apps(root, saved) or any(
        flatten(starting_point(item, saved)[0]) != flatten(item.today)
        for item in covered_files(root)
    )
    if not moved:
        return None
    return (
        f"The {preset} preset has changed since this repo last took it: run "
        f"`osprey init --revalidate` to apply the changes."
    )
