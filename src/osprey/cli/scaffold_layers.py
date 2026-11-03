"""Materializing a preset's tier or app files into an existing repo.

``osprey init`` writes ``tiers/<name>.yml`` and ``apps/<name>.yml`` for every
tier and app the new repo's roster names. This module is the way back in for a
repo that already exists: one that lost a file, one whose operator wants the
installed definition again, or one created from a preset without a web-terminal
stack that adopts another preset's tiers or apps later.

It runs the same pipeline init runs (``profile_cmd._layer_profile_texts``):
the preset's roster decides which tiers and apps there are, each is emitted as
an explicit file with its comments and a provenance block, and the roster's
privilege rules are checked before anything is written. Only the writing
differs, because here the files may already be there: a file whose content
matches is left untouched, and a file that differs is the operator's and is
replaced only under ``force``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from osprey.utils.workspace import APPS_DIRNAME, TIERS_DIRNAME

#: The two kinds of layer file this module writes, named by their folder.
LayerKind = Literal["tiers", "apps"]

#: What happened to one file: written fresh, replaced under ``force``, already
#: identical, or left alone because it differs and ``force`` was not passed.
LayerAction = Literal["created", "replaced", "unchanged", "refused"]


@dataclass(frozen=True)
class LayerResult:
    """The outcome for one tier or app file.

    Attributes:
        relative: The repo-relative path (``tiers/readonly.yml``).
        action: See :data:`LayerAction`.
    """

    relative: str
    action: LayerAction


def plan_layer_files(
    repo_root: Path,
    repo_name: str,
    preset_name: str,
    kind: LayerKind,
    *,
    force: bool = False,
) -> tuple[list[LayerResult], dict[str, str]]:
    """Decide, without writing, what each tier or app file of ``preset_name`` needs.

    Args:
        repo_root: The deployment repo the files land in.
        repo_name: The repo's display name, which titles each tier
            (``"My Assistant (readonly)"``) — the profile's ``name``.
        preset_name: The bundled deployment preset whose roster names the
            tiers and apps. It need not be the repo's own preset.
        kind: ``"tiers"`` or ``"apps"``.
        force: Replace a file that exists with different content.

    Returns:
        ``(results, texts)``: one :class:`LayerResult` per file the preset's
        roster names, tiers or apps sorted by name, and the emitted text of
        every file whose action is ``created`` or ``replaced``.

    Raises:
        click.UsageError: If the preset's roster names a tier or app that cannot
            be materialized, or would hand deployment editing to a terminal that
            must not have it — init's own refusals, word for word.
        BuildProfileError: If the preset does not resolve.
    """
    # Function-local: these pull the whole build-profile chain in, and the CLI's
    # lazy-import budget is measured on what importing a command group costs.
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.cli.profile_cmd import _layer_profile_texts, _parsed_layer_deltas

    if kind not in (TIERS_DIRNAME, APPS_DIRNAME):
        raise ValueError(f"kind must be {TIERS_DIRNAME!r} or {APPS_DIRNAME!r}, not {kind!r}")

    preset, _preset_dir = resolve_build_profile(None, preset_name)
    # The repo root is where the reader stands, so headers name `tiers/<n>.yml`
    # with no directory prefix — the same text init writes into a repo root.
    all_texts = _layer_profile_texts(preset, repo_name, "")
    # A bare entry the repo ships a file for is written as that file's path,
    # as init writes it, so a file init made reads as unchanged here.
    from osprey.cli.init_revalidate import shipped_shadow_texts

    texts = shipped_shadow_texts(
        repo_root,
        {r: text for r, text in all_texts.items() if r.startswith(f"{kind}/")},
    )
    # The parse is the check that the emission left valid YAML behind; it
    # covers every file before the first write, as at init.
    _parsed_layer_deltas(texts)

    results: list[LayerResult] = []
    to_write: dict[str, str] = {}
    for relative, text in texts.items():
        destination = repo_root / relative
        if not destination.exists():
            action: LayerAction = "created"
        elif destination.read_text(encoding="utf-8") == text:
            action = "unchanged"
        elif force:
            action = "replaced"
        else:
            action = "refused"
        results.append(LayerResult(relative=relative, action=action))
        if action in ("created", "replaced"):
            to_write[relative] = text
    return results, to_write


def write_layer_files(repo_root: Path, texts: dict[str, str]) -> None:
    """Write the texts :func:`plan_layer_files` returned, creating the folder lazily.

    A run that writes nothing leaves no empty ``tiers/`` or ``apps/`` behind.
    """
    for relative, text in texts.items():
        destination = repo_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")


def unnamed_by_roster(repo_profile: Any, results: list[LayerResult], kind: LayerKind) -> list[str]:
    """The files among ``results`` that this repo's own roster does not name.

    A file written from another preset (``--from``) sits unread until the
    repo's roster names it: a tier by a person, ``default_tier``, an
    authorization role or ``dispatch.tier``, an app by ``shared:``. The same
    reading init uses to decide which files a repo gets.

    Args:
        repo_profile: The repo's resolved build profile.
        results: What :func:`plan_layer_files` returned.
        kind: ``"tiers"`` or ``"apps"``.

    Returns:
        The repo-relative paths nothing in the roster names, in ``results``' order.
    """
    from osprey.cli.profile_cmd import _roster_layers

    dispatch = getattr(repo_profile, "dispatch", None)
    tiers, apps = _roster_layers(repo_profile.config, dispatch.tier if dispatch else None)
    named = set(tiers if kind == TIERS_DIRNAME else apps)
    return [result.relative for result in results if Path(result.relative).stem not in named]
