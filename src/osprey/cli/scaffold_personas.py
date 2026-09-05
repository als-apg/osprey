"""Materializing a repo's persona delta files from a preset's catalog.

A deployment that stands up one web terminal per persona keeps one file per
persona in ``personas/``, beside its ``profile.yml``. Each file is a pure DELTA:
the persona preset's own layer, with no ``extends:``, merged over the host
profile implicitly because of where it sits. ``osprey init`` emits those files
when it creates the repo from a preset that already carries the catalog.

This module is the other way in — a repo that was created from a *different*
preset (a plain ``hello-world`` deployment, say) and later adopts a persona
stack by copying the ``modules.web_terminals`` block in. The catalog then names
personas the host preset never had, so the deltas have to be emitted from the
preset that owns them rather than from the repo's own lineage.

That is the one thing separating this from init's path, and it is why the
emission here calls :func:`~.build_profile_emit.emit_persona_delta_yaml`
directly instead of going through init's
``profile_cmd._persona_profile_texts``. That helper answers a question this
module cannot: whether each persona preset is a delta over *the host's own*
preset, and whether both render the same data bundle. Both hold by construction
at init time and neither holds here — the host is ``hello-world`` and the
personas come from ``control-assistant`` — so running its checks would refuse
exactly the case this module exists for.

What is kept is the check that actually protects the repo: every emitted text is
parsed before a single file is written, so a preset whose ``extends:`` surgery
left broken YAML behind costs nothing.

Writing the files is only half of it. :func:`repoint_persona_catalog` finishes
the adoption by pointing the pasted-in catalog at those files, so the entries
name what the repo now holds rather than the presets they were copied from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from osprey.cli.profile_root import PERSONA_DIRNAME
from osprey.errors import BuildProfileError


@dataclass(frozen=True)
class PersonaReport:
    """What one persona emission did, in catalog order.

    Attributes:
        written: Repo-relative paths of the files this run wrote — new files,
            plus the ones ``force`` replaced.
        skipped: Repo-relative paths that already existed and were left exactly
            as they are. Empty when ``force`` was passed.
        names: Every persona name in the preset's catalog, whether its file was
            written or skipped. The caller reporting "5 of 5 present" needs the
            catalog's own size, which neither of the other two lists carries.
    """

    written: list[str]
    skipped: list[str]
    names: list[str]


def emit_persona_files(
    repo_root: Path,
    repo_name: str,
    preset_name: str,
    *,
    force: bool = False,
) -> PersonaReport:
    """Write ``personas/<name>.yml`` for every persona ``preset_name`` catalogs.

    Nothing is written until every delta has been emitted and parsed, so a
    catalog with one bad entry leaves the repo untouched rather than half
    populated.

    Args:
        repo_root: The deployment repo the files land in. ``personas/`` is
            created under it, and only when there is something to write.
        repo_name: The repo's display name, which titles each emitted persona
            (``"ALS Assistant (readonly)"``). The host profile's ``name``, not
            the directory.
        preset_name: The bundled preset whose ``modules.web_terminals.personas``
            catalog is the source. It need not be the repo's own preset — that
            is the case this module is for.
        force: Rewrite a persona file that already exists. Without it an
            existing file is left alone and reported as skipped, because it is
            the operator's to edit once it is in the repo.

    Returns:
        A :class:`PersonaReport` whose three lists are all in catalog order.
        A preset that catalogs no personas yields a report with three empty
        lists; whether that is an error is the caller's question, since a caller
        may be probing rather than asking.

    Raises:
        BuildProfileError: If the preset does not resolve, if a catalog entry
            names no usable persona preset or a name that cannot be a file name,
            or if an emitted delta is not valid YAML. Every message names the
            persona it came from.
    """
    # Function-local: this pulls in the whole build-profile chain and the
    # profile command module behind it, and the CLI's lazy-import budget is
    # measured on what importing a command group costs.
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.cli.build_profile_emit import emit_persona_delta_yaml, persona_catalog
    from osprey.cli.profile_cmd import _parsed_persona_deltas

    preset, _preset_dir = resolve_build_profile(None, preset_name)
    catalog = persona_catalog(preset.config)

    texts: dict[str, str] = {}
    for name, entry in catalog.items():
        _reject_unusable_name(name)
        persona_preset = entry.get("build_profile")
        if not isinstance(persona_preset, str) or not persona_preset:
            raise BuildProfileError(
                f"Persona {name!r} in preset {preset_name!r} declares no build_profile, "
                f"so there is no preset to emit its delta from."
            )
        try:
            texts[name] = emit_persona_delta_yaml(
                preset_name=persona_preset,
                profile_name=f"{repo_name} ({name})",
                profile_filename=f"{PERSONA_DIRNAME}/{name}.yml",
            )
        except BuildProfileError as exc:
            # The emission names the persona's own preset; only the caller here
            # knows which catalog entry asked for it.
            raise BuildProfileError(f"Persona {name!r}: {exc}") from exc

    # The only proof the ``extends:`` line surgery left valid YAML behind, and
    # it covers every delta before the first write.
    _parsed_persona_deltas(texts)

    persona_dir = repo_root / PERSONA_DIRNAME
    written: list[str] = []
    skipped: list[str] = []
    for name, text in texts.items():
        destination = persona_dir / f"{name}.yml"
        relative = f"{PERSONA_DIRNAME}/{name}.yml"
        if destination.exists() and not force:
            skipped.append(relative)
            continue
        # Created lazily: a run that writes nothing leaves no empty directory
        # behind for the next reader to wonder about.
        persona_dir.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        written.append(relative)

    return PersonaReport(written=written, skipped=skipped, names=list(catalog))


def repoint_persona_catalog(repo_root: Path, names: Sequence[str]) -> int:
    """Point the repo's persona catalog at the sibling files just emitted.

    The other half of the adoption. :func:`emit_persona_files` puts the deltas
    in ``personas/``; until the catalog names them, every entry still resolves
    to the bundled preset it was copied from, so a build would render the
    preset's own data tree instead of the facility's.

    The rewrite goes through the same channel ``osprey set`` writes by, so
    ``profile.yml`` keeps its comments and is replaced atomically. Each key is
    written at the DEEPEST spelling — see
    :func:`~.build_profile_emit.persona_catalog_layer` for why that is the only
    spelling that survives beside the module subtree the operator pasted in.

    Args:
        repo_root: The deployment repo whose ``profile.yml`` is edited. Its
            directory name keys each persona's ``project``/``project_path``,
            which is why a shipped preset cannot spell them.
        names: Personas to repoint — ordinarily ``PersonaReport.names`` from the
            emission that just ran. A name the repo's catalog does not carry is
            skipped silently: it names no entry to rewrite, and whether that is
            worth reporting is the command layer's question.

    Returns:
        How many catalog entries were rewritten. Entries already pointing at
        their sibling file are left alone, so a second call returns ``0`` and
        does not touch the file at all.

    Raises:
        BuildProfileError: If the repo has no readable ``profile.yml``, or a
            catalog name cannot be a file name under ``personas/``.
    """
    # Function-local for the same reason as the emission above: this reaches the
    # whole build-profile chain, and the CLI's lazy-import budget is measured on
    # what importing a command group costs.
    from osprey.cli.build_profile import resolve_build_profile, write_back_cli_overrides
    from osprey.cli.build_profile_emit import persona_catalog, persona_catalog_layer
    from osprey.cli.build_profile_resolve import PROFILE_FILENAME

    profile_path = repo_root / PROFILE_FILENAME
    profile, _profile_dir = resolve_build_profile(profile_path, None)
    catalog = persona_catalog(profile.config)

    stale: list[str] = []
    for name in names:
        entry = catalog.get(name)
        if entry is None:
            continue
        _reject_unusable_name(name)
        if entry.get("build_profile") != f"{PERSONA_DIRNAME}/{name}.yml":
            stale.append(name)
    if not stale:
        return 0

    layer = persona_catalog_layer(stale, repo_name=repo_root.name)
    # `write_back_cli_overrides` takes the pairs an operator would have typed,
    # and its own flattening turns `config.a.b.c=v` back into the single dotted
    # `config:` key the layer spells — so the fragment reaches the file exactly
    # as `persona_catalog_layer` wrote it, with no second spelling to keep in
    # step. Every value here is a repo-relative path built from a name already
    # checked as a plain file name, so none of them needs quoting.
    set_pairs = tuple(f"config.{key}={value}" for key, value in layer["config"].items())
    write_back_cli_overrides(profile_path, set_pairs=set_pairs)
    return len(stale)


def _reject_unusable_name(name: str) -> None:
    """Refuse a persona name that cannot become a file name under ``personas/``.

    ``..`` needs naming explicitly: ``Path("..").name`` is ``".."``, not the
    empty string, so a plain-name check alone would let it through and the
    emission would write a ``personas/..yml`` nobody meant.

    Raises:
        BuildProfileError: If the name is empty, is a relative-path token, or
            carries a path separator.
    """
    if not name or name in (".", "..") or Path(name).name != name:
        raise BuildProfileError(
            f"Persona name {name!r} becomes a file name under {PERSONA_DIRNAME}/, "
            f"so it must be a plain name — no path separators, and not empty."
        )
