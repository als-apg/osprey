"""Scaffold Gallery Service — bridges BuildArtifactCatalog + TemplateManager for the web UI.

Stateless service class instantiated per-request with a project directory.
Provides list/get/diff/claim/save/unclaim operations that the frontend
gallery consumes via the ``/api/scaffold`` route family.

Ownership is not written to ``config.yml`` from here as a matter of course.
Where a claim durably belongs depends on what this process can reach —
the profile, a per-user volume, or neither — and
:mod:`osprey.interfaces.web_terminal.ownership` answers that once. This module
routes its writes accordingly and reads the union of the surfaces.
"""

from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import yaml

from osprey.cli.templates.manager import TemplateManager
from osprey.interfaces.web_terminal.ownership import (
    OwnershipMode,
    OwnershipStore,
    generated_project_paths,
    is_pristine_from_image,
    merge_user_owned,
    profile_artifacts,
    refuse_generated_path,
    refuse_unstorable_path,
    rehydrate,
    resolve_ownership,
)
from osprey.services.build_artifacts.catalog import BuildArtifact, BuildArtifactCatalog
from osprey.services.build_artifacts.ownership import (
    get_user_owned,
    update_config_add_user_owned,
    update_config_remove_user_owned,
    update_manifest_add_user_owned,
    update_manifest_remove_user_owned,
)
from osprey.utils.config import resolve_env_vars

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, annotation only
    from osprey.cli.scaffold_cmd import ClaimTarget

logger = logging.getLogger(__name__)

#: The file that carries a directory-shaped artifact's body. Among the
#: categories the gallery can create, only ``skills`` is a directory, and a
#: skill directory *is* its ``SKILL.md`` as far as Claude Code is concerned —
#: without one there is nothing to load.
DIRECTORY_ENTRY_FILE = "SKILL.md"


def restore_scaffold_bodies(project_dir: Path) -> list[str]:
    """Put volume-owned artifact bodies back into the project tree.

    Called once at web-terminal startup, which is the only moment that works.
    The agent runtime reads the image-baked tree and knows nothing about the
    volume, so a container recreated with a fresh tree would run the framework's
    original artifacts while the gallery displayed the operator's edited ones —
    a divergence with nothing to report it. Doing this on first gallery open
    instead would leave that window open for as long as nobody opens the
    gallery, which is most of the time.

    A no-op outside a deployed container, and a no-op once tree and store agree.

    Returns:
        Canonical names whose body was restored.
    """
    ownership = resolve_ownership(project_dir)
    if ownership.store is None:
        return []

    service = ScaffoldGalleryService(project_dir)
    restored = rehydrate(
        project_dir, ownership.store, ownership.store.read(), service._framework_or_none
    )
    if restored:
        logger.info(
            "Restored %d user-owned artifact(s) from the claude-config volume: %s",
            len(restored),
            ", ".join(sorted(restored)),
        )
    return restored


# Language inference from file extension
_EXT_LANG = {
    ".md": "markdown",
    ".py": "python",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
}

# Front-matter extraction patterns
_FM_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
_PY_FM_RE = re.compile(r'^(?:#!.*\n)?"""\n---\n(.*?)\n---', re.DOTALL)

# Subdirectories under .claude/ to scan for artifact files
_CLAUDE_SUBDIRS = ("agents", "commands", "hooks", "output-styles", "rules", "skills")

# Config-tier files: (relative_path, relative_path) — checked first in scan order
_CONFIG_FILES = (
    ("CLAUDE.md", "CLAUDE.md"),
    (".mcp.json", ".mcp.json"),
    (".claude/settings.json", ".claude/settings.json"),
)


class ScaffoldGalleryService:
    """Service for the Scaffold Gallery web UI.

    Instantiated per-request with the project directory. All methods are
    synchronous since the web terminal runs them in a thread pool.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self._registry = BuildArtifactCatalog.default()
        self._ownership = resolve_ownership(project_dir)
        self._manager: TemplateManager | None = None
        self._ctx: dict[str, Any] | None = None
        self._refresh_ownership()

    def _load_config(self) -> dict[str, Any]:
        config_file = self.project_dir / "config.yml"
        if not config_file.exists():
            return {}
        with open(config_file, encoding="utf-8") as f:
            return resolve_env_vars(yaml.safe_load(f) or {})

    # ── Ownership ─────────────────────────────────────────────────────

    @property
    def _store(self) -> OwnershipStore | None:
        """The durable volume store, or None outside a deployed container."""
        return self._ownership.store

    def _refresh_ownership(self) -> None:
        """Re-read every ownership surface and recompute their union."""
        self._config = self._load_config()
        self._config_owned = get_user_owned(self._config)
        raw = self._store.read() if self._store is not None else {}

        # Drop records naming a file the build generates, on the way out of the
        # store. Containment stops such a record from ever holding or restoring
        # a body, but a record is more than a body: left in place it flows into
        # the union below and the gallery tells the operator they own the
        # write-safety layer's config, shows them the framework's text as
        # theirs, and fails their save. The store outlives the image, so it can
        # hold a record the write guard would refuse — the read side has to be
        # as skeptical as the write side.
        self._records = {
            name: record
            for name, record in raw.items()
            if not self._names_generated_output(name, record.output_path)
        }
        merged = [
            name
            for name in merge_user_owned(self._config_owned, self._records)
            if not self._names_generated_output(name, None)
        ]

        # An artifact sitting in the profile's convention directory is owned
        # whether or not config.yml says so yet: a claim moves it there and the
        # registration follows at the next build. Deriving it from the profile
        # rather than waiting for that build is what keeps claim → edit → save
        # working in the meantime.
        self._profile_artifacts = (
            profile_artifacts(self._ownership.profile_root) if self._is_profile_mode else {}
        )
        self._profile_owned = set(self._profile_artifacts)
        self._user_owned = merged + [n for n in sorted(self._profile_owned) if n not in merged]

    def _names_generated_output(self, name: str, output_path: str | None) -> bool:
        """Whether an ownership entry points at a file the build generates.

        Checked by every spelling the entry could reach the file under, because
        a record written by an older OSPREY carries whatever that OSPREY chose:
        the path it recorded, the catalog's path for the name, or the path the
        name maps to on its own. One of the three is enough — ``hooks/hook-config``
        and ``hooks/hook_config.json`` name the same generated file, and a check
        that caught only one spelling would be no check at all.
        """
        candidates = {output_path} if output_path else set()
        art = self._registry.get(name)
        if art is not None:
            candidates.add(art.output_path)
        candidates.add(self._canonical_to_path(name))
        generated = generated_project_paths()
        return any(candidate in generated for candidate in candidates if candidate)

    @property
    def _is_profile_mode(self) -> bool:
        """The profile is reachable, so its copies are the ones in play."""
        return self._ownership.mode is OwnershipMode.PROFILE

    @property
    def _delegates_claims(self) -> bool:
        """Claims belong to the CLI path — including when it will refuse them.

        A project that names a profile it cannot reach has nowhere durable to
        record a claim, so the gallery hands the claim over exactly as it would
        in profile mode and lets the CLI's refusal stand. Recording it in
        config.yml instead would make the gallery a way around the single-writer
        contract, and the next build would erase the entry anyway.
        """
        return self._ownership.mode in (OwnershipMode.PROFILE, OwnershipMode.DEGRADED)

    def _profile_file(self, name: str | None) -> Path | None:
        """The profile's own copy of *name*, when the profile holds one.

        A claim moves the artifact out of the project, so from then on the
        profile file — not the project's — is what the operator means when they
        open it, and the only place a save is not thrown away by the next
        ``osprey build``. Only single files qualify: a skill or a service is a
        whole directory the gallery does not edit as text.
        """
        if name is None:
            return None
        art = self._profile_artifacts.get(name)
        if art is None or art.is_directory or not art.path.is_file():
            return None
        return art.path

    def _output_path_for(self, name: str) -> str:
        """The project-relative output path a canonical name maps to."""
        art = self._registry.get(name)
        if art is not None:
            return art.output_path
        held = self._profile_artifacts.get(name)
        if held is not None:
            return held.output_path
        return self._canonical_to_path(name)

    def _stored_output_path(self, name: str) -> str:
        """Output path for an owned artifact, honoring what the store recorded.

        A store entry keeps the path its claim was made against, which stays
        right for an artifact a later image dropped from the registry — the
        claim outlives the image that carried it.
        """
        record = self._records.get(name)
        if record is not None and record.output_path:
            return record.output_path
        return self._output_path_for(name)

    def _record_claim(
        self, name: str, content: str | None = None, output_path: str | None = None
    ) -> None:
        """Durably record *name* as user-owned on whichever surface applies.

        *output_path* overrides the derived path for callers that already know
        it — notably ``create_artifact``, whose hook artifacts end in ``.py``
        rather than the ``.md`` the canonical-name mapping assumes.
        """
        if output_path is None:
            output_path = self._output_path_for(name)
        self._refuse_unownable(name, output_path)

        if self._ownership.mode is OwnershipMode.VOLUME:
            # The project tree is image-baked: a config.yml edit here would
            # report success and vanish with the container.
            store = self._store
            assert store is not None  # VOLUME mode always carries the store
            store.claim(name, output_path, content)
        else:
            self._require_durable_config_surface()
            update_config_add_user_owned(self.project_dir, name)
            manager, ctx = self._ensure_template_context()
            update_manifest_add_user_owned(self.project_dir, manager, ctx, name)
        self._refresh_ownership()

    def _refuse_unownable(self, name: str, output_path: str) -> None:
        """Apply the claimability rules the CLI applies, on the modes it does not see.

        PROFILE and DEGRADED hand claims to ``osprey scaffold claim``, which
        refuses a generated path itself. VOLUME and CONFIG record ownership
        here instead, so without this the gallery would be a way around a rule
        the CLI enforces — and in a container the refusal is the load-bearing
        one, because a frozen ``hook_config.json`` comes back over the
        regenerated file at every start.
        """
        refuse_generated_path(name, output_path)
        if self._ownership.mode is OwnershipMode.VOLUME:
            refuse_unstorable_path(name, output_path)

    def _record_release(self, name: str) -> None:
        """Durably record *name* as no longer user-owned."""
        if self._ownership.mode is OwnershipMode.VOLUME:
            # A tombstone, not a deletion: config.yml claims this name and
            # only an explicit record can outrank it.
            store = self._store
            assert store is not None  # VOLUME mode always carries the store
            store.release(name, self._stored_output_path(name))
        else:
            self._require_durable_config_surface()
            update_config_remove_user_owned(self.project_dir, name)
            if self._registry.get(name) is not None:
                update_manifest_remove_user_owned(self.project_dir, name)
        self._refresh_ownership()

    def _require_durable_config_surface(self) -> None:
        """Refuse to write config.yml for a project whose profile owns it.

        Only a project that names no profile at all still keeps ownership in
        its own config.yml. Where a profile is named but unreachable there is
        nowhere durable to write, and the refusal says so in the CLI's own
        currency — ``ScaffoldClaimError``, which the route family renders as a
        conflict with the message intact.

        Claims and registrations never reach this: they are handed to
        :func:`~osprey.cli.scaffold_cmd.claim_into_profile`, which refuses in
        its own words. This guards the paths that do not go through it —
        creating an artifact, and releasing one.
        """
        if self._ownership.mode is not OwnershipMode.DEGRADED:
            return
        from osprey.cli.scaffold_cmd import OWNERSHIP_LIVES_IN_THE_PROFILE, ScaffoldClaimError

        raise ScaffoldClaimError(
            f"{self.project_dir} was built from a profile, and that profile cannot be "
            "reached from here. Nothing was written.\n\n"
            f"{OWNERSHIP_LIVES_IN_THE_PROFILE}\n\n"
            "  Restore the profile this project names — its manifest records the path as\n"
            "  build_args.profile_path_abs — or rebuild from the profile you want to own\n"
            "  this artifact, and try again."
        )

    def _write_body(self, output_path: str, content: str, name: str | None = None) -> None:
        """Write an artifact body to whichever surfaces must carry it.

        The profile's copy is the source of truth where there is one. Otherwise
        the project tree always gets the write — the running agent reads from
        there — and a deployed container additionally gets the volume copy,
        which is the one that survives recreation. A read-only image tree
        degrades to store-only rather than failing the save outright.

        On the volume the save is recorded as a claim, not just written as a
        body. Ownership derived from the image's ``config.yml`` carries no store
        record, and a restore walks the records — so a body saved without one
        would sit on the volume that outlived the container and never be put
        back. Saving an artifact you already own is a claim on it by any honest
        reading.
        """
        if self._is_directory_artifact(name):
            raise ValueError(
                f"'{name}' is a whole directory, not a file the gallery edits as text. "
                "Edit the files inside it instead."
            )

        profile_file = self._profile_file(name)
        if profile_file is not None:
            profile_file.write_text(content, encoding="utf-8")
            return

        if self._store is not None:
            if name is not None:
                self._store.claim(name, output_path, content)
            else:  # pragma: no cover - every save path knows its artifact
                self._store.write_content(output_path, content)

        target = self.project_dir / output_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError:
            if self._store is None:
                raise
            logger.warning(
                "Project tree at %s is not writable; %s is held on the claude-config volume only.",
                self.project_dir,
                output_path,
            )

    def _read_body(self, output_path: str, name: str | None = None) -> str | None:
        """Read an artifact body from the surface that owns it.

        Profile first: a claim moved the artifact there, so that copy is the
        one the next build reads and the one an edit has to land on.

        Otherwise the store and the project tree can disagree, and which one
        wins is the same question :func:`~...ownership.rehydrate` answers when
        it decides whether to overwrite: a tree copy that still renders
        identical to its framework template is the image's, so the durable copy
        is the newer of the two and wins. A tree copy that does not is somebody
        else's edit — made through the terminal, or by the agent — and it wins,
        because rehydrate has already declined to overwrite it. Preferring the
        store unconditionally would show the operator a file that is not what
        the agent is reading, and quietly revert that edit at the next save.
        """
        profile_file = self._profile_file(name)
        if profile_file is not None:
            return profile_file.read_text(encoding="utf-8")

        fpath = self.project_dir / output_path
        current = self._safe_read(fpath) if fpath.is_file() else None

        if self._store is not None:
            stored = self._store.read_content(output_path)
            if stored is not None:
                if current is None or current == stored:
                    return stored
                if name is not None and is_pristine_from_image(
                    name, current, self._framework_or_none
                ):
                    return stored
                return current

        return current

    def _is_directory_artifact(self, name: str | None) -> bool:
        """Whether *name* is a whole directory rather than a single file.

        A skill or a service is a directory on both surfaces the gallery knows
        about, so both are consulted: the catalog for what the framework ships,
        the profile for what a claim moved there.
        """
        if name is None:
            return False
        art = self._registry.get(name)
        if art is not None and art.is_directory:
            return True
        held = self._profile_artifacts.get(name)
        return held is not None and held.is_directory

    def _framework_or_none(self, name: str) -> str | None:
        """Render an artifact's framework template, or None if there isn't one."""
        art = self._registry.get(name)
        if art is None or art.is_directory:
            return None
        return self._render_framework(art)

    # ── List ──────────────────────────────────────────────────────────

    def list_artifacts(self) -> list[dict[str, Any]]:
        """Return all artifacts with status and metadata.

        Filesystem-first: only artifacts whose files exist on disk are
        returned, enriched with registry metadata where available.
        Custom user-owned files (registered via :meth:`register_untracked`)
        are also included.
        """
        result: list[dict[str, Any]] = []
        registry_names: set[str] = set()

        # Phase 1: filesystem-driven discovery
        for _abs_path, rel_path in self._scan_all_files():
            art = self._registry.get_by_output(rel_path)

            if art is not None:
                # Registry-backed artifact that exists on disk
                registry_names.add(art.canonical_name)
                is_owned = art.canonical_name in self._user_owned
                category = (
                    art.canonical_name.split("/")[0] if "/" in art.canonical_name else "config"
                )
                fm = self._read_front_matter_from_disk(rel_path, art)
                summary = fm.get("summary") or art.description
                description = fm.get("description") or art.description

                result.append(
                    {
                        "name": art.canonical_name,
                        "category": category,
                        "summary": summary,
                        "description": description,
                        "output_path": art.output_path,
                        "status": "user-owned" if is_owned else "framework",
                        "custom": False,
                        "language": self._infer_language(art.output_path),
                    }
                )
            else:
                # Not in registry — check if it's a registered custom artifact
                canonical = self._path_to_canonical(rel_path)
                if canonical in self._user_owned and canonical not in registry_names:
                    registry_names.add(canonical)
                    fm = self._read_front_matter_from_disk(rel_path)
                    summary = fm.get("summary", "(custom user file)")
                    description = fm.get(
                        "description", "(custom user file — no framework template)"
                    )
                    category = canonical.split("/")[0] if "/" in canonical else "other"
                    result.append(
                        {
                            "name": canonical,
                            "category": category,
                            "summary": summary,
                            "description": description,
                            "output_path": rel_path,
                            "status": "user-owned",
                            "custom": True,
                            "language": self._infer_language(rel_path),
                        }
                    )

        # Phase 2: artifacts the profile holds. A claim moved these out of the
        # project, so the filesystem pass cannot see them — and without this
        # they would vanish from the gallery the moment they were claimed,
        # until the next build copied them back in.
        for name in sorted(self._profile_owned - registry_names):
            held = self._profile_artifacts[name]
            if held.is_directory:
                continue
            art = self._registry.get(name)
            fm = self._front_matter(self._safe_read(held.path) or "")
            fallback = art.description if art is not None else "(profile-owned file)"
            result.append(
                {
                    "name": name,
                    "category": name.split("/")[0] if "/" in name else "other",
                    "summary": fm.get("summary") or fallback,
                    "description": fm.get("description") or fallback,
                    "output_path": held.output_path,
                    "status": "user-owned",
                    "custom": art is None,
                    "language": self._infer_language(held.output_path),
                }
            )

        return result

    # ── Content retrieval ─────────────────────────────────────────────

    def get_content(self, name: str) -> dict[str, Any]:
        """Get the active content for an artifact (always reads from disk)."""
        art = self._registry.get(name)
        if art is None:
            return self._get_custom_content(name)

        disk_content = self._read_user_file(art)
        if disk_content is not None:
            source = "user-owned" if art.canonical_name in self._user_owned else "framework"
            return {
                "content": disk_content,
                "source": source,
                "language": self._infer_language(art.output_path),
            }
        # File not on disk — render from template as fallback
        return {
            "content": self._render_framework(art),
            "source": "framework",
            "language": self._infer_language(art.output_path),
        }

    def get_framework_content(self, name: str) -> str:
        """Render and return the framework template content."""
        art = self._get_artifact(name)
        return self._render_framework(art)

    def get_override_content(self, name: str) -> str | None:
        """Read the user-owned file content, or None if not user-owned."""
        art = self._get_artifact(name)
        if art.canonical_name not in self._user_owned:
            return None
        return self._read_user_file(art)

    # ── Diff ──────────────────────────────────────────────────────────

    def compute_diff(self, name: str) -> dict[str, Any]:
        """Compute unified diff between framework and user-owned file."""
        art = self._get_artifact(name)
        if art.canonical_name not in self._user_owned:
            raise FileNotFoundError(f"'{name}' is not user-owned — no diff available")

        user_content = self._read_user_file(art)
        if user_content is None:
            raise FileNotFoundError(f"User-owned file not found: {art.output_path}")

        framework_content = self._render_framework(art)
        framework_lines = framework_content.splitlines(keepends=True)
        user_lines = user_content.splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                framework_lines,
                user_lines,
                fromfile=f"framework:{art.template_path}",
                tofile=f"yours:{art.output_path}",
            )
        )

        additions = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
        deletions = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))

        return {
            "unified_diff": "".join(diff_lines),
            "has_diff": len(diff_lines) > 0,
            "additions": additions,
            "deletions": deletions,
        }

    # ── Claim ─────────────────────────────────────────────────────────

    def scaffold_override(self, name: str) -> dict[str, Any]:
        """Claim an artifact for in-place editing."""
        art = self._get_artifact(name)

        if name in self._user_owned:
            raise FileExistsError(f"'{name}' is already user-owned")

        if self._delegates_claims:
            return self._claim_into_profile(name)

        # Before anything is rendered or written: a refusal must leave no file
        # behind, and this path would otherwise materialize the artifact first.
        self._refuse_unownable(name, art.output_path)

        # Render framework content if file doesn't exist
        output_file = self.project_dir / art.output_path
        if not output_file.exists():
            content = self._render_framework(art)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            if output_file.suffix == ".py":
                output_file.chmod(output_file.stat().st_mode | 0o755)
        else:
            content = output_file.read_text(encoding="utf-8")

        self._record_claim(name, content)

        return {
            "status": "claimed",
            "output_path": art.output_path,
            "content": content,
        }

    def _claim_into_profile(self, name: str) -> dict[str, Any]:
        """Hand a claim to the CLI path that owns it.

        ``osprey scaffold claim`` moves the artifact into its profile
        convention slot and leaves registration to the next build's convention
        scan. The gallery calls it rather than reimplementing it: a second
        implementation of "what a claim means" is the thing this whole model
        removes. Nothing is rendered here first — a name the project does not
        actually carry is a refusal the CLI is entitled to make.
        """
        from osprey.cli.scaffold_cmd import claim_into_profile

        # ScaffoldClaimError propagates untouched: every refusal names what was
        # refused and what to do instead, and routes/scaffold.py renders it as
        # a 409 with that text intact. Re-wrapping it here would cost the
        # operator the only part of the failure worth reading.
        outcome = claim_into_profile(self.project_dir, name)

        self._refresh_ownership()
        content = ""
        if outcome.profile_path.is_file():
            content = outcome.profile_path.read_text(encoding="utf-8")

        return {
            "status": "claimed",
            "output_path": outcome.target.project_rel,
            "content": content,
            "profile_path": str(outcome.profile_path),
            "moved_to_profile": True,
        }

    # ── Create ─────────────────────────────────────────────────────────

    def create_artifact(self, category: str, name: str, content: str = "") -> dict[str, Any]:
        """Create a new custom artifact, on whichever surface will keep it.

        Every path here comes from ``resolve_claim_target`` rather than being
        assembled locally, because each creatable category is a profile
        convention and each convention already decides the shape of one entry:
        a rule is a markdown file, a hook is a script known by its filename, a
        skill is a directory. Assembling ``.claude/<category>/<name>.md`` by
        hand gets two of the three wrong: it drops a hook's ``.py`` from the
        name ownership records, and it writes a skill as a flat file that
        Claude Code does not load at all.
        """
        allowed = {"agents", "rules", "hooks", "skills", "commands", "output-styles"}
        if category not in allowed:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {sorted(allowed)}")

        from osprey.cli.scaffold_cmd import resolve_claim_target

        # A hook carries its suffix in the name it is owned under; every other
        # creatable shape derives its own. Resolving also validates the name,
        # so `..` and absolute paths are refused before anything is written.
        artifact_name = f"{category}/{name}.py" if category == "hooks" else f"{category}/{name}"
        target = resolve_claim_target(artifact_name)

        canonical_name = target.name
        output_path = (
            f"{target.project_rel}/{DIRECTORY_ENTRY_FILE}"
            if target.is_directory
            else target.project_rel
        )

        if self._registry.get(canonical_name):
            raise ValueError(f"'{canonical_name}' is a framework artifact — use claim instead")
        if canonical_name in self._user_owned:
            raise FileExistsError(f"'{canonical_name}' already exists")

        if not content:
            content = self._starter_content(category, name)

        if self._delegates_claims:
            return self._create_in_profile(target, output_path, content)

        full_path = self.project_dir / output_path
        if full_path.exists():
            raise FileExistsError(f"File already exists at {output_path}")

        # Checked before the file is written, so a refusal leaves nothing behind.
        self._refuse_unownable(canonical_name, output_path)

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        if full_path.suffix == ".py":
            full_path.chmod(full_path.stat().st_mode | 0o755)

        self._record_claim(canonical_name, content, output_path=output_path)

        return {
            "status": "created",
            "canonical_name": canonical_name,
            "output_path": output_path,
            "content": content,
        }

    def _create_in_profile(
        self, target: ClaimTarget, output_path: str, content: str
    ) -> dict[str, Any]:
        """Author a new artifact directly in the profile that owns the project.

        Where a profile is reachable it is the source of truth, so a new
        artifact belongs in its convention directory and nowhere else. Writing
        the project tree and registering it in ``config.yml`` — what this used
        to do — created exactly the second source of truth the profile model
        removes: the registration is derived output the next build regenerates
        from the profile, so it would be dropped, and the file beside it pruned
        as unowned. The operator would have watched the artifact be created and
        then quietly disappear one build later.

        Registration is left to the next build's convention scan, the same
        contract a claim gets. Until that build runs the gallery still shows the
        artifact as owned, because it derives profile ownership directly rather
        than waiting for the scan (see :meth:`_refresh_ownership`).
        """
        # A degraded project names a profile it cannot reach, so there is
        # nowhere durable to author into. Refuse before anything is written.
        self._require_durable_config_surface()

        profile_root = self._ownership.profile_root
        if profile_root is None:  # pragma: no cover - profile mode means it resolved
            raise FileNotFoundError(f"No reachable profile to create '{target.name}' in")

        slot = profile_root / target.profile_rel
        body = slot / DIRECTORY_ENTRY_FILE if target.is_directory else slot

        if slot.exists():
            raise FileExistsError(f"'{target.name}' is already in the profile: {slot}")
        if (self.project_dir / target.project_rel).exists():
            # The project already carries this artifact, so this is a claim —
            # and a claim is what moves it without discarding what it holds.
            raise FileExistsError(
                f"File already exists at {target.project_rel} — claim it instead of creating it."
            )

        body.parent.mkdir(parents=True, exist_ok=True)
        body.write_text(content, encoding="utf-8")
        if body.suffix == ".py":
            body.chmod(body.stat().st_mode | 0o755)

        self._refresh_ownership()

        return {
            "status": "created",
            "canonical_name": target.name,
            "output_path": output_path,
            "content": content,
            "profile_path": str(body),
            "created_in_profile": True,
        }

    @staticmethod
    def _starter_content(category: str, name: str) -> str:
        """Return minimal starter content for a new artifact."""
        display = name.replace("-", " ").replace("_", " ").title()
        if category == "hooks":
            return (
                f'"""Hook: {display}."""\n\nimport sys\nimport json\n\n\n'
                f"def main():\n    pass\n\n\n"
                f'if __name__ == "__main__":\n    main()\n'
            )
        if category == "skills":
            return (
                f"---\nname: {name}\ndescription: {display}\n---\n\n"
                f"# {display}\n\nDescribe this skill's purpose and workflow.\n"
            )
        return f"# {display}\n\nDescribe this {category.rstrip('s')} here.\n"

    # ── Save ──────────────────────────────────────────────────────────

    def save_override(self, name: str, content: str) -> dict[str, Any]:
        """Write content to a user-owned file."""
        out = self._stored_output_path(name)
        # Refused ahead of the ownership check, so a generated file gets the
        # reason it is refused rather than "claim it first" — advice that would
        # only lead the operator to the same refusal one step later. Ownership
        # filtering should mean it never reaches here at all; without this it
        # would fail instead as a store write it cannot make durable.
        refuse_generated_path(name, out)

        if name not in self._user_owned:
            raise FileNotFoundError(f"'{name}' is not user-owned — claim first")

        self._write_body(out, content, name)

        return {"status": "saved", "path": out}

    # ── Unclaim ──────────────────────────────────────────────────────

    def unoverride(self, name: str, delete_file: bool = False) -> dict[str, Any]:
        """Release ownership, restoring framework management."""
        if name not in self._user_owned:
            raise FileNotFoundError(f"'{name}' is not user-owned")

        held = self._profile_artifacts.get(name)
        if held is not None:
            # The profile supplies this one, so its ownership is derived: the
            # next build's convention scan registers it again from the profile.
            # Dropping the project's record would change nothing that survives
            # that build, and reporting "removed" for it would be false — the
            # artifact stays owned, stays on disk, and comes back registered.
            # Giving it up means removing it from the profile, which is not a
            # thing this surface does.
            from osprey.cli.scaffold_cmd import still_supplied_by_profile_message

            return {
                "status": "still-supplied-by-profile",
                "deleted_file": False,
                "restored_file": False,
                "profile_path": str(held.path),
                "message": still_supplied_by_profile_message(str(held.path)),
            }

        is_custom = self._registry.get(name) is None

        self._record_release(name)

        deleted = False
        restored = False
        if delete_file and is_custom:
            # Custom artifact (no framework template) — delete the file
            out = self.project_dir / self._canonical_to_path(name)
            if out.exists():
                out.unlink()
                deleted = True
        elif delete_file and not is_custom:
            # Framework artifact — restore file to rendered template
            art = self._registry.get(name)
            try:
                content = self._render_framework(art)
                out = self.project_dir / art.output_path
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(content, encoding="utf-8")
                restored = True
            except Exception:
                pass  # ownership already removed; stale file stays

        return {"status": "removed", "deleted_file": deleted, "restored_file": restored}

    # ── Untracked file detection ─────────────────────────────────────

    @property
    def _scan_dirs(self) -> tuple[str, ...]:
        """Categories where users may create new .md files (excludes hooks and config)."""
        return tuple(sorted(self._registry.categories - {"config", "hooks"}))

    def scan_untracked(self) -> list[dict[str, Any]]:
        """Find files in .claude/ that are active in Claude Code but not managed.

        Scans .claude/{rules,agents,commands,skills}/ for .md files whose
        output path doesn't match any registered artifact and whose derived
        canonical name isn't already in ``scaffold.user_owned``.
        """
        claude_dir = self.project_dir / ".claude"
        if not claude_dir.is_dir():
            return []

        registered_outputs: set[str] = {a.output_path for a in self._registry.all_artifacts()}

        untracked: list[dict[str, Any]] = []
        for subdir_name in self._scan_dirs:
            subdir = claude_dir / subdir_name
            if not subdir.is_dir():
                continue
            for fpath in sorted(subdir.rglob("*.md")):
                if not fpath.is_file() or fpath.name.startswith("."):
                    continue
                rel_path = str(fpath.relative_to(self.project_dir))
                if rel_path in registered_outputs:
                    continue
                canonical = self._path_to_canonical(rel_path)
                if canonical in self._user_owned:
                    continue
                preview = self._safe_preview(fpath)
                untracked.append(
                    {
                        "canonical_name": canonical,
                        "output_path": rel_path,
                        "category": canonical.split("/")[0] if "/" in canonical else "other",
                        "preview": preview,
                    }
                )
        return untracked

    def register_untracked(self, canonical_name: str) -> dict[str, Any]:
        """Register an untracked file by adding it to ``scaffold.user_owned``."""
        # Ownership is checked first: an artifact already claimed into the
        # profile is no longer in the project tree, and "already registered" is
        # the accurate answer to a second attempt — "file not found" is not.
        if canonical_name in self._user_owned:
            raise FileExistsError(f"'{canonical_name}' is already registered")
        output_path = self._canonical_to_path(canonical_name)
        full_path = self.project_dir / output_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found on disk: {output_path}")

        if self._delegates_claims:
            # The file's durable home is the profile, exactly as for a claim.
            # Registering it in the project's config.yml instead would be
            # rewritten by the next build from the profile — the registration
            # would report success and then be gone.
            outcome = self._claim_into_profile(canonical_name)
            return {
                "status": "registered",
                "canonical_name": canonical_name,
                "profile_path": outcome["profile_path"],
                "moved_to_profile": True,
            }

        # Carry the body into the store too: registering an untracked file in a
        # container that never persists it would lose the very file being
        # registered on the next container recreation.
        self._record_claim(
            canonical_name,
            self._safe_read(full_path),
            output_path=output_path,
        )

        return {"status": "registered", "canonical_name": canonical_name}

    @staticmethod
    def _safe_read(fpath: Path) -> str | None:
        """Read a file's text, or None if it is not readable as text."""
        try:
            return fpath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            return None

    def delete_untracked(self, canonical_name: str) -> dict[str, Any]:
        """Delete an untracked file from disk."""
        if self._registry.get(canonical_name) is not None:
            raise ValueError(f"'{canonical_name}' is a framework artifact — use unoverride instead")
        output_path = self._canonical_to_path(canonical_name)
        full_path = self.project_dir / output_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found on disk: {output_path}")

        full_path.unlink()
        return {"status": "deleted", "output_path": output_path}

    @staticmethod
    def _path_to_canonical(rel_path: str) -> str:
        """Derive a canonical name from a relative output path.

        ``.claude/rules/test.md`` -> ``rules/test``. Only ``.md`` is stripped:
        a hook keeps its ``.py``, because that is the name it is owned under
        (the inverse of :meth:`_canonical_to_path`).
        """
        # Strip leading .claude/
        name = rel_path
        if name.startswith(".claude/"):
            name = name[len(".claude/") :]
        # Strip .md extension
        if name.endswith(".md"):
            name = name[: -len(".md")]
        return name

    @staticmethod
    def _canonical_to_path(canonical_name: str) -> str:
        """Convert a canonical name back to a relative output path.

        ``.md`` is appended only to a name that carries no suffix of its own.
        Hook artifacts are executable ``.py`` scripts named by filename in
        ``settings.json``, so the name a hook is known by *includes* its suffix
        (``hooks/osprey_cf_feedback_capture.py``) — appending ``.md`` to that
        would name a file that does not exist, and the body of every claimed
        hook would be stored against a path nothing ever reads back.
        """
        if PurePosixPath(canonical_name).suffix:
            return f".claude/{canonical_name}"
        return f".claude/{canonical_name}.md"

    @staticmethod
    def _safe_preview(fpath: Path, max_chars: int = 200) -> str:
        """Read the first ``max_chars`` characters of a file for preview."""
        try:
            text = fpath.read_text(encoding="utf-8")
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "..."
        except (UnicodeDecodeError, PermissionError, OSError):
            return "(unable to read)"

    # ── Private helpers ───────────────────────────────────────────────

    def _get_custom_content(self, name: str) -> dict[str, Any]:
        """Read content for a custom user-owned artifact (no framework template)."""
        if name not in self._user_owned:
            raise KeyError(f"Unknown artifact: '{name}'")
        output_path = self._stored_output_path(name)
        content = self._read_body(output_path, name)
        if content is None:
            raise FileNotFoundError(f"Custom file not found: {output_path}")
        return {
            "content": content,
            "source": "user-owned",
            "language": self._infer_language(output_path),
        }

    def _scan_all_files(self) -> list[tuple[Path, str]]:
        """Yield all existing Claude Code files as (absolute_path, rel_path) pairs.

        Mirrors ClaudeCodeFileService._collect_targets() but used for prompt
        gallery discovery.  Config-tier files come first, then rglob on each
        subdirectory.
        """
        targets: list[tuple[Path, str]] = []

        # Config-tier files
        for rel_path, _ in _CONFIG_FILES:
            fpath = self.project_dir / rel_path
            if fpath.is_file():
                targets.append((fpath, rel_path))

        # Subdirectory files
        claude_dir = self.project_dir / ".claude"
        for subdir_name in _CLAUDE_SUBDIRS:
            subdir = claude_dir / subdir_name
            if not subdir.is_dir():
                continue
            for fpath in sorted(subdir.rglob("*")):
                if fpath.is_file() and not fpath.name.startswith("."):
                    rel = str(fpath.relative_to(self.project_dir))
                    targets.append((fpath, rel))

        return targets

    def _read_front_matter_from_disk(
        self,
        rel_path: str,
        art: BuildArtifact | None = None,
    ) -> dict[str, str]:
        """Extract front-matter from on-disk content, falling back to rendered template.

        Reads the file at *rel_path* first (reflects user modifications).
        Falls back to ``_render_framework(art)`` only if the file is absent
        AND a ``BuildArtifact`` is provided.  Uses both ``_FM_RE`` and
        ``_PY_FM_RE`` patterns so ``.py`` hook files are handled correctly.
        """
        content: str | None = None
        fpath = self.project_dir / rel_path
        if fpath.is_file():
            try:
                content = fpath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError, OSError):
                pass

        if content is None and art is not None:
            try:
                content = self._render_framework(art)
            except Exception:
                return {}

        if content is None:
            return {}
        return self._front_matter(content)

    @staticmethod
    def _front_matter(content: str) -> dict[str, str]:
        """Parse front-matter fields out of artifact text.

        Handles both the markdown form and the ``.py`` hook form, whose
        front-matter sits inside the module docstring.
        """
        match = _FM_RE.match(content) or _PY_FM_RE.match(content)
        if not match:
            return {}

        fields: dict[str, str] = {}
        for line in match.group(1).split("\n"):
            kv = re.match(r'^(\w[\w-]*):\s*"?(.*?)"?\s*$', line)
            if kv:
                fields[kv.group(1)] = kv.group(2)
        return fields

    def _get_artifact(self, name: str) -> BuildArtifact:
        """Look up an artifact by name, raising if unknown."""
        art = self._registry.get(name)
        if art is None:
            raise KeyError(f"Unknown artifact: '{name}'")
        return art

    def _ensure_template_context(self) -> tuple[TemplateManager, dict[str, Any]]:
        """Return cached (manager, context) pair, creating on first call."""
        if self._manager is None:
            self._manager = TemplateManager()
            from osprey.cli.templates.claude_code import build_claude_code_context

            self._ctx = build_claude_code_context(
                self._manager.template_root, self._manager.jinja_env, self.project_dir, self._config
            )
        return self._manager, self._ctx  # type: ignore[return-value]

    def _render_framework(self, art: BuildArtifact) -> str:
        """Render the framework template with the current config context."""
        manager, ctx = self._ensure_template_context()
        claude_code_dir = manager.template_root / "claude_code"
        template_file = claude_code_dir / art.template_path

        if not template_file.exists():
            return f"# Template not found: {art.template_path}\n"

        if template_file.suffix == ".j2":
            template_rel = f"claude_code/{art.template_path}"
            template = manager.jinja_env.get_template(template_rel)
            return template.render(**ctx)
        else:
            return template_file.read_text(encoding="utf-8")

    def _read_user_file(self, art: BuildArtifact) -> str | None:
        """Read the user's copy of an artifact from the surface that holds it."""
        return self._read_body(art.output_path, art.canonical_name)

    @staticmethod
    def _infer_language(output_path: str) -> str:
        """Infer language from the output file extension."""
        ext = PurePosixPath(output_path).suffix
        return _EXT_LANG.get(ext, "text")
