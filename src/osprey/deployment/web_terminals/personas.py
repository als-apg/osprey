"""Roster and persona identity resolution for multi-user web terminal deployments.

Turns the raw ``modules.web_terminals`` roster into explicit per-user identity:
normalized ``{"name", "index"}`` entries (:func:`normalize_users`) and fully
resolved image/project/container-dir identity per user
(:func:`resolve_personas`). Port arithmetic lives separately in
:mod:`osprey.deployment.web_terminals.ports`.
"""

from typing import Any

# Task 2.5: the only wired value for `modules.web_terminals.mcp.topology`. Every
# other value (including the recognized-but-rejected `shared_http`) is
# fail-closed at render time — see render.py's `_check_mcp_topology()`. Lives
# in this neutral module so lint.py and render.py can both import it without
# one depending on the other. This key is scoped to the shared
# framework-MCP tier only; it has no bearing on a facility's own
# `claude_code.servers` custom entries (those render through the unrelated
# per-project `.mcp.json` pipeline, untouched by this module).
SUPPORTED_MCP_TOPOLOGY = "per_container_stdio"


def as_dict(value: Any) -> dict[str, Any]:
    """Read a config section defensively: anything not a dict becomes empty."""
    return value if isinstance(value, dict) else {}


def normalize_users(users_raw: Any) -> list[dict[str, Any]]:
    """Normalize ``modules.web_terminals.users`` into explicit ``{"name", "index"}`` dicts.

    This is the canonical users-normalizer for the module; render.py delegates to it
    directly rather than re-parsing the raw roster itself. Its purpose is to let a
    decommission migration *freeze* a bare roster's positional indices onto explicit
    ``index`` fields before removing a user — once frozen, deleting an earlier entry
    can no longer shift a later user's port allocation, because the survivor's index
    no longer depends on its position in the list.

    Legacy bare strings (``"alice"``) normalize to ``{"name": "alice", "index":
    <raw list position>}``, matching the fallback ``render.py`` already uses so
    ports stay identical to what a pre-existing all-strings roster produces today.
    Already-explicit object entries (``{"name": ..., "index": ...}``) pass through
    with their explicit index preserved, regardless of list position. This makes
    the function idempotent: normalizing an already-normalized list is a no-op,
    which is what lets the freeze step be applied without checking whether a
    roster has already been frozen.

    Malformed entries — anything that isn't a string, and any dict missing a
    string ``name`` or an int ``index`` — are dropped rather than raising
    (well-formedness is lint.py's job). ``bool`` is a subclass of ``int`` in
    Python, so a dict entry
    like ``{"name": "alice", "index": True}`` would technically satisfy
    ``isinstance(index, int)``; this function deliberately treats ``bool`` as an
    invalid index type and drops such entries, since ``index: true/false`` in a
    facility config is a config typo (e.g. a YAML boolean where an int was meant),
    not a meaningful index.

    Args:
        users_raw: The raw ``modules.web_terminals.users`` value. Anything other
            than a list (including ``None``) is treated as an empty roster.

    Returns:
        New ``{"name": str, "index": int}`` dicts in config-declaration order.
        Input dicts are never mutated or returned by reference.
    """
    if not isinstance(users_raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for position, entry in enumerate(users_raw):
        if isinstance(entry, str):
            normalized.append({"name": entry, "index": position})
        elif isinstance(entry, dict):
            name = entry.get("name")
            index = entry.get("index")
            if isinstance(name, str) and isinstance(index, int) and not isinstance(index, bool):
                normalized.append({"name": name, "index": index})
    return normalized


def effective_image_source(web_terminals: dict[str, Any]) -> str:
    """Coerce ``modules.web_terminals.image_source`` to one of the two real modes.

    ``"local"`` only for the exact literal ``"local"``; every other value
    (unset, ``"registry"``, or an invalid literal) resolves to ``"registry"``.
    This is the single source of that coercion — :func:`resolve_personas`,
    render, and lint all route through it so an invalid literal is treated
    identically everywhere (and reported once, separately, by lint's
    ``_check_unknown_image_source``).
    """
    return "local" if web_terminals.get("image_source") == "local" else "registry"


def _persona_ref_by_name(raw_users: Any) -> dict[str, str]:
    """Recover each roster entry's ``persona`` reference from the raw roster.

    normalize_users() drops any `persona` field off each surviving entry;
    recover it here, keyed by name (the same key every other per-user artifact
    in this module — compose service names, volume names — is keyed by), since
    normalize_users()'s own index-freezing contract is orthogonal to persona
    resolution.
    """
    refs: dict[str, str] = {}
    if isinstance(raw_users, list):
        for raw_entry in raw_users:
            if isinstance(raw_entry, dict):
                name = raw_entry.get("name")
                persona = raw_entry.get("persona")
                if isinstance(name, str) and isinstance(persona, str) and persona:
                    refs[name] = persona
    return refs


def resolve_personas(
    web_terminals: dict[str, Any],
    registry_cfg: dict[str, Any],
    facility_prefix: str,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Resolve each roster entry's persona reference into its image/project identity.

    Layered on :func:`normalize_users`'s output — that function's signature and
    drop-don't-raise contract stay untouched, and this function re-derives each
    surviving entry's ``persona`` field (which :func:`normalize_users` discards)
    from the raw roster by matching on ``name``.

    A ``persona`` reference resolves in this order: the entry's own ``persona:``
    key, else ``modules.web_terminals.default_persona``, else ``None`` (no persona
    system in effect for this entry at all — every config predating persona
    catalogs resolves this way for every entry). Resolving against the catalog
    (``modules.web_terminals.personas.<name>: {project, project_path,
    build_profile}``) then follows these naming rules:

    * **No persona in effect** (``persona`` is ``None``): today's exact values —
      ``image`` is ``<registry_url>/web-terminal:latest`` (unsuffixed, same
      string the compose template built directly before this function existed),
      ``project`` and ``container_project_dir`` are ``<facility_prefix>-assistant``
      / ``/app/<facility_prefix>-assistant``. This is the zero-migration path: a
      config with no ``personas`` catalog at all resolves every entry here.
    * **Default persona** (resolved ``persona`` equals ``default_persona``, and
      a catalog entry exists for it): registry mode keeps the same un-suffixed
      ``<registry_url>/web-terminal:latest`` image (so the default persona's
      *image* never changes when a catalog is introduced); local mode still
      builds ``<persona.project>-<persona>:local`` like every other persona.
      ``container_project_dir`` stays pinned to ``/app/<facility_prefix>-assistant``
      regardless of the catalog entry's own ``project`` — the existing per-user
      agent-data volume must keep resolving to the same in-container path it did
      before personas existed.
    * **Non-default persona**: registry mode uses
      ``<registry_url>/web-terminal-<persona>:latest``; local mode uses
      ``<persona.project>-<persona>:local`` (same rule as the default persona);
      ``container_project_dir`` is derived from the persona's own
      ``/app/<project>``.

    Args:
        web_terminals: The already-dict-coerced ``modules.web_terminals`` section
            (``users``, ``personas``, ``default_persona``, ``image_source``).
        registry_cfg: The already-dict-coerced top-level ``registry`` section
            (only ``url`` is read).
        facility_prefix: ``facility.prefix``, used for the zero-migration /
            default-persona project dir and image (``<prefix>-assistant``).
        strict: When ``True`` (render/build/seed callers), an unresolvable
            persona reference — an explicit or inherited ``persona:`` naming a
            catalog entry that doesn't exist, or ``image_source: local`` with no
            catalog/``default_persona`` configured at all — raises
            ``ValueError``. When ``False`` (lifecycle verbs), the same
            conditions degrade gracefully: an unresolved entry falls back to the
            zero-migration values (so a stale/bad persona reference never blocks
            ``decommission``/``prune``/``nuke``) instead of raising.

    Returns:
        One ``{"name", "index", "persona", "image", "project",
        "container_project_dir", "extra_mounts"}`` dict per surviving
        :func:`normalize_users` entry, in the same order. ``persona`` is the
        resolved catalog key, or ``None`` when no persona is in effect for that
        entry. ``extra_mounts`` is the persona's ``extra_mounts`` list (compose
        volume strings applied to every user of that persona), defaulting to
        ``[]`` — both when no persona is in effect and when the catalog entry
        sets none.

    Raises:
        ValueError: See ``strict`` above.
    """
    normalized = normalize_users(web_terminals.get("users"))

    personas_raw = web_terminals.get("personas")
    personas_catalog: dict[str, Any] = personas_raw if isinstance(personas_raw, dict) else {}

    default_persona_name = web_terminals.get("default_persona")
    if not isinstance(default_persona_name, str) or not default_persona_name:
        default_persona_name = None

    image_source = effective_image_source(web_terminals)

    registry_url = ""
    if isinstance(registry_cfg, dict):
        url = registry_cfg.get("url")
        if isinstance(url, str):
            registry_url = url

    persona_ref_by_name = _persona_ref_by_name(web_terminals.get("users"))

    if (
        strict
        and image_source == "local"
        and (not personas_catalog or default_persona_name is None)
    ):
        raise ValueError(
            "modules.web_terminals.image_source: local requires both a "
            "modules.web_terminals.personas catalog and default_persona to be "
            "configured"
        )

    default_project = f"{facility_prefix}-assistant"
    default_container_dir = f"/app/{facility_prefix}-assistant"
    default_image = f"{registry_url}/web-terminal:latest"

    def _zero_migration_entry(name: str, index: int, persona: str | None) -> dict[str, Any]:
        """The zero-migration resolution: today's exact pre-persona values, with
        ``persona`` carried through for logging (``None`` when no persona is in
        effect, or the unresolvable reference on the lenient degrade path).
        ``extra_mounts`` is empty here — the zero-migration path has no catalog
        entry to read persona-level host mounts from."""
        return {
            "name": name,
            "index": index,
            "persona": persona,
            "image": default_image,
            "project": default_project,
            "container_project_dir": default_container_dir,
            "extra_mounts": [],
        }

    resolved: list[dict[str, Any]] = []
    for entry in normalized:
        name = entry["name"]
        index = entry["index"]
        persona_ref = persona_ref_by_name.get(name) or default_persona_name

        if persona_ref is None:
            # No persona system in effect for this entry — zero-migration path.
            resolved.append(_zero_migration_entry(name, index, None))
            continue

        catalog_entry = personas_catalog.get(persona_ref)
        if not isinstance(catalog_entry, dict):
            if strict:
                raise ValueError(
                    f"user {name!r} references persona {persona_ref!r}, which has "
                    "no entry in modules.web_terminals.personas"
                )
            # Lenient degrade (lifecycle verbs): keep the requested persona name
            # visible for logging, but fall back to the zero-migration values so
            # a stale/bad reference never blocks a lifecycle verb.
            resolved.append(_zero_migration_entry(name, index, persona_ref))
            continue

        project = catalog_entry.get("project")
        if not isinstance(project, str) or not project:
            project = default_project

        # Persona-level host mounts, applied to every user of this persona. A
        # non-list drops to []; individual non-string/empty entries are dropped
        # (well-formedness — the colon-part syntax — is lint's job).
        extra_mounts_raw = catalog_entry.get("extra_mounts")
        extra_mounts = (
            [mount for mount in extra_mounts_raw if isinstance(mount, str) and mount]
            if isinstance(extra_mounts_raw, list)
            else []
        )

        is_default = persona_ref == default_persona_name

        if image_source == "local":
            image = f"{project}-{persona_ref}:local"
        elif is_default:
            image = default_image
        else:
            image = f"{registry_url}/web-terminal-{persona_ref}:latest"

        container_project_dir = f"/app/{project}"

        resolved.append(
            {
                "name": name,
                "index": index,
                "persona": persona_ref,
                "image": image,
                "project": project,
                "container_project_dir": container_project_dir,
                "extra_mounts": extra_mounts,
            }
        )

    return resolved
