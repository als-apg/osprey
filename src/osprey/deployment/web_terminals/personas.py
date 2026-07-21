"""Roster and persona identity resolution for multi-user web terminal deployments.

Turns the raw ``modules.web_terminals`` roster into explicit per-user identity:
normalized ``{"name", "index"}`` entries (:func:`normalize_users`) and fully
resolved image/project/container-dir identity per user
(:func:`resolve_personas`). Port arithmetic lives separately in
:mod:`osprey.deployment.web_terminals.ports`.
"""

import os
import re
from typing import Any

# Matches ${VAR} and $VAR env references inside modules.web_terminals.image_tag.
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

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

    An object entry's optional ``display_name`` (a human-facing window/tab title,
    surfaced downstream as the per-user ``OSPREY_WEB_APP_NAME``) is carried through
    onto the normalized entry when it is a string, and dropped defensively
    otherwise (a non-string ``display_name`` is a config typo, reported separately
    by lint). Bare-string entries never carry one. Keeping this passthrough here
    means the normalized entry stays the single object :func:`resolve_personas`
    reads a user's identity off of, rather than re-deriving the field from the raw
    roster the way ``persona`` is.

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
        New ``{"name": str, "index": int}`` dicts (plus an optional
        ``"display_name": str`` key when the entry carried a string one) in
        config-declaration order. Input dicts are never mutated or returned by
        reference.
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
                normalized_entry: dict[str, Any] = {"name": name, "index": index}
                display_name = entry.get("display_name")
                if isinstance(display_name, str):
                    normalized_entry["display_name"] = display_name
                normalized.append(normalized_entry)
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


def resolve_image_tag(web_terminals: dict[str, Any]) -> str:
    """Resolve ``modules.web_terminals.image_tag`` to a literal registry tag.

    Defaults to ``"latest"`` (unset or non-string), then expands any ``${VAR}``
    / ``$VAR`` references against the **process environment at render time**, so
    the rendered compose artifact self-carries a fixed literal tag rather than a
    compose-side ``${...}`` the runtime would re-interpolate at ``up`` time. A
    referenced variable that is unset expands to the empty string (unlike
    :func:`os.path.expandvars`, which would leave the reference in place and leak
    a ``${...}`` into the output); an image_tag that resolves entirely empty is a
    lint warning (``_check_image_tag_empty``), never a silent bad tag.

    This is the single source of that resolution — :func:`resolve_personas` and
    lint both route through it so the tag is read and expanded identically.
    """
    raw = web_terminals.get("image_tag")
    if not isinstance(raw, str):
        raw = "latest"
    return _ENV_REF_RE.sub(lambda m: os.environ.get(m.group(1) or m.group(2), ""), raw)


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
    build_profile}``) then follows these naming rules. In registry mode every
    image carries the tag :func:`resolve_image_tag` resolves from
    ``modules.web_terminals.image_tag`` (``<tag>`` below, default ``latest``);
    local ``:local`` images are unaffected by that field:

    * **No persona in effect** (``persona`` is ``None``): today's exact values —
      ``image`` is ``<registry_url>/web-terminal:<tag>`` (unsuffixed, the same
      string the compose template built directly before this function existed
      whenever ``<tag>`` is its ``latest`` default),
      ``project`` and ``container_project_dir`` are ``<facility_prefix>-assistant``
      / ``/app/<facility_prefix>-assistant``. This is the zero-migration path: a
      config with no ``personas`` catalog at all resolves every entry here.
    * **Default persona** (resolved ``persona`` equals ``default_persona``, and
      a catalog entry exists for it): registry mode keeps the same un-suffixed
      ``<registry_url>/web-terminal:<tag>`` image (so the default persona's
      *image* never changes when a catalog is introduced); local mode still
      builds ``<persona.project>-<persona>:local`` like every other persona.
      ``container_project_dir`` stays pinned to ``/app/<facility_prefix>-assistant``
      regardless of the catalog entry's own ``project`` — the existing per-user
      agent-data volume must keep resolving to the same in-container path it did
      before personas existed.
    * **Non-default persona**: registry mode uses
      ``<registry_url>/web-terminal-<persona>:<tag>``; local mode uses
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
        "container_project_dir", "extra_mounts", "seed_base"}`` dict per
        surviving :func:`normalize_users` entry, in the same order. ``persona``
        is the resolved catalog key, or ``None`` when no persona is in effect for
        that entry. ``extra_mounts`` is the persona's ``extra_mounts`` list
        (compose volume strings applied to every user of that persona),
        defaulting to ``[]`` — both when no persona is in effect and when the
        catalog entry sets none. ``seed_base`` is the catalog entry's
        ``seed_base`` (a bool; anything else is defensively coerced to
        ``True``), and always ``True`` for the zero-migration / lenient-degrade
        paths — it controls whether the shared base context is prepended when
        seeding this entry's ``CLAUDE.md``. An optional ``"display_name"`` key is
        added — carried through from :func:`normalize_users` — only when the entry
        declared a non-empty string one (render emits it as
        ``OSPREY_WEB_APP_NAME``); it is omitted entirely otherwise, so a roster
        with no ``display_name`` resolves byte-identically to before this field
        existed.

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
    image_tag = resolve_image_tag(web_terminals)

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
    default_image = f"{registry_url}/web-terminal:{image_tag}"

    def _with_display_name(entry: dict[str, Any], display_name: Any) -> dict[str, Any]:
        """Attach an optional ``display_name`` to a resolved entry, mirroring
        render.py's conditional-``sublabel`` convention: the key is present only
        for a non-empty string, so an absent/empty one leaves the entry
        byte-identical to a pre-``display_name`` resolution."""
        if isinstance(display_name, str) and display_name:
            entry["display_name"] = display_name
        return entry

    def _zero_migration_entry(
        name: str, index: int, persona: str | None, display_name: Any
    ) -> dict[str, Any]:
        """The zero-migration resolution: today's exact pre-persona values, with
        ``persona`` carried through for logging (``None`` when no persona is in
        effect, or the unresolvable reference on the lenient degrade path) and the
        optional ``display_name`` passed through unchanged. ``extra_mounts`` is
        empty here — the zero-migration path has no catalog entry to read
        persona-level host mounts from. ``seed_base`` is ``True`` — the shared
        base-context prepend has always been mandatory for a
        no-persona/zero-migration entry, and opting out is only expressible
        through a catalog entry."""
        return _with_display_name(
            {
                "name": name,
                "index": index,
                "persona": persona,
                "image": default_image,
                "project": default_project,
                "container_project_dir": default_container_dir,
                "extra_mounts": [],
                "seed_base": True,
            },
            display_name,
        )

    resolved: list[dict[str, Any]] = []
    for entry in normalized:
        name = entry["name"]
        index = entry["index"]
        display_name = entry.get("display_name")
        persona_ref = persona_ref_by_name.get(name) or default_persona_name

        if persona_ref is None:
            # No persona system in effect for this entry — zero-migration path.
            resolved.append(_zero_migration_entry(name, index, None, display_name))
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
            resolved.append(_zero_migration_entry(name, index, persona_ref, display_name))
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

        # seed_base: whether this persona's users get the shared base context
        # prepended ahead of their own extra context at seed time. Defaults to
        # True (the historical, always-prepend behavior); a non-bool value is a
        # config typo that lint reports separately, so coerce it back to the
        # safe default here rather than propagating garbage.
        seed_base = catalog_entry.get("seed_base")
        if not isinstance(seed_base, bool):
            seed_base = True

        is_default = persona_ref == default_persona_name

        if image_source == "local":
            image = f"{project}-{persona_ref}:local"
        elif is_default:
            image = default_image
        else:
            image = f"{registry_url}/web-terminal-{persona_ref}:{image_tag}"

        container_project_dir = f"/app/{project}"

        resolved.append(
            _with_display_name(
                {
                    "name": name,
                    "index": index,
                    "persona": persona_ref,
                    "image": image,
                    "project": project,
                    "container_project_dir": container_project_dir,
                    "extra_mounts": extra_mounts,
                    "seed_base": seed_base,
                },
                display_name,
            )
        )

    return resolved
