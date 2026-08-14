"""Roster and persona identity resolution for multi-user web terminal deployments.

Turns the raw ``modules.web_terminals`` roster into explicit per-user identity:
normalized ``{"name", "index"}`` entries (:func:`normalize_users`) and fully
resolved image/project/container-dir identity per user
(:func:`resolve_personas`). Callers that write the roster *back* to
``config.yml`` rather than render from it use :func:`freeze_user_indices`, which
keeps the authored keys the normalizer projects away. Also home to the
username→env-var-suffix mapping (:func:`env_var_suffix`) and its collision detector
(:func:`env_var_suffix_collisions`), which credential provisioning, the auth
sidecar and lint share so a user's credentials are keyed identically everywhere.
Port arithmetic lives separately in :mod:`osprey.deployment.web_terminals.ports`.
"""

import os
import re
from collections.abc import Iterable
from typing import Any

# Matches ${VAR} and $VAR env references inside modules.web_terminals.image_tag.
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

# The only wired value for `modules.web_terminals.mcp.topology`. Every
# other value (including the recognized-but-rejected `shared_http`) is
# fail-closed at render time — see render.py's `_check_mcp_topology()`. Lives
# in this neutral module so lint.py and render.py can both import it without
# one depending on the other. This key is scoped to the shared
# framework-MCP tier only; it has no bearing on a facility's own
# `claude_code.servers` custom entries (those render through the unrelated
# per-project `.mcp.json` pipeline, untouched by this module).
SUPPORTED_MCP_TOPOLOGY = "per_container_stdio"

# Usernames become nginx `location` keys and URL path segments (`/<user>/...`), so
# they're held to a stricter charset than a bare "no reserved collision" check.
# Public and defined here, alongside `env_var_suffix`, because this module owns
# what a roster username *is*: lint's scaffold-time rule, render's fail-closed
# gate and `auth_credentials`' deploy-time gate all import it from here, so the
# three cannot drift apart.
#
# Apply it with `.fullmatch()`, never `.match()`: Python's `$` also matches
# *before* a trailing newline, so `.match()` accepts "alice\n" — a name that
# goes on to render into an nginx location key mid-directive.
USERNAME_CHARSET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


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
    ports stay identical to what an all-strings roster produces.
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

    An object entry's optional ``theme`` (that user's default web UI theme --
    a theme family like ``desy`` or a concrete id like ``desy-light``, surfaced
    downstream as the per-user ``OSPREY_WEB_THEME``) is carried through on
    exactly the same terms. Validity of the *value* is not checked here: the
    web terminal resolves it at startup and warns+falls back on an unknown one,
    and lint reports a non-string separately.

    An object entry's optional ``oidc_subject`` (the value of the configured OIDC
    claim -- ``sub`` by default -- that identifies this roster user at the IdP) is
    carried through on the same terms, with one deliberate difference: an *empty*
    string is dropped along with every non-string, where ``display_name`` and
    ``theme`` would keep it. This field is an authorization mapping, not a
    cosmetic one -- carrying ``""`` through would let an identity whose claim is
    missing or empty match a roster user. Dropping it instead leaves that user
    with no mapping at all, which the callback answers with 403. Only the
    *non-secret* side of the mapping ever lives in config.yml; password hashes
    never do (they live in ``.env.auth``, keyed by :func:`env_var_suffix`).

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
        New ``{"name": str, "index": int}`` dicts (plus optional
        ``"display_name"``, ``"theme"`` and ``"oidc_subject"`` string keys when
        the entry carried them) in config-declaration order. Input dicts are
        never mutated or returned by reference.
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
                theme = entry.get("theme")
                if isinstance(theme, str):
                    normalized_entry["theme"] = theme
                oidc_subject = entry.get("oidc_subject")
                if isinstance(oidc_subject, str) and oidc_subject:
                    normalized_entry["oidc_subject"] = oidc_subject
                normalized.append(normalized_entry)
    return normalized


def freeze_user_indices(users_raw: Any) -> list[dict[str, Any]]:
    """The roster **as authored**, with :func:`normalize_users`' indices frozen onto it.

    :func:`normalize_users` answers "who is on the roster, at which index" and
    deliberately projects each entry down to the handful of fields the render
    reads. That projection is the right input for rendering and the WRONG thing
    to write back to ``config.yml``: every key it does not know about —
    ``persona`` above all — would be dropped from the file, and a roster whose
    ``persona:`` keys are gone re-resolves every survivor onto
    ``default_persona`` (see :func:`resolve_personas`, which reads ``persona``
    off the raw roster). For a user removal that is a silent privilege change,
    not a cosmetic one, whenever the default persona is the more privileged.

    So this function starts from :func:`normalize_users`' output — survival and
    index-freezing stay entirely that function's contract, with no second copy
    of its rules here — and re-attaches the keys the author actually wrote.
    Entries are matched by ``name``, the same key
    :func:`_persona_ref_by_name` and every other per-user artifact in this
    module (container names, volume names) is keyed by. A roster listing one
    name twice (a duplicate-name config error lint reports separately) therefore
    gives both entries the *last* authored entry's extra keys; each still keeps
    its own frozen index.

    Values are carried through verbatim, including ones
    :func:`normalize_users` would drop as malformed (a non-string
    ``display_name``, say). Removing one user must not quietly rewrite another
    user's config: what the author wrote about the survivors goes back to the
    file unchanged, and lint keeps reporting it.

    Args:
        users_raw: The raw ``modules.web_terminals.users`` value.

    Returns:
        One dict per surviving :func:`normalize_users` entry, in the same order,
        carrying every key its authored entry had plus an explicit ``index``. A
        bare-string entry becomes ``{"name", "index"}``, since a string carries
        nothing else — unless the same name is also spelled as an object
        somewhere in the list, which is the duplicate-name case above. Input
        dicts are never mutated.
    """
    authored_by_name: dict[str, dict[str, Any]] = {}
    if isinstance(users_raw, list):
        for entry in users_raw:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                authored_by_name[entry["name"]] = entry

    frozen: list[dict[str, Any]] = []
    for normalized in normalize_users(users_raw):
        entry = dict(authored_by_name.get(normalized["name"], {}))
        # normalize_users wins on the keys it owns: `index` is the frozen one,
        # not whatever the file said, and `name` is already identical.
        entry.update(normalized)
        frozen.append(entry)
    return frozen


def env_var_suffix(username: str) -> str:
    """Map a roster username to the suffix its per-user env vars are keyed by.

    Uppercase, with ``-`` replaced by ``_`` — so ``alice-b`` keys
    ``OSPREY_AUTH_PW_HASH_ALICE_B``. This is the single definition of that
    mapping; credential provisioning, the sidecar's env lookup, and lint all
    route through it so a username can never be keyed one way at mint time and
    another at verify time.

    The mapping is intentionally total and lossy: it neither validates the
    username charset nor rejects anything. Two distinct usernames can therefore
    collide onto one suffix (``alice-b`` and ``alice_b``), which is exactly what
    :func:`env_var_suffix_collisions` exists to detect — enforcement is the
    caller's (a hard raise on the deploy preflight path, an ERROR in lint), not
    this function's.
    """
    return username.upper().replace("-", "_")


def env_var_suffix_collisions(usernames: Iterable[str]) -> dict[str, list[str]]:
    """Find roster usernames that :func:`env_var_suffix` maps onto one suffix.

    Without this check ``alice-b`` and ``alice_b`` would silently share a single
    ``OSPREY_AUTH_PW_HASH_ALICE_B`` entry — one user's password would open the
    other's terminal, which is precisely the isolation the auth feature exists to
    establish.

    A username repeated verbatim in the roster is *not* a collision here: it is
    one user listed twice (a duplicate-name config error reported separately),
    not two users sharing a credential. Only distinct names count.

    Args:
        usernames: Roster usernames — typically ``entry["name"]`` for each
            :func:`normalize_users` entry. Non-string items are ignored, matching
            this module's drop-don't-raise convention.

    Returns:
        ``{suffix: [colliding usernames]}`` for suffixes claimed by two or more
        distinct usernames; empty when the roster is unambiguous. Suffix keys and
        the names under each are sorted, so a lint or preflight message built
        from this is byte-stable across runs.
    """
    by_suffix: dict[str, set[str]] = {}
    for username in usernames:
        if isinstance(username, str):
            by_suffix.setdefault(env_var_suffix(username), set()).add(username)
    return {suffix: sorted(names) for suffix, names in sorted(by_suffix.items()) if len(names) > 1}


def roster_user_names(web_terminals: Any) -> list[str]:
    """The web-terminal user names a ``modules.web_terminals`` subtree declares.

    The whole rule in one place: a module that is not switched on has no roster
    at all, and the entries of one that is are read through
    :func:`normalize_users` rather than off the raw list. Both matter to callers
    who then create directories per user — ``osprey init`` seeding a
    per-user context slot and the build copying into it must agree on the roster
    down to the last name, or the build looks for a directory nobody made.

    Deliberately takes the *subtree*, not a whole config: its two callers reach
    it by different routes (a profile's ``config:`` block, which needs
    ``effective_web_terminals`` to fold the dotted keys; a built project's
    ``config.yml``, which is already nested), and only what they do with it
    afterwards is shared.

    Args:
        web_terminals: The ``modules.web_terminals`` subtree. Anything that is
            not a mapping is an absent module, so the roster is empty.

    Returns:
        User names in roster order; empty for a disabled or absent module.
    """
    subtree = as_dict(web_terminals)
    if not subtree.get("enabled"):
        return []
    return [entry["name"] for entry in normalize_users(subtree.get("users"))]


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

    * **No persona in effect** (``persona`` is ``None``): the facility defaults —
      ``image`` is ``<registry_url>/web-terminal:<tag>`` (unsuffixed, the same
      string the compose template names directly whenever ``<tag>`` is its
      ``latest`` default),
      ``project`` and ``container_project_dir`` are ``<facility_prefix>-assistant``
      / ``/app/<facility_prefix>-assistant``. This is the zero-migration path: a
      config with no ``personas`` catalog at all resolves every entry here.
    * **Default persona** (resolved ``persona`` equals ``default_persona``, and
      a catalog entry exists for it): registry mode keeps the same un-suffixed
      ``<registry_url>/web-terminal:<tag>`` image (so the default persona's
      *image* never changes when a catalog is introduced); local mode still
      builds ``<persona.project>-<persona>:local`` like every other persona.
      ``container_project_dir`` is ``/app/<project>`` from the catalog entry,
      exactly as for any other persona: the image is built FROM that project, so
      pinning the directory to the facility default would name a path that
      image does not have. A catalog that gives the default persona a project
      other than ``<facility_prefix>-assistant`` therefore moves where its
      users' agent-data volume mounts — the volume itself is unchanged and
      keeps its contents, but they are no longer at the path the container
      reads.
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
        seeding this entry's ``CLAUDE.md``. Optional ``"display_name"`` and
        ``"theme"`` keys are added — carried through from
        :func:`normalize_users` — only when the entry declared a non-empty string
        one (render emits them as ``OSPREY_WEB_APP_NAME`` and
        ``OSPREY_WEB_THEME``); each is omitted entirely otherwise, so a roster
        declaring neither resolves byte-identically to before these fields
        existed. An optional ``"oidc_subject"`` key rides through on the same
        terms, so the auth sidecar's roster→identity mapping is read off the same
        resolved entry as everything else rather than re-derived from the raw
        roster.

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

    def _with_optional_fields(entry: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
        """Attach the optional per-user fields to a resolved entry, mirroring
        render.py's conditional-``sublabel`` convention: a key is present only
        for a non-empty string, so a roster declaring none leaves the entry
        byte-identical to a resolution from before these fields existed."""
        for field in ("display_name", "theme", "oidc_subject"):
            value = source.get(field)
            if isinstance(value, str) and value:
                entry[field] = value
        return entry

    def _zero_migration_entry(
        name: str, index: int, persona: str | None, source: dict[str, Any]
    ) -> dict[str, Any]:
        """The zero-migration resolution: the pre-persona values, with
        ``persona`` carried through for logging (``None`` when no persona is in
        effect, or the unresolvable reference on the lenient degrade path) and the
        optional per-user fields passed through unchanged. ``extra_mounts`` is
        empty here — the zero-migration path has no catalog entry to read
        persona-level host mounts from. ``seed_base`` is ``True`` — the shared
        base-context prepend is mandatory for a no-persona/zero-migration entry,
        and opting out is only expressible through a catalog entry."""
        return _with_optional_fields(
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
            source,
        )

    resolved: list[dict[str, Any]] = []
    for entry in normalized:
        name = entry["name"]
        index = entry["index"]
        persona_ref = persona_ref_by_name.get(name) or default_persona_name

        if persona_ref is None:
            # No persona system in effect for this entry — zero-migration path.
            resolved.append(_zero_migration_entry(name, index, None, entry))
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
            resolved.append(_zero_migration_entry(name, index, persona_ref, entry))
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
        # True (always prepend); a non-bool value is a
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
            _with_optional_fields(
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
                entry,
            )
        )

    return resolved
