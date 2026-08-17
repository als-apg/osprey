"""``.env.users`` generation for multi-user web-terminal deploys.

Local-mode deploys generate ``.env.users`` from the deployment's env chain
(``.env.shared`` then ``.env``, later winning — a filtered subset: runtime
credentials in, build/CI-only variables out); registry-mode deploys only
exists-check it (CI is expected to have produced it). Called from
:func:`osprey.deployment.web_terminals.provision.deploy_up_web_terminals`.
"""

import os
import re
from pathlib import Path

import yaml

from osprey.cli.output import report_fact
from osprey.deployment.errors import ComposeInterpolationError
from osprey.deployment.web_terminals.personas import effective_image_source
from osprey.utils.dotenv import (
    ENV_CHAIN_FILENAMES,
    ENV_LOCAL_FILENAME,
    chain_files,
    compose_unsafe_vars,
    format_env_line,
    merge_chain,
    parse_dotenv_file,
)
from osprey.utils.logger import get_logger

logger = get_logger("deployment.lifecycle")

#: The env file every per-user web-terminal container runs with. Named for who
#: reads it (the users' containers) rather than for a deployment mode, since a
#: local development deploy generates the same file a production one does.
USERS_ENV_FILENAME = ".env.users"

#: The name this artifact used to carry. Kept only so :func:`migrate_users_env`
#: can move an existing file onto the current name; nothing else reads it, and
#: nothing ever writes it again.
LEGACY_USERS_ENV_FILENAME = ".env.production"


def migrate_users_env(project_root: str | Path) -> Path | None:
    """Move a leftover ``.env.production`` onto :data:`USERS_ENV_FILENAME`.

    Called once per ``osprey up``, before anything reads the file (see
    ``_start_stack``). The artifact is gitignored and never regenerated when
    present, so a deployment that predates the rename carries the operator's
    only copy of the web tier's runtime secrets under the old name — losing it
    would silently fall through to "registry-mode deploys expect this file"
    (see :func:`ensure_env_production`) on a host that had a working deploy
    minutes earlier.

    Both files can exist at once, and which one is authoritative is not a
    guess: registry-mode CI renders a fresh ``.env.users`` on the host just
    before ``osprey up``, while the old file — gitignored, so untouched by the
    ``git reset --hard`` that precedes a CI checkout — survives from an earlier
    pipeline. The new file is therefore always kept, and the leftover is
    deleted rather than left to look like a second source of truth. Either way
    the operator gets one line naming both paths.

    A moved file lands at mode ``0600`` whatever it carried before, matching
    every other write of this artifact.

    :param project_root: Project root directory holding both files.
    :return: Path to ``.env.users`` when a file was moved or a leftover
        removed, ``None`` when there was no old file to act on.
    :raises OSError: When something that is not a regular file already occupies
        the new name; the old file is left where it is.
    """
    root = Path(project_root)
    legacy_path = root / LEGACY_USERS_ENV_FILENAME
    users_path = root / USERS_ENV_FILENAME
    if not legacy_path.is_file():
        return None

    # is_file, not exists: deleting the operator's only copy of the web tier's
    # secrets is justified by a real file standing in its place and by nothing
    # else. Anything else at that name (a directory, a dangling symlink) falls
    # through to os.replace below, which fails loudly with the leftover intact.
    if users_path.is_file():
        legacy_path.unlink()
        report_fact(
            logger,
            f"removed leftover {LEGACY_USERS_ENV_FILENAME} ({USERS_ENV_FILENAME} is the "
            "current name and already exists)",
        )
        return users_path

    # os.replace, not shutil.move: an atomic same-directory rename, so the
    # secrets never exist under two names at once.
    os.replace(legacy_path, users_path)
    # Enforced, not inherited: the mode travels with the rename, and a file
    # written before this artifact was created at 0600 (or hand-authored at the
    # umask) would otherwise arrive under the new name still world-readable.
    # Every other write of this file lands at 0600; a migrated one does too.
    os.chmod(users_path, 0o600)
    report_fact(
        logger,
        f"{LEGACY_USERS_ENV_FILENAME} renamed → {USERS_ENV_FILENAME}",
        wrote=(
            USERS_ENV_FILENAME,
            f"renamed from {LEGACY_USERS_ENV_FILENAME}; the web terminals' runtime "
            "secrets live under this name now",
        ),
    )
    return users_path


#: The two forms a config value may use to reference an env var: ``${NAME}``
#: and ``${NAME:-default}``. Anchored, because the values this module reads are
#: reference LITERALS — a whole config value that is nothing but the reference —
#: and a partial match would report a variable the reader never actually
#: depends on. The unbraced ``$NAME`` spelling is deliberately NOT matched:
#: nothing this module reads is written that way, and treating it as a
#: reference would make any value containing a bare ``$`` look like one.
_ENV_REFERENCE_RE = re.compile(r"\A\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}\Z")

#: Where a project's telemetry credentials live in its own ``config.yml``.
_TELEMETRY_CONFIG_PATH = ("claude_code", "telemetry", "openobserve")

#: The observability store's account NAME — an email address, not a secret.
#: A fixed name rather than a config-declared one, like ``TZ`` and
#: ``ARIEL_DSN``: the shipped configs reference it under this spelling and
#: ``osprey up`` writes it into the deploy env chain under it too.
_TELEMETRY_USER_ENV_VAR = "ZO_ROOT_USER_EMAIL"


def _env_reference(value: object) -> tuple[str, bool] | None:
    """Read a config value that IS an env-var reference.

    The telemetry credential keys hold the reference itself (``user:
    ${ZO_ROOT_USER_EMAIL:-root@example.com}``) rather than the *name* of a
    variable the way ``llm.api_key_env_var`` does, so telling a reference from
    a plain literal — and a bare reference from one carrying its own default —
    takes reading the value, which is what this does.

    The distinction matters because the two forms fail differently in a
    container: a reference with a default quietly falls back to it when the
    variable is unset, while a bare one is left in the config verbatim (see
    :func:`osprey_connectors.config.resolve_env_vars`) and reaches the store as
    a literal ``${...}`` string.

    :param value: A raw config value; anything that is not a string, and any
        string that is not exactly one reference, reads as a plain literal.
    :return: ``(var_name, has_default)``, or ``None`` for a plain literal.
    """
    if not isinstance(value, str):
        return None
    match = _ENV_REFERENCE_RE.match(value)
    if match is None:
        return None
    return match.group(1), match.group(2) is not None


def _telemetry_credentials(cfg: dict) -> dict:
    """The ``claude_code.telemetry.openobserve`` block of one project's config."""
    node: object = cfg
    for key in _TELEMETRY_CONFIG_PATH:
        if not isinstance(node, dict):
            return {}
        node = node.get(key)
    return node if isinstance(node, dict) else {}


def _referenced_persona_names(config: dict) -> list[str]:
    """Every persona name this roster runs: the default plus each user's own.

    Sorted, so everything derived from the roster reports in a stable order.
    Names are taken as written; whether the catalog knows one is a separate
    question (see :func:`_referenced_persona_entries`).
    """
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    referenced: set[str] = set()
    default_persona = web_terminals.get("default_persona")
    if isinstance(default_persona, str) and default_persona:
        referenced.add(default_persona)
    users = web_terminals.get("users")
    for user in users if isinstance(users, list) else []:
        if isinstance(user, dict) and isinstance(user.get("persona"), str) and user["persona"]:
            referenced.add(user["persona"])
    return sorted(referenced)


def _referenced_persona_entries(config: dict) -> list[tuple[str, dict]]:
    """``(persona_name, catalog entry)`` for every persona this roster runs.

    Referenced names resolved against ``modules.web_terminals.personas``; a
    name with no catalog entry contributes nothing here.
    """
    catalog = ((config.get("modules") or {}).get("web_terminals") or {}).get("personas")
    catalog = catalog if isinstance(catalog, dict) else {}
    entries: list[tuple[str, dict]] = []
    for persona_name in _referenced_persona_names(config):
        entry = catalog.get(persona_name)
        if isinstance(entry, dict):
            entries.append((persona_name, entry))
    return entries


def _persona_config_yml(project_root: Path, entry: dict) -> Path | None:
    """Where a catalog entry's rendered ``config.yml`` would be.

    Path join: an absolute ``project_path`` stands on its own; a relative one
    resolves against the deploy project root, same as every other cwd-relative
    assumption on this path. An entry naming no project has no config to read.
    """
    project_path_raw = entry.get("project_path")
    if not isinstance(project_path_raw, str) or not project_path_raw:
        return None
    return Path(project_root, project_path_raw) / "config.yml"


def _load_config_yml(path: Path) -> dict | None:
    """Load a rendered ``config.yml``; ``None`` when it cannot be read as one."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _telemetry_credential_references(
    config: dict, project_root: Path, field: str, *, bare_only: bool
) -> dict[str, str]:
    """``{var: origin}`` for one telemetry credential key across every config in play.

    The deploy config and each referenced persona project's rendered
    ``config.yml`` are read the same way :func:`_claude_code_auth_secret_vars`
    reads them, since a per-user container runs its persona's project and it is
    that project's telemetry block which decides what the agent inside presents
    to the observability store.

    :param field: Key under ``claude_code.telemetry.openobserve`` to read.
    :param bare_only: When true, report only references with no ``:-default``
        of their own — the ones that cannot resolve to anything on their own.
    """
    references: dict[str, str] = {}

    def _record(cfg: dict, source: str) -> None:
        reference = _env_reference(_telemetry_credentials(cfg).get(field))
        if reference is None:
            return
        var, has_default = reference
        if bare_only and has_default:
            return
        references.setdefault(var, f"claude_code.telemetry.openobserve.{field} {source}")

    for persona_name, entry in _referenced_persona_entries(config):
        config_yml = _persona_config_yml(project_root, entry)
        if config_yml is None or not config_yml.is_file():
            # Silent, unlike _claude_code_auth_secret_vars: that function warns
            # about the same unreadable project already, and saying it twice
            # per deploy would read as two separate problems.
            continue
        persona_config = _load_config_yml(config_yml)
        if persona_config is None:
            continue
        _record(persona_config, f"(persona {persona_name!r})")

    _record(config, "(deploy config)")
    return references


def _telemetry_credential_requirements(config: dict, project_root: Path) -> dict[str, str]:
    """``{var: origin}`` for telemetry passwords a config REQUIRES from the env chain.

    A bare ``${VAR}`` password reference — no ``:-default`` — is a config
    asserting that the variable is set: unset, it is left in the rendered
    config verbatim and the agent authenticates to the observability store with
    a literal ``${VAR}`` string. This reports those variables so
    :func:`ensure_env_production` can refuse a deploy that would ship one,
    exactly as it refuses a missing provider auth secret.

    The result feeds the missing-variable gate ONLY. It must never reach
    :func:`_build_env_production_subset`: the store password is the store's
    single admin credential, and ``.env.users`` is handed to every persona
    alike, so copying it there would grant every read-only terminal admin read
    of every transcript in the store.
    """
    return _telemetry_credential_references(config, project_root, "password", bare_only=True)


def _telemetry_user_references(config: dict, project_root: Path) -> dict[str, str]:
    """``{var: origin}`` for the telemetry account-name variables a config references.

    Both reference forms count here, unlike
    :func:`_telemetry_credential_requirements`: a defaulted reference resolves
    to the shipped placeholder address, which is the wrong account whenever the
    deploy configured the store under a different one.
    """
    return _telemetry_credential_references(config, project_root, "user", bare_only=False)


def _copy_named_env_var(var_name: str | None, source: dict[str, str], dest: dict[str, str]) -> None:
    """Copy ``source[var_name]`` into ``dest[var_name]`` iff both are present.

    ``var_name`` is itself a config-declared *name* (e.g. ``llm.api_key_env_var``
    resolves to ``"CBORG_API_KEY"``), not a literal value — the indirection
    every external-credential entry in :func:`_build_env_production_subset`
    goes through, since the config records where a facility keeps a secret and
    never the secret itself. A ``var_name`` that is unset (module
    misconfigured) or absent from ``source`` (operator never set it) is
    silently skipped — never fabricated, matching every other var-presence
    check in this module.
    """
    if not var_name or var_name not in source:
        return
    dest[var_name] = source[var_name]


def _claude_code_auth_secret_vars(
    config: dict, project_root: Path
) -> tuple[dict[str, str], dict[str, str]]:
    """Auth-secret env-var names every ``claude_code.provider`` in play needs.

    This is the web-terminal counterpart of the launch-time secret injection
    in :mod:`osprey.build.claude_code_resolver`: a per-user web container runs
    its persona project's agent, which authenticates via the provider named in
    that project's ``claude_code.provider`` — and the *only* env its container
    sees is ``docker-compose.web.yml``'s ``env_file: .env.users``. A
    generated ``.env.users`` that misses the provider's secret var
    produces terminals that come up healthy and fail authentication on the
    first prompt.

    Returns two ``{var_name: origin}`` dicts (origin is a human-readable
    source description for error messages):

    - **required** — vars some deployed web container actually authenticates
      with: each referenced persona project's provider (persona catalogs), or
      the deploy config's own provider when no persona catalog is configured
      (the zero-migration path, where the web image is the facility project
      itself).
    - **extra** — vars worth *copying* when present but not worth failing
      over: the deploy config's own provider when a persona catalog is in
      play (per-user containers run persona projects, not the deploy
      project).

    Referenced personas whose ``project_path`` isn't rendered or readable yet
    contribute nothing — a broken catalog entry is lint's / strict
    ``resolve_personas``'s error to report, and
    :func:`verify_persona_renders` REFUSES a deploy whose persona projects are
    missing *before* :func:`ensure_env_production` runs, so on every deploy path
    that reaches generation the rendered configs are on disk. A provider name known
    neither to ``CLAUDE_CODE_PROVIDERS`` nor to the config's own
    ``api.providers`` is likewise skipped here (the resolver raises its own
    actionable error for that at launch).
    """
    from osprey.build.claude_code_resolver import provider_auth_secret_env

    def _provider_var(cfg: dict) -> tuple[str, str | None] | None:
        provider = (cfg.get("claude_code") or {}).get("provider")
        if not isinstance(provider, str) or not provider:
            return None
        api_providers = (cfg.get("api") or {}).get("providers")
        if not isinstance(api_providers, dict):
            api_providers = None
        return provider, provider_auth_secret_env(provider, api_providers)

    catalog = ((config.get("modules") or {}).get("web_terminals") or {}).get("personas")
    catalog = catalog if isinstance(catalog, dict) else {}
    referenced = _referenced_persona_names(config)
    entries = _referenced_persona_entries(config)

    required: dict[str, str] = {}
    extra: dict[str, str] = {}

    for persona_name, entry in entries:
        config_yml = _persona_config_yml(project_root, entry)
        if config_yml is None:
            continue
        if not config_yml.is_file():
            # Said out loud rather than skipped: a persona whose project cannot
            # be found contributes no auth secret, so its terminals come up
            # healthy and fail authentication on the first prompt — the exact
            # failure this function exists to prevent. Silence here reads as
            # "that persona needs nothing", which is a claim we cannot make.
            logger.warning(
                "Persona %r: no config.yml at %s, so its provider's auth secret "
                "cannot be determined and will not be included in .env.users. "
                "Terminals running this persona may fail authentication.",
                persona_name,
                config_yml,
            )
            continue
        persona_config = _load_config_yml(config_yml)
        if persona_config is None:
            continue
        resolved = _provider_var(persona_config)
        if resolved is None:
            continue
        provider, var = resolved
        if var and var not in required:
            required[var] = f"claude_code.provider {provider!r} (persona {persona_name!r})"

    own = _provider_var(config)
    if own is not None:
        provider, var = own
        if var and var not in required:
            origin = f"claude_code.provider {provider!r} (deploy config)"
            if catalog and referenced:
                extra[var] = origin
            else:
                required[var] = origin

    return required, extra


def _build_env_production_subset(
    config: dict,
    dotenv: dict[str, str],
    claude_code_secret_vars: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the module-conditional subset written into ``.env.users``.

    ``.env.users`` is the env file every per-user web-terminal container
    runs with (``docker-compose.web.yml``'s ``env_file:``), so this function
    *is* the definition of which secrets a web terminal is entitled to see —
    the list below is the specification, not a restatement of one kept
    elsewhere. Both routes that produce the file come through here — the
    deploy path (:func:`ensure_env_production`) and ``osprey users env``,
    which renders this same subset to stdout or a named file — so neither can drift from the other or introduce a second
    spec. What earns a place is a credential the agent inside that container
    presents to a system OUTSIDE the deploy:

    - ``llm.api_key_env_var`` — the LLM provider key, unconditional.
    - ``claude_code_secret_vars`` — the auth-secret vars resolved by
      :func:`_claude_code_auth_secret_vars` (the ``claude_code.provider``
      of the deploy config and of every referenced persona project), passed
      in by :func:`ensure_env_production`. Same chain-presence rule as
      every other entry; whether an *absent* var is an error is
      :func:`ensure_env_production`'s call, not this function's.
    - ``modules.olog.{username,password}_env_var`` — the electronic-logbook
      account, only if ``modules.olog.enabled``.
    - ``modules.wiki_search.token_env_var`` — the wiki credential, only if
      ``modules.wiki_search.enabled``.
    - ``ARIEL_DSN`` — only if ``modules.ariel.enabled``, from
      ``modules.ariel.dsn`` directly. Unlike every entry above this is read
      from ``config``, not ``dotenv``: the DSN is itself a literal config
      value, not the *name* of an env var holding one.
    - ``ZO_ROOT_USER_EMAIL`` — the observability store's account NAME, copied
      when the chain sets it and silently skipped otherwise. A fixed key rather
      than a config-declared one, like the two entries below: the shipped
      telemetry block references it under this spelling and ``osprey up``
      writes it into the deploy env chain under it too. An account name is not
      a secret; it is here because a terminal that emits telemetry has to name
      the same account the store was configured with, and the reference's own
      fallback address is the wrong one on any deploy that set this.
    - ``TZ`` — always, from ``facility.timezone`` (default ``"UTC"``, matching
      the schema's own documented default), likewise a literal config value.

    The store's PASSWORD is deliberately not here beside its account name.
    ``ZO_ROOT_USER_PASSWORD`` is the observability store's single admin
    credential — the same value the store container itself is configured with —
    so a copy in this rosterwide file would grant every persona, read-only ones
    included, admin read of every transcript the store holds. It stays out for
    the same reason the service tokens do (see below), and
    :func:`_telemetry_credential_requirements` exists to REPORT a config that
    depends on it rather than to satisfy one.

    NEVER included, by construction (this function never reads them at all):
    build-time credentials — the CI provider token, the container-registry
    login, every external-project pull token — and the tokens OSPREY's own
    deployed services authenticate to each other with
    (``EVENT_DISPATCHER_TOKEN``, ``DISPATCH_WORKER_TOKEN``,
    ``BLUESKY_LAUNCH_TOKEN`` and the rest of ``_SERVICE_TOKEN_VARS`` in
    :mod:`osprey.deployment.container_lifecycle`, minted per deploy under
    those fixed names). Build-time credentials are nothing a web terminal
    presents to anyone; the containers that need a service token read the deploy
    ``.env`` the main compose file hands them.

    The reason the SERVICE tokens are excluded is narrower, and it is about this
    file rather than about the tokens: ``.env.users`` is a single file
    handed to EVERY per-user container alike. It cannot say "alice but not bob",
    so anything placed here is granted to every persona in the roster —
    including read-only ones, whose entire purpose is not to hold write-capable
    credentials. A per-user entitlement therefore has to be expressed somewhere
    that can distinguish users, and that is the per-user ``environment:`` block
    in ``docker-compose.web.yml``. See
    :func:`osprey.deployment.web_terminals.render.render_web_terminals`, whose
    ``dispatcher_personas``, ``ariel_personas`` and ``launch_token_personas``
    arguments each carry the subset of the roster entitled to one credential —
    ``EVENT_DISPATCHER_TOKEN``, ``ARIEL_DB_PASSWORD`` and
    ``BLUESKY_LAUNCH_TOKEN`` respectively — emitted into that user's own
    ``environment:`` block and interpolated by compose from the deploy ``.env``,
    so the secret never lands in a rendered artifact either.

    So this is NOT the claim that no web terminal ever presents a service token —
    the EVENTS panel's proxy presents the dispatcher token server-side, so the
    browser never holds it, and the ``bluesky`` MCP server presents the launch
    token to the bridge from inside the terminal container. It is the narrower
    and still-load-bearing claim that no
    service token is granted *rosterwide from here*. Note what that buys and
    what it does not: a container that receives one of these tokens shares its
    process namespace with the agent, which can read it, so every grant is a
    deliberate per-persona decision and never a default.

    ``BLUESKY_LAUNCH_TOKEN`` is the sharpest case, because it arms a queue start
    — an agent that holds it can put hardware in motion. Its entitlement
    predicate,
    :func:`osprey.deployment.web_terminals.personas.config_needs_launch_token`,
    requires BOTH ``control_system.writes_enabled: true`` AND an enabled
    ``bluesky`` MCP server in that persona's own rendered config. A read-only
    persona is spelled ``writes_enabled: false``, so it cannot satisfy that pair
    and can never be handed the token. That guarantee exists only because the
    grant is per-persona; a copy of the token in this file would hand it to
    exactly the personas the predicate is there to exclude.

    One nuance applies to all three credentials alike. A roster entry that names
    no persona — the zero-migration path, where the web image IS the deploy
    project — consults no persona set at all; the render answers it straight
    from the deploy config, via ``config_needs_launch_token``,
    ``config_needs_dispatcher_token`` or ``config_needs_ariel_password``. An
    empty persona set therefore does NOT mean "this credential is granted
    nowhere": persona-less entries are decided independently of it.

    This is the security spec for this function: a var absent from the
    enumerated list above can never appear in the returned dict, regardless of
    what the input env chain contains.

    :param config: Raw deploy config (facility fields merged in — see
        ``modules.web_terminals.image_source`` in :func:`ensure_env_production`).
    :param dotenv: The operator's env chain, already merged via
        :func:`osprey.utils.dotenv.merge_chain` — or a single named secrets
        file parsed via :func:`osprey.utils.dotenv.parse_dotenv_file`, when
        the caller was handed one instead.
    :return: The subset to write into ``.env.users``, in stable
        (insertion) order.
    """
    subset: dict[str, str] = {}

    llm = config.get("llm") or {}
    _copy_named_env_var(llm.get("api_key_env_var"), dotenv, subset)

    for var_name in claude_code_secret_vars or {}:
        _copy_named_env_var(var_name, dotenv, subset)

    modules = config.get("modules") or {}

    olog = modules.get("olog") or {}
    if olog.get("enabled"):
        _copy_named_env_var(olog.get("username_env_var"), dotenv, subset)
        _copy_named_env_var(olog.get("password_env_var"), dotenv, subset)

    wiki_search = modules.get("wiki_search") or {}
    if wiki_search.get("enabled"):
        _copy_named_env_var(wiki_search.get("token_env_var"), dotenv, subset)

    ariel = modules.get("ariel") or {}
    if ariel.get("enabled"):
        dsn = ariel.get("dsn")
        if dsn:
            subset["ARIEL_DSN"] = str(dsn)

    # Fixed key, chain-presence rule: the store's account name crosses, its
    # admin password never does (see the security spec above).
    _copy_named_env_var(_TELEMETRY_USER_ENV_VAR, dotenv, subset)

    facility = config.get("facility") or {}
    subset["TZ"] = str(facility.get("timezone") or "UTC")

    return subset


def users_env_generation_problem(config: dict, project_root: str | Path) -> str | None:
    """Whether generating ``.env.users`` would leave web terminals unauthenticated.

    The gate :func:`ensure_env_production` raises on, expressed as a question so
    it can also be ASKED — by the collect-all preflight, which reports every
    cheaply checkable refusal at once instead of costing the operator a deploy
    attempt per finding. One function rather than two, so the probe and the
    writer can never disagree about which variables are required or about how
    the refusal is worded.

    Answers ``None`` — nothing to report — for the three cases that are not this
    refusal, in the order :func:`ensure_env_production` settles them:

    * ``.env.users`` already on disk. An existing file is never regenerated, so
      there is no generation to have a problem with (whether its CONTENTS are
      adequate is :func:`_warn_if_env_production_lacks_credentials`' warning,
      not a refusal).
    * Registry mode. Nothing is generated there at all.
    * No env-chain file. There is nothing to generate FROM, which is its own
      refusal with its own remedy and not this one.

    Pure: it parses the env chain and each referenced persona's rendered
    ``config.yml``, and writes nothing anywhere. The ambient environment is read
    only for the presence check behind the shell hint — never for a value, and
    never as a source (see :func:`ensure_env_production` on why the chain on
    disk is the whole source).

    :param config: Raw deploy config.
    :param project_root: Project root; ``.env.users`` and the env-chain files
        are resolved relative to it.
    :return: The refusal sentence, or ``None`` when generation would succeed.
    """
    root = Path(project_root)
    users_env_path = root / USERS_ENV_FILENAME
    if users_env_path.is_file():
        return None

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if effective_image_source(web_terminals) != "local":
        return None

    sources = chain_files(root)
    if not sources:
        return None
    sources_desc = " + ".join(str(path) for path in sources)
    env_path = root / ENV_LOCAL_FILENAME

    dotenv = merge_chain(root)
    required_cc_vars, _extra_cc_vars = _claude_code_auth_secret_vars(config, root)
    # Reported, never copied: these are variables a telemetry block depends on
    # that this file is not allowed to carry, so they join the gate below and
    # nothing else. Keeping them out of the {**required, **extra} pair handed to
    # _build_env_production_subset is what stops the store's admin password from
    # being written into a file every persona reads.
    telemetry_vars = _telemetry_credential_requirements(config, root)

    # "Missing" means missing from the MERGED chain: a var only .env.shared
    # sets is set.
    missing = {
        var: origin
        for var, origin in {**telemetry_vars, **required_cc_vars}.items()
        if var not in dotenv
    }
    if not missing:
        return None

    needs = "; ".join(f"{origin} needs {var}" for var, origin in missing.items())
    # The chain on disk stays the only SOURCE (see ensure_env_production's
    # determinism note) — but when a missing var is sitting right there in the
    # shell, say so and hand over the exact copy-in command instead of leaving
    # the operator to discover the .env-only rule by archaeology. Presence
    # check only; the value itself is never read into the message.
    exported = [var for var in missing if os.environ.get(var)]
    shell_hint = ""
    if exported:
        # One store, one command. ``env_path`` is the deployment repo's own
        # ``.env`` at the repo root — source, not render — so appending to
        # it IS the durable write; there is no second, profile-side copy to
        # name and no rebuild needed to carry the value anywhere. (This
        # replaced a two-.env model where the project's ``.env`` was
        # derived from the profile's and a write to the wrong one was
        # dropped by the next build. Under the four-zone layout the
        # profile and the secret store share a root, so that distinction no
        # longer exists — and telling an operator their write will be
        # dropped would now be false.)
        names = ", ".join(exported)
        verb = "are" if len(exported) > 1 else "is"
        copy_cmds = " && ".join(f'echo "{var}=${var}" >> {env_path}' for var in exported)
        shell_hint = (
            f" Note: {names} {verb} exported in the current shell, but this "
            f"deploy reads only the env chain on disk ({sources_desc}) "
            "(generation never reads the ambient environment). Copy it in "
            f"with: {copy_cmds}"
        )
    # Named separately because the remedy is only half the usual one: the
    # chain is where these belong, but this file will still not carry them,
    # so an operator who adds one and expects it to reach the terminals has
    # to be told what actually happens instead.
    telemetry_missing = [var for var in missing if var in telemetry_vars]
    telemetry_note = ""
    if telemetry_missing:
        telemetry_names = ", ".join(telemetry_missing)
        telemetry_verb = "are" if len(telemetry_missing) > 1 else "is"
        telemetry_note = (
            f" Note: {telemetry_names} {telemetry_verb} the observability store's single admin "
            "credential, so this file never carries it — one .env.users is "
            "handed to every persona alike, read-only ones included, and admin "
            "access to that store reads every transcript in it. Setting it in "
            "the chain configures the store itself; a scoped ingest credential "
            "is what will let a terminal authenticate to the store. A telemetry "
            "block that names its own fallback (${VAR:-default}) is not asked "
            "for here at all."
        )
    return (
        f"Generating {users_env_path} from {sources_desc} would leave web "
        f"terminals unauthenticated: {needs}, set in none of them. Add the "
        f"missing variable(s) to {env_path}, or author .env.users "
        "yourself (an existing file is never regenerated) if this deploy "
        f"authenticates another way.{telemetry_note}{shell_hint}"
    )


def ensure_env_production(config: dict, project_root: str | Path) -> Path:
    """Ensure ``<project_root>/.env.users`` exists, generating it when possible.

    ``docker-compose.web.yml`` (see :func:`deploy_up_web_terminals`) declares
    ``env_file: .env.users`` unconditionally, so compose hard-fails before
    a single container starts if that file is missing. This resolves it up
    front, with different rules per ``modules.web_terminals.image_source``
    (default ``"registry"``):

    - **Already present** (either mode): returned as-is, untouched. This is
      always checked first, so an operator-authored or previously-generated
      file is never clobbered. When the config declares LLM credentials
      (``llm.api_key_env_var`` or any ``claude_code.provider`` in play — see
      :func:`_claude_code_auth_secret_vars`) and the existing file contains
      *none* of them, a warning names the missing var(s) — a stale file from
      before a provider change otherwise produces web terminals that fail
      authentication with nothing in the deploy output to say why.
    - **Registry mode, absent**: raises. Registry-mode deploys expect the file
      to have been rendered already by ``osprey users env``,
      which the emitted CI pipeline's deploy job runs
      on the host between ``osprey build`` and ``osprey up``. This
      function only exists-checks in that mode, it never generates, because
      there is no local env chain this system is licensed to treat as the
      authoritative source of a registry-mode deploy's secrets.
    - **Local mode, absent, an env-chain file present**: generated via
      :func:`_build_env_production_subset` (the module-conditional subset,
      including every ``claude_code`` auth secret resolved by
      :func:`_claude_code_auth_secret_vars`) and written with mode ``0600``
      from the moment the file is created — the same permission convention
      :func:`_ensure_service_tokens` uses for minted tokens. A *required*
      ``claude_code`` auth secret absent from the whole chain raises instead
      of generating: the resulting file would produce healthy-looking terminals
      that fail authentication on their first prompt (authoring
      ``.env.users`` directly remains the bypass for deploys that
      authenticate another way). A telemetry password reference that carries no
      default of its own (see :func:`_telemetry_credential_requirements`) joins
      the same gate: the variable it names is reported when the chain does not
      set it, and is never written into the generated file either way.
    - **Local mode, absent, the whole chain absent too**: raises, before any
      compose invocation — there is nothing to generate from and no file to
      fall back on.

    Values come from the merged env chain — :func:`osprey.utils.dotenv.merge_chain`
    reads ``.env.shared`` then ``.env``, so a key both files set takes the
    ``.env`` value and a key only the shared defaults carry is still delivered.
    The chain on disk is the whole source: the ambient process/shell
    environment is never read (unlike :func:`_ensure_service_tokens`'s
    ``_effective_value``), which keeps the generated file deterministic and
    independent of whatever happens to be exported in the caller's shell.

    :param config: Raw deploy config.
    :param project_root: Project root directory; ``.env.users`` and the
        env-chain files are all resolved relative to it.
    :return: Path to the existing or newly-generated ``.env.users``.
    :raises RuntimeError: per the absent-file rules above, with an actionable
        message naming the missing file(s) and how to resolve it.
    """
    root = Path(project_root)
    users_env_path = root / USERS_ENV_FILENAME
    if users_env_path.is_file():
        _warn_if_env_production_lacks_credentials(config, root, users_env_path)
        # Scan the file we are about to hand to every web terminal, not just
        # one this run generated. Registry mode -- the DEFAULT -- never reaches
        # the generator below at all: CI assembles .env.users from masked
        # variables and ships it beside the image. An operator-authored file
        # takes the same path, since an existing one is never regenerated. Both
        # carry values OSPREY never saw, which is exactly the case the
        # generate-path check cannot cover (same reasoning as
        # provision._raise_if_auth_env_would_be_interpolated).
        offenders = compose_unsafe_vars(parse_dotenv_file(users_env_path))
        if offenders:
            raise ComposeInterpolationError(offenders, users_env_path)
        return users_env_path

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if effective_image_source(web_terminals) != "local":
        raise RuntimeError(
            f"{users_env_path} not found. Registry-mode web-terminal deploys "
            "(modules.web_terminals.image_source: registry, the default) expect "
            "this file to have been rendered already -- `osprey users env` "
            "writes it, and the emitted CI pipeline's deploy "
            "job runs that on the host just before `osprey up`, which does "
            "not generate it in this mode. Either run `osprey users env` "
            "(or supply .env.users directly), or set "
            "modules.web_terminals.image_source: local to generate it from the "
            "env chain."
        )

    # The env chain, not the root .env alone: a key the committed defaults
    # carry (a proxy endpoint, a shared auth var) is as real a source for a
    # web terminal as one the host-local .env carries, and .env still wins on
    # any key both set. With no .env.shared on disk the chain is [.env] and
    # the merge is that file's own parse, byte for byte.
    sources = chain_files(root)
    if not sources:
        chain_names = ", ".join(ENV_CHAIN_FILENAMES)
        raise RuntimeError(
            f"Neither {users_env_path} nor an env-chain file ({chain_names}) "
            f"was found in {root}. Local-mode web-terminal deploys "
            "(modules.web_terminals.image_source: local) need one of them: create "
            ".env.users directly, or create .env so osprey up can derive the "
            "module-conditional subset of .env.users from it."
        )
    sources_desc = " + ".join(str(path) for path in sources)

    dotenv = merge_chain(root)
    required_cc_vars, extra_cc_vars = _claude_code_auth_secret_vars(config, root)

    # Unlike every optional module var above (silently skipped when absent —
    # see _copy_named_env_var), a missing claude_code auth secret means some
    # web container comes up healthy and fails authentication on its first
    # prompt, with nothing in the deploy output to say why. Fail HERE, before
    # any compose invocation, naming the exact var and both remedies. The
    # question and the sentence answering it both live in
    # users_env_generation_problem, so the collect-all preflight asks exactly
    # what this raises on rather than a second approximation of it.
    problem = users_env_generation_problem(config, root)
    if problem is not None:
        raise RuntimeError(problem)

    subset = _build_env_production_subset(config, dotenv, {**required_cc_vars, **extra_cc_vars})

    # Every value above is a verbatim copy out of the operator's env chain (or,
    # for ARIEL_DSN, straight out of facility config), and this file is handed to
    # every per-user web terminal as `env_file: .env.users`. A `$` in any
    # of them is interpolated away en route to the container. Checked before the
    # open() below, not after: a refused deploy must not leave a half-written
    # secrets file that a later run would mistake for one the operator authored
    # (an existing .env.users is never regenerated).
    offenders = compose_unsafe_vars(subset)
    if offenders:
        raise ComposeInterpolationError(offenders, users_env_path)

    # format_env_line, not a bare f-string: a value that needs quoting to survive
    # a re-read (leading/trailing whitespace, an embedded space or `#`) is
    # rendered so every .env parser downstream — ours, and whichever compose
    # implementation reads this env_file: — hands the container the value the
    # chain actually holds instead of a truncated one.
    lines = "".join(f"{format_env_line(key, value)}\n" for key, value in subset.items())
    # Create with mode 0600 from the FIRST byte on disk, not write-then-chmod:
    # write_text() would create the file at the process umask (typically
    # 0644) and write every secret before a later os.chmod tightened
    # permissions, leaving a window on a multi-user host where a co-tenant
    # could read it. os.open with O_CREAT + an explicit mode is atomic --
    # there is no instant the file exists at a wider mode.
    fd = os.open(users_env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(lines)
    # Belt-and-suspenders: also covers the file already existing (e.g. a
    # leftover from a prior run) with a wider mode O_CREAT wouldn't have
    # reset on its own.
    os.chmod(users_env_path, 0o600)

    report_fact(
        logger,
        f"Generated {users_env_path} from {sources_desc} (mode 0600): {', '.join(subset)}",
    )

    return users_env_path


def _warn_if_env_production_lacks_credentials(
    config: dict, project_root: Path, users_env_path: Path
) -> None:
    """Warn when an existing ``.env.users`` is missing credentials the config names.

    The never-clobber rule (see :func:`ensure_env_production`) means a file
    generated before a provider change — or before the generator knew about
    ``claude_code`` providers at all — keeps being shipped into every web
    container verbatim. When the config declares LLM credentials and the file
    contains none of them, the deploy would succeed with terminals that fail
    authentication on their first prompt; this warning is the only breadcrumb.
    Advisory by design: an operator-authored file may authenticate another
    way, so nothing here blocks the deploy or touches the file.

    Two arms, evaluated independently — either can fire on its own, and a file
    that satisfies one says nothing about the other:

    - The LLM arm is all-or-nothing on purpose: a file carrying ANY of the
      provider secrets in play is a file someone is maintaining, and naming the
      rest of them would fire on every deploy that authenticates a subset of
      its personas another way.
    - The telemetry arm asks about one variable at a time, because the
      observability account name has no such alternative: a file that omits it
      leaves the terminals naming the reference's own fallback address, which
      is the wrong account on any deploy that configured the store under a
      different one.
    """
    _warn_if_telemetry_account_absent(config, project_root, users_env_path)

    required_cc_vars, extra_cc_vars = _claude_code_auth_secret_vars(config, project_root)
    expected: dict[str, str] = dict(extra_cc_vars)
    expected.update(required_cc_vars)
    llm_var = (config.get("llm") or {}).get("api_key_env_var")
    if isinstance(llm_var, str) and llm_var:
        expected.setdefault(llm_var, "llm.api_key_env_var")
    if not expected:
        return
    try:
        present = parse_dotenv_file(users_env_path)
    except OSError:
        return  # unreadable file surfaces as compose's own env_file error
    if any(var in present for var in expected):
        return
    expectations = "; ".join(f"{var} ({origin})" for var, origin in expected.items())
    logger.warning(
        f"{users_env_path} exists but contains none of the LLM "
        f"credential(s) this config's providers need: {expectations}. Web "
        "terminals will fail authentication unless this deploy authenticates "
        "another way. Delete the file to regenerate it from the env chain, or "
        "add the variable(s) to it directly."
    )


def _warn_if_telemetry_account_absent(
    config: dict, project_root: Path, users_env_path: Path
) -> None:
    """Warn when an existing ``.env.users`` omits the telemetry account name.

    The second, independently-evaluated arm of
    :func:`_warn_if_env_production_lacks_credentials` — a file that carries an
    LLM credential and therefore satisfies the arm above can still omit this
    one, which is the common shape for a file written before the generator
    copied it at all.

    Names only, never values: the advisory says which variable is missing and
    where the config asks for it, and reads nothing out of either file beyond
    whether the key is present.
    """
    expected = _telemetry_user_references(config, project_root)
    if not expected:
        return
    try:
        present = parse_dotenv_file(users_env_path)
    except OSError:
        return  # unreadable file surfaces as compose's own env_file error
    absent = {var: origin for var, origin in expected.items() if var not in present}
    if not absent:
        return
    expectations = "; ".join(f"{var} ({origin})" for var, origin in absent.items())
    logger.warning(
        f"{users_env_path} exists but does not set the observability account "
        f"name(s) this config's telemetry references: {expectations}. Web "
        "terminals emitting telemetry will name whatever that reference "
        "resolves to inside the container — its own fallback address, or the "
        "placeholder verbatim when it has none — so the store rejects their "
        "records unless its account happens to match. Delete the file to "
        "regenerate it from the env chain, or add the variable(s) to it "
        "directly."
    )
