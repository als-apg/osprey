"""``.env.production`` generation for multi-user web-terminal deploys.

Local-mode deploys generate ``.env.production`` from ``.env`` (a filtered
subset: runtime credentials in, build/CI-only variables out); registry-mode
deploys only exists-check it (CI is expected to have produced it). Called from
:func:`osprey.deployment.web_terminals.provision.deploy_up_web_terminals`.
"""

import os
from pathlib import Path

import yaml

from osprey.deployment.web_terminals.personas import effective_image_source
from osprey.utils.dotenv import parse_dotenv_file
from osprey.utils.logger import get_logger

logger = get_logger("deployment.lifecycle")


def _copy_named_env_var(var_name: str | None, source: dict[str, str], dest: dict[str, str]) -> None:
    """Copy ``source[var_name]`` into ``dest[var_name]`` iff both are present.

    ``var_name`` is itself a config-declared *name* (e.g. ``llm.api_key_env_var``
    resolves to ``"CBORG_API_KEY"``), not a literal value — this is the
    ``${env.${config.X.Y_env_var}}`` indirection the ``.gitlab-ci.yml`` template
    (see :func:`_build_env_production_subset`) uses for every secret it
    assembles. A ``var_name`` that is unset (module misconfigured) or absent
    from ``source`` (operator never set it) is silently skipped — never
    fabricated, matching every other var-presence check in this module.
    """
    if not var_name or var_name not in source:
        return
    dest[var_name] = source[var_name]


def _claude_code_auth_secret_vars(
    config: dict, project_root: Path
) -> tuple[dict[str, str], dict[str, str]]:
    """Auth-secret env-var names every ``claude_code.provider`` in play needs.

    This is the web-terminal counterpart of the launch-time secret injection
    in :mod:`osprey.cli.claude_code_resolver`: a per-user web container runs
    its persona project's agent, which authenticates via the provider named in
    that project's ``claude_code.provider`` — and the *only* env its container
    sees is ``docker-compose.web.yml``'s ``env_file: .env.production``. A
    generated ``.env.production`` that misses the provider's secret var
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
    :func:`deploy_up_web_terminals` auto-renders missing personas *before*
    :func:`ensure_env_production` runs, so on every deploy path that reaches
    generation the rendered configs are on disk. A provider name known
    neither to ``CLAUDE_CODE_PROVIDERS`` nor to the config's own
    ``api.providers`` is likewise skipped here (the resolver raises its own
    actionable error for that at launch).
    """
    from osprey.cli.claude_code_resolver import provider_auth_secret_env

    def _provider_var(cfg: dict) -> tuple[str, str | None] | None:
        provider = (cfg.get("claude_code") or {}).get("provider")
        if not isinstance(provider, str) or not provider:
            return None
        api_providers = (cfg.get("api") or {}).get("providers")
        if not isinstance(api_providers, dict):
            api_providers = None
        return provider, provider_auth_secret_env(provider, api_providers)

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    catalog = web_terminals.get("personas")
    catalog = catalog if isinstance(catalog, dict) else {}

    referenced: set[str] = set()
    default_persona = web_terminals.get("default_persona")
    if isinstance(default_persona, str) and default_persona:
        referenced.add(default_persona)
    users = web_terminals.get("users")
    for user in users if isinstance(users, list) else []:
        if isinstance(user, dict) and isinstance(user.get("persona"), str) and user["persona"]:
            referenced.add(user["persona"])

    required: dict[str, str] = {}
    extra: dict[str, str] = {}

    for persona_name in sorted(referenced):
        entry = catalog.get(persona_name)
        if not isinstance(entry, dict):
            continue
        project_path_raw = entry.get("project_path")
        if not isinstance(project_path_raw, str) or not project_path_raw:
            continue
        # Path join: an absolute project_path stands on its own; a relative
        # one resolves against the deploy project root, same as every other
        # cwd-relative assumption on this path.
        config_yml = Path(project_root, project_path_raw) / "config.yml"
        if not config_yml.is_file():
            continue
        try:
            with config_yml.open("r", encoding="utf-8") as fh:
                persona_config = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(persona_config, dict):
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
    """Build the module-conditional CI subset for a local-mode ``.env.production``.

    Mirrors the shipped facility-scaffolding CI template's own
    ``.env.production`` assembly step (``osprey-build-deploy`` skill's
    ``templates/core/.gitlab-ci.yml``, ``osprey-build`` job, lines 80-98) —
    the same var set that job composes from masked CI variables, here sourced
    from the operator's local ``.env`` instead of ``$CI_*`` secrets:

    - ``llm.api_key_env_var`` — the LLM provider key, unconditional.
    - ``claude_code_secret_vars`` — the auth-secret vars resolved by
      :func:`_claude_code_auth_secret_vars` (the ``claude_code.provider``
      of the deploy config and of every referenced persona project), passed
      in by :func:`ensure_env_production`. Same ``.env``-presence rule as
      every other entry; whether an *absent* var is an error is
      :func:`ensure_env_production`'s call, not this function's.
    - ``modules.olog.{username,password}_env_var`` — only if ``modules.olog.enabled``.
    - ``modules.wiki_search.token_env_var`` — only if ``modules.wiki_search.enabled``.
    - ``modules.event_dispatcher.token_env_var`` — only if
      ``modules.event_dispatcher.enabled``. Deliberately NOT
      ``sidecar_token_env_var`` — see the exclusion list below.
    - ``ARIEL_DSN`` — only if ``modules.ariel.enabled``, from
      ``modules.ariel.dsn`` directly. Unlike every other entry above, the CI
      template substitutes this value straight from facility-config
      (``ARIEL_DSN=${config.modules.ariel.dsn}``, no ``${env.*}``
      indirection) because the DSN is itself a literal config value, not the
      *name* of an env var holding one — so it is read from ``config``, not
      ``dotenv``.
    - ``TZ`` — always, from ``facility.timezone`` (default ``"UTC"``, matching
      the facility-config schema's own documented default), likewise a
      literal config value with no ``${env.*}`` indirection in the template.

    NEVER included, by construction (this function never reads these config
    paths at all): the CI/registry provider token (``ci.token_env_var`` /
    legacy ``gitlab.token_env_var``), ``registry.token_env_var``, the
    dispatcher's ``sidecar_token_env_var``, or any ``registry.external_projects``
    entry's ``token_env_var``. Those all gate build/push/CI-registry access,
    not anything the running containers need, and none of them belongs in a
    per-deploy runtime secrets file. This is the security spec for this
    function: a var absent from the enumerated list above can never appear in
    the returned dict, regardless of what the input ``.env`` contains.

    :param config: Raw deploy config (facility fields merged in — see
        ``modules.web_terminals.image_source`` in :func:`ensure_env_production`).
    :param dotenv: The operator's ``.env``, already parsed via
        :func:`osprey.utils.dotenv.parse_dotenv_file`.
    :return: The subset to write into ``.env.production``, in stable
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

    event_dispatcher = modules.get("event_dispatcher") or {}
    if event_dispatcher.get("enabled"):
        # NEVER sidecar_token_env_var -- see the exclusion list above.
        _copy_named_env_var(event_dispatcher.get("token_env_var"), dotenv, subset)

    ariel = modules.get("ariel") or {}
    if ariel.get("enabled"):
        dsn = ariel.get("dsn")
        if dsn:
            subset["ARIEL_DSN"] = str(dsn)

    facility = config.get("facility") or {}
    subset["TZ"] = str(facility.get("timezone") or "UTC")

    return subset


def ensure_env_production(config: dict, project_root: str | Path) -> Path:
    """Ensure ``<project_root>/.env.production`` exists, generating it when possible.

    ``docker-compose.web.yml`` (see :func:`deploy_up_web_terminals`) declares
    ``env_file: .env.production`` unconditionally, so compose hard-fails before
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
    - **Registry mode, absent**: raises. Registry-mode deploys expect a CI
      pipeline (the ``osprey-build-deploy`` skill's ``.gitlab-ci.yml``,
      ``osprey-build`` job) to have produced this file already — this
      function only exists-checks in that mode, it never generates, because
      there is no local ``.env`` this system is licensed to treat as the
      authoritative source of CI-provisioned secrets.
    - **Local mode, absent, ``.env`` present**: generated via
      :func:`_build_env_production_subset` (the module-conditional CI subset,
      including every ``claude_code`` auth secret resolved by
      :func:`_claude_code_auth_secret_vars`) and written with mode ``0600``
      from the moment the file is created — the same permission convention
      :func:`_ensure_service_tokens` uses for minted tokens. A *required*
      ``claude_code`` auth secret absent from ``.env`` raises instead of
      generating: the resulting file would produce healthy-looking terminals
      that fail authentication on their first prompt (authoring
      ``.env.production`` directly remains the bypass for deploys that
      authenticate another way).
    - **Local mode, absent, ``.env`` absent too**: raises, before any compose
      invocation — there is nothing to generate from and no file to fall back
      on.

    Every secret value this generates comes solely from the parsed ``.env``
    (never the ambient process/shell environment, unlike
    :func:`_ensure_service_tokens`'s ``_effective_value``): ``.env`` is the
    canonical local secrets store for this deploy, so reading only from it
    keeps the generated file deterministic and independent of whatever
    happens to be exported in the caller's shell.

    :param config: Raw deploy config.
    :param project_root: Project root directory; ``.env.production`` and
        ``.env`` are both resolved relative to it.
    :return: Path to the existing or newly-generated ``.env.production``.
    :raises RuntimeError: per the absent-file rules above, with an actionable
        message naming the missing file(s) and how to resolve it.
    """
    root = Path(project_root)
    env_production_path = root / ".env.production"
    if env_production_path.is_file():
        _warn_if_env_production_lacks_credentials(config, root, env_production_path)
        return env_production_path

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if effective_image_source(web_terminals) != "local":
        raise RuntimeError(
            f"{env_production_path} not found. Registry-mode web-terminal deploys "
            "(modules.web_terminals.image_source: registry, the default) expect "
            "this file to be produced by CI (see the osprey-build-deploy skill's "
            ".gitlab-ci.yml osprey-build job) and shipped alongside the pulled "
            "image context -- osprey deploy up does not generate it in this mode. "
            "Either supply .env.production directly, or set "
            "modules.web_terminals.image_source: local to generate it from .env."
        )

    env_path = root / ".env"
    if not env_path.is_file():
        raise RuntimeError(
            f"Neither {env_production_path} nor {env_path} was found. Local-mode "
            "web-terminal deploys (modules.web_terminals.image_source: local) need "
            "one of them: create .env.production directly, or create .env so "
            "osprey deploy up can derive .env.production's module-conditional CI "
            "subset from it."
        )

    dotenv = parse_dotenv_file(env_path)
    required_cc_vars, extra_cc_vars = _claude_code_auth_secret_vars(config, root)

    # Unlike every optional module var above (silently skipped when absent —
    # see _copy_named_env_var), a missing claude_code auth secret means some
    # web container comes up healthy and fails authentication on its first
    # prompt, with nothing in the deploy output to say why. Fail HERE, before
    # any compose invocation, naming the exact var and both remedies.
    missing = {var: origin for var, origin in required_cc_vars.items() if var not in dotenv}
    if missing:
        needs = "; ".join(f"{origin} needs {var}" for var, origin in missing.items())
        # .env stays the only secret SOURCE (see the determinism note above) —
        # but when a missing var is sitting right there in the caller's shell,
        # say so and hand over the exact copy-in command instead of leaving
        # the operator to discover the .env-only rule by archaeology. Presence
        # check only; the value itself is never read into the message.
        exported = [var for var in missing if os.environ.get(var)]
        shell_hint = ""
        if exported:
            copy_cmds = " && ".join(f'echo "{var}=${var}" >> {env_path}' for var in exported)
            verb = "are" if len(exported) > 1 else "is"
            shell_hint = (
                f" Note: {', '.join(exported)} {verb} exported in the current "
                f"shell, but .env is the canonical secrets store for this "
                f"deploy (generation never reads the ambient environment). "
                f"Copy it in with: {copy_cmds}"
            )
        raise RuntimeError(
            f"Generating {env_production_path} from {env_path} would leave web "
            f"terminals unauthenticated: {needs}, not set in {env_path}. Add "
            "the missing variable(s) there, or author .env.production yourself "
            "(an existing file is never regenerated) if this deploy "
            f"authenticates another way.{shell_hint}"
        )

    subset = _build_env_production_subset(config, dotenv, {**required_cc_vars, **extra_cc_vars})

    lines = "".join(f"{key}={value}\n" for key, value in subset.items())
    # Create with mode 0600 from the FIRST byte on disk, not write-then-chmod:
    # write_text() would create the file at the process umask (typically
    # 0644) and write every secret before a later os.chmod tightened
    # permissions, leaving a window on a multi-user host where a co-tenant
    # could read it. os.open with O_CREAT + an explicit mode is atomic --
    # there is no instant the file exists at a wider mode.
    fd = os.open(env_production_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(lines)
    # Belt-and-suspenders: also covers the file already existing (e.g. a
    # leftover from a prior run) with a wider mode O_CREAT wouldn't have
    # reset on its own.
    os.chmod(env_production_path, 0o600)

    logger.key_info(
        "Generated %s from %s (mode 0600): %s",
        env_production_path,
        env_path,
        ", ".join(subset),
    )

    return env_production_path


def _warn_if_env_production_lacks_credentials(
    config: dict, project_root: Path, env_production_path: Path
) -> None:
    """Warn when an existing ``.env.production`` carries no LLM credential.

    The never-clobber rule (see :func:`ensure_env_production`) means a file
    generated before a provider change — or before the generator knew about
    ``claude_code`` providers at all — keeps being shipped into every web
    container verbatim. When the config declares LLM credentials and the file
    contains none of them, the deploy would succeed with terminals that fail
    authentication on their first prompt; this warning is the only breadcrumb.
    Advisory by design: an operator-authored file may authenticate another
    way, so nothing here blocks the deploy or touches the file.
    """
    required_cc_vars, extra_cc_vars = _claude_code_auth_secret_vars(config, project_root)
    expected: dict[str, str] = dict(extra_cc_vars)
    expected.update(required_cc_vars)
    llm_var = (config.get("llm") or {}).get("api_key_env_var")
    if isinstance(llm_var, str) and llm_var:
        expected.setdefault(llm_var, "llm.api_key_env_var")
    if not expected:
        return
    try:
        present = parse_dotenv_file(env_production_path)
    except OSError:
        return  # unreadable file surfaces as compose's own env_file error
    if any(var in present for var in expected):
        return
    expectations = "; ".join(f"{var} ({origin})" for var, origin in expected.items())
    logger.warning(
        f"{env_production_path} exists but contains none of the LLM "
        f"credential(s) this config's providers need: {expectations}. Web "
        "terminals will fail authentication unless this deploy authenticates "
        "another way. Delete the file to regenerate it from .env, or add the "
        "variable(s) to it directly."
    )
