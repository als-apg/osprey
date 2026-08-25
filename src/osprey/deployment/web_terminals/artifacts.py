"""Render-and-write seam for the multi-user web-terminal deployment artifacts.

:func:`osprey.deployment.web_terminals.render.render_web_terminals` produces the
artifacts in memory (``docker-compose.web.yml``, ``nginx/nginx.conf``,
``nginx/landing.html``, plus one ``nginx/templates/secret-<user>.conf.template``
per gated roster user) as a ``{relative_path: content}`` mapping. This module is
the single place that decides *where on disk* those relative paths land, so every
consumer agrees on one location:

* ``osprey up`` renders and writes them at bring-up, then includes the web
  compose file in the ``compose up`` invocation.
* the lifecycle verbs (``decommission``/``prune``) re-render and re-write them after
  editing the roster, so the deployed nginx routing and compose services match the
  new roster.

If bring-up and the lifecycle verbs wrote to different directories, a decommission
would update artifacts that ``up`` never reads. Routing every writer through this
one helper makes that class of drift impossible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from osprey.deployment.errors import DeploymentError
from osprey.deployment.web_terminals.auth_credentials import (
    AUTH_ENV_FILENAME,
    terminal_secret_var,
)
from osprey.deployment.web_terminals.personas import (
    normalize_users,
    personas_needing_archiver_password,
    personas_needing_ariel_mirror,
    personas_needing_ariel_password,
    personas_needing_dispatcher_token,
    personas_needing_facility_bundle,
    personas_needing_graphdb_password,
    personas_needing_launch_token,
    personas_not_denying_bash,
)
from osprey.deployment.web_terminals.render import (
    clear_nginx_templates_dir,
    render_web_terminals,
)
from osprey.utils.dotenv import ENV_LOCAL_FILENAME, parse_dotenv_file
from osprey.utils.workspace import AUDIT_DIR_RELPATH, BUILD_DIR_NAME

#: Filename of the rendered web-stack compose file, as
#: :func:`~osprey.deployment.web_terminals.render.render_web_terminals` keys it.
WEB_COMPOSE_FILENAME = "docker-compose.web.yml"


class BashLaunchTokenConflictError(DeploymentError):
    """A persona would hold ``BLUESKY_LAUNCH_TOKEN`` while its agent may run a shell.

    ``BLUESKY_LAUNCH_TOKEN`` exists so a single chat approval is enough to arm a
    Bluesky queue start, which is physical accelerator hardware motion. That
    approval gates the ``queue_start`` MCP tool — and only that tool. An agent
    that may also run ``Bash`` can read the token straight out of its own
    environment and arm the queue with a plain HTTP call, reaching the same
    hardware with no approval prompt at any point. The chat gate is then
    decoration.

    So the two are refused *together* rather than shipped and documented: the
    deploy stops before a single web-terminal artifact is written, while the
    operator still has the context to fix it. Fail-closed, like
    :class:`~osprey.deployment.errors.ComposeInterpolationError` and for the same
    reason — a warning about this would scroll past and leave a stack running.

    Deliberately not a ``ValueError``. ``force_recreate_auth_sidecar`` catches
    ``(ValueError, OSError)`` around its best-effort re-render and downgrades it
    to a warning; a safety refusal that a credential rotation could swallow is
    not a refusal.

    Three boundaries this guard does **not** cover, none of them accidental:

    * **Persona-less roster entries are out of reach.** Both sides of the
      intersection are keyed on persona names, so a roster entry naming no
      persona — the zero-migration path, where the web image *is* the deploy
      project — appears in neither set and is not bound by this check.
      :func:`~osprey.deployment.web_terminals.render.render_web_terminals`
      answers that entry's entitlement independently, via
      :func:`~osprey.deployment.web_terminals.personas.config_needs_launch_token`
      on the deploy config itself. This is consistent with the feature's scope:
      the single-user path already resolves this token today, so it is not new
      exposure — but a reader who assumes the guard covers every entitled
      container will be wrong.
    * **The check is render-scoped, not image-scoped.** It reads the rendered
      project directory, which equals the image's ``.claude/settings.json`` only
      as of that image's last build: ``COPY . /app/<project>/`` bakes the
      directory in, and the per-user compose services declare only ``image:``,
      with no ``build:`` stanza to rebuild one at start. A render that *adds* a
      ``Bash`` deny without an image rebuild therefore passes this guard while
      the running image is still permissive. That is why the remedy below says
      rebuild, not merely "restore the deny".
    * **It assumes the agent does not run in bypass-permissions mode.** An agent
      launched with permissions bypassed ignores its deny list wholesale, and a
      ``Bash`` deny then proves nothing. No such flag appears anywhere on the
      web-terminal launch path today, so the assumption holds — but nothing here
      would fail if a future change broke it.

    Args:
        personas: The offending persona names — every persona that is both
            entitled to the launch token and shipping a settings artifact that
            does not deny ``Bash``. All of them are named in the message; an
            operator fixing one at a time would otherwise redeploy once per
            offender to discover the next.
    """

    def __init__(self, personas: Iterable[str]) -> None:
        self.personas = sorted(personas)
        named = ", ".join(repr(name) for name in self.personas)
        super().__init__(
            f"Refusing to deploy: {named} would be granted BLUESKY_LAUNCH_TOKEN "
            f"while also permitted to run a shell. Each named persona is entitled to "
            f"the token — its rendered config.yml sets control_system.writes_enabled: "
            f"true AND leaves the bluesky MCP server enabled — and its shipped "
            f'.claude/settings.json does not list "Bash" in permissions.deny (a '
            f"missing or unparseable settings.json counts the same — it is not "
            f"evidence of a deny). The token arms a queue start, which moves "
            f"accelerator hardware; the chat approval gates only the queue_start "
            f"tool, so an agent with a shell can read the token out of its own "
            f"environment and arm a queue with no approval at all. Fix either half "
            f'of the conflict: restore "Bash" to permissions.deny for that persona '
            f"and REBUILD its image (or republish and re-pull it in registry mode) "
            f"— a re-render alone is not enough, because the settings.json read here "
            f"is the one baked into the image, and the per-user services declare only "
            f"`image:`, so nothing rebuilds at start. Or withdraw the entitlement by "
            f"moving that persona off the bluesky server "
            f"(claude_code.servers.bluesky.enabled: false) or off writes "
            f"(control_system.writes_enabled: false)."
        )


def check_bash_launch_token_conflict(config: Any, project_root: Path | str) -> set[str]:
    """Refuse the deploy if any launch-token persona's shipped settings permit ``Bash``.

    The whole guard in one callable, because it runs at **three** points and the
    three must never be able to disagree about what a conflict is:

    * :func:`preflight_web_terminals` — the fail-fast gate. Refusing there is what
      keeps a conflicted deploy from paying for the minutes-long project-image
      build, the archiver/ARIEL store staging, and the minting *and printing* of
      ``.env.users`` and auth credentials for a stack that will never come up.
      That last part is the property ``preflight_web_terminals``' own docstring
      commits to, so the check belongs ahead of :func:`ensure_env_production`.
    * :func:`~osprey.deployment.web_terminals.lifecycle.decommission_user` — ahead
      of its ``config_replace_list`` roster edit. The seam below would refuse this
      path anyway, but only *after* the roster edit had been written, leaving
      ``config.yml`` updated, the artifacts stale, and the container and volume
      removal never run. It is handed the **post-removal** roster, and that is
      load-bearing rather than incidental: decommissioning the offending persona's
      own last user drops that persona from the referenced set and so must still
      succeed — it is the one remediation that needs no image rebuild. A future
      change that probes the pre-removal roster silently takes that away.
    * :func:`write_web_terminal_artifacts` — belt and braces, and the only cover
      for :func:`~osprey.deployment.web_terminals.provision.force_recreate_auth_sidecar`,
      whose pre-recreate re-render reaches the render seam through neither of the
      call sites above. The seam has to answer for itself.

    No call site is redundant; removing any one of them leaves a real path
    uncovered. The decommission check is the easiest to mistake for duplication,
    because the seam behind it means deleting it breaks no test about *safety* —
    only the one about ordering
    (``test_decommission_refuses_before_touching_the_roster_when_a_survivor_is_conflicted``).

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values resolve
            against it.

    Returns:
        The personas entitled to ``BLUESKY_LAUNCH_TOKEN`` — returned rather than
        recomputed so the caller that goes on to render hands the *same* set down,
        and the set the guard cleared cannot drift from the set the render grants.

    Raises:
        BashLaunchTokenConflictError: One or more entitled personas ship a
            ``.claude/settings.json`` that does not deny ``Bash``.
    """
    launch_token_personas = personas_needing_launch_token(config, project_root)
    if offenders := bash_launch_token_offenders(config, project_root):
        raise BashLaunchTokenConflictError(offenders)
    return launch_token_personas


def bash_launch_token_offenders(config: Any, project_root: Path | str) -> set[str]:
    """The personas in conflict, computed without refusing anything.

    The same intersection :func:`check_bash_launch_token_conflict` refuses on,
    split out so the collect-all preflight can ASK the question without raising
    on the answer. The raising wrapper stays exactly as it was for its three
    call sites — this is a second reader of one predicate, never a second
    definition of what a conflict is.

    Reads two rendered directories and nothing else: no file is written, no
    process is started, and nothing about the deployment changes. That purity is
    what licenses calling it from the middle of a start sequence, before any
    provisioning step has been skipped or repeated.

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it.

    Returns:
        Every persona both entitled to ``BLUESKY_LAUNCH_TOKEN`` and shipping
        settings that do not deny ``Bash``. Empty when there is no conflict.
    """
    return personas_needing_launch_token(config, project_root) & personas_not_denying_bash(
        config, project_root
    )


def web_artifacts_dir(repo_root: Path | str) -> Path:
    """Where a repo's rendered web-terminal artifacts live: ``<repo>/build``.

    They are render output — regenerated from the profile by every build and by
    every roster verb — so they belong in the disposable zone with the rest of
    it, and never at the repo root, whose contents are tracked source.
    """
    return Path(repo_root) / BUILD_DIR_NAME


def web_compose_file(repo_root: Path | str) -> Path:
    """The rendered ``build/docker-compose.web.yml`` of the repo at *repo_root*.

    One spelling for the file every web-stack invocation is pinned to, so the
    writer, the ``-f`` argument, the "is anything deployed here?" probe and the
    image-drift reconcile cannot disagree about which file that is. They did:
    the probe looked in the working directory while the render wrote elsewhere,
    which is how a password rotation could report success having recreated
    nothing.
    """
    return web_artifacts_dir(repo_root) / WEB_COMPOSE_FILENAME


def auth_env_digest(project_root: str | Path) -> str:
    """sha256 hex digest of ``.env.auth``'s current bytes under ``project_root``.

    The digest is rendered as a label on the auth sidecar's compose service
    (see :data:`~osprey.deployment.web_terminals.render.AUTH_ENV_DIGEST_LABEL`),
    so a content change to the file becomes a service-definition change — the
    one recreate trigger every compose implementation honours. An absent (or
    unreadable) file digests as empty content rather than raising: the render
    must never crash over it, and on the deploy path preflight has already
    either created the file or aborted. Erring toward the empty sentinel is
    fail-safe — at worst it flips the label and costs one sidecar recreate,
    never a stale credential.
    """
    try:
        content = (Path(project_root) / AUTH_ENV_FILENAME).read_bytes()
    except OSError:
        content = b""
    return hashlib.sha256(content).hexdigest()


def _terminal_secrets(config: Any, root: Path) -> dict[str, str] | None:
    """The roster's operator secrets as the deploy ``.env`` currently holds them.

    Read back off disk here rather than threaded down from the mint, because
    every caller of the write seam re-renders from config alone — the roster
    verbs and ``force_recreate_auth_sidecar`` never ran the mint themselves —
    and because :func:`render_web_terminals` reads no filesystem of its own.
    Same shape as :func:`auth_env_digest` above, and for the same reason: every
    disk-derived input is resolved at this seam.

    The values are used only to answer *whether* each roster user has a secret;
    :func:`~osprey.deployment.web_terminals.render._terminal_secret_artifacts`
    renders the variable NAME and never the value, which is what keeps the
    secret out of ``build/`` and out of every image layer.

    The empty-versus-``None`` distinction is the whole contract with the render,
    and it is a statement about *provisioning*, not about the roster:

    * A mapping — even a partial one — means this deployment HAS been
      provisioned, so the render may hold it to the full roster. A user missing
      from it (added to the roster since the last ``osprey up``, or blanked by
      hand) refuses the render with a message naming the user and the variable
      to set. That is the failure worth having: the alternative is a terminal
      that 401s every request from behind a healthy-looking proxy.
    * ``None`` means no roster user has a secret at all, so provisioning has not
      run here and this render has nothing to speak for. Snippets are skipped
      rather than refused. This is not an open door: nginx.conf.j2 emits an
      unconditional ``include`` for each authenticated location, and nginx
      treats a missing include as a hard startup failure — so such a stack still
      never serves a terminal. It fails at container start naming a path instead
      of at render naming a user, which is the worse error but the honest one
      for a caller that only wants to *see* what config renders to.

    On the deploy path neither degraded case arises: ``osprey up``'s preflight
    runs ``ensure_terminal_secrets`` for the whole roster and refuses the deploy
    before reaching here if any secret could not be established.

    Args:
        config: The parsed deploy config; its ``modules.web_terminals.users``
            roster decides which variables are looked up.
        root: The deployment repo root, which holds the deploy ``.env``.

    Returns:
        ``{username: secret}`` for every roster user with a non-blank value, or
        ``None`` when that mapping would be empty.
    """
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    env_path = root / ENV_LOCAL_FILENAME
    stored = parse_dotenv_file(env_path) if env_path.is_file() else {}
    secrets = {}
    for entry in normalize_users(web_terminals.get("users")):
        name = entry["name"]
        value = stored.get(terminal_secret_var(name), "").strip()
        if value:
            secrets[name] = value
    return secrets or None


def resolve_render_inputs(config: Any, repo_root: Path | str) -> dict[str, Any]:
    """Every disk-derived input the deploy hands :func:`render_web_terminals`.

    Resolved HERE and passed down, because ``render_web_terminals()`` reads no
    filesystem of its own (see its docstring): the ``.env.auth`` digest; which
    personas declare the EVENTS panel and so need the dispatcher's bearer,
    which configure ARIEL and so need its Postgres password, which both allow
    writes and run the bluesky server and so may arm a queue start, which
    configure a graph store and so need its Neo4j password — each in their own
    per-user environment block; which name a facility-knowledge bundle or run
    a qmd export and so get the deployment's bundle or mirror bind-mounted,
    and the groups those shared directories (and every user's audit zone)
    were provisioned with; and the roster's operator secrets.

    The one seam between "what is on disk" and "what the render is told", so a
    test that wants the render the deploy actually produces asks this rather
    than restating a subset of the inputs — a subset is exactly how a mount the
    deploy emits can be invisible to a test that never passed its entitlement.

    Fails closed BEFORE any render, so a refused deploy leaves no half-written
    artifacts behind: a persona holding ``BLUESKY_LAUNCH_TOKEN`` whose shipped
    settings still permit ``Bash`` can arm a queue start from a shell, with none
    of the chat approval the token exists to make sufficient. ``osprey up`` and
    ``decommission_user`` each already refused this earlier, where refusing is
    cheaper and cannot half-apply; ``force_recreate_auth_sidecar``'s
    pre-recreate re-render reaches here having passed neither, which is why the
    seam checks for itself. The entitled set comes back from the guard so the
    render is handed exactly the set that was cleared.

    Args:
        config: The parsed facility config.
        repo_root: The deployment repo root.

    Returns:
        Keyword arguments for :func:`render_web_terminals`.

    Raises:
        BashLaunchTokenConflictError: See :func:`check_bash_launch_token_conflict`.
    """
    from osprey.deployment.compose_generator import (
        resolve_ariel_mirror_dir,
        resolve_facility_bundle_dir,
        shared_corpus_gid,
    )

    root = Path(repo_root)
    launch_token_personas = check_bash_launch_token_conflict(config, root)
    return {
        "auth_env_digest": auth_env_digest(root),
        "dispatcher_personas": personas_needing_dispatcher_token(config, root),
        "ariel_personas": personas_needing_ariel_password(config, root),
        "launch_token_personas": launch_token_personas,
        "graphdb_personas": personas_needing_graphdb_password(config, root),
        "archiver_password_personas": personas_needing_archiver_password(config, root),
        "facility_bundle_personas": personas_needing_facility_bundle(config, root),
        # A pure read, like every other disk-derived input here: the deploy path
        # provisions the bundle directory before this render (see
        # deploy_up_web_terminals), and this asks what group it ended up with so
        # each entitled service can join it. An unprovisioned directory answers
        # None and emits no `group_add` — the honest render, since joining a
        # group that was never established would be a guess.
        "facility_bundle_gid": shared_corpus_gid(resolve_facility_bundle_dir(config, root)),
        # The ARIEL mirror and each user's audit zone: the same pure reads, for
        # the same reason, of directories the deploy provisioned before this
        # render. The audit zone's group is read off its root, which the
        # per-user subdirectories inherit.
        "ariel_mirror_personas": personas_needing_ariel_mirror(config, root),
        "ariel_mirror_gid": shared_corpus_gid(resolve_ariel_mirror_dir(config, root)),
        "audit_gid": shared_corpus_gid(root / AUDIT_DIR_RELPATH),
        # The roster's operator secrets, read back off the deploy .env (see
        # _terminal_secrets for the None case and why it is not an open door).
        # Without this the render emits no per-user snippet at all, while
        # nginx.conf.j2 still emits the `include` that reads one — an nginx that
        # refuses to start, pointing at a path nothing ever wrote.
        "terminal_secrets": _terminal_secrets(config, root),
    }


def write_web_terminal_artifacts(config: Any, repo_root: Path | str | None = None) -> list[Path]:
    """Render the web-terminal artifacts into the repo's ``build/`` zone.

    The artifacts' relative paths (``docker-compose.web.yml``,
    ``nginx/nginx.conf``, ``nginx/landing.html``, and one
    ``nginx/templates/secret-<user>.conf.template`` per gated roster user) are
    preserved beneath the destination; parent directories (e.g. ``nginx/``) are
    created as needed. The compose file and its ``nginx/`` subtree must stay
    co-located, which writing them together under one destination guarantees.

    ``nginx/templates/`` is emptied before anything is written, and this is the
    only artifact directory treated that way. Every other file is overwritten in
    place by each render, so a stale one cannot survive; the snippets are keyed
    by USERNAME, so a decommissioned user's snippet is overwritten by nothing
    and would otherwise live forever — and nginx's entrypoint envsubsts every
    file it finds there, so it would go on injecting a header for a user who is
    no longer on the roster. Clearing also materializes the directory on a first
    render, which the compose file bind-mounts and would otherwise have the
    container runtime create root-owned.

    Two directories, deliberately, because the two kinds of file have opposite
    lifetimes. The artifacts are render output and land in ``<repo>/build``;
    ``.env.auth`` is a durable 0600 credential store and stays at the repo root,
    with the deployment's other secrets. Compose sees them as one: every
    relative path in the rendered file — including the sidecar's
    ``env_file: .env.auth`` — resolves against the pinned project directory,
    which IS the repo root (see
    :func:`~osprey.deployment.compose_generator.compose_base_cmd`). So the
    digest below is read from the repo root, not from the destination: it has to
    describe exactly the file the rendered stack will read.

    Every writer (``osprey up``, the roster verbs' re-render, and
    :func:`~osprey.deployment.web_terminals.provision.force_recreate_auth_sidecar`
    just before each forced recreate) goes through here, and each renders AFTER
    its ``.env.auth`` mutation, so the digest is current on every path that
    reaches a compose ``up``.

    Args:
        config: The parsed facility config, passed straight through to
            :func:`render_web_terminals` (raises ``ValueError`` on an unrenderable
            config, e.g. a TLS seam enabled without cert/key).
        repo_root: The deployment repo. Defaults to the one
            :func:`~osprey.deployment.compose_generator.resolve_repo_root`
            derives from the config. There is deliberately no way to write the
            artifacts anywhere but this repo's ``build/``: a second destination
            is how bring-up and the roster verbs came to act on different
            copies. A caller that only wants to *see* the render calls
            :func:`render_web_terminals` and writes it wherever it likes.

    Returns:
        The list of files written, in the render mapping's iteration order.

    Raises:
        BashLaunchTokenConflictError: A persona is entitled to
            ``BLUESKY_LAUNCH_TOKEN`` and its shipped ``.claude/settings.json``
            does not deny ``Bash``. Raised before the render, so a refused
            deploy writes nothing. Every writer routes through here, so every
            path that could put such a stack on disk — bring-up, the roster
            verbs' re-render, the pre-recreate re-render — refuses it. Two of
            those refuse earlier still (see
            :func:`check_bash_launch_token_conflict`); this is the backstop that
            makes the property hold for all of them.
        ValueError: From the render — including, once this deployment has any
            operator secret provisioned at all, a roster user that has none of
            its own (see :func:`_terminal_secrets`). Raised before any file is
            written or cleared.
    """
    from osprey.deployment.compose_generator import resolve_repo_root

    root = Path(repo_root) if repo_root is not None else resolve_repo_root(config)
    artifacts = render_web_terminals(config, **resolve_render_inputs(config, root))
    dest = web_artifacts_dir(root)
    # Before the first write, never after: a render that raised above must not
    # have removed the snippets the currently-deployed stack is still reading,
    # and a stale snippet must not survive into the set written below.
    clear_nginx_templates_dir(dest)
    written: list[Path] = []
    for relative_path, content in artifacts.items():
        target = dest / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written
