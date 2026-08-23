"""Post-deploy endpoint summary.

Every ``osprey up`` ends by saying what is reachable where, derived
from the published host ports in the rendered compose files (the same source
:mod:`osprey.deployment.host_ports` preflights) — so the summary needs no
per-facility knowledge. A service on the host's network namespace publishes no
port and appears in no ``ports:`` block, so its bound ports are derived from the
rendered config the same way the preflight derives them; leaving them out would
have made a host-mode deploy's summary silently short of the very services the
operator asked for. A web-terminal line is always included: a project
*without* a web tier says "(not configured)" explicitly, turning "nothing
listens on the landing port" from a silent absence into a stated fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from osprey.cli import output
from osprey.deployment.compose_generator import resolve_project_name
from osprey.deployment.host_ports import (
    _WILDCARD_HOSTS,
    _WORKER_SERVICE_PREFIX,
    derive_host_network_bindings,
    parse_host_port_bindings,
)
from osprey.utils.logger import get_logger

logger = get_logger("deployment.summary")

# Framework services fronted by HTTP, shown as clickable URLs. Everything else
# (databases, channel-access gateways, ...) is shown as a bare address. Keyed
# on compose service names, mirroring host_ports._SERVICE_REMEDY_KEYS.
_HTTP_SERVICES = {
    "openobserve",
    "event-dispatcher",
    "bluesky-bridge",
    "tiled",
    "bluesky-web",
}


def _http_service(service: str) -> bool:
    """Whether a service is fronted by HTTP, so its address is shown as a URL.

    Workers are indexed (``dispatch-worker-1``, ``-2``, …) off one shared name,
    so the index is dropped before the lookup — the same reduction
    :mod:`osprey.deployment.host_ports` makes to find their remedy key. They
    reach the summary only on the host network, where they bind directly.
    """
    return service in _HTTP_SERVICES or service.startswith(f"{_WORKER_SERVICE_PREFIX}-")


def endpoint_entries(config: dict, compose_files: list[str]) -> list[tuple[str, str]]:
    """Where a deployment's services are reachable, as ``(service, address)`` pairs.

    The single derivation of "what answers where": the text summary below lays
    these out, and the closing summary card takes the URLs off the same list, so
    no reader of a deployment can describe it differently from another. It says
    where a service *would* answer — the published ports are read out of the
    rendered compose files, not probed.

    Two sources, because a deployment has two ways to reach a host port. What is
    published lives in the compose files; what a host-namespace service binds
    directly lives only in the rendered config, and is derived from it by the
    same function the port preflight uses, so the summary and the preflight
    cannot name different ports for the same service.

    :param config: Loaded configuration dictionary
    :param compose_files: Rendered compose file paths, spelled absolutely or
        resolvable from the working directory — they are opened here
    """
    entries: list[tuple[str, str]] = []

    try:
        bindings = parse_host_port_bindings(compose_files)
    except Exception:
        bindings = []
    try:
        bindings = bindings + derive_host_network_bindings(config)
    except Exception:
        pass
    for binding in sorted(bindings, key=lambda b: (b.service, b.host_port)):
        host = "127.0.0.1" if binding.host_ip in _WILDCARD_HOSTS else binding.host_ip
        address = f"{host}:{binding.host_port}"
        if _http_service(binding.service):
            address = f"http://{address}"
        if binding.host_network:
            address = f"{address}  (host network)"
        entries.append((binding.service, address))

    landing_url = landing_page_url(config)
    if landing_url:
        entries.append(("web terminal", f"{landing_url}  (landing page)"))
    else:
        entries.append(("web terminal", "(not configured in this project)"))

    return entries


def landing_page_url(config: dict) -> str | None:
    """Where this deployment's landing page answers, or ``None`` if it has none.

    The one derivation of that address: the endpoint list above spells it into a
    row, and the closing call to action hands it to the operator as the thing to
    open. A deployment with ``modules.web_terminals`` disabled has no landing
    page at all, which is what ``None`` says.

    THE EXTERNAL ORIGIN, not the loopback address, whenever one can be derived.
    Every terminal checks a mutating request's ``Origin`` against exactly that
    value, so a browser that arrived on ``http://127.0.0.1:<port>`` loads every
    page and then has each write, approval and chat message refused with a 403
    the operator can only see inside a container. The loopback address is the
    fallback for a deployment that declares no origin at all — one with the
    module enabled and no roster, which has no terminal to be wrong about —
    and :func:`as_built_closing_facts` records which of the two this is so the
    closing line can say so.

    :param config: Loaded configuration dictionary
    """
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    nginx_port = web_terminals.get("nginx_port")
    if not (web_terminals.get("enabled") and isinstance(nginx_port, int)):
        return None
    try:
        from osprey.deployment.web_terminals.render import deployment_external_origin

        return deployment_external_origin(config)
    except Exception as exc:
        # Advisory, like everything else this module derives: a config with no
        # `deploy.fqdn` and no `external_origin` still has a landing page on the
        # deploy host, and the closing card marks the address as local-only.
        logger.debug(f"External origin unavailable, falling back to loopback: {exc}")
        return f"http://127.0.0.1:{nginx_port}"


def as_built_endpoint_entries(repo_root: Path | str) -> list[tuple[str, str]]:
    """The endpoint entries of the deployment ``repo_root`` has BUILT.

    The same two facts every other reader of a deployment starts from — the
    rendered ``build/config.yml`` and the compose files it names — so a caller
    holding nothing but the repo (the summary card at the end of a verb) reaches
    the same answer as one that was handed the config. A repo with nothing
    rendered has no endpoints to declare and answers with an empty list.

    Compose paths come back relative to the repo root and are anchored on it
    here, because :func:`endpoint_entries` opens them itself.
    """
    from osprey.deployment.container_lifecycle import as_built_compose_files

    root = Path(repo_root)
    config = _as_built_config(root)
    if config is None:
        return []
    files = [
        str(path if path.is_absolute() else root / path)
        for path in (Path(name) for name in as_built_compose_files(config, root))
    ]
    return endpoint_entries(config, files)


def _as_built_config(root: Path) -> dict | None:
    """The rendered ``build/config.yml`` of the deployment at ``root``, or ``None``.

    The one load the ``as_built_*`` readers start from, so a caller that wants
    two facts about a built deployment pays for one parse and cannot get the two
    answers out of two different renders.
    """
    from osprey.deployment.container_lifecycle import as_built_config_path
    from osprey.utils.config import load_project_config

    config_path = as_built_config_path(root)
    if not config_path.is_file():
        return None
    return load_project_config(str(config_path), wrap_errors=True)


@dataclass(frozen=True)
class ClosingFacts:
    """What a finished ``up`` needs in order to say where to go next.

    :param landing_url: The landing page's address, or ``None`` for a
        deployment that runs no web terminals.
    :param logins: ``(username, password)`` pairs safe to print -- the roster
        logins still carrying the password ``profile.yml`` declared. Empty
        whenever the operator owns the credentials; see
        :func:`~osprey.deployment.web_terminals.auth_credentials.seeded_logins`.
    :param token_login_users: Roster users who can only be reached by opening
        their own ``?token=`` URL, in roster order. These are the entries no
        login wall stands in front of -- the whole roster when
        ``auth.method`` is ``none``, and the ``login: false`` entries otherwise
        -- so the terminal's own gate is the only one and the URL is the only
        way through it. NOT the URLs themselves: each carries that user's
        operator secret, and a deploy's closing output is read over shoulders,
        pasted into tickets and captured by CI logs. The verb that prints one
        (``osprey users login-url <user>``) is what goes here instead.
    :param landing_url_is_external_origin: Whether ``landing_url`` is the origin
        the terminals actually check a mutating request against, or the loopback
        fallback :func:`landing_page_url` uses when no origin can be derived.
        False means the address serves the landing page and nothing beyond it --
        a browser that arrives there is refused on every write -- and the closing
        line has to say so rather than present it as the way in.
    """

    landing_url: str | None
    logins: tuple[tuple[str, str], ...]
    token_login_users: tuple[str, ...] = ()
    landing_url_is_external_origin: bool = True


def as_built_closing_facts(repo_root: Path | str) -> ClosingFacts:
    """The closing facts of the deployment ``repo_root`` has BUILT.

    Read off the same rendered ``build/config.yml`` every other ``as_built_*``
    reader uses, so the address printed at the end of a deploy is the address
    the endpoint list printed a few lines above it.

    Advisory: anything unreadable -- no build, a config that will not parse, an
    ``auth`` stanza naming a method that does not exist -- comes back as "no
    landing page, no logins" rather than as an error. This is the last thing a
    successful deploy prints, and it must not be what fails it.
    """
    try:
        root = Path(repo_root)
        config = _as_built_config(root)
        if config is None:
            return ClosingFacts(landing_url=None, logins=())
        return ClosingFacts(
            landing_url=landing_page_url(config),
            logins=tuple(_seeded_logins(root, config)),
            token_login_users=tuple(token_login_users(config)),
            landing_url_is_external_origin=_external_origin_is_derivable(config),
        )
    except Exception as exc:
        logger.debug(f"Closing facts skipped: {exc}")
        return ClosingFacts(landing_url=None, logins=())


def _external_origin_is_derivable(config: dict) -> bool:
    """Whether this deployment declares the origin its terminals check against.

    The same call :func:`landing_page_url` makes, asked as a yes/no so the
    closing line can tell "open this" from "this shows you the landing page and
    nothing else works from it". Kept as its own reader rather than inferred
    from the URL's shape: a deployment whose ``deploy.fqdn`` genuinely IS a
    loopback address would be misread by any string test.
    """
    try:
        from osprey.deployment.web_terminals.render import deployment_external_origin

        return bool(deployment_external_origin(config))
    except Exception:
        return False


def _seeded_logins(root: Path, config: dict) -> list[tuple[str, str]]:
    """The profile-seeded logins of this deployment's roster, in roster order.

    Gated on there being a login to have: with ``auth.method`` at anything but
    ``password`` no OSPREY-held credential exists (``none`` has none at all, and
    under ``oidc`` the facility's identity provider holds them), so a password
    named here would be one nothing ever checks. Roster entries carrying
    ``login: false`` sit outside the wall and are skipped for the same reason --
    the same predicate credential provisioning itself uses.
    """
    from osprey.deployment.web_terminals.auth_credentials import seeded_logins
    from osprey.deployment.web_terminals.personas import entry_requires_login, normalize_users
    from osprey.deployment.web_terminals.render import _auth_tls_context

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if not web_terminals.get("enabled"):
        return []
    if _auth_tls_context(web_terminals).get("auth_method") != "password":
        return []
    names = [
        entry["name"]
        for entry in normalize_users(web_terminals.get("users"))
        if entry_requires_login(entry)
    ]
    return seeded_logins(root, names)


def token_login_users(config: dict) -> list[str]:
    """The roster users whose terminal has no login wall in front of it.

    The complement of :func:`_seeded_logins`, and derived from the same two
    predicates so the two cannot disagree about who has a login. Public because
    ``osprey users login-url`` asks the same question before it prints anything:
    the URL is inert for a user behind the wall, so the verb refuses there, and
    a second spelling of "who has a login page" could send an operator a live
    secret the deployment would then ignore. With
    ``auth.method`` at ``none`` that is everybody: nginx runs no login flow and
    injects no operator secret, so the per-user app's own token->cookie gate is
    the only one and a browser gets past it exactly once, by opening that user's
    ``?token=`` URL. With authentication on it is the ``login: false`` entries,
    which sit outside the wall for the same reason and reach their terminal the
    same way.

    Returns names, never URLs. The URL carries a live credential; the closing
    card is not a place to put one.
    """
    from osprey.deployment.web_terminals.personas import entry_requires_login, normalize_users
    from osprey.deployment.web_terminals.render import _auth_tls_context

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if not web_terminals.get("enabled"):
        return []
    walled = _auth_tls_context(web_terminals).get("auth_method") != "none"
    return [
        entry["name"]
        for entry in normalize_users(web_terminals.get("users"))
        if not (walled and entry_requires_login(entry))
    ]


def _summary_title(config: dict) -> str:
    """The heading both the printed section and the logged block carry."""
    return f"Service endpoints ({resolve_project_name(config)}):"


def _summary_text(title: str, entries: list[tuple[str, str]]) -> str:
    """Lay ``entries`` out under ``title`` as the block a caller wants as text."""
    return "\n".join([title] + [f"  {service:<20} {address}" for service, address in entries])


def format_endpoint_summary(config: dict, compose_files: list[str]) -> str:
    """Render :func:`endpoint_entries` as the text block a deploy or report prints.

    :param config: Loaded configuration dictionary
    :param compose_files: Rendered compose file paths, spelled absolutely or
        resolvable from the working directory — they are opened here
    :return: Multi-line summary text
    """
    return _summary_text(_summary_title(config), endpoint_entries(config, compose_files))


def log_endpoint_summary(config: dict, compose_files: list[str]) -> None:
    """Print the endpoint summary, and keep it in the record for sinks.

    Where a deployment answers is what the operator ran ``osprey up`` to find
    out, so it is printed as the verb's own output rather than logged: an
    INFO record is not rendered on a normal run, and a fact that only a
    ``--verbose`` run shows is a fact the deploy did not report. Every exit
    path of ``deploy_up`` calls this function, so all of them inherit the
    printed form from here.

    The logged block stays as it was, at the same level, so file and
    aggregation sinks keep the whole summary in one record. It reaches a
    terminal only on a ``--verbose`` run, where the transcript is what was
    asked for.

    Advisory: a summary that cannot be derived is reported to the debug record
    and never fails a deploy that otherwise succeeded.

    :param config: Loaded configuration dictionary
    :param compose_files: Rendered compose file paths, spelled absolutely or
        resolvable from the working directory — they are opened here
    """
    try:
        title = _summary_title(config)
        entries = endpoint_entries(config, compose_files)
        output.section(title, entries)
        logger.key_info(_summary_text(title, entries))
    except Exception as exc:
        logger.debug(f"Endpoint summary skipped: {exc}")
