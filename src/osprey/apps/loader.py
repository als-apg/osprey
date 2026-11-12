"""Discover and validate a deployment's applications."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from osprey.apps.agents import AGENTS_DIR, DELEGATED_TOOLS, AppAgent, parse_agent
from osprey.apps.errors import AppManifestError, Violation
from osprey.apps.manifest import NO_HOOKS_MESSAGE, AppManifest, parse_manifest
from osprey.apps.origin import (
    MANIFEST_FILE,
    AppOrigin,
    deployment_origin,
    installed_apps,
    origin_of,
)
from osprey.apps.rules import (
    AppSelection,
    LoginSelection,
    Reserved,
    check_pins,
    deployment_packages,
    fixed_names,
    framework_pins,
    refuse_shadowed_packages,
    shadows_installed,
    validate,
)
from osprey.utils.workspace import APPS_DIRNAME, PROFILE_FILENAME

if TYPE_CHECKING:
    from osprey.cli.build_profile_model import BuildProfile

APPS_DIR = "apps"

#: The key the root render's selection goes by among the logins: the root file's
#: name, which no login file's stem can spell.
ROOT_LOGIN = PROFILE_FILENAME


@dataclass(frozen=True)
class DeploymentApps:
    """What a checked deployment runs of its ``apps/``.

    Attributes:
        manifests: The applications the profile's ``apps:`` block names.
        logins: Each render's selection: :data:`ROOT_LOGIN` for the root, the
            file's stem for a login.
        unnamed: The ``apps/<name>/`` directories the block does not name: reported,
            neither built nor installed, and read only for a package's name.
        agents: Each application's checked subagents, by application name, in the
            order its manifest lists them.
    """

    manifests: tuple[AppManifest, ...] = ()
    logins: Mapping[str, LoginSelection] = field(default_factory=dict)
    unnamed: tuple[str, ...] = ()
    agents: Mapping[str, tuple[AppAgent, ...]] = field(default_factory=dict)


def load(
    deployment_root: Path,
    names: Collection[str] | None = None,
    *,
    installed: Mapping[str, AppOrigin] | None = None,
) -> list[AppManifest]:
    """Load the applications of a deployment, from both sources, sorted by name.

    An application is ``apps/<name>/app.yml`` under ``deployment_root``, or the
    ``app.yml`` of a package an installed distribution declares
    (:mod:`osprey.apps.origin`). Each manifest carries its
    :attr:`~osprey.apps.manifest.AppManifest.origin`. Only ``app.yml`` is read;
    other files in an application directory (its ``.provenance`` sidecar,
    fragments, code) are left alone. Hidden entries and ``__pycache__`` in
    ``apps/`` are skipped. An installed application is read only when named.

    Args:
        deployment_root: The deployment root.
        names: The applications to read, every other one left unread; ``None``
            reads every ``apps/`` directory and no installed application.
        installed: The installed applications, from
            :func:`~osprey.apps.origin.installed_apps`; ``None`` reads the
            environment's.

    Raises:
        AppManifestError: A manifest is refused, its ``name`` differs from its
            directory or entry point, a directory holds no ``app.yml``, an
            ``apps/<name>/`` directory has an installed application's name, or
            ``apps/`` holds a plain ``<name>.yml`` login file.
    """
    if installed is None:
        installed = installed_apps()
    apps_dir = deployment_root / APPS_DIR
    entries = (
        [e for e in sorted(apps_dir.iterdir(), key=lambda p: p.name) if not _skipped(e)]
        if apps_dir.is_dir()
        else []
    )
    shadowing = shadows_installed([e.name for e in entries if e.is_dir()], installed)
    if shadowing:
        raise AppManifestError.from_violations(shadowing)
    manifests: list[AppManifest] = []
    for entry in entries:
        if entry.is_file():
            if entry.suffix in (".yml", ".yaml"):
                raise AppManifestError(
                    entry.stem,
                    "",
                    "plain_app_file",
                    f"{APPS_DIR}/{entry.name} is a login file; apps/ holds application "
                    "directories only. Run `osprey init --revalidate` to move it to tiers/",
                )
            continue
        if names is not None and entry.name not in names:
            continue
        manifest_path = entry / MANIFEST_FILE
        if not manifest_path.is_file():
            raise AppManifestError(
                entry.name,
                "",
                "missing_manifest",
                f"{APPS_DIR}/{entry.name}/ has no {MANIFEST_FILE}",
            )
        manifest = _read_manifest(manifest_path, entry.name, None)
        if manifest.name != entry.name:
            raise AppManifestError(
                entry.name,
                "name",
                "name_matches_directory",
                f"name {manifest.name!r} must equal the directory name {entry.name!r}",
            )
        manifests.append(manifest.bind_origin(deployment_origin(deployment_root, entry.name)))
    for name in sorted(set(names or ()) & set(installed)):
        origin = installed[name]
        manifest = _read_manifest(origin.directory / MANIFEST_FILE, name, origin)
        if manifest.name != name:
            raise AppManifestError.from_violations(
                [
                    Violation(
                        name,
                        "name",
                        "name_matches_entry_point",
                        f"name {manifest.name!r} must equal the entry-point name {name!r}",
                        source=origin.manifest_source,
                    )
                ]
            )
        manifests.append(manifest.bind_origin(origin))
    return sorted(manifests, key=lambda m: m.name)


def _skipped(entry: Path) -> bool:
    """Whether an entry of ``apps/`` is no application: hidden, or ``__pycache__``."""
    return entry.name.startswith(".") or entry.name == "__pycache__"


def _read_manifest(path: Path, name: str, origin: AppOrigin | None) -> AppManifest:
    """Parse one ``app.yml``; a refusal names *origin*'s file when it is installed."""
    source = origin.manifest_source if origin is not None else None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AppManifestError.from_violations(
            [Violation(name, "", "yaml", f"not valid YAML: {exc}", source=source)]
        ) from None
    return parse_manifest(data, app=name, source=source)


def check_deployment(root: Path) -> DeploymentApps:
    """Run every check a deployment's applications face before anything is written.

    Loads the applications the profile names, leaving every other ``apps/<name>/``
    unread but for a package's name, then validates them against each render's selection
    and the names the profile already uses, refuses a hand-written key inside a namespace
    the projection owns, a projected port that collides, a package that would move a
    pinned version, and a requirement only an unlisted directory holds. ``osprey
    build`` and ``osprey app check`` both run this, so they report identically.
    The profile is read as the build reads it, this host's variant included.

    Args:
        root: The deployment root.

    Returns:
        What the deployment runs, with each render's selection.

    Raises:
        AppManifestError: Carrying every violation of the failing step.
        BuildProfileError: The profile or a login file does not resolve.
    """
    from osprey.cli.application_material import application_dirs
    from osprey.cli.build_profile import resolve_build_document
    from osprey.cli.profile_root import layer_files
    from osprey.cli.variant_selection import resolve_variant_selection

    variant = resolve_variant_selection(root)
    overlays = (variant.path,) if variant.path is not None else ()
    host = resolve_build_document(root / PROFILE_FILENAME, None, overlays).profile
    installed = installed_apps()
    manifests = tuple(load(root, host.apps, installed=installed))
    profiles: dict[str, BuildProfile] = {ROOT_LOGIN: host}
    for _kind, layer in layer_files(root):
        profiles[layer.stem] = resolve_build_document(layer, None, overlays).profile
    logins: dict[str, LoginSelection] = {}
    if manifests:
        cards = _shared_cards(host)
        access = _access_resolver(host)
        logins = {
            login: _login_selection(
                profile, manifests, cards.get(login, ()), login=login, access=access
            )
            for login, profile in profiles.items()
        }
        validate(manifests, logins=logins, reserved=_reserved(host, profiles.values(), installed))
    _refuse_projected_keys(manifests, profiles)
    if manifests:
        _refuse_unbuildable_services(root, manifests, host)
        _refuse_missing_skills(root, manifests)
        _refuse_hooks(root, manifests)
        agents = _check_agents(root, manifests)
    else:
        agents = {}
    dependencies = list(host.dependencies or [])
    check_pins(deployment_packages(root, dependencies, apps=host.apps), framework_pins())
    refuse_shadowed_packages(
        root,
        dependencies,
        apps=host.apps,
        environment_packages=list(host.environment.packages or []),
    )
    return DeploymentApps(
        manifests=manifests,
        logins=logins,
        agents=agents,
        unnamed=tuple(d.name for d in application_dirs(root) if d.name not in host.apps),
    )


def _login_selection(
    profile: BuildProfile,
    manifests: Iterable[AppManifest],
    shared_cards: Iterable[str],
    *,
    login: str,
    access: Callable[[Any, str, str], str],
) -> LoginSelection:
    """One render's selection, read from its merged profile.

    *access* (:func:`_access_resolver`) resolves an entry's authored ``access:``
    for *login*; an entry that spells none is every user of the login and
    carries ``None``.
    """
    from osprey.cli.build_profile_merge import _rendered_control_system
    from osprey_connectors.types import tier_writes

    cards = tuple(shared_cards)
    entries = {m.name: profile.apps.get(m.name) for m in manifests}
    return LoginSelection(
        audience="shared" if cards else "person",
        apps={
            name: AppSelection(
                shown=profile.app_shown(name),
                access=None
                if entry is None or entry.access is None
                else access(entry.access, login, name),
            )
            for name, entry in entries.items()
        },
        tier=tier_writes(_rendered_control_system(profile.config)),
        shared_cards=cards,
    )


def _access_resolver(host: BuildProfile) -> Callable[[Any, str, str], str]:
    """How an ``apps.<name>.access`` value resolves for a login, at build.

    Exactly as a shared card's set is expanded at render: parsed by the roster's
    one parser, then every ``self``, ``roster`` and ``tier:`` member spelled out
    to the identities the sidecar forwards, in their canonical spelling
    (:func:`~osprey.deployment.web_terminals.roster.expand_login_principals`),
    so the terminal compares a forwarded identity against ``user:`` and
    ``domain:`` members alone and keeps no roster. A set no browser caller can
    match is refused
    (:func:`~osprey.deployment.web_terminals.roster.app_access_refusal`, the
    predicate lint reports).

    Returns:
        ``(authored, login, app) -> wire value`` of the resolved set.

    Raises:
        BuildProfileError: From the returned callable, when the set admits no
            browser caller.
    """
    from osprey.cli.build_profile_emit import effective_web_terminals
    from osprey.deployment.web_terminals.render import _auth_tls_context
    from osprey.deployment.web_terminals.roster import (
        APP_ENTRY_KIND,
        access_wire_value,
        app_access_refusal,
        expand_login_principals,
        resolve_access_principals,
        resolve_cards,
    )
    from osprey_connectors.errors import BuildProfileError

    web_terminals = effective_web_terminals(host.config)
    enabled = bool(web_terminals.get("enabled"))
    cards = resolve_cards(web_terminals, {}, strict=False) if enabled else []
    try:
        context = _auth_tls_context(web_terminals) if enabled else None
    except ValueError:
        context = None
    method = context["auth_method"] if context is not None else None
    claim = context["auth_oidc_claim"] if context is not None else None

    def resolve(authored: Any, login: str, app: str) -> str:
        principals = resolve_access_principals(
            {"kind": APP_ENTRY_KIND, "name": app, "login": login, "access": authored}
        )
        resolved = expand_login_principals(
            principals, cards, login=login, method=method, claim=claim
        )
        refusal = app_access_refusal(resolved, login=login, app=app, method=method)
        if refusal is not None:
            raise BuildProfileError(refusal[1])
        return access_wire_value(resolved)

    return resolve


def _shared_cards(host: BuildProfile) -> dict[str, list[str]]:
    """The roster cards that make each login's audience shared, by login.

    A login named by ``shared:`` is its own card; a person card counts for the
    login it resolves to when it admits anyone beyond its owner, however it got
    there. A card whose login is decided at sign-in, or that resolves to none, is
    no build fact.
    """
    from osprey.cli.build_profile_emit import effective_web_terminals
    from osprey.cli.build_profile_merge import shared_login_names
    from osprey.deployment.web_terminals.roster import (
        UnresolvedRoleError,
        authored_people,
        effective_tier,
        entry_is_shared,
        resolve_authorization_roles,
        sign_in_decides_tier,
    )

    web_terminals = effective_web_terminals(host.config)
    if not web_terminals.get("enabled"):
        return {}
    cards: dict[str, list[str]] = {
        name: [name] for name in sorted(shared_login_names({"config": host.config}))
    }
    try:
        roles = resolve_authorization_roles(web_terminals)
    except ValueError:
        roles = {}
    for entry in authored_people(web_terminals):
        if not isinstance(entry, Mapping) or not entry_is_shared(dict(entry)):
            continue
        if sign_in_decides_tier(entry, web_terminals):
            continue
        try:
            login = effective_tier(entry, roles, web_terminals.get("default_tier"), strict=True)
        except UnresolvedRoleError:
            continue
        if login:
            cards.setdefault(login, []).append(str(entry.get("name")))
    return cards


def _reserved(
    host: BuildProfile, profiles: Iterable[BuildProfile], installed: Mapping[str, AppOrigin]
) -> Reserved:
    """The names applications may not take, over the root and every login.

    The installed applications' names come from discovery, so none is hand-kept.
    """
    from osprey.cli.build_profile_emit import effective_web_terminals
    from osprey.cli.build_profile_panels import _spelled_panel_ids
    from osprey.deployment.web_terminals.roster import roster_user_names
    from osprey.profiles.web_panels import BUILTIN_PANELS, UNIVERSAL_PANELS

    profiles = list(profiles)
    panels = {pid for profile in profiles for pid in _spelled_panel_ids(profile.config)}
    return Reserved(
        services=frozenset(key for profile in profiles for key in profile.services or {}),
        panels=frozenset(panels) | BUILTIN_PANELS | UNIVERSAL_PANELS,
        servers=frozenset(name for profile in profiles for name in profile.mcp_servers or {}),
        names=fixed_names(),
        roster_users=frozenset(roster_user_names(effective_web_terminals(host.config))),
        installed=dict(installed),
    )


def _config_leaves(config: Any, prefix: str = "") -> Iterable[str]:
    """Every dotted path a ``config:`` block writes, in any spelling."""
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            yield from _config_leaves(value, path)
        else:
            yield path


def _refuse_projected_keys(
    manifests: Iterable[AppManifest], profiles: Mapping[str, BuildProfile]
) -> None:
    """Refuse a ``config:`` key inside a namespace the projection writes.

    The projection is the one writer of these keys, so a spelling beside it would
    be overwritten or carried along without a word. ``apps`` and
    ``dispatch.denied_app_tools`` are refused with no application listed too: an
    absent list is what keeps ``dispatch_denied_tools`` failing closed. A panel id
    or a profile service or server is the rules' to refuse
    (:class:`~osprey.apps.rules.Reserved`).
    """
    owners = {"apps": "", "dispatch.denied_app_tools": ""}
    agent_owners: dict[str, str] = {}
    for m in manifests:
        owners.update({f"services.{key}": m.name for key in m.containers})
        if m.tools is not None:
            owners[f"claude_code.servers.{m.name}"] = m.name
        agent_owners.update({f"claude_code.agents.{agent}": m.name for agent in m.agents})
    found: dict[str, Violation] = {}
    for login, profile in profiles.items():
        source = login if login == ROOT_LOGIN else f"tiers/{login}.yml"
        for path in _config_leaves(profile.config):
            for namespace, app in agent_owners.items():
                if path == namespace or path.startswith(f"{namespace}."):
                    found.setdefault(
                        path,
                        Violation(
                            app,
                            f"config.{path}",
                            "app_agent_override",
                            "an app's agent is on where the app is shown; set "
                            f"`apps.{app}.shown: false` instead of this key",
                            source=source,
                        ),
                    )
            for namespace, app in owners.items():
                if path == namespace or path.startswith(f"{namespace}."):
                    found.setdefault(
                        path,
                        Violation(
                            app,
                            f"config.{path}",
                            "projected_namespace",
                            f"the build projects {namespace} from the application manifests; "
                            "remove this key",
                            source=source,
                        ),
                    )
    if found:
        raise AppManifestError.from_violations(list(found.values()))


def _refuse_unbuildable_services(
    root: Path, manifests: tuple[AppManifest, ...], host: BuildProfile
) -> None:
    """Refuse a projected service the build could not place.

    The projection against the root profile, so a colliding port or a missing
    compose fragment stops the build before the first render rather than inside it.
    """
    from osprey.cli.build_injectors import project_app_services

    found: list[Violation] = []
    for key, service in project_app_services(host, manifests).items():
        template = service.template
        if not template.startswith("osprey.") and not (root / template).is_dir():
            found.append(
                Violation(
                    str(service.config["app"]),
                    f"containers.{key}",
                    "fragment_missing",
                    f"an image container renders from a compose fragment, and {template}/ "
                    "does not exist",
                )
            )
    if found:
        raise AppManifestError.from_violations(found)


def _refuse_hooks(root: Path, manifests: tuple[AppManifest, ...]) -> None:
    """Refuse every place an application could bring hook code into the operator session.

    A ``hooks/`` directory at the application root, a ``.claude/`` directory anywhere
    under it (a settings file there can carry hooks), and a ``hooks`` key in a listed
    skill's frontmatter. An unlisted skill is never installed, so it is not read. The
    directory is the one the manifest came from.
    """
    from osprey.cli.validate_claude_artifacts import _parse_frontmatter

    found: list[Violation] = []
    for manifest in manifests:
        origin = origin_of(manifest, root)
        app_dir = origin.directory
        home = f"{APPS_DIRNAME}/{manifest.name}"
        carriers = [app_dir / "hooks"] if (app_dir / "hooks").is_dir() else []
        carriers += sorted(p for p in app_dir.rglob(".claude") if p.is_dir())
        for carrier in carriers:
            relative = carrier.relative_to(app_dir).as_posix()
            source = origin.manifest_source if origin.installed else home
            found.append(
                Violation(manifest.name, relative, "no_hooks", NO_HOOKS_MESSAGE, source=source)
            )
        for skill in manifest.skills:
            relative = f"skills/{skill}/SKILL.md"
            skill_file = app_dir / relative
            if not skill_file.is_file():
                continue
            frontmatter = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))
            if "hooks" in (frontmatter or {}):
                source = origin.manifest_source if origin.installed else f"{home}/{relative}"
                found.append(
                    Violation(
                        manifest.name,
                        f"skills.{skill}.hooks",
                        "no_hooks",
                        NO_HOOKS_MESSAGE,
                        source=source,
                    )
                )
    if found:
        raise AppManifestError.from_violations(found)


def _refuse_missing_skills(root: Path, manifests: tuple[AppManifest, ...]) -> None:
    """Refuse a listed skill with no ``SKILL.md``, or a name two applications both list.

    The file is looked for in the directory the manifest came from.
    """
    found: list[Violation] = []
    owners: dict[str, str] = {}
    for manifest in manifests:
        origin = origin_of(manifest, root)
        for skill in manifest.skills:
            relative = f"skills/{skill}/SKILL.md"
            if not (origin.directory / relative).is_file():
                where = (
                    f"{origin.directory}/{relative} (installed by {origin.installer})"
                    if origin.installed
                    else f"{APPS_DIRNAME}/{manifest.name}/{relative}"
                )
                found.append(
                    Violation(
                        manifest.name,
                        f"skills.{skill}",
                        "skill_missing",
                        f"{where} does not exist",
                        source=origin.manifest_source,
                    )
                )
            if skill in owners:
                found.append(
                    Violation(
                        manifest.name,
                        f"skills.{skill}",
                        "skill_collision",
                        f"{owners[skill]} already carries a skill named {skill!r}",
                        source=origin.manifest_source,
                    )
                )
            owners.setdefault(skill, manifest.name)
    if found:
        raise AppManifestError.from_violations(found)


def _check_agents(root: Path, manifests: Iterable[AppManifest]) -> dict[str, tuple[AppAgent, ...]]:
    """Read and check every listed agent file; return each application's agents.

    Refused, every violation at once: a listed agent with no file
    (``agent_missing``), a name two applications list (``agent_collision``), a
    name of one of OSPREY's own agents whether or not that agent is on
    (``agent_reserved``), the file's own refusals (:func:`parse_agent`), a tool
    ``mcp__<app>__<tool>`` the application does not declare
    (``agent_tool_unknown``), a delegated tool the agent does not hold
    (``delegate_not_held``), delegated tools the note never names
    (``delegates_unused``) and a note naming delegated tools there are none of
    (``delegates_missing``). An agent may share a name with a skill of its own
    application: the two live in different directories.
    """
    from osprey.registry.mcp import FRAMEWORK_AGENTS

    found: list[Violation] = []
    owners: dict[str, str] = {}
    agents: dict[str, tuple[AppAgent, ...]] = {}
    for manifest in manifests:
        origin = origin_of(manifest, root)
        parsed: list[AppAgent] = []
        for name in manifest.agents:
            relative = f"{AGENTS_DIR}/{name}.md"
            path = origin.directory / relative
            source = (
                origin.manifest_source
                if origin.installed
                else f"{APPS_DIRNAME}/{manifest.name}/{relative}"
            )
            listed_at = f"agents.{name}"
            if name in owners:
                found.append(
                    Violation(
                        manifest.name,
                        listed_at,
                        "agent_collision",
                        f"{owners[name]} already carries an agent named {name!r}",
                        source=origin.manifest_source,
                    )
                )
            owners.setdefault(name, manifest.name)
            if name in FRAMEWORK_AGENTS:
                found.append(
                    Violation(
                        manifest.name,
                        listed_at,
                        "agent_reserved",
                        f"{name!r} is the name of one of OSPREY's own agents; rename the agent",
                        source=origin.manifest_source,
                    )
                )
                continue
            if not path.is_file():
                where = f"{path} (installed by {origin.installer})" if origin.installed else source
                found.append(
                    Violation(
                        manifest.name,
                        listed_at,
                        "agent_missing",
                        f"{where} does not exist",
                        source=origin.manifest_source,
                    )
                )
                continue
            try:
                agent = parse_agent(
                    path.read_text(encoding="utf-8"), app=manifest.name, name=name, source=source
                )
            except AppManifestError as exc:
                found.extend(exc.violations)
                continue
            found.extend(_agent_tool_violations(manifest, agent, source))
            parsed.append(agent)
        if parsed:
            agents[manifest.name] = tuple(parsed)
    if found:
        raise AppManifestError.from_violations(found)
    return agents


def _agent_tool_violations(
    manifest: AppManifest, agent: AppAgent, source: str | None
) -> list[Violation]:
    """The tool rules of one agent against its application's declared tools."""
    tools = manifest.tools
    declared = (
        {*tools.read, *(entry.tool for entry in tools.approval)} if tools is not None else set()
    )
    prefix = f"mcp__{manifest.name}__"
    at = f"agents.{agent.name}"
    found: list[Violation] = []
    unknown = [
        t for t in agent.tools if t.startswith(prefix) and t.removeprefix(prefix) not in declared
    ]
    if unknown:
        found.append(
            Violation(
                manifest.name,
                f"{at}.tools",
                "agent_tool_unknown",
                f"{', '.join(unknown)} is not in the application's tools.read or tools.approval",
                source=source,
            )
        )
    not_tools = [d for d in agent.delegates if d not in declared]
    if not_tools:
        found.append(
            Violation(
                manifest.name,
                f"{at}.delegates",
                "agent_tool_unknown",
                f"{', '.join(not_tools)} is not in the application's tools.read or "
                "tools.approval; delegates names the application's own tools",
                source=source,
            )
        )
    not_held = [d for d in agent.delegates if d in declared and f"{prefix}{d}" not in agent.tools]
    if not_held:
        found.append(
            Violation(
                manifest.name,
                f"{at}.delegates",
                "delegate_not_held",
                f"delegates names {', '.join(not_held)}, which the agent does not hold; "
                "add " + ", ".join(f"{prefix}{d}" for d in not_held) + " to its tools",
                source=source,
            )
        )
    if agent.delegates and DELEGATED_TOOLS not in agent.delegation:
        found.append(
            Violation(
                manifest.name,
                f"{at}.delegation",
                "delegates_unused",
                f"the note never names the delegated tools; write {DELEGATED_TOOLS} where "
                "the brief says which tools to leave to the agent",
                source=source,
            )
        )
    if not agent.delegates and DELEGATED_TOOLS in agent.delegation:
        found.append(
            Violation(
                manifest.name,
                f"{at}.delegates",
                "delegates_missing",
                f"the note names {DELEGATED_TOOLS} but delegates lists no tool; list the "
                "tools to leave to the agent, or write the note without it",
                source=source,
            )
        )
    return found
