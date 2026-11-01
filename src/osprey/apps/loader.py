"""Discover and validate a deployment's applications."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from osprey.apps.errors import AppManifestError, Violation
from osprey.apps.manifest import AppManifest, parse_manifest
from osprey.apps.rules import (
    AppSelection,
    LoginSelection,
    Reserved,
    check_pins,
    deployment_packages,
    fixed_names,
    framework_pins,
    validate,
)
from osprey.utils.workspace import PROFILE_FILENAME

if TYPE_CHECKING:
    from osprey.cli.build_profile_model import BuildProfile

APPS_DIR = "apps"
MANIFEST_FILE = "app.yml"

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
            and nothing of them is read, built or installed.
    """

    manifests: tuple[AppManifest, ...] = ()
    logins: Mapping[str, LoginSelection] = field(default_factory=dict)
    unnamed: tuple[str, ...] = ()


def load(deployment_root: Path, names: Collection[str] | None = None) -> list[AppManifest]:
    """Load the ``apps/<name>/app.yml`` files under ``deployment_root``, sorted by name.

    Only ``app.yml`` is read; other files in an application directory (its
    ``.provenance`` sidecar, fragments, code) are left alone. Hidden entries and
    ``__pycache__`` in ``apps/`` are skipped.

    Args:
        deployment_root: The deployment root.
        names: The applications to read, every other directory left unread;
            ``None`` reads them all.

    Raises:
        AppManifestError: A manifest is refused, its ``name`` differs from its
            directory, a directory holds no ``app.yml``, or ``apps/`` holds a
            plain ``<name>.yml`` login file.
    """
    apps_dir = deployment_root / APPS_DIR
    if not apps_dir.is_dir():
        return []
    manifests: list[AppManifest] = []
    for entry in sorted(apps_dir.iterdir(), key=lambda p: p.name):
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
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
        try:
            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise AppManifestError(entry.name, "", "yaml", f"not valid YAML: {exc}") from None
        manifest = parse_manifest(data, app=entry.name)
        if manifest.name != entry.name:
            raise AppManifestError(
                entry.name,
                "name",
                "name_matches_directory",
                f"name {manifest.name!r} must equal the directory name {entry.name!r}",
            )
        manifests.append(manifest)
    return sorted(manifests, key=lambda m: m.name)


def check_deployment(root: Path) -> DeploymentApps:
    """Run every check a deployment's applications face before anything is written.

    Loads the applications the profile names, leaving every other ``apps/<name>/``
    unread, then validates them against each render's selection and the names the profile already uses, refuses a
    hand-written key inside a namespace the projection owns, a projected port
    that collides, and a package that would move a pinned version. ``osprey
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
    manifests = tuple(load(root, host.apps))
    profiles: dict[str, BuildProfile] = {ROOT_LOGIN: host}
    for _kind, layer in layer_files(root):
        profiles[layer.stem] = resolve_build_document(layer, None, overlays).profile
    logins: dict[str, LoginSelection] = {}
    if manifests:
        cards = _shared_cards(host)
        logins = {
            login: _login_selection(profile, manifests, cards.get(login, ()))
            for login, profile in profiles.items()
        }
        validate(manifests, logins=logins, reserved=_reserved(host, profiles.values()))
    _refuse_projected_keys(manifests, profiles)
    if manifests:
        _refuse_unbuildable_services(root, manifests, host)
    check_pins(
        deployment_packages(root, list(host.dependencies or []), apps=host.apps),
        framework_pins(),
    )
    return DeploymentApps(
        manifests=manifests,
        logins=logins,
        unnamed=tuple(d.name for d in application_dirs(root) if d.name not in host.apps),
    )


def _login_selection(
    profile: BuildProfile, manifests: Iterable[AppManifest], shared_cards: Iterable[str]
) -> LoginSelection:
    """One render's selection, read from its merged profile."""
    from osprey.cli.build_profile_merge import _rendered_control_system
    from osprey_connectors.types import tier_writes

    cards = tuple(shared_cards)
    return LoginSelection(
        audience="shared" if cards else "person",
        apps={m.name: AppSelection(shown=profile.app_shown(m.name)) for m in manifests},
        tier=tier_writes(_rendered_control_system(profile.config)),
        shared_cards=cards,
    )


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


def _reserved(host: BuildProfile, profiles: Iterable[BuildProfile]) -> Reserved:
    """The names applications may not take, over the root and every login."""
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
    for m in manifests:
        owners.update({f"services.{key}": m.name for key in m.containers})
        if m.tools is not None:
            owners[f"claude_code.servers.{m.name}"] = m.name
    found: dict[str, Violation] = {}
    for login, profile in profiles.items():
        source = login if login == ROOT_LOGIN else f"tiers/{login}.yml"
        for path in _config_leaves(profile.config):
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
