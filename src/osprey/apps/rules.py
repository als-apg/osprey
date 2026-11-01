"""The rules a deployment's applications obey before anything is rendered.

:func:`validate` checks what holds for every render: each manifest on its own,
the manifests against each other, and against the names the resolved profile
and OSPREY already use. :func:`check_pins` refuses an application package that
would move a version the framework pins, before the joint resolve would fail
without naming who asked; :func:`deployment_packages` lists the packages it
reads. Each collects every violation and raises one
:class:`~osprey.apps.errors.AppManifestError`, so ``osprey app check`` and
``osprey build`` report the same list.
"""

from __future__ import annotations

import posixpath
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from osprey.apps.errors import AppManifestError, Violation
from osprey.apps.manifest import AppManifest, launch_token_env
from osprey.bluesky_tool_names import QUEUE_CONTROL_TOOLS, QUEUE_TOOLS

__all__ = [
    "HOOK_DISPATCHED_TOOLS",
    "QUEUE_TOOL_PREFIX",
    "AppSelection",
    "LoginSelection",
    "Reserved",
    "builtin_credential_names",
    "check_pins",
    "deployment_packages",
    "fixed_names",
    "framework_pins",
    "validate",
]

#: Built-in tools the approval hook dispatches on their short name alone, whatever
#: server they come from. An application tool spelled the same would be read as
#: the built-in. Restated from ``osprey_approval.py`` (a hook script this package
#: cannot import); ``tests/apps/test_manifest_rules.py`` pins them to the hook.
HOOK_DISPATCHED_TOOLS: tuple[str, ...] = (
    "execute",
    "execute_file",
    "pyaml_measure",
    "channel_write",
)

#: The prefix that makes a tool lane-addressed: the hooks carve such tools out of
#: the posture gates (``_QUEUE_TOOL_PREFIX`` and ``_LANE_ADDRESSED_TOOL`` in
#: ``osprey_approval.py``, ``_is_lane_addressed`` in ``osprey_writes_check.py``,
#: which reads ``hook_config.json``'s ``lane_addressed_tools``). Restated from
#: ``osprey_approval.py``; the test pins the spelling to the hook.
QUEUE_TOOL_PREFIX = "queue_"

#: Every field an approval binds is spelled with this prefix; the approval itself
#: records :data:`APPROVED_TARGET`, so an entry may require it without binding it.
APPROVED_PREFIX = "approved_"
APPROVED_TARGET = "approved_target"


@dataclass(frozen=True)
class AppSelection:
    """One login's choice for one application: whether it is shown."""

    shown: bool


@dataclass(frozen=True)
class LoginSelection:
    """What one login's render selects, built by the caller from its merged config.

    Attributes:
        audience: ``shared`` when ``shared:`` names the login or any roster card
            resolving to it is shared; ``person`` otherwise.
        apps: The login's ``apps:`` block, by application name.
        tier: The login's ``control_system.writes`` tier.
        shared_cards: The roster cards that make the audience ``shared``; empty
            for ``person``.
    """

    audience: Literal["person", "shared"]
    apps: Mapping[str, AppSelection]
    tier: Literal["never", "asked", "allowed"]
    shared_cards: tuple[str, ...]

    def arms(self, app: str) -> bool:
        """Whether this login arms *app*: shows it, under a tier that writes.

        ``asked`` and ``allowed`` both arm; which writes then need a human's
        approval, and which machine they reach, is decided per write at runtime.
        """
        selection = self.apps.get(app)
        return selection is not None and selection.shown and self.tier != "never"


@dataclass(frozen=True)
class Reserved:
    """Names an application may not take, one namespace each.

    Attributes:
        services: The resolved profile's service keys.
        panels: Its ``web.panels`` ids, built-ins included.
        servers: Its ``mcp_servers`` names.
        names: :func:`fixed_names`.
        roster_users: The ``modules.web_terminals`` roster's user names, each
            a per-user spool and state directory an audit identity may not take.
    """

    services: frozenset[str]
    panels: frozenset[str]
    servers: frozenset[str]
    names: frozenset[str]
    roster_users: frozenset[str] = frozenset()


@lru_cache(maxsize=1)
def fixed_names() -> frozenset[str]:
    """Names OSPREY keeps for itself whether or not a profile enables them.

    The framework's MCP server names, the built-in panel ids and OSPREY's own
    service keys. An application is also its projected server name, so one
    named like a framework server is refused even where that server is off.
    """
    from osprey.cli.build_profile_load import bundled_service_names
    from osprey.profiles.web_panels import BUILTIN_PANELS, UNIVERSAL_PANELS
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    return (
        frozenset(FRAMEWORK_SERVERS) | BUILTIN_PANELS | UNIVERSAL_PANELS | bundled_service_names()
    )


@lru_cache(maxsize=1)
def builtin_credential_names() -> frozenset[str]:
    """The environment variables OSPREY mints for its own services, lane tokens included."""
    from osprey.deployment.container_lifecycle import builtin_token_vars

    return builtin_token_vars()


def validate(
    manifests: Sequence[AppManifest],
    *,
    logins: Mapping[str, LoginSelection],
    reserved: Reserved,
) -> None:
    """Check the rules that hold for every render of the deployment.

    Args:
        manifests: Every application of the deployment.
        logins: Every login's selection, by login name.
        reserved: The names applications may not take.

    Raises:
        AppManifestError: Carrying every violation found.
    """
    found: list[Violation] = []
    seen_names: set[str] = set()
    container_owner: dict[str, str] = {}
    panel_owner: dict[str, str] = {}
    identity_owner: dict[str, str] = {}
    for manifest in manifests:
        found.extend(_check_app(manifest, reserved, container_owner, panel_owner, seen_names))
        found.extend(_check_audit_identity(manifest, reserved, identity_owner))
    found.extend(_check_arming(manifests, logins))
    if found:
        raise AppManifestError.from_violations(found)


def _check_arming(
    manifests: Sequence[AppManifest], logins: Mapping[str, LoginSelection]
) -> list[Violation]:
    """The rules for each application a login arms.

    An armed application with write tools needs a container that drives the
    target, or its writes reach nothing; a shared login arms no application.
    """
    out: list[Violation] = []
    for login, selection in logins.items():
        for m in manifests:
            if not selection.arms(m.name):
                continue
            writes = m.tools is not None and bool(m.tools.approval)
            if writes and not any(c.drives_target for c in m.containers.values()):
                out.append(
                    Violation(
                        m.name,
                        "containers",
                        "armed_drives_target",
                        f"login {login} arms {m.name}, whose write tools need a container "
                        "that sets drives_target: true",
                    )
                )
            if selection.audience == "shared":
                cards = ", ".join(selection.shared_cards) or "shared:"
                out.append(
                    Violation(
                        "",
                        f"logins.{login}",
                        "shared_login_tier",
                        f"login {login} is shared (by {cards}) and arms application {m.name}; "
                        "a shared login renders control_system.writes: never",
                    )
                )
    return out


def _check_audit_identity(
    m: AppManifest, reserved: Reserved, identity_owner: dict[str, str]
) -> list[Violation]:
    """An application's audit identity names its own spool and state directories.

    So it may not be a service's own identity, another application's, or a
    roster user's: each of those already writes there.
    """
    from osprey.deployment.web_terminals.render import _RESERVED_AUDIT_IDENTITY_RE

    identity = m.audit_identity
    if _RESERVED_AUDIT_IDENTITY_RE.fullmatch(identity):
        rule, message = "audit_identity_reserved", f"{identity!r} is a service's own identity"
    elif identity in identity_owner:
        rule = "audit_identity_unique"
        message = f"{identity!r} is already the audit identity of {identity_owner[identity]!r}"
    elif identity in reserved.roster_users:
        rule = "audit_identity_roster"
        message = f"{identity!r} is a user of the web-terminal roster"
    else:
        identity_owner[identity] = m.name
        return []
    return [Violation(m.name, "audit_identity", rule, message)]


def _check_app(
    m: AppManifest,
    reserved: Reserved,
    container_owner: dict[str, str],
    panel_owner: dict[str, str],
    seen_names: set[str],
) -> list[Violation]:
    out: list[Violation] = []

    def refuse(field: str, rule: str, message: str) -> None:
        out.append(Violation(m.name, field, rule, message))

    if m.name in seen_names:
        refuse("name", "name_unique", f"another application is also named {m.name!r}")
    seen_names.add(m.name)
    if m.name in reserved.names:
        refuse("name", "reserved_name", f"{m.name!r} is a name OSPREY keeps for itself")

    for key, container in m.containers.items():
        at = f"containers.{key}"
        if container.drives_target and container.runtime != "osprey":
            refuse(
                f"{at}.drives_target",
                "drives_target_runtime",
                "a container that drives the target runs runtime: osprey",
            )
        if container.control_identity and not container.drives_target:
            refuse(
                f"{at}.control_identity",
                "control_identity_drives_target",
                "control_identity is for a container that sets drives_target: true",
            )
        if key in reserved.services:
            refuse(at, "container_key_unique", f"{key!r} is already a service of the profile")
        elif key in container_owner:
            refuse(
                at,
                "container_key_unique",
                f"{key!r} is already a container of application {container_owner[key]!r}",
            )
        else:
            container_owner[key] = m.name

    panel = m.panel
    if panel is not None:
        if panel.container not in m.containers:
            refuse(
                "panel.container",
                "panel_container",
                f"the panel is served by {panel.container!r}, which is not a container "
                f"of this application ({', '.join(m.containers)})",
            )
        if panel.id in reserved.panels:
            refuse("panel.id", "panel_id_unique", f"{panel.id!r} is already a web panel")
        elif panel.id in panel_owner:
            refuse(
                "panel.id",
                "panel_id_unique",
                f"{panel.id!r} is already the panel of application {panel_owner[panel.id]!r}",
            )
        else:
            panel_owner[panel.id] = m.name
        if panel.launch_token:
            env = launch_token_env(m.name)
            if env in builtin_credential_names():
                refuse(
                    "panel.launch_token",
                    "launch_token_env",
                    f"the launch token's variable {env} is already one of OSPREY's own "
                    "credentials; rename the application",
                )

    for index, entry in enumerate(m.vendor):
        if not _inside(entry.into):
            refuse(
                f"vendor.{index}.into",
                "vendor_into",
                f"{entry.into!r} is not a path inside apps/{m.name}/",
            )

    if m.tools is not None:
        out.extend(_check_tools(m, reserved))
    return out


def _check_tools(m: AppManifest, reserved: Reserved) -> list[Violation]:
    tools = m.tools
    assert tools is not None
    out: list[Violation] = []

    def refuse(field: str, rule: str, message: str) -> None:
        out.append(Violation(m.name, field, rule, message))

    if m.name in reserved.servers:
        refuse(
            "tools",
            "server_name_unique",
            f"the tool server is named for the application, and {m.name!r} is already "
            "an MCP server of the profile",
        )
    if tools.shown_when == "panel" and m.panel is None:
        refuse("tools.shown_when", "shown_when_panel", "shown_when: panel needs a panel")

    named = [(f"tools.read.{i}", name) for i, name in enumerate(tools.read)]
    named += [(f"tools.approval.{i}.tool", e.tool) for i, e in enumerate(tools.approval)]
    for field, name in named:
        if name in HOOK_DISPATCHED_TOOLS:
            refuse(
                field,
                "builtin_tool_name",
                f"{name!r} is a built-in tool the approval hook handles by name",
            )
        elif (
            name in QUEUE_CONTROL_TOOLS or name in QUEUE_TOOLS or name.startswith(QUEUE_TOOL_PREFIX)
        ):
            refuse(
                field,
                "queue_tool_name",
                f"{name!r} is a queue tool name ({QUEUE_TOOL_PREFIX}*), which the write "
                "gates address by lane",
            )

    get_routes = {r.path for r in m.panel.agent_routes if r.method == "GET"} if m.panel else set()
    for index, entry in enumerate(tools.approval):
        at = f"tools.approval.{index}"
        if entry.state_path is not None:
            if m.panel is None:
                refuse(
                    f"{at}.state_path",
                    "state_path_panel",
                    f"{entry.tool} fetches state from the panel, and the application has none",
                )
            elif entry.state_path not in get_routes:
                refuse(
                    f"{at}.state_path",
                    "state_path_route",
                    f"{entry.state_path} must be a GET agent route of the panel",
                )
            if "deny_in_jobs" in tools.model_fields_set and not tools.deny_in_jobs:
                refuse(
                    "tools.deny_in_jobs",
                    "state_path_in_jobs",
                    f"{entry.tool} binds fetched state, which a job cannot fetch; it is "
                    "denied in jobs, so do not set deny_in_jobs: false",
                )
        bound = [
            (f"{at}.bind.state_fields.{i}.field", s.field)
            for i, s in enumerate(entry.bind.state_fields)
        ]
        bound += [
            (f"{at}.bind.input_digests.{i}.field", d.field)
            for i, d in enumerate(entry.bind.input_digests)
        ]
        produced = {APPROVED_TARGET}
        for field, name in bound:
            if not name.startswith(APPROVED_PREFIX):
                refuse(
                    field,
                    "bound_field_name",
                    f"{entry.tool} binds {name!r}; a bound field is named {APPROVED_PREFIX}*",
                )
            elif name in produced:
                refuse(
                    field,
                    "bound_twice",
                    f"{entry.tool} binds {name!r} more than once"
                    if name != APPROVED_TARGET
                    else f"{entry.tool} binds {name!r}, which the approval records itself",
                )
            produced.add(name)
        required = [(f"{at}.require.all_of.{i}", f) for i, f in enumerate(entry.require.all_of)]
        required += [
            (f"{at}.require.one_of.{i}", f)
            for i, group in enumerate(entry.require.one_of)
            for f in group
        ]
        for i, digest in enumerate(entry.bind.input_digests):
            if digest.field not in entry.require.all_of:
                refuse(
                    f"{at}.bind.input_digests.{i}.field",
                    "digest_required",
                    f"{entry.tool} binds the input digest {digest.field!r}, which "
                    "require.all_of must name: every bound digest is required",
                )
        for field, name in required:
            if name not in produced:
                refuse(
                    field,
                    "approved_fields",
                    f"{entry.tool} requires {name!r}, which is neither {APPROVED_TARGET} "
                    "nor a field its bind produces",
                )
    return out


def _inside(into: str) -> bool:
    """Whether the relative path *into* stays inside the directory it is joined to."""
    if not into or "\\" in into or into.startswith("/"):
        return False
    normal = posixpath.normpath(into)
    return normal not in (".", "..") and not normal.startswith("../")


# --- pins ---------------------------------------------------------------------


def framework_pins() -> dict[str, str]:
    """The installed ``osprey-framework``'s exact (``==``) requirements, base and every extra.

    Keyed by canonical distribution name.
    """
    import importlib.metadata

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pins: dict[str, str] = {}
    for raw in importlib.metadata.requires("osprey-framework") or ():
        req = Requirement(raw)
        specifiers = list(req.specifier)
        if len(specifiers) == 1 and specifiers[0].operator == "==":
            pins[canonicalize_name(req.name)] = specifiers[0].version
    return pins


def check_pins(packages: Sequence[Path], pins: Mapping[str, str]) -> None:
    """Refuse a package whose requirements would move a pinned version.

    Reads each ``pyproject.toml``'s ``[project] dependencies`` as written; no
    resolver runs and nothing is fetched.

    Args:
        packages: ``pyproject.toml`` paths, from :func:`deployment_packages`.
        pins: Version by distribution name, from :func:`framework_pins`.

    Raises:
        AppManifestError: Naming, for each moved pin, the package, the
            requirement and the pinned version.
    """
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    pinned = {canonicalize_name(name): version for name, version in pins.items()}
    found: list[Violation] = []
    for path in packages:
        owner = path.parent.name
        source = str(path)
        try:
            project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
            dependencies = list(project.get("dependencies", []))
        except (OSError, tomllib.TOMLDecodeError, AttributeError, TypeError) as exc:
            found.append(Violation(owner, "", "pyproject", f"cannot be read: {exc}", source=source))
            continue
        for index, raw in enumerate(dependencies):
            field = f"project.dependencies.{index}"
            try:
                req = Requirement(str(raw))
            except InvalidRequirement as exc:
                found.append(Violation(owner, field, "pyproject", str(exc), source=source))
                continue
            name = canonicalize_name(req.name)
            if name not in pinned:
                continue
            version = pinned[name]
            wanted = req.url or str(req.specifier)
            if req.url or not req.specifier.contains(version, prereleases=True):
                found.append(
                    Violation(
                        owner,
                        field,
                        "pin_moved",
                        f"{owner} requires {req.name} {wanted}, which moves it from "
                        f"{version}, the version OSPREY pins",
                        source=source,
                    )
                )
    if found:
        raise AppManifestError.from_violations(found)


def deployment_packages(
    root: Path, dependencies: Sequence[str], *, apps: Iterable[str]
) -> list[Path]:
    """The ``pyproject.toml`` of every Python package the deployment builds.

    Each listed ``apps/<name>/`` holding one, sorted by name, then each
    local-path entry of the profile's ``dependencies`` (``./x``, ``~/x``, ``/x``,
    ``-e`` forms, or ``name @ file://...``) that holds one, in order, once each.
    ``apps/`` holds applications only: a directory the profile does not list is
    switched off, and a package that is no application is a ``dependencies`` entry.

    Args:
        root: The deployment root.
        dependencies: The profile's ``dependencies`` entries.
        apps: The applications the profile's ``apps:`` block lists.
    """
    found: list[Path] = []
    for name in sorted(apps):
        candidate = root / "apps" / name / "pyproject.toml"
        if candidate.is_file():
            found.append(candidate.resolve())
    for raw in dependencies:
        directory = local_path(raw, root)
        if directory is None:
            continue
        candidate = (directory / "pyproject.toml").resolve()
        if candidate.is_file() and candidate not in found:
            found.append(candidate)
    return found


def local_path(raw: str, root: Path) -> Path | None:
    """The path a ``dependencies`` entry names on this host, or ``None`` for an index one."""
    spec = raw.strip()
    if spec.startswith("-e "):
        spec = spec[3:].strip()
    if spec.startswith((".", "~", "/")):
        path = Path(spec).expanduser()
        return path if path.is_absolute() else root / path
    if " @ " in spec:
        url = urlparse(spec.split(" @ ", 1)[1].strip())
        if url.scheme == "file":
            return Path(unquote(url.path))
    return None
