"""Templates and render contexts for ``osprey deploy scaffold``.

The scaffolding verb emits two files into a facility repo — a CI pipeline and a
post-deploy health check — and both are Jinja templates under
``osprey/templates/deploy/``. This module owns three things the verb needs and
nothing else: which template a platform selects, what context each template
expects, and how to turn a profile into that context.

The split matters because the templates are held to a byte-for-byte golden
(``tests/deployment/goldens/``). Layout decisions — column alignment, comment
wording, divider widths — belong to the templates; the facts a facility supplies
belong here, where they can be derived once and tested directly.

Every input is a profile fact. Deployment coordinates come from the ``deploy:``
block via :func:`osprey.cli.build_profile_deploy.parse_deploy_block`; everything
else is read from the profile proper. The only value that is not a profile fact
is the project name, which is the facility repo's directory name and therefore
known only at emission time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

import osprey

from .build_profile_deploy import DeployConfig

#: Directory holding the shipped deploy templates.
TEMPLATE_DIR: Path = Path(osprey.__file__).parent / "templates" / "deploy"

#: CI platform -> pipeline template. A plain dict rather than a registration
#: API: the set of platforms with a shipped pipeline is closed, small, and
#: already enumerated by ``build_profile_deploy.SUPPORTED_CI_PLATFORMS`` — a
#: second way to add one would just be a second place to forget.
CI_TEMPLATES: dict[str, str] = {"gitlab": "gitlab-ci.yml.j2"}

#: The post-deploy health check, emitted for every platform.
VERIFY_TEMPLATE: str = "verify.sh.j2"

#: Provenance markers the emitted files carry. A file whose first lines name one
#: of these was written by the scaffolder and may be re-emitted; anything else is
#: treated as hand-written.
CI_MARKER: str = "deploy/gitlab-ci"
VERIFY_MARKER: str = "deploy/verify"

#: The CI-only variable holding the deploy host's SSH private key. Fixed rather
#: than profile-named: it authenticates the pipeline to the host and is never
#: part of the deployment's own environment, so the profile has no business
#: declaring it under ``env.required``.
DEPLOY_SSH_KEY_VAR: str = "OSPREY_DEPLOY_SSH_KEY"

#: Where the profile is checked out inside the facility repo, and therefore the
#: path every emitted file refers to it by.
PROFILE_PATH: str = "profile/profile.yml"

#: Probes run on the deploy host itself, so every target is loopback regardless
#: of how the outside world reaches the service.
PROBE_HOST: str = "127.0.0.1"


@dataclass(frozen=True)
class Probe:
    """One health check in the emitted ``verify.sh``.

    Attributes:
        kind: ``http`` for an endpoint that answers a GET, ``tcp`` for a
            listener that speaks something else.
        label: Text inside the quotes, ``<service>: <what> on <port>``.
        port: Host port the probe targets. Also the sort key within a group.
        path: URL path, for ``http`` probes.
        comment: Lines emitted above the probe, explaining a non-obvious choice.
    """

    kind: str
    label: str
    port: int
    path: str = "/"
    comment: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        """The URL an ``http`` probe fetches."""
        return f"http://{PROBE_HOST}:{self.port}{self.path}"

    @property
    def target(self) -> str:
        """The probe helper's arguments after the label."""
        return self.url if self.kind == "http" else f"{PROBE_HOST} {self.port}"


@dataclass(frozen=True)
class ProbeGroup:
    """A selectable section of ``verify.sh``.

    Attributes:
        id: Name the operator passes as an argument to run only this group.
        divider: Section title in the source's own comment divider.
        heading: Title printed when the group runs.
        comment: Lines emitted under the divider.
        probes: The group's probes, ordered by port.
    """

    id: str
    divider: str
    heading: str
    probes: tuple[Probe, ...]
    comment: tuple[str, ...] = ()


@dataclass
class CIContext:
    """Everything ``gitlab-ci.yml.j2`` renders from.

    Attributes:
        facility_name: The profile's ``name:`` — the pipeline's title.
        osprey_version: Installed framework version, for the provenance header.
        project_name: Facility repo directory name; every path keys off it.
        build_dir: Where ``osprey build`` lands, ``build/<project_name>``.
        requirement: pip requirement pinning the floor the profile declares.
        registry_url: Full image-name prefix, or ``None`` when no registry is
            configured and none is needed.
        registry_host: The half ``docker login`` takes.
        registry_token_var: Variable naming the registry credential.
        deploy_host: Address the pipeline SSHes to.
        deploy_user: SSH user owning the checkout.
        deploy_path: The checkout's absolute path on that host.
        service_images: Profile-owned services that get an image-build job.
        external_projects: Other projects' images this deployment also pulls.
    """

    facility_name: str
    osprey_version: str
    project_name: str
    build_dir: str
    requirement: str
    registry_url: str | None
    registry_host: str | None
    registry_token_var: str | None
    deploy_host: str
    deploy_user: str
    deploy_path: str
    service_images: list[str] = field(default_factory=list)
    external_projects: list[dict[str, str]] = field(default_factory=list)

    @property
    def has_images_stage(self) -> bool:
        """Whether anything is built or pulled between validate and deploy."""
        return bool(self.registry_url) and bool(self.service_images or self.external_projects)


@dataclass
class VerifyContext:
    """Everything ``verify.sh.j2`` renders from.

    Attributes:
        facility_name: The profile's ``name:`` — the script's title.
        osprey_version: Installed framework version, for the provenance header.
        groups: Probe groups, in the order they run.
    """

    facility_name: str
    osprey_version: str
    groups: list[ProbeGroup] = field(default_factory=list)

    @property
    def usage_group(self) -> str:
        """The group the usage comment shows as an example argument."""
        return self.groups[0].id if self.groups else ""


def render(template_name: str, context: CIContext | VerifyContext) -> str:
    """Render one deploy template.

    Args:
        template_name: File name under :data:`TEMPLATE_DIR`.
        context: The matching context dataclass.

    Returns:
        The rendered file, ready to write verbatim.
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    return env.get_template(template_name).render(
        ctx=context,
        ssh_key_var=DEPLOY_SSH_KEY_VAR,
        profile_path=PROFILE_PATH,
    )


def build_ci_context(
    profile: dict[str, Any],
    deploy: DeployConfig,
    profile_dir: Path,
    project_name: str,
    osprey_version: str | None = None,
) -> CIContext:
    """Derive the CI pipeline's context from a profile.

    Args:
        profile: The resolved raw profile dict.
        deploy: The parsed ``deploy:`` block.
        profile_dir: Directory holding ``profile.yml``, used to find which
            declared services carry a Dockerfile.
        project_name: The facility repo's directory name.
        osprey_version: Version for the provenance header; defaults to the
            installed framework's.

    Returns:
        The context :data:`CI_TEMPLATES` renders against.
    """
    registry_url = deploy.registry.url if deploy.registry else None
    return CIContext(
        facility_name=str(profile.get("name", project_name)),
        osprey_version=osprey_version or osprey.__version__,
        project_name=project_name,
        build_dir=f"build/{project_name}",
        requirement=_pip_requirement(profile.get("requires_osprey_version")),
        registry_url=registry_url,
        registry_host=registry_url.split("/", 1)[0] if registry_url else None,
        registry_token_var=deploy.registry.token_env_var if deploy.registry else None,
        deploy_host=deploy.host.fqdn or deploy.host.name,
        deploy_user=deploy.host.user,
        deploy_path=deploy.host.project_path,
        service_images=service_image_names(profile, profile_dir),
        external_projects=[
            {
                "name": project.name,
                "url": project.url,
                "image": project.image,
                "token_env_var": project.token_env_var or "",
            }
            for project in deploy.external_projects
        ],
    )


def build_verify_context(
    profile: dict[str, Any],
    osprey_version: str | None = None,
) -> VerifyContext:
    """Derive the post-deploy health check's context from a profile.

    Args:
        profile: The resolved raw profile dict.
        osprey_version: Version for the provenance header; defaults to the
            installed framework's.

    Returns:
        The context :data:`VERIFY_TEMPLATE` renders against.
    """
    groups: list[ProbeGroup] = []

    service_probes = _service_probes(profile)
    if service_probes:
        groups.append(
            ProbeGroup(
                id="services",
                divider="Deployed services",
                heading="Services",
                probes=service_probes,
            )
        )

    web_probes = _web_probes(profile)
    if web_probes:
        groups.append(
            ProbeGroup(
                id="web",
                divider="Web terminal tier",
                heading="Web terminals",
                comment=(
                    "One container per roster user behind nginx. A user whose terminal is down is",
                    "a person who cannot work, so each is named rather than summarized.",
                ),
                probes=web_probes,
            )
        )

    return VerifyContext(
        facility_name=str(profile.get("name", "OSPREY")),
        osprey_version=osprey_version or osprey.__version__,
        groups=groups,
    )


def service_image_names(profile: dict[str, Any], profile_dir: Path) -> list[str]:
    """List the profile-owned services that earn an image-build job.

    A service qualifies by being declared under the profile's ``services:``
    block *and* carrying a Dockerfile. The first half is what excludes the
    packaged services: the virtual accelerator's service directory is copied
    into a materialized profile with a Dockerfile of its own, but it is declared
    by the ``virtual_accelerator:`` injector rather than by ``services:``, and
    its image comes from its own upstream.

    Args:
        profile: The resolved raw profile dict.
        profile_dir: Directory holding ``profile.yml``.

    Returns:
        Service names, in declaration order.
    """
    declared = profile.get("services")
    if not isinstance(declared, dict):
        return []
    return [
        name for name in declared if (profile_dir / "services" / str(name) / "Dockerfile").is_file()
    ]


def _pip_requirement(requires: Any) -> str:
    """Turn ``requires_osprey_version`` into a pip requirement."""
    specifier = str(requires).strip() if isinstance(requires, str) else ""
    return f"osprey-framework{specifier}"


def _service_probes(profile: dict[str, Any]) -> tuple[Probe, ...]:
    """Probe every service the deployment actually runs.

    Ordered by port rather than by declaration: the deployed set is assembled
    from three unrelated places (the ``config:`` list, the virtual-accelerator
    injector, the ``services:`` block), so port order is the only ordering that
    is stable under an edit to any one of them — and it reads, top to bottom,
    the way ``docker ps`` does.
    """
    probes: list[Probe] = []

    virtual_accelerator = profile.get("virtual_accelerator")
    if isinstance(virtual_accelerator, dict):
        port = _as_port(virtual_accelerator.get("port"), 5064)
        probes.append(
            Probe(
                kind="tcp",
                label=f"virtual-accelerator: Channel Access on {port}",
                port=port,
            )
        )

    config = _config_block(profile)
    deployed = _dotted(config, "deployed_services")
    if isinstance(deployed, list) and "openobserve" in deployed:
        port = _as_port(_dotted(config, "services.openobserve.port"), 5080)
        probes.append(
            Probe(
                kind="http",
                label=f"openobserve: telemetry store on {port}",
                port=port,
                path="/healthz",
            )
        )

    probes.extend(_facility_service_probes(profile))
    return tuple(sorted(probes, key=lambda probe: probe.port))


def _facility_service_probes(profile: dict[str, Any]) -> list[Probe]:
    """Probe the facility's own services, named by what they serve.

    A service that publishes a port an ``mcp_servers:`` entry also names is that
    server's container, so the probe borrows the tools the entry allows for its
    description — and stops at the listener, because MCP endpoints do not answer
    a bare GET.
    """
    declared = profile.get("services")
    if not isinstance(declared, dict):
        return []

    probes: list[Probe] = []
    for name, spec in declared.items():
        port = _as_port((spec or {}).get("config", {}).get("port"), 0)
        if not port:
            continue
        tools = _mcp_tools_on_port(profile, port)
        if tools:
            probes.append(
                Probe(
                    kind="tcp",
                    label=f"{name}: {tools} on {port}",
                    port=port,
                    comment=(
                        "A bare GET to an MCP endpoint is not a request it answers, "
                        "so this stops at",
                        "the listener — the same thing the container's own healthcheck "
                        "settles for.",
                    ),
                )
            )
        else:
            probes.append(Probe(kind="tcp", label=f"{name}: service on {port}", port=port))
    return probes


def _mcp_tools_on_port(profile: dict[str, Any], port: int) -> str:
    """Name the tools an ``mcp_servers:`` entry on ``port`` allows."""
    servers = profile.get("mcp_servers")
    if not isinstance(servers, dict):
        return ""
    for spec in servers.values():
        if not isinstance(spec, dict) or _as_port(spec.get("port"), 0) != port:
            continue
        allow = (spec.get("permissions") or {}).get("allow")
        if isinstance(allow, list) and allow:
            return ", ".join(str(tool).replace("_", " ") for tool in allow)
    return ""


def _web_probes(profile: dict[str, Any]) -> tuple[Probe, ...]:
    """Probe the nginx landing page and one terminal per roster user."""
    web = _dotted(_config_block(profile), "modules.web_terminals")
    if not isinstance(web, dict) or not web.get("enabled"):
        return ()

    probes = [
        Probe(
            kind="http",
            label=f"nginx: landing page on {_as_port(web.get('nginx_port'), 9080)}",
            port=_as_port(web.get("nginx_port"), 9080),
        )
    ]

    base = _as_port(web.get("web_base_port"), 9091)
    default_persona = str(web.get("default_persona", ""))
    users = web.get("users")
    for position, entry in enumerate(users if isinstance(users, list) else []):
        if isinstance(entry, str):
            name, index, persona = entry, position, default_persona
        elif isinstance(entry, dict):
            name = str(entry.get("name", ""))
            index = _as_port(entry.get("index"), position)
            persona = str(entry.get("persona") or default_persona)
        else:
            continue
        if not name:
            continue
        port = base + index
        suffix = f" ({persona})" if persona else ""
        probes.append(Probe(kind="http", label=f"{name}{suffix} on {port}", port=port))

    return tuple(sorted(probes, key=lambda probe: probe.port))


def _config_block(profile: dict[str, Any]) -> dict[str, Any]:
    """The profile's ``config:`` overrides, or an empty mapping."""
    config = profile.get("config")
    return config if isinstance(config, dict) else {}


def _dotted(config: dict[str, Any], key: str) -> Any:
    """Read a config override written either dotted or nested.

    ``config:`` entries are dotted keys by convention, but a nested mapping is
    accepted by the loader, so both spellings have to resolve here too.
    """
    if key in config:
        return config[key]
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _as_port(value: Any, default: int) -> int:
    """Coerce a profile-supplied port, falling back to the framework default."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
