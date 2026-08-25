"""What an attached render is told about the services its host deploys.

An *attached* project (``deploy_services: false`` — every web-terminal persona)
scaffolds no services and so never reaches the build-step injectors that write
``services.<name>`` blocks into a deploying project's config. Its in-container
clients still resolve their endpoints from those blocks — the bluesky MCP
server through :func:`osprey.bluesky_bridge_connection.resolve_bridge_url`, the
VA connector through
:func:`osprey_connectors.control_system.va_connector.fill_gateway_ports`, hybrid
logbook search through :func:`osprey.deployment.qmd_service.resolve_qmd_service_config`
— and each either bottoms out in a compiled-in default when the block is
absent, or refuses at first use inside a per-user container. At the shipped
defaults a default equals the port the host publishes, so the drift is
invisible until an operator moves a port on the hosting profile and every
persona keeps dialing the old one.

The build therefore PROJECTS: for an attached render built beside its hosting
deployment, the client-facing facts the Reach Contract registry
(:mod:`osprey.deployment.reach`) declares are copied from the host's rendered
config into the attached render on the ordinary config-override path, exactly
as :func:`osprey.cli.build_profile_archiver.va_archiver_config_overrides` does
for the archive: the build already knows where every service is, and no
persona restates it. Built with no host in its repo, an attached profile is
projected from what its app template deploys at the shipped defaults (it
extends a deployment of that template), with its own ``config:`` laid over
them. A deploying project gets nothing from here; its injectors write the
full blocks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from osprey.deployment.reach import REACH_CONTRACTS, dotted_get, project_attached_overrides


def attached_render_overrides(
    host_config: Mapping[str, Any] | None,
    rendered_config: Mapping[str, Any],
    *,
    selected_panels: Iterable[str] = (),
) -> dict[str, Any]:
    """The dotted keys projected into an attached render from its host's render.

    Thin, named entry point over
    :func:`osprey.deployment.reach.project_attached_overrides` — the build's
    side of the contract — so the build reads one function and the registry
    stays the one place the keys are declared.

    Args:
        host_config: The hosting deployment's rendered config — or, for an
            attached profile built with no host in its repo, the app
            template rendered as a deployment with the profile's ``config:``
            over it. ``None`` projects nothing.
        rendered_config: The attached render's config after its own
            ``config:`` overlay — what the projection's gates read.
        selected_panels: The attached profile's ``web_panels`` selection.

    Returns:
        Dotted config keys to apply after the profile's own; empty without a
        host.
    """
    return project_attached_overrides(host_config, rendered_config, selected_panels=selected_panels)


def _spelled_values(config: Any, dotted_key: str) -> list[tuple[str, Any]]:
    """Every way *config* writes *dotted_key*, with the value each gives.

    A ``config:`` block may spell one leaf as the whole dotted key
    (``services.bluesky.port``), as a dotted prefix over a mapping
    (``services.bluesky: {port: …}``), fully nested
    (``services: {bluesky: {port: …}}``), or any mix of the two — every
    split of the segments is legal YAML that
    :func:`osprey.cli.build_profile_model._config_lookup` reads, and all of
    them reach the same rendered leaf. Each spelling found is reported the way
    it was written, so a refusal can name the line to remove.
    """
    return _spellings(config, dotted_key.split("."), [])


def _spellings(node: Any, segments: list[str], written: list[str]) -> list[tuple[str, Any]]:
    """Walk every split of *segments* through the mapping *node*."""
    if not isinstance(node, dict):
        return []
    found: list[tuple[str, Any]] = []
    for cut in range(1, len(segments) + 1):
        head = ".".join(segments[:cut])
        if head not in node:
            continue
        rest = segments[cut:]
        if not rest:
            found.append((": ".join([*written, head]), node[head]))
        else:
            found.extend(_spellings(node[head], rest, [*written, head]))
    return found


def reach_override_errors(config: Any, projected: Mapping[str, Any]) -> list[str]:
    """Refuse an attached profile whose ``config:`` contradicts a projected key.

    The build derives *projected* from the hosting deployment's render, so a
    ``config:`` entry for the same key is a second home for one fact. The
    two are allowed to agree — a persona profile INHERITS the hosting
    profile's ``config:`` dotted keys, so a host that moved a port there
    spells it in every persona's merged config too, and those spellings ARE
    the host's value — and refused when they disagree, because that case is
    the silent one: a persona dialing a port the host stopped publishing gets
    connection refused with nothing to say which of the two spellings was
    stale. A hand-copied value that agrees today is refused the day the host
    moves and it does not. Built alone, the profile's ``config:`` is laid
    over the app template's defaults before projection, so it can only agree.

    Args:
        config: The attached profile's ``config:`` block, as merged.
        projected: What :func:`attached_render_overrides` will apply.

    Returns:
        One error per contradicting spelling; empty when there is none.
    """
    errors: list[str] = []
    for dotted_key, value in projected.items():
        for spelling, spelled in _spelled_values(config, dotted_key):
            if spelled == value:
                continue
            errors.append(
                f"config spells {spelling!r} as {spelled!r}, but the hosting deployment's "
                f"render says {dotted_key} is {value!r}, and the build copies that into "
                f"every attached render — a second copy that disagrees would dial the "
                f"wrong place. Remove it from config: (the host's value is what this "
                f"render is told), or move the service on the hosting deployment. A "
                f"persona file materialized by an earlier release pinned this key "
                f"itself; delete that line — the build now supplies it."
            )
    return errors


def selected_panel_errors(
    selected_panels: Iterable[str], rendered_config: Mapping[str, Any], *, told_by: str
) -> list[str]:
    """Refuse an attached profile whose selected tab was told no address.

    A ``web_panels:`` entry for a panel the Reach Contract projects (the
    EVENTS and BLUESKY tabs) is a consumer switched on by the PROFILE, which
    the rendered config alone cannot tell: the render carries a
    ``web.panels.<id>`` block only once something wrote one. After projection
    the block is there whenever the host ran the sidecar; when it is not, the
    tab would silently be missing from the render — the operator selected it
    and nothing said it was dropped.

    Args:
        selected_panels: The attached profile's ``web_panels`` selection.
        rendered_config: The attached render's config after projection.
        told_by: What the render was told from, for the message.

    Returns:
        One error per selected contract panel with no ``url``.
    """
    selected = set(selected_panels)
    errors: list[str] = []
    for contract in REACH_CONTRACTS.values():
        for panel in sorted({key.panel for key in contract.projected if key.panel}):
            if panel not in selected or dotted_get(rendered_config, f"web.panels.{panel}.url"):
                continue
            errors.append(
                f"web_panels selects {panel!r} but this render carries no "
                f"web.panels.{panel}.url — {told_by} runs no {contract.service}, so nothing "
                f"told this attached render where the tab's sidecar is. Deploy the "
                f"{contract.service} service on the hosting deployment, name the tab's target "
                f"under `config:` (web.panels.{panel}.url), or drop the panel."
            )
    return errors


def orphan_panel_fragments(
    selected_panels: Iterable[str], rendered_config: Mapping[str, Any]
) -> list[str]:
    """The ``web.panels.<id>`` blocks an attached render must drop.

    A persona profile inherits the hosting profile's ``config:`` keys, so a
    host that pins ``web.panels.events.path`` — the documented way to move
    the dashboard's route — hands every persona a ``web.panels.events``
    fragment. For a persona that selects the tab the projection completes it
    with the host's ``url``; for one that does not, the fragment is a tab this
    persona never asked for, with an EMPTY url — the web terminal makes a
    custom panel of any such block — that would either fail at first click or
    be refused by :func:`osprey.deployment.reach.reach_errors` as a consumer
    with nothing to dial. Neither is what the operator meant, so the fragment
    goes: a tab the Reach Contract projects exists in an attached render on
    the strength of the profile's selection alone.

    Args:
        selected_panels: The attached profile's ``web_panels`` selection.
        rendered_config: The attached render's config after projection.

    Returns:
        Dotted ``web.panels.<id>`` keys to delete, registry order.
    """
    selected = set(selected_panels)
    keys: list[str] = []
    for contract in REACH_CONTRACTS.values():
        for panel in sorted({key.panel for key in contract.projected if key.panel}):
            if panel in selected:
                continue
            block = dotted_get(rendered_config, f"web.panels.{panel}")
            if isinstance(block, Mapping) and not block.get("url"):
                keys.append(f"web.panels.{panel}")
    return keys
