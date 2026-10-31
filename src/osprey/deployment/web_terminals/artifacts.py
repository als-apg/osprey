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
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from osprey.agent_runner.tool_names import OPEN_MODE_EGRESS_TOOLS
from osprey.bluesky_bridge_connection import lane_env_prefix
from osprey.deployment.errors import DeploymentError
from osprey.deployment.web_terminals.auth_credentials import (
    AUTH_ENV_FILENAME,
    terminal_secret_var,
)
from osprey.deployment.web_terminals.render import (
    PROXY_ENV_NAMES,
    _auth_tls_context,
    clear_nginx_templates_dir,
    render_web_terminals,
)
from osprey.deployment.web_terminals.roster import (
    as_dict,
    launch_token_writes_key,
    login_titles_in,
    referenced_render_project_dirs,
    rendered_configs,
    renders_needing_archiver_ca_bundles,
    renders_needing_archiver_credentials,
    renders_needing_ariel_mirror,
    renders_needing_ariel_password,
    renders_needing_dispatcher_token,
    renders_needing_facility_bundle,
    renders_needing_graphdb_password,
    renders_needing_launch_token_by_lane,
    renders_needing_phoebus_handles,
    renders_with_open_shell,
    roster_entries,
    settings_json_deny_entries,
    settings_json_is_rendered,
)
from osprey_connectors.dotenv import ENV_LOCAL_FILENAME, merge_chain, parse_dotenv_file
from osprey_connectors.types import WRITES_KEY
from osprey_connectors.workspace import BUILD_DIR_NAME

#: Filename of the rendered web-stack compose file, as
#: :func:`~osprey.deployment.web_terminals.render.render_web_terminals` keys it.
WEB_COMPOSE_FILENAME = "docker-compose.web.yml"

#: The one deploy-config key that waives the Bash/launch-token refusal. Root of
#: the deploy config, boolean ``true`` and nothing else (see
#: :func:`dangerously_allow_bash`). Named so it cannot be typed innocently, in
#: the spirit of Claude Code's dangerous skip-permissions flag: for a dev box
#: whose only operator is the person reading the banner, never for a deployment
#: that arms a live machine.
DANGEROUSLY_ALLOW_BASH_KEY = "dangerously_allow_bash"


class DangerouslyAllowBashValueError(DeploymentError):
    """``dangerously_allow_bash`` is set to something other than ``true``.

    A key this loud cannot be half-set. ``"true"``, ``1``, ``yes`` and every
    other spelling are refused as a config error rather than read as either
    on (a silent waiver) or off (a refusal that reads as the key not working).
    """

    def __init__(self, value: Any) -> None:
        self.value = value
        super().__init__(
            f"{DANGEROUSLY_ALLOW_BASH_KEY} is {value!r}; the only value it takes is the "
            f"boolean true. Remove the key to keep the Bash/launch-token refusal."
        )


def dangerously_allow_bash(config: Any) -> bool:
    """Whether the deploy config waives the Bash/launch-token refusal.

    Absent, ``null`` and ``false`` all read as off — byte-for-byte the refusal.
    ``true`` reads as on. Anything else raises
    :class:`DangerouslyAllowBashValueError`, on every reader of the predicate
    and whether or not a conflict exists, so a mis-set key is caught on the
    first ``osprey up`` rather than on the first conflict.

    Args:
        config: The parsed deploy config.
    """
    value = (config or {}).get(DANGEROUSLY_ALLOW_BASH_KEY)
    if value is None or value is False:
        return False
    if value is True:
        return True
    raise DangerouslyAllowBashValueError(value)


class BashLaunchTokenConflictError(DeploymentError):
    """A render would hold ``BLUESKY_LAUNCH_TOKEN`` while its agent may run a shell.

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

    Two boundaries this guard does **not** cover, neither of them accidental:

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
        tiers: The offending render names, across every lane — every render
            that is both entitled to some lane's launch token and shipping a
            settings artifact that leaves a shell (``Bash`` or ``Monitor``) open. All of them are named in the message; an
            operator fixing one at a time would otherwise redeploy once per
            offender to discover the next.
        by_lane: The same offenders split by the lane that entitles them, so the
            message names WHICH token each render would hold. Optional: a
            caller holding only the flat set (the collect-all preflight, which
            asks :func:`bash_launch_token_offenders`) still gets the refusal and
            the remedy, without the per-lane breakdown.
        writes_keys: Per lane, the config key that decides that lane's token
            entitlement (:func:`~osprey.deployment.web_terminals.roster.launch_token_writes_key`),
            which is the login tier's ``control_system.writes`` on every lane.
    """

    def __init__(
        self,
        tiers: Iterable[str],
        *,
        by_lane: Mapping[str, Iterable[str]] | None = None,
        writes_keys: Mapping[str, str] | None = None,
    ) -> None:
        self.tiers = sorted(tiers)
        self.renders_by_lane = {
            lane: sorted(names) for lane, names in sorted((by_lane or {}).items())
        }
        named = ", ".join(repr(name) for name in self.tiers)
        breakdown = "".join(
            f"\n  - lane {lane!r} ({lane_env_prefix(lane)}_LAUNCH_TOKEN): "
            f"{', '.join(repr(name) for name in names)} — armed by "
            f"{(writes_keys or {}).get(lane, WRITES_KEY)} and "
            f"claude_code.servers.bluesky.enabled"
            for lane, names in self.renders_by_lane.items()
        )
        writes_remedy = f"{WRITES_KEY}: never in that render's tier"
        super().__init__(
            f"Refusing to deploy: {named} would be granted a Bluesky launch token "
            f"while also permitted to run a shell.{breakdown}\n"
            f"Each named render is entitled to that lane's token — its rendered "
            f"config.yml allows writes ({WRITES_KEY}: allowed) AND leaves "
            f"the bluesky MCP server enabled — and its shipped "
            f'.claude/settings.json does not list both "Bash" and "Monitor" in '
            f"permissions.deny (Monitor runs shell commands too; a missing or unparseable settings.json counts the same — it is not "
            f"evidence of a deny). The token arms a queue start on that lane, which "
            f"moves accelerator hardware; the chat approval gates only the queue_start "
            f"tool, so an agent with a shell can read the token out of its own "
            f"environment and arm a queue with no approval at all. Fix either half "
            f'of the conflict: restore "Bash" and "Monitor" to permissions.deny for that render '
            f"and REBUILD its image (or republish and re-pull it in registry mode) "
            f"— a re-render alone is not enough, because the settings.json read here "
            f"is the one baked into the image, and the per-user services declare only "
            f"`image:`, so nothing rebuilds at start. Or withdraw the entitlement by "
            f"moving that render off the bluesky server "
            f"(claude_code.servers.bluesky.enabled: false) or off writes "
            f"({writes_remedy})."
        )


def check_bash_launch_token_conflict(config: Any, project_root: Path | str) -> dict[str, set[str]]:
    """Refuse the deploy if any launch-token render's shipped settings leave a shell (``Bash`` or ``Monitor``) open.

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
      load-bearing rather than incidental: decommissioning the offending render's
      own last user drops that render from the referenced set and so must still
      succeed — it is the one remediation that needs no image rebuild. A future
      change that probes the pre-removal roster silently takes that away.
    * :func:`write_web_terminal_artifacts` — belt and braces, and the only cover
      for :func:`~osprey.deployment.web_terminals.provision.force_recreate_auth_sidecar`,
      whose pre-recreate re-render reaches the render seam through neither of the
      call sites above. The seam has to answer for itself.

    :data:`DANGEROUSLY_ALLOW_BASH_KEY` waives the refusal -- and ONLY the
    refusal: the entitlement map below is returned unchanged, so the waived
    render is granted exactly the token it would have been. It is honoured
    here and in :func:`bash_launch_token_offenders` off one reader
    (:func:`dangerously_allow_bash`), so the three call sites and the
    collect-all preflight cannot disagree about whether it applies.

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
        ``{lane: renders}`` — the renders entitled to each lane's launch token,
        returned rather than recomputed so the caller that goes on to render
        hands the *same* sets down, and what the guard cleared cannot drift from
        what the render grants. A lane nobody may arm is absent, per
        :func:`~osprey.deployment.web_terminals.roster.renders_needing_launch_token_by_lane`.

    Raises:
        BashLaunchTokenConflictError: One or more renders entitled on SOME lane
            ship a ``.claude/settings.json`` that does not deny both ``Bash`` and
            ``Monitor``.
            Any lane's token is enough: a shell reads whichever one the container
            holds, and every lane arms real hardware motion on its own target.
    """
    # Read FIRST, so a mis-set key is a config error even on a roster with no
    # conflict to waive.
    waived = dangerously_allow_bash(config)
    entitled_by_lane = renders_needing_launch_token_by_lane(config, project_root)
    permitting = renders_with_open_shell(config, project_root)
    offenders_by_lane = {
        lane: conflicted
        for lane, tiers in entitled_by_lane.items()
        if (conflicted := tiers & permitting)
    }
    if offenders_by_lane and not waived:
        # Read once more, only on the refusal path: the remedy names the key
        # that decides THAT lane's posture in THAT render's config, and a lane
        # whose offenders disagree about it (two renders, two live connector
        # types) names both rather than picking one.
        render_configs = rendered_configs(config, project_root)
        writes_keys = {
            lane: " or ".join(
                sorted({launch_token_writes_key(render_configs.get(tier), lane) for tier in tiers})
            )
            for lane, tiers in offenders_by_lane.items()
        }
        raise BashLaunchTokenConflictError(
            set().union(*offenders_by_lane.values()),
            by_lane=offenders_by_lane,
            writes_keys=writes_keys,
        )
    return entitled_by_lane


def bash_launch_token_offenders(config: Any, project_root: Path | str) -> set[str]:
    """The renders in conflict, computed without refusing anything.

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
        Every render both entitled to SOME lane's launch token and shipping
        settings that leave a shell open — the union across lanes, because one
        token in the container's environment is one shell away from arming its
        lane. Empty when there is no conflict.
    """
    if dangerously_allow_bash(config):
        return set()
    return _offenders_ignoring_the_waiver(config, project_root)


def dangerously_allowed_bash_renders(config: Any, project_root: Path | str) -> set[str]:
    """The renders :data:`DANGEROUSLY_ALLOW_BASH_KEY` waved through.

    What the deploy's banner names: exactly the set
    :func:`bash_launch_token_offenders` would have returned with the key off,
    computed by the same predicate so the banner and the refusal cannot
    disagree about who is in conflict. Empty whenever the key is off, and
    empty on a conflict-free roster with the key on -- a banner naming nobody
    would train operators to ignore it.

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it.
    """
    if not dangerously_allow_bash(config):
        return set()
    return _offenders_ignoring_the_waiver(config, project_root)


def _offenders_ignoring_the_waiver(config: Any, project_root: Path | str) -> set[str]:
    """The conflict set with :data:`DANGEROUSLY_ALLOW_BASH_KEY` left out of it."""
    entitled = set().union(*renders_needing_launch_token_by_lane(config, project_root).values())
    return entitled & renders_with_open_shell(config, project_root)


# ---------------------------------------------------------------------------
# The OPEN-mode egress gate
#
# A twin of the Bash/launch-token guard above, and deliberately shaped like it:
# same fail-closed read of the same image-baked artifact, same non-raising twin
# for the collect-all preflight, same call sites.
# The question it asks is different — not "may this render arm hardware from a
# shell?" but "may this render reach the deployment's own terminals?" — but a
# second shape for a second gate is how two guards drift into disagreeing about
# what a rendered settings.json says.
# ---------------------------------------------------------------------------


#: The :func:`open_mode_missing_by_render` value meaning "there is no rendered
#: ``.claude/settings.json`` for this offender on this host at all" — as opposed
#: to a rendered artifact that lifts particular entries, which is reported as
#: the tuple of entries it lifts.
#:
#: Empty rather than a marker string, so the two cases can never be confused for
#: a tool name and the ordinary "which tools are missing" read stays a plain
#: tuple. The distinction exists because the two states have different remedies:
#: a lifted entry is fixed by restoring it and rebuilding, while a missing
#: render is fixed by rendering the project first (``osprey build``) — and an
#: operator told to add a deny entry to a file that does not exist is told
#: nothing they can act on.
UNRENDERED_SETTINGS: tuple[str, ...] = ()


class OpenModeEgressError(DeploymentError):
    """An OPEN deployment ships a render that may still reach the host network.

    Under ``auth.method: none`` nginx vouches for every request it proxies: it
    injects each user's operator secret on every location, so
    *anything* that can reach nginx is served that user's terminal. That is the
    posture's whole point for a human — no login, no magic link — and its whole
    danger for an agent, because the agent's own tools reach nginx from inside
    the deployment, over loopback, indistinguishable by source address from the
    operator's browser.

    The python executor's open-mode socket guard refuses the deployment's own
    web ports from executed code, but it covers exactly that one path. A shell,
    ``WebFetch``/``WebSearch`` or a Playwright browser reaches those ports
    straight past it, in-process guard or not. So open mode is refused unless
    every render's shipped settings deny all of
    :data:`~osprey.agent_runner.tool_names.OPEN_MODE_EGRESS_TOOLS` — the deploy
    stops before a single web-terminal artifact is written, while the operator
    still has the context to choose between the two real remedies.

    Deliberately not a ``ValueError``, for the reason
    :class:`BashLaunchTokenConflictError` spells out:
    ``force_recreate_auth_sidecar`` catches ``(ValueError, OSError)`` around its
    best-effort re-render and downgrades it to a warning, and a safety refusal a
    credential rotation can swallow is not a refusal.

    Two boundaries carried forward from the Bash guard unchanged, because this
    gate reads the same artifact:

    * **The check is render-scoped, not image-scoped.** It reads the rendered
      project directory, which equals the image's ``.claude/settings.json``
      only as of that image's last build: ``COPY . /app/<project>/`` bakes the
      directory in, and the per-user compose services declare only ``image:``,
      with no ``build:`` stanza to rebuild one at start. A render that *adds*
      the denies without an image rebuild therefore passes this gate while the
      running image is still permissive — which is why the remedy says rebuild.
    * **It assumes the agent does not run in bypass-permissions mode**, where a
      deny list is ignored wholesale and proves nothing. No such flag appears
      on the web-terminal launch path today.

    Args:
        missing_by_render: :func:`open_mode_missing_by_render`'s answer — the
            subset of :data:`~osprey.agent_runner.tool_names.OPEN_MODE_EGRESS_TOOLS`
            each offender fails to deny, keyed by offender, with
            :data:`UNRENDERED_SETTINGS` for an offender with no rendered
            artifact at all (which gets the render remedy instead of a deny
            list it cannot edit). Every offender is named, so an operator
            fixing one at a time does not redeploy once per offender to
            discover the next, and each is named with the entry it actually
            lifted rather than the whole set. :attr:`renders` is its key set,
            exposed because that is the shape callers compare against.
    """

    def __init__(self, missing_by_render: Mapping[str, Iterable[str]]) -> None:
        self.missing_by_render = {
            tier: list(tools) for tier, tools in sorted(missing_by_render.items())
        }
        self.tiers = sorted(self.missing_by_render)
        named = ", ".join(repr(name) for name in self.tiers)
        # The union of what the offenders are actually missing, so the tools
        # named in the headline are the tools named in the remedy.
        reaching = [
            tool
            for tool in OPEN_MODE_EGRESS_TOOLS
            # An offender with no rendered artifact denies nothing, so it puts
            # the WHOLE set back in reach rather than none of it.
            if any(not tools or tool in tools for tools in self.missing_by_render.values())
        ]
        tools_named = ", ".join(repr(tool) for tool in reaching)
        breakdown = "".join(
            (
                f"\n  - {tier!r} does not deny: {', '.join(repr(tool) for tool in tools)}"
                if tools
                else (
                    f"\n  - {tier!r} has no rendered .claude/settings.json on this "
                    f"host — run `osprey build` to render it, then rebuild its image"
                )
            )
            for tier, tools in self.missing_by_render.items()
        )
        super().__init__(
            f"Refusing to deploy: auth.method 'none' (open) lets nginx vouch for every "
            f"terminal, and render(s) {named} may still reach the host network via "
            f"{tools_named}.{breakdown}\n"
            f"Open mode injects each user's operator secret on every "
            f"location, so anything that reaches nginx is served that user's terminal "
            f"— including an agent running inside another user's terminal, which "
            f"arrives over loopback and is indistinguishable from the operator's own "
            f"browser. The python executor refuses the deployment's own web ports from "
            f"executed code, but that guard covers only executed code: a shell, a web "
            f"fetch or a Playwright browser reaches those ports straight past it. Every "
            f"render's shipped .claude/settings.json must therefore list all of "
            f"{', '.join(repr(tool) for tool in OPEN_MODE_EGRESS_TOOLS)} in "
            f"permissions.deny (a missing or unparseable settings.json counts the same "
            f"— it is not evidence of a deny). Set "
            f"modules.web_terminals.auth.method to 'token' to keep the magic-link wall, "
            f"or deny {tools_named} in those renders, RENDER them on this host "
            f"(`osprey build`) and rebuild their images — in registry mode, republish "
            f"the images from a render that carries the denies and re-pull them, and "
            f"keep this host's render in step. A re-render alone is not enough, "
            f"because the settings.json that RUNS is the one baked into the image and "
            f"the per-user services declare only `image:`, so nothing rebuilds at "
            f"start; a re-pull alone is not enough either, because the settings.json "
            f"read HERE is this host's render, which open mode requires in both "
            f"image-source modes."
        )


def check_open_mode_requirements(config: Any, project_root: Path | str) -> None:
    """Refuse an OPEN deploy whose renders can still reach the host network.

    The twin of :func:`check_bash_launch_token_conflict`, wired into the same
    three call points for the same reasons, so the two gates cannot come to
    disagree about which deploys are allowed to start:

    * :func:`~osprey.deployment.web_terminals.provision.preflight_web_terminals`
      — the fail-fast gate, ahead of
      :func:`~osprey.deployment.web_terminals.provision.ensure_env_production`,
      so a deployment that will be refused never pays for the project-image
      build or has credentials minted and printed for it.
    * :func:`~osprey.deployment.web_terminals.lifecycle.decommission_user` —
      ahead of its roster edit, handed the **post-removal** roster. That is
      load-bearing rather than incidental here too: decommissioning the
      offending render's own last user drops it from the referenced set, so
      that removal must still succeed. It is the one remediation that needs no
      image rebuild.
    * :func:`resolve_render_inputs` — the render seam, and so the cover for
      every writer of this deployment's artifacts that reaches a render without
      passing the preflight above: :func:`write_web_terminal_artifacts` routes
      through it, which is how
      :func:`~osprey.deployment.web_terminals.provision.deploy_up_web_terminals`'
      own re-render is answered for. The Bash guard's third caller,
      :func:`~osprey.deployment.web_terminals.provision.force_recreate_auth_sidecar`'s
      pre-recreate re-render, is unreachable under this posture — open mode runs
      no auth sidecar to recreate — and the gate sits at the shared seam anyway
      rather than teaching one seam which of its two guards each caller needs.

    The claim is scoped to that seam, which is the path that writes THIS
    deployment's artifacts. ``osprey scaffold web-terminals render``
    (:mod:`osprey.cli.scaffold_cmd`) calls
    :func:`~osprey.deployment.web_terminals.render.render_web_terminals`
    directly, into an output directory of the operator's choosing, and stays
    ungated — an inspection surface, exactly as it is for the Bash twin. It is
    not silent there: the lint rule ``web_terminals.open_mode_egress`` runs on
    that path off the same predicate and refuses the render unless ``--no-lint``
    is passed.

    Every other auth method returns immediately: ``token`` keeps the magic-link
    exchange, ``password``/``oidc`` put a login wall in front of the roster, and
    in none of the three is nginx vouching for whoever reaches it. The posture is
    read through
    :func:`~osprey.deployment.web_terminals.render._auth_tls_context`'s derived
    booleans rather than off the method name, so a fifth method cannot fork a
    branch nobody updated.

    Like
    :func:`~osprey.deployment.web_terminals.roster.settings_json_denies`, this
    reads the **rendered** project directory, which equals the image's
    ``.claude/settings.json`` only as of that image's last build — a render
    newer than its image can report a deny the running image does not yet
    carry, which is why the refusal says rebuild rather than merely "add the
    deny".

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it.

    Raises:
        OpenModeEgressError: The deployment is open and at least one referenced
            render ships settings that do not deny every tool in
            :data:`~osprey.agent_runner.tool_names.OPEN_MODE_EGRESS_TOOLS`.
    """
    if missing := open_mode_missing_by_render(config, project_root):
        raise OpenModeEgressError(missing)


def open_mode_offenders(config: Any, project_root: Path | str) -> set[str]:
    """The renders an OPEN deploy would be refused for, computed without raising.

    The same predicate :func:`check_open_mode_requirements` refuses on, split
    out so a caller can ASK the question without raising on the answer — a
    second reader of one predicate, never a second definition of what an
    offender is. The names alone; :func:`open_mode_missing_by_render` is the
    same answer with the per-offender detail attached, and this is that answer's
    key set.

    Reads rendered directories and nothing else: no file is written, no process
    is started, and nothing about the deployment changes. That purity is what
    licenses calling it from the middle of a start sequence, and from lint.

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it.

    Returns:
        Every render whose shipped settings do not deny the whole of
        :data:`~osprey.agent_runner.tool_names.OPEN_MODE_EGRESS_TOOLS` — including
        one with no rendered settings artifact at all. Empty when the deployment is not open, and empty when
        it is open and clean.
    """
    return set(open_mode_missing_by_render(config, project_root))


def open_mode_missing_by_render(
    config: Any, project_root: Path | str
) -> dict[str, tuple[str, ...]]:
    """Per offender, the egress entries its shipped settings fail to deny.

    The one place the raising gate and the ask-only readers get their answer, so
    they cannot disagree — including about whether the deployment is open at
    all, which is settled here rather than at each caller. It is also what a
    caller building an :class:`OpenModeEgressError` of its own passes as
    ``missing_by_render`` so its message names the same entries the raising
    gate would have named. Asking :func:`open_mode_offenders` instead and
    phrasing the refusal against the whole set is supported, but sends the
    operator through every entry of the set to find the one that is lifted.

    Pure in the same sense as :func:`open_mode_offenders`, which is derived from
    this.

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it.

    Returns:
        ``{offender: entries}``, each tuple in
        :data:`~osprey.agent_runner.tool_names.OPEN_MODE_EGRESS_TOOLS` order,
        and :data:`UNRENDERED_SETTINGS` (the empty tuple) for an offender with
        no rendered artifact on this host. ``{}`` when the deployment is not
        open, or when every offender-to-be denies the whole set.
    """
    if not deployment_is_open(config):
        return {}
    missing: dict[str, tuple[str, ...]] = {}
    # One roster walk, and one read of each render's settings.json diffed
    # against the whole tuple — not one walk per entry. The offender still has
    # to be named with the entries IT is missing (a render that lifted only
    # `WebFetch` is a different problem, and a different remedy, from one
    # shipping no settings.json at all), and the diff answers that from the same
    # single read.
    for tier, project_dir in referenced_render_project_dirs(config, project_root).items():
        if (gap := _open_mode_gap(project_dir)) is not None:
            missing[tier] = gap
    return missing


def _open_mode_gap(project_dir: Path | None) -> tuple[str, ...] | None:
    """What one rendered project fails to deny, or ``None`` when it denies it all.

    Fails closed on every artifact it cannot read, exactly as
    :func:`~osprey.deployment.web_terminals.roster.settings_json_denies` does,
    and splits that failure in two so the refusal can be acted on: an artifact
    that is simply not there on this host reports
    :data:`UNRENDERED_SETTINGS` — render it — while one that is there and
    unreadable, or there and permissive, reports the entries it does not deny.

    Args:
        project_dir: The rendered project directory whose
            ``.claude/settings.json`` this offender ships, or ``None`` for a
            render whose catalog entry names no ``project_path`` — which ships
            no artifact at all and is reported the same way an unrendered one
            is.

    Returns:
        ``None`` when every entry in
        :data:`~osprey.agent_runner.tool_names.OPEN_MODE_EGRESS_TOOLS` is denied;
        :data:`UNRENDERED_SETTINGS` when there is no artifact on this host; else
        the entries that are missing, in
        :data:`~osprey.agent_runner.tool_names.OPEN_MODE_EGRESS_TOOLS` order.
    """
    if project_dir is None or not settings_json_is_rendered(project_dir):
        return UNRENDERED_SETTINGS
    denied = settings_json_deny_entries(project_dir)
    gap = tuple(tool for tool in OPEN_MODE_EGRESS_TOOLS if denied is None or tool not in denied)
    return gap or None


def deployment_is_open(config: Any) -> bool:
    """Whether this deployment runs OPEN — nginx vouching for every terminal.

    The gate's own posture question, and exported because one other decision
    hangs off the same answer: open mode requires a local render on this
    host (it is the artifact this gate reads), so
    :func:`~osprey.deployment.web_terminals.provision.render_problem`
    demands one in BOTH image-source modes when this returns ``True``. Routed
    here rather than re-derived there, so the mode that requires the render and
    the gate that reads it can never disagree about which deployments are open.

    Asked through
    :func:`~osprey.deployment.web_terminals.render._auth_tls_context`, the
    single definition of what an ``auth`` stanza means, and off its
    ``open_perimeter`` boolean rather than the method name — the same boolean
    the compose render stamps the perimeter from, so the gate and the stamp
    cannot come to disagree about which deployments are open.

    Degrades to ``False`` on the one input that function rejects — an
    ``auth.method`` naming a method that does not exist. Such a config renders
    nothing at all: the render raises on it and lint reports
    ``web_terminals.unknown_auth_method``, both independently of this gate. It
    is not an open deployment, and answering the question with a ``ValueError``
    would put a raise inside :func:`open_mode_offenders`, whose callers are
    promised one that does not raise.

    Args:
        config: The parsed deploy config.

    Returns:
        ``True`` only for a deployment whose ``auth.method`` resolves to the
        open posture.
    """
    web_terminals = as_dict(as_dict(as_dict(config).get("modules")).get("web_terminals"))
    try:
        context = _auth_tls_context(web_terminals)
    except ValueError:
        return False
    return bool(context["open_perimeter"])


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
    for entry in roster_entries(web_terminals):
        name = entry["name"]
        value = stored.get(terminal_secret_var(name), "").strip()
        if value:
            secrets[name] = value
    return secrets or None


def proxy_env_names_with_a_value(root: Path) -> tuple[str, ...]:
    """The members of :data:`PROXY_ENV_NAMES` compose will interpolate to a value.

    Read where compose reads them, in its order: this process's environment
    first, then the merged env chain. A name that resolves to nothing there
    is left out, so the render writes it under neither spelling.

    Args:
        root: The deployment repo root, which holds the env chain.

    Returns:
        The names with a non-blank value, in :data:`PROXY_ENV_NAMES` order.
    """
    chain = merge_chain(root)
    return tuple(
        name for name in PROXY_ENV_NAMES if os.environ.get(name, chain.get(name, "")).strip()
    )


def resolve_render_inputs(config: Any, repo_root: Path | str) -> dict[str, Any]:
    """Every disk-derived input the deploy hands :func:`render_web_terminals`.

    Resolved HERE and passed down, because ``render_web_terminals()`` reads no
    filesystem of its own (see its docstring): the ``.env.auth`` digest; which
    renders declare the EVENTS panel and so need the dispatcher's bearer,
    which configure ARIEL and so need its Postgres password, which both allow
    writes and run the bluesky server and so may arm a queue start, which
    configure a graph store and so need its Neo4j password — each in their own
    per-user environment block; which name a facility-knowledge bundle or run
    a qmd export and so get the deployment's bundle or mirror bind-mounted,
    and the groups those shared directories (and every user's spool and state directories)
    were provisioned with; the roster's operator secrets; and which proxy
    settings hold a value where compose will read them (this process's
    environment, then the chain); and the facility's display name, from the
    build's facility identity; and each shared login's title, the top-level
    ``name:`` of its ``tiers/<name>.yml``, which titles the card's landing
    entry, browser tab and sign-in chooser entry.

    The one seam between "what is on disk" and "what the render is told", so a
    test that wants the render the deploy actually produces asks this rather
    than restating a subset of the inputs — a subset is exactly how a mount the
    deploy emits can be invisible to a test that never passed its entitlement.

    Fails closed BEFORE any render, so a refused deploy leaves no half-written
    artifacts behind: a render holding ``BLUESKY_LAUNCH_TOKEN`` whose shipped
    settings still leave a shell open can arm a queue start from a shell, with none
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
        OpenModeEgressError: See :func:`check_open_mode_requirements`.
    """
    from osprey.deployment.compose_generator import (
        resolve_ariel_mirror_dir,
        resolve_facility_bundle_dir,
        shared_corpus_gid,
    )
    from osprey.utils.facility import facility_name
    from osprey.utils.workspace import TIERS_DIRNAME

    root = Path(repo_root)
    launch_token_renders_by_lane = check_bash_launch_token_conflict(config, root)
    # The second fail-closed gate, in the same position and for the same reason:
    # an open deployment whose renders can still reach its own terminals must
    # leave no half-written artifacts behind either.
    check_open_mode_requirements(config, root)
    return {
        "auth_env_digest": auth_env_digest(root),
        "dispatcher_renders": renders_needing_dispatcher_token(config, root),
        "ariel_renders": renders_needing_ariel_password(config, root),
        # Every lane's slice: the render grants each render one launch token
        # per lane whose target that render is armed for, and the guard above
        # cleared every lane, so no grant reaches the render unchecked.
        "launch_token_renders": launch_token_renders_by_lane,
        "graphdb_renders": renders_needing_graphdb_password(config, root),
        "archiver_credential_renders": renders_needing_archiver_credentials(config, root),
        "archiver_ca_bundle_renders": renders_needing_archiver_ca_bundles(config, root),
        "facility_bundle_renders": renders_needing_facility_bundle(config, root),
        # A pure read, like every other disk-derived input here: the deploy path
        # provisions the bundle directory before this render (see
        # deploy_up_web_terminals), and this asks what group it ended up with so
        # each entitled service can join it. An unprovisioned directory answers
        # None and emits no `group_add` — the honest render, since joining a
        # group that was never established would be a guess.
        "facility_bundle_gid": shared_corpus_gid(resolve_facility_bundle_dir(config, root)),
        # The ARIEL mirror: the same pure read, for the same reason, of a
        # directory the deploy provisioned before this render. The per-user
        # spool and state subdirectories read back no gid here — their group is joined
        # inside the container by the entrypoint, off the mounted directory.
        "ariel_mirror_renders": renders_needing_ariel_mirror(config, root),
        "ariel_mirror_gid": shared_corpus_gid(resolve_ariel_mirror_dir(config, root)),
        # A pure read of each render's rendered config, like the grants above.
        "phoebus_handle_renders": renders_needing_phoebus_handles(config, root),
        # The roster's operator secrets, read back off the deploy .env (see
        # _terminal_secrets for the None case and why it is not an open door).
        # Without this the render emits no per-user snippet at all, while
        # nginx.conf.j2 still emits the `include` that reads one — an nginx that
        # refuses to start, pointing at a path nothing ever wrote.
        "terminal_secrets": _terminal_secrets(config, root),
        "proxy_env_names": proxy_env_names_with_a_value(root),
        "facility_name": facility_name(
            web_artifacts_dir(root), as_dict(config).get("project_name")
        ),
        "shared_titles": login_titles_in(root / TIERS_DIRNAME),
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
        BashLaunchTokenConflictError: A render is entitled to
            ``BLUESKY_LAUNCH_TOKEN`` and its shipped ``.claude/settings.json``
            leaves a shell open. Raised before the render, so a refused
            deploy writes nothing. Every writer routes through here, so every
            path that could put such a stack on disk — bring-up, the roster
            verbs' re-render, the pre-recreate re-render — refuses it. Two of
            those refuse earlier still (see
            :func:`check_bash_launch_token_conflict`); this is the backstop that
            makes the property hold for all of them.
        OpenModeEgressError: The deployment is open (``auth.method: none``) and
            some referenced render's shipped ``.claude/settings.json`` does not
            deny every tool in
            :data:`~osprey.agent_runner.tool_names.OPEN_MODE_EGRESS_TOOLS` — including the
            case where there is no rendered artifact to read at all, which open
            mode requires on this host. Raised before the render on every
            writer, for the same reason and with the same backstop role as the
            conflict above (see :func:`check_open_mode_requirements`).
        ValueError: From the render — including, once this deployment has any
            operator secret provisioned at all, a roster user that has none of
            its own (see :func:`_terminal_secrets`). Raised before any file is
            written or cleared.
    """
    from osprey.deployment.compose_generator import resolve_repo_root

    root = Path(repo_root) if repo_root is not None else resolve_repo_root(config)
    inputs = resolve_render_inputs(config, root)
    # After the two refusals above, so a refused render still writes nothing.
    # Every terminal the overlay is about to name a record login file for has
    # one before the overlay exists, and so does the history page's gate
    # secret the nginx sidecar reads: a roster verb can add a terminal between
    # two starts, and a compose document naming a missing `env_file` fails
    # every command run against it.
    from osprey.agent_record.credentials import ensure_agent_record_credentials

    ensure_agent_record_credentials(root, config, reader_keys=False)
    artifacts = render_web_terminals(config, **inputs)
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
