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
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from osprey.bluesky_bridge_connection import LANE_KEYS, lane_env_prefix
from osprey.deployment.errors import DeploymentError
from osprey.deployment.web_terminals.auth_credentials import (
    AUTH_ENV_FILENAME,
    terminal_secret_var,
)
from osprey.deployment.web_terminals.personas import (
    as_dict,
    config_needs_launch_token_for,
    launch_token_writes_key,
    normalize_users,
    personas_needing_archiver_password,
    personas_needing_ariel_mirror,
    personas_needing_ariel_password,
    personas_needing_dispatcher_token,
    personas_needing_facility_bundle,
    personas_needing_graphdb_password,
    personas_needing_launch_token_by_lane,
    personas_not_denying_bash,
    referenced_persona_project_dirs,
    rendered_persona_configs,
    settings_json_denies_bash,
    settings_json_deny_entries,
    settings_json_is_rendered,
)
from osprey.deployment.web_terminals.render import (
    _auth_tls_context,
    clear_nginx_templates_dir,
    render_web_terminals,
)
from osprey.utils.dotenv import ENV_LOCAL_FILENAME, parse_dotenv_file
from osprey.utils.workspace import BUILD_DIR_NAME
from osprey_connectors.types import WRITES_ENABLED_KEY

#: Filename of the rendered web-stack compose file, as
#: :func:`~osprey.deployment.web_terminals.render.render_web_terminals` keys it.
WEB_COMPOSE_FILENAME = "docker-compose.web.yml"

#: The offender :class:`BashLaunchTokenConflictError` and
#: :class:`OpenModeEgressError` name for the roster entries that run no persona
#: (the zero-migration path). Those entries share
#: the deploy project's own image, config and ``.claude/settings.json``, so
#: they are one offender however many of them there are — and a spelling no
#: persona catalog key can collide with.
ZERO_MIGRATION_OFFENDER = "(no persona: the deploy project)"

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

    Persona-less roster entries are bound too. Both sides of the persona
    intersection are keyed on persona names, so an entry naming no persona —
    the zero-migration path, where the web image *is* the deploy project —
    appears in neither set; the guard therefore asks the same two questions of
    the deploy project itself, once per lane: entitlement via
    :func:`~osprey.deployment.web_terminals.personas.config_needs_launch_token_for`
    on the deploy config, exactly as
    :func:`~osprey.deployment.web_terminals.render.render_web_terminals` grants
    that entry its per-lane tokens, and the deny via the deploy project's own
    ``.claude/settings.json``. A conflict is reported under
    :data:`ZERO_MIGRATION_OFFENDER` — one offender however many persona-less
    entries there are, named on every lane that entitles it.

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
        personas: The offending persona names, across every lane — every persona
            that is both entitled to some lane's launch token and shipping a
            settings artifact that does not deny ``Bash`` — plus
            :data:`ZERO_MIGRATION_OFFENDER` when the persona-less entries are in
            that state on some lane. All of them are named in the message; an
            operator fixing one at a time would otherwise redeploy once per
            offender to discover the next.
        by_lane: The same offenders split by the lane that entitles them, so the
            message names WHICH token each persona would hold. Optional: a
            caller holding only the flat set (the collect-all preflight, which
            asks :func:`bash_launch_token_offenders`) still gets the refusal and
            the remedy, without the per-lane breakdown.
        writes_keys: Per lane, the config key that decides that lane's write
            posture (:func:`~osprey.deployment.web_terminals.personas.launch_token_writes_key`).
            Named per lane rather than deployment-wide because that is the key
            an operator actually edits to withdraw one lane's entitlement.
    """

    def __init__(
        self,
        personas: Iterable[str],
        *,
        by_lane: Mapping[str, Iterable[str]] | None = None,
        writes_keys: Mapping[str, str] | None = None,
    ) -> None:
        self.personas = sorted(personas)
        self.personas_by_lane = {
            lane: sorted(names) for lane, names in sorted((by_lane or {}).items())
        }
        named = ", ".join(repr(name) for name in self.personas)
        breakdown = "".join(
            f"\n  - lane {lane!r} ({lane_env_prefix(lane)}_LAUNCH_TOKEN): "
            f"{', '.join(repr(name) for name in names)} — armed by "
            f"{(writes_keys or {}).get(lane, WRITES_ENABLED_KEY)} and "
            f"claude_code.servers.bluesky.enabled"
            for lane, names in self.personas_by_lane.items()
        )
        # A caller holding only the flat set has no lane to point at, so the
        # remedy describes the key instead of referring back to a line that
        # would not be there.
        writes_remedy = (
            "the control_system.connector.<type>.writes_enabled named above, set to false"
            if self.personas_by_lane
            else "control_system.connector.<type>.writes_enabled: false for the "
            "target the lane drives"
        )
        zero_migration_note = (
            (
                f" {ZERO_MIGRATION_OFFENDER!r} stands for the roster entries that run "
                f"no persona at all: they run the deploy project itself, so the "
                f"entitlement is the deploy config's own write posture for the target "
                f"each named lane drives and its bluesky server, and the settings.json "
                f"read is the deploy project's .claude/settings.json."
            )
            if ZERO_MIGRATION_OFFENDER in self.personas
            else ""
        )
        super().__init__(
            f"Refusing to deploy: {named} would be granted a Bluesky launch token "
            f"while also permitted to run a shell.{breakdown}\n"
            f"Each named persona is entitled to that lane's token — its rendered "
            f"config.yml arms writes for the control target the lane drives AND leaves "
            f"the bluesky MCP server enabled — and its shipped "
            f'.claude/settings.json does not list "Bash" in permissions.deny (a '
            f"missing or unparseable settings.json counts the same — it is not "
            f"evidence of a deny). The token arms a queue start on that lane, which "
            f"moves accelerator hardware; the chat approval gates only the queue_start "
            f"tool, so an agent with a shell can read the token out of its own "
            f"environment and arm a queue with no approval at all. Fix either half "
            f'of the conflict: restore "Bash" to permissions.deny for that persona '
            f"and REBUILD its image (or republish and re-pull it in registry mode) "
            f"— a re-render alone is not enough, because the settings.json read here "
            f"is the one baked into the image, and the per-user services declare only "
            f"`image:`, so nothing rebuilds at start. Or withdraw the entitlement by "
            f"moving that persona off the bluesky server "
            f"(claude_code.servers.bluesky.enabled: false) or off that lane's writes "
            f"({writes_remedy}).{zero_migration_note}"
        )


def check_bash_launch_token_conflict(config: Any, project_root: Path | str) -> dict[str, set[str]]:
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

    :data:`DANGEROUSLY_ALLOW_BASH_KEY` waives the refusal -- and ONLY the
    refusal: the entitlement map below is returned unchanged, so the waived
    persona is granted exactly the token it would have been. It is honoured
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
        ``{lane: personas}`` — the personas entitled to each lane's launch token,
        returned rather than recomputed so the caller that goes on to render
        hands the *same* sets down, and what the guard cleared cannot drift from
        what the render grants. A lane nobody may arm is absent, per
        :func:`~osprey.deployment.web_terminals.personas.personas_needing_launch_token_by_lane`.
        :data:`ZERO_MIGRATION_OFFENDER` is deliberately NOT in it: this value is
        handed to :func:`~osprey.deployment.web_terminals.render.render_web_terminals`
        as ``launch_token_personas`` and matched against real persona names, and
        a persona-less entry's grant is answered there from the deploy config
        directly. The sentinel belongs to the refusal, never to the grant.

    Raises:
        BashLaunchTokenConflictError: One or more personas entitled on SOME lane
            ship a ``.claude/settings.json`` that does not deny ``Bash``, or the
            roster's persona-less entries are in that same state (reported as
            :data:`ZERO_MIGRATION_OFFENDER`, see :func:`_personaless_lanes`).
            Any lane's token is enough: a shell reads whichever one the container
            holds, and every lane arms real hardware motion on its own target.
    """
    # Read FIRST, so a mis-set key is a config error even on a roster with no
    # conflict to waive.
    waived = dangerously_allow_bash(config)
    entitled_by_lane = personas_needing_launch_token_by_lane(config, project_root)
    permitting = personas_not_denying_bash(config, project_root)
    offenders_by_lane = {
        lane: conflicted
        for lane, personas in entitled_by_lane.items()
        if (conflicted := personas & permitting)
    }
    # The persona intersection above cannot see roster entries that run no
    # persona; they are bound per lane against the deploy project itself.
    for lane in _personaless_lanes(config, project_root):
        offenders_by_lane.setdefault(lane, set()).add(ZERO_MIGRATION_OFFENDER)
    if offenders_by_lane and not waived:
        # Read once more, only on the refusal path: the remedy names the key
        # that decides THAT lane's posture in THAT persona's config, and a lane
        # whose offenders disagree about it (two personas, two live connector
        # types) names both rather than picking one.
        persona_configs = rendered_persona_configs(config, project_root)
        writes_keys = {
            lane: " or ".join(
                sorted(
                    {
                        # The sentinel has no entry in persona_configs — it IS
                        # the deploy config — and `.get` would answer None,
                        # which resolves to the wrong key rather than to none.
                        launch_token_writes_key(
                            config
                            if persona == ZERO_MIGRATION_OFFENDER
                            else persona_configs.get(persona),
                            lane,
                        )
                        for persona in personas
                    }
                )
            )
            for lane, personas in offenders_by_lane.items()
        }
        raise BashLaunchTokenConflictError(
            set().union(*offenders_by_lane.values()),
            by_lane=offenders_by_lane,
            writes_keys=writes_keys,
        )
    return entitled_by_lane


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
        Every persona both entitled to SOME lane's launch token and shipping
        settings that do not deny ``Bash`` — the union across lanes, because one
        token in the container's environment is one shell away from arming its
        lane — plus :data:`ZERO_MIGRATION_OFFENDER` when the roster's
        persona-less entries are in that state on some lane (see
        :func:`_personaless_lanes`). Empty when there is no conflict.
    """
    if dangerously_allow_bash(config):
        return set()
    return _offenders_ignoring_the_waiver(config, project_root)


def dangerously_allowed_bash_personas(config: Any, project_root: Path | str) -> set[str]:
    """The personas :data:`DANGEROUSLY_ALLOW_BASH_KEY` waved through.

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
    entitled = set().union(*personas_needing_launch_token_by_lane(config, project_root).values())
    offenders = entitled & personas_not_denying_bash(config, project_root)
    if _personaless_lanes(config, project_root):
        offenders.add(ZERO_MIGRATION_OFFENDER)
    return offenders


def _personaless_lanes(config: Any, project_root: Path | str) -> set[str]:
    """The lanes on which the roster's persona-less entries are in conflict.

    The one place both fix sites ask the question, so the raising guard and the
    ask-only reader cannot disagree about what a persona-less conflict is.

    The persona intersection is keyed on persona names, so a roster entry naming
    no persona — the zero-migration path, where the web image *is* the deploy
    project — appears on neither side of it. Such an entry is entitled exactly
    the way :func:`~osprey.deployment.web_terminals.render.render_web_terminals`
    grants it its tokens: :func:`config_needs_launch_token_for` on the deploy
    config itself, asked once per lane, because that entry can hold every lane's
    token the deploy config arms. Its shipped settings artifact is the deploy
    project's own ``.claude/settings.json``, which is a property of the deploy
    project and not of any one lane — so the deny is asked once and, when it is
    absent, every armed lane is in conflict.

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root, whose ``.claude/settings.json`` is
            the settings artifact those entries ship.

    Returns:
        Every lane whose token the persona-less entries would hold while the
        deploy project permits a shell. Empty when the roster has no
        persona-less entry, when the deploy project denies ``Bash``, or when no
        lane entitles it.
    """
    if not _roster_has_personaless_entries(config):
        return set()
    if settings_json_denies_bash(project_root):
        return set()
    return {lane for lane in LANE_KEYS if config_needs_launch_token_for(config, lane)}


def _roster_has_personaless_entries(config: Any) -> bool:
    """True if some roster entry runs no persona and so runs the deploy project.

    Mirrors the resolution ``render_web_terminals`` applies, through the same
    :func:`~osprey.deployment.web_terminals.personas.effective_persona` it
    uses: a ``default_persona`` covers every entry that names none, a ``role:``
    resolves to the persona the ``authorization`` block binds it to, and a
    bare-string entry or an object entry carrying neither a ``persona`` nor a
    resolvable ``role`` is persona-less. Well-formedness of the roster and of
    the authorization block is lint's job — a malformed entry is simply not
    counted here, matching ``normalize_users`` dropping it, and an
    unparseable ``authorization`` block binds no role.
    """
    from .personas import effective_persona, resolve_authorization_roles

    web_terminals = as_dict(as_dict(as_dict(config).get("modules")).get("web_terminals"))
    default_persona = web_terminals.get("default_persona")
    try:
        roles = resolve_authorization_roles(web_terminals)
    except ValueError:
        roles = {}
    users = web_terminals.get("users")
    for user in users if isinstance(users, list) else []:
        if isinstance(user, str):
            return True
        if isinstance(user, dict) and not effective_persona(
            user, roles, default_persona, strict=False
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# The OPEN-mode egress gate
#
# A twin of the Bash/launch-token guard above, and deliberately shaped like it:
# same fail-closed read of the same image-baked artifact, same non-raising twin
# for the collect-all preflight, same persona-less sentinel, same call sites.
# The question it asks is different — not "may this persona arm hardware from a
# shell?" but "may this persona reach the deployment's own terminals?" — but a
# second shape for a second gate is how two guards drift into disagreeing about
# what a rendered settings.json says.
# ---------------------------------------------------------------------------


#: The ``permissions.deny`` entries every persona must ship before a deployment
#: may run OPEN (``modules.web_terminals.auth.method: none``). Each is a
#: host-network egress path an agent can take from *outside* the python
#: executor, which is where the open-mode socket guard sits: a shell, the two
#: web tools, and the Playwright browser server.
#:
#: Every entry is spelled exactly as
#: :data:`~osprey.cli.templates.claude_code.DENY_DEFAULTS` spells it — that
#: tuple is what ``settings.json.j2`` writes into the artifact this gate reads,
#: and the comparison is literal (see
#: :func:`~osprey.deployment.web_terminals.personas.settings_json_denies`).
#: A strict subset of it, deliberately: ``Edit`` writes files rather than
#: reaching the network, and the context7 MCP server reaches a documentation
#: host rather than this deployment's own terminals. Written out rather than
#: derived by filtering ``DENY_DEFAULTS``, so that a rename there fails a test
#: loudly instead of silently dropping an entry from this gate and weakening it
#: (``test_the_open_mode_egress_tools_are_spelled_as_the_template_ships_them``).
OPEN_MODE_EGRESS_TOOLS: tuple[str, ...] = (
    "Bash",
    "WebFetch",
    "WebSearch",
    "mcp__plugin_playwright_playwright__*",
)

#: The :func:`open_mode_missing_by_persona` value meaning "there is no rendered
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
    """An OPEN deployment ships a persona that may still reach the host network.

    Under ``auth.method: none`` nginx vouches for every request it proxies: it
    injects each user's operator secret on every non-exempt location, so
    *anything* that can reach nginx is served that user's terminal. That is the
    posture's whole point for a human — no login, no magic link — and its whole
    danger for an agent, because the agent's own tools reach nginx from inside
    the deployment, over loopback, indistinguishable by source address from the
    operator's browser.

    The python executor's open-mode socket guard refuses the deployment's own
    web ports from executed code, but it covers exactly that one path. A shell,
    ``WebFetch``/``WebSearch`` or a Playwright browser reaches those ports
    straight past it, in-process guard or not. So open mode is refused unless
    every persona's shipped settings deny all of
    :data:`OPEN_MODE_EGRESS_TOOLS` — the deploy stops before a single
    web-terminal artifact is written, while the operator still has the context
    to choose between the two real remedies.

    Deliberately not a ``ValueError``, for the reason
    :class:`BashLaunchTokenConflictError` spells out:
    ``force_recreate_auth_sidecar`` catches ``(ValueError, OSError)`` around its
    best-effort re-render and downgrades it to a warning, and a safety refusal a
    credential rotation can swallow is not a refusal.

    Persona-less roster entries are bound the same way they are there: an entry
    naming no persona runs the deploy project itself, so the settings artifact
    read for it is the deploy project's own ``.claude/settings.json`` and it is
    reported under :data:`ZERO_MIGRATION_OFFENDER`.

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
        missing_by_persona: :func:`open_mode_missing_by_persona`'s answer — the
            subset of :data:`OPEN_MODE_EGRESS_TOOLS` each offender fails to
            deny, keyed by offender, with :data:`ZERO_MIGRATION_OFFENDER`
            standing for the persona-less entries and
            :data:`UNRENDERED_SETTINGS` for an offender with no rendered
            artifact at all (which gets the render remedy instead of a deny
            list it cannot edit). Every offender is named, so an operator
            fixing one at a time does not redeploy once per offender to
            discover the next, and each is named with the entry it actually
            lifted rather than the whole set. :attr:`personas` is its key set,
            exposed because that is the shape callers compare against.
    """

    def __init__(self, missing_by_persona: Mapping[str, Iterable[str]]) -> None:
        self.missing_by_persona = {
            persona: list(tools) for persona, tools in sorted(missing_by_persona.items())
        }
        self.personas = sorted(self.missing_by_persona)
        named = ", ".join(repr(name) for name in self.personas)
        # The union of what the offenders are actually missing, so the tools
        # named in the headline are the tools named in the remedy.
        reaching = [
            tool
            for tool in OPEN_MODE_EGRESS_TOOLS
            # An offender with no rendered artifact denies nothing, so it puts
            # the WHOLE set back in reach rather than none of it.
            if any(not tools or tool in tools for tools in self.missing_by_persona.values())
        ]
        tools_named = ", ".join(repr(tool) for tool in reaching)
        breakdown = "".join(
            (
                f"\n  - {persona!r} does not deny: {', '.join(repr(tool) for tool in tools)}"
                if tools
                else (
                    f"\n  - {persona!r} has no rendered .claude/settings.json on this "
                    f"host — run `osprey build` to render it, then rebuild its image"
                )
            )
            for persona, tools in self.missing_by_persona.items()
        )
        zero_migration_note = (
            (
                f" {ZERO_MIGRATION_OFFENDER!r} stands for the roster entries that run "
                f"no persona at all: they run the deploy project itself, so the "
                f"settings.json read for them is the deploy project's own "
                f".claude/settings.json."
            )
            if ZERO_MIGRATION_OFFENDER in self.personas
            else ""
        )
        super().__init__(
            f"Refusing to deploy: auth.method 'none' (open) lets nginx vouch for every "
            f"terminal, and persona(s) {named} may still reach the host network via "
            f"{tools_named}.{breakdown}\n"
            f"Open mode injects each user's operator secret on every non-exempt "
            f"location, so anything that reaches nginx is served that user's terminal "
            f"— including an agent running inside another user's terminal, which "
            f"arrives over loopback and is indistinguishable from the operator's own "
            f"browser. The python executor refuses the deployment's own web ports from "
            f"executed code, but that guard covers only executed code: a shell, a web "
            f"fetch or a Playwright browser reaches those ports straight past it. Every "
            f"persona's shipped .claude/settings.json must therefore list all of "
            f"{', '.join(repr(tool) for tool in OPEN_MODE_EGRESS_TOOLS)} in "
            f"permissions.deny (a missing or unparseable settings.json counts the same "
            f"— it is not evidence of a deny). Set "
            f"modules.web_terminals.auth.method to 'token' to keep the magic-link wall, "
            f"or deny {tools_named} in those personas, RENDER them on this host "
            f"(`osprey build`) and rebuild their images — in registry mode, republish "
            f"the images from a render that carries the denies and re-pull them, and "
            f"keep this host's render in step. A re-render alone is not enough, "
            f"because the settings.json that RUNS is the one baked into the image and "
            f"the per-user services declare only `image:`, so nothing rebuilds at "
            f"start; a re-pull alone is not enough either, because the settings.json "
            f"read HERE is this host's render, which open mode requires in both "
            f"image-source modes.{zero_migration_note}"
        )


def check_open_mode_requirements(config: Any, project_root: Path | str) -> None:
    """Refuse an OPEN deploy whose personas can still reach the host network.

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
      offending persona's own last user drops it from the referenced set, so
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
    :func:`~osprey.deployment.web_terminals.personas.settings_json_denies`, this
    reads the **rendered** project directory, which equals the image's
    ``.claude/settings.json`` only as of that image's last build — a render
    newer than its image can report a deny the running image does not yet
    carry, which is why the refusal says rebuild rather than merely "add the
    deny".

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it, and it is itself the settings artifact the
            persona-less roster entries ship.

    Raises:
        OpenModeEgressError: The deployment is open and at least one referenced
            persona — or the deploy project itself, on behalf of the
            persona-less entries — ships settings that do not deny every tool in
            :data:`OPEN_MODE_EGRESS_TOOLS`.
    """
    if missing := open_mode_missing_by_persona(config, project_root):
        raise OpenModeEgressError(missing)


def open_mode_offenders(config: Any, project_root: Path | str) -> set[str]:
    """The personas an OPEN deploy would be refused for, computed without raising.

    The same predicate :func:`check_open_mode_requirements` refuses on, split
    out so a caller can ASK the question without raising on the answer — a
    second reader of one predicate, never a second definition of what an
    offender is. The names alone; :func:`open_mode_missing_by_persona` is the
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
        Every persona whose shipped settings do not deny the whole of
        :data:`OPEN_MODE_EGRESS_TOOLS` — including one with no rendered
        settings artifact at all — plus :data:`ZERO_MIGRATION_OFFENDER` when the
        roster's persona-less entries are in that state. Empty when the
        deployment is not open, and empty when it is open and clean.
    """
    return set(open_mode_missing_by_persona(config, project_root))


def open_mode_missing_by_persona(
    config: Any, project_root: Path | str
) -> dict[str, tuple[str, ...]]:
    """Per offender, the egress entries its shipped settings fail to deny.

    The one place the raising gate and the ask-only readers get their answer, so
    they cannot disagree — including about whether the deployment is open at
    all, which is settled here rather than at each caller. It is also what a
    caller building an :class:`OpenModeEgressError` of its own passes as
    ``missing_by_persona`` so its message names the same entries the raising
    gate would have named. Asking :func:`open_mode_offenders` instead and
    phrasing the refusal against the whole set is supported, but sends the
    operator through four entries to find the one that is lifted.

    Pure in the same sense as :func:`open_mode_offenders`, which is derived from
    this.

    Args:
        config: The parsed deploy config.
        project_root: Deploy project root; relative ``project_path`` values
            resolve against it, and it is itself the artifact the persona-less
            roster entries ship.

    Returns:
        ``{offender: entries}``, each tuple in :data:`OPEN_MODE_EGRESS_TOOLS`
        order, and :data:`UNRENDERED_SETTINGS` (the empty tuple) for an offender
        with no rendered artifact on this host. ``{}`` when the deployment is
        not open, or when every offender-to-be denies the whole set.
    """
    if not deployment_is_open(config):
        return {}
    missing: dict[str, tuple[str, ...]] = {}
    # One roster walk, and one read of each persona's settings.json diffed
    # against the whole tuple — not one walk per entry. The offender still has
    # to be named with the entries IT is missing (a persona that lifted only
    # `WebFetch` is a different problem, and a different remedy, from one
    # shipping no settings.json at all), and the diff answers that from the same
    # single read.
    for persona, project_dir in referenced_persona_project_dirs(config, project_root).items():
        if (gap := _open_mode_gap(project_dir)) is not None:
            missing[persona] = gap
    # The persona walk above is keyed on persona names and so cannot see roster
    # entries that run no persona; they ship the deploy project's own artifact.
    if (
        _roster_has_personaless_entries(config)
        and (gap := _open_mode_gap(Path(project_root))) is not None
    ):
        missing[ZERO_MIGRATION_OFFENDER] = gap
    return missing


def _open_mode_gap(project_dir: Path | None) -> tuple[str, ...] | None:
    """What one rendered project fails to deny, or ``None`` when it denies it all.

    Fails closed on every artifact it cannot read, exactly as
    :func:`~osprey.deployment.web_terminals.personas.settings_json_denies` does,
    and splits that failure in two so the refusal can be acted on: an artifact
    that is simply not there on this host reports
    :data:`UNRENDERED_SETTINGS` — render it — while one that is there and
    unreadable, or there and permissive, reports the entries it does not deny.

    Args:
        project_dir: The rendered project directory whose
            ``.claude/settings.json`` this offender ships, or ``None`` for a
            persona whose catalog entry names no ``project_path`` — which ships
            no artifact at all and is reported the same way an unrendered one
            is.

    Returns:
        ``None`` when every entry in :data:`OPEN_MODE_EGRESS_TOOLS` is denied;
        :data:`UNRENDERED_SETTINGS` when there is no artifact on this host; else
        the entries that are missing, in :data:`OPEN_MODE_EGRESS_TOOLS` order.
    """
    if project_dir is None or not settings_json_is_rendered(project_dir):
        return UNRENDERED_SETTINGS
    denied = settings_json_deny_entries(project_dir)
    gap = tuple(tool for tool in OPEN_MODE_EGRESS_TOOLS if denied is None or tool not in denied)
    return gap or None


def deployment_is_open(config: Any) -> bool:
    """Whether this deployment runs OPEN — nginx vouching for every terminal.

    The gate's own posture question, and exported because one other decision
    hangs off the same answer: open mode requires a local persona render on this
    host (it is the artifact this gate reads), so
    :func:`~osprey.deployment.web_terminals.provision.persona_render_problem`
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
        OpenModeEgressError: See :func:`check_open_mode_requirements`.
    """
    from osprey.deployment.compose_generator import (
        resolve_ariel_mirror_dir,
        resolve_facility_bundle_dir,
        shared_corpus_gid,
    )

    root = Path(repo_root)
    launch_token_personas_by_lane = check_bash_launch_token_conflict(config, root)
    # The second fail-closed gate, in the same position and for the same reason:
    # an open deployment whose personas can still reach its own terminals must
    # leave no half-written artifacts behind either.
    check_open_mode_requirements(config, root)
    return {
        "auth_env_digest": auth_env_digest(root),
        "dispatcher_personas": personas_needing_dispatcher_token(config, root),
        "ariel_personas": personas_needing_ariel_password(config, root),
        # Every lane's slice: the render grants each persona one launch token
        # per lane whose target that persona is armed for, and the guard above
        # cleared every lane, so no grant reaches the render unchecked.
        "launch_token_personas": launch_token_personas_by_lane,
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
        # The ARIEL mirror: the same pure read, for the same reason, of a
        # directory the deploy provisioned before this render. The per-user
        # audit subdirectories read back no gid here — their group is joined
        # inside the container by the entrypoint, off the mounted directory.
        "ariel_mirror_personas": personas_needing_ariel_mirror(config, root),
        "ariel_mirror_gid": shared_corpus_gid(resolve_ariel_mirror_dir(config, root)),
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
        OpenModeEgressError: The deployment is open (``auth.method: none``) and
            some referenced persona's shipped ``.claude/settings.json`` does not
            deny every tool in :data:`OPEN_MODE_EGRESS_TOOLS` — including the
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
