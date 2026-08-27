"""Cross-surface regressions for a roster that binds personas through ``role:``.

A ``role:`` is a second way to say what ``persona:`` already said. That is the
whole risk: five surfaces read the roster to decide which privilege set a
terminal runs with — the compose/nginx/landing render, the lint belt at both
altitudes, per-persona env provisioning, and the ``osprey init`` composition
card — and a role that reaches only four of them hands one surface a privilege
set another never saw.

So these tests do not check that roles "work". They check that a role-only
roster and the ``persona:``-pinned roster it stands for are INDISTINGUISHABLE
downstream, by comparing the actual artifacts byte for byte rather than a
subset of them, and each comparison is paired with a sensitivity control that
re-binds one role and asserts the same comparison NOTICES. Without the control
a comparison of two artifacts nothing reads would pass forever.

**The one place the twins are no longer identical, and why.** A role started out
as purely a second spelling of a ``persona:`` pin, and while that was all it was,
every artifact matched byte for byte. It is now also a SECOND, independent fact:
under ``auth.method: password`` the auth sidecar mints that user's session with
the role as the privilege the login was granted, rendered as one
``OSPREY_AUTH_ROSTER_ROLE_<SUFFIX>`` line per role-carrying entry. A ``persona:``
pin conveys none of that — a pinned roster grants no session roles at all — so
the twins are genuinely different deployments in exactly that respect, and an
equality that still held would mean the privilege never reached the sidecar.

The comparison is therefore pinned to the exact DELTA rather than dropped: strip
those lines and the compose overlay must still be byte-identical, the other two
artifacts are still compared whole and unmodified, and the stripped lines
themselves are asserted to be exactly one per role-carrying entry naming the
right user and the right role. That is strictly more than ``==`` pinned before —
it proves the persona resolution is untouched AND bounds what the roles feature
is allowed to add. If a future change makes a role reach any OTHER byte of any
artifact, this still fails.

Three regressions, in the order a deployment meets them:

* the twins render, lint, provision and print identically (:class:`TestRoleOnlyRosterIsIndistinguishable`);
* a ``login: false`` entry that reaches a deployment-editing persona through a
  role still hits the existing guard, at lint and at materialization
  (:class:`TestRoleBoundPrivilegeStillTripsTheGuard`) — a role must not be a
  way around a refusal a pin cannot get around;
* ``osprey users remove`` rewrites a role-only roster without touching the
  survivors' bindings (:class:`TestUsersRemoveKeepsSurvivorRoleKeys`) — the
  removal path rebuilds roster entries, and a rebuild that dropped ``role:``
  (or replaced it with a ``persona:``) would silently re-tier everyone left.

The bundled ``control-assistant`` preset is the exemplar throughout: its roster
is already the shape this matters for — a privileged ``admin`` tier, an
unprivileged ``ariel`` tier served with no login, and a ``default_persona``
some entries inherit — so a regression that let the two spellings diverge shows
up in a real deployment's artifacts rather than in a fixture built to show it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import click
import pytest

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_emit import effective_web_terminals
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.profile_card import format_profile_card
from osprey.cli.profile_cmd import _parsed_persona_deltas, _persona_profile_texts
from osprey.cli.users_cmd import _drop_user_from_profile_roster
from osprey.deployment.web_terminals import env_production
from osprey.deployment.web_terminals.lint import lint_profile_config, lint_web_terminals
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.services.auth_sidecar.routes.recheck import ENV_ROSTER_ROLE_PREFIX

UNAUTHENTICATED_CODE = "web_terminals.unauthenticated_privileged_terminal"

#: The one artifact key the twins are allowed to differ in, and only by the lines
#: below.
_COMPOSE = "docker-compose.web.yml"

#: Imported from the sidecar that reads these variables back, not spelled here:
#: a rename on that side must fail as a rename, not as a comparison that quietly
#: stopped stripping anything and started passing on equality again.
_ROLE_VAR_PREFIX = ENV_ROSTER_ROLE_PREFIX


def _is_session_role_line(line: str) -> bool:
    """Whether *line* is one rendered ``- "OSPREY_AUTH_ROSTER_ROLE_<SUFFIX>=..."`` env item.

    Anchored at the start of the item rather than a substring match, so a
    comment or a value that merely mentions the prefix cannot be stripped or
    counted as a role line.
    """
    return line.strip().startswith(f'- "{_ROLE_VAR_PREFIX}')


def _session_role_lines(overlay: str) -> list[str]:
    """The auth sidecar's per-user session-role env lines, in render order.

    Whole lines, indentation and quoting included, so the assertion that pins
    their contents pins the rendered text rather than a normalization of it.
    """
    return [line for line in overlay.splitlines() if _is_session_role_line(line)]


def _without_session_roles(rendered: dict[str, str]) -> dict[str, str]:
    """The rendered artifacts with those lines removed from the compose overlay.

    The other two artifacts are returned untouched: a role has no business
    reaching the nginx fragment or the landing page, and stripping there would
    hide it if one ever did.
    """
    stripped = dict(rendered)
    stripped[_COMPOSE] = "\n".join(
        line for line in rendered[_COMPOSE].splitlines() if not _is_session_role_line(line)
    )
    # splitlines()/join drops a trailing newline the render emits; put it back so
    # the comparison stays byte-for-byte rather than byte-for-byte-modulo-EOF.
    if rendered[_COMPOSE].endswith("\n"):
        stripped[_COMPOSE] += "\n"
    return stripped


#: The dotted key a profile's ``config:`` block spells its web-terminal subtree
#: with. Profiles use the dotted spelling; a rendered deploy config uses the
#: nested one, which is why the two altitudes are built by separate helpers.
_PROFILE_WEB_KEY = "modules.web_terminals"


# ---------------------------------------------------------------------------
# The twins: one roster, two spellings
# ---------------------------------------------------------------------------


def _role_bound(web_terminals: dict[str, Any]) -> dict[str, Any]:
    """The same web-terminal subtree with every ``persona:`` pin moved to a role.

    Each pinned persona ``P`` becomes a declared role ``P-role`` bound to ``P``,
    and the entry carries the role instead of the pin. Entries that pinned
    nothing are left alone — they inherit ``default_persona`` in both spellings,
    which is the pre-roles resolution these regressions must not disturb.

    Deliberately NOT a rename of the persona: the role name differs from the
    persona name in every case, so any surface that quietly reported the role
    where it means the persona (or resolved one as the other) shows up as a
    difference rather than passing by coincidence.
    """
    twin = copy.deepcopy(web_terminals)
    roles: dict[str, Any] = {}
    for entry in twin.get("users", []):
        persona = entry.pop("persona", None)
        if not persona:
            continue
        role = f"{persona}-role"
        roles[role] = {"persona": persona}
        entry["role"] = role
    twin["authorization"] = {"roles": roles}
    return twin


def _rebind(web_terminals: dict[str, Any], role: str, persona: str) -> dict[str, Any]:
    """The role-bound subtree with one role pointed at a different persona.

    The sensitivity control behind every byte comparison here: the artifacts
    must differ from the pinned twin's once a role binds something else. A
    comparison that could not tell these apart would be pinning nothing.
    """
    twin = copy.deepcopy(web_terminals)
    twin["authorization"]["roles"][role] = {"persona": persona}
    return twin


def _profile_twins() -> tuple[dict[str, Any], dict[str, Any]]:
    """The exemplar profile's ``config:`` block, pinned and role-bound."""
    resolved, _preset_dir = resolve_build_profile(None, "control-assistant", (), ())
    pinned = copy.deepcopy(resolved.config)
    role_bound = copy.deepcopy(pinned)
    role_bound[_PROFILE_WEB_KEY] = _role_bound(role_bound[_PROFILE_WEB_KEY])
    return pinned, role_bound


def _deploy_config(web_terminals: dict[str, Any]) -> dict[str, Any]:
    """A rendered-altitude deploy config around one web-terminal subtree.

    ``deploy.fqdn`` is required to render the landing URL once the roster is
    non-empty, and ``registry.url`` is what per-user image references are built
    from; both are identical for the two twins, so neither can be what makes an
    artifact match.
    """
    return {
        "facility": {"prefix": "test"},
        "deploy": {"fqdn": "terminals.example.org"},
        "registry": {"url": "registry.example.org/test"},
        "modules": {"web_terminals": copy.deepcopy(web_terminals)},
    }


@pytest.fixture(scope="session")
def exemplar_web_terminals() -> dict[str, Any]:
    """The bundled ``control-assistant`` roster and catalog, as the preset ships it."""
    resolved, _preset_dir = resolve_build_profile(None, "control-assistant", (), ())
    return effective_web_terminals(resolved.config)


@pytest.fixture(scope="session")
def exemplar_deltas() -> dict[str, Any]:
    """The parsed persona deltas the composition card reads privileges from.

    Derived from the catalog, not the roster, so one set serves both twins —
    which is itself the point: moving a pin to a role must not change which
    deltas are in play.
    """
    resolved, _preset_dir = resolve_build_profile(None, "control-assistant", (), ())
    texts = _persona_profile_texts(resolved, "Exemplar", "", "control-assistant")
    return _parsed_persona_deltas(texts)


# ---------------------------------------------------------------------------
# Regression 1 — the two spellings are indistinguishable downstream
# ---------------------------------------------------------------------------


class TestRoleOnlyRosterIsIndistinguishable:
    """Every artifact a role-only roster produces, against the pinned twin's."""

    def test_the_three_rendered_artifacts_differ_only_by_the_session_roles(
        self, exemplar_web_terminals: dict[str, Any]
    ) -> None:
        """The compose overlay, the nginx fragment and the landing page, whole.

        Compared as the render returns them — all three keys, full text — not a
        sampled service block: a role that reached the compose file but not the
        landing page would be a deployment whose cards and whose containers
        disagree about who runs what.

        The exemplar is a ``password`` deployment, so the role-bound twin's
        overlay legitimately carries the sidecar's per-user session roles and the
        pinned twin's carries none (see the module docstring). Those lines are
        stripped and asserted separately; everything else — every image, project,
        volume, port, mount and label — must still match byte for byte, which is
        the property this class exists for.
        """
        # Arrange
        pinned = _deploy_config(exemplar_web_terminals)
        role_bound = _deploy_config(_role_bound(exemplar_web_terminals))

        # Act
        rendered_pinned = render_web_terminals(pinned)
        rendered_role_bound = render_web_terminals(role_bound)

        # Assert
        assert sorted(rendered_pinned) == [
            "docker-compose.web.yml",
            "nginx/landing.html",
            "nginx/nginx.conf",
        ]
        # The pinned twin declares no roles at all, so it must carry no session
        # role anywhere — the baseline the delta below is measured against.
        assert _session_role_lines(rendered_pinned[_COMPOSE]) == []
        assert _without_session_roles(rendered_role_bound) == rendered_pinned

    def test_the_session_roles_are_exactly_the_roster_bindings(
        self, exemplar_web_terminals: dict[str, Any]
    ) -> None:
        """The other half: what those stripped lines say.

        The test above bounds the delta; this one pins its contents, so the
        allowance cannot quietly widen into "the compose file may differ by any
        line whose name starts with the prefix". One line per role-carrying
        entry, in roster order, each naming that user's variable and that
        user's role — including ``ariel``, whose entry is opted out of the login
        wall and whose role is therefore inert but still rendered, because the
        roster the sidecar keys this table from lists it.
        """
        # Arrange
        role_bound = _deploy_config(_role_bound(exemplar_web_terminals))

        # Act
        overlay = render_web_terminals(role_bound)[_COMPOSE]

        # Assert
        assert _session_role_lines(overlay) == [
            f'      - "{_ROLE_VAR_PREFIX}ALICE=readwrite-role"',
            f'      - "{_ROLE_VAR_PREFIX}BOB=readonly-role"',
            f'      - "{_ROLE_VAR_PREFIX}ARIEL=ariel-role"',
            f'      - "{_ROLE_VAR_PREFIX}CAROL=admin-role"',
        ]

    def test_the_render_comparison_notices_a_role_bound_elsewhere(
        self, exemplar_web_terminals: dict[str, Any]
    ) -> None:
        """The control: re-bind one role and the artifacts must diverge.

        Asserted on the STRIPPED overlays as well as the whole ones. Re-binding
        changes which persona a role names, not the role names themselves, so
        the session-role lines are identical either way — and a control that
        compared only the whole text would from now on pass on the mere presence
        of those lines, whatever happened to the images underneath them.
        """
        # Arrange: alice's role now names the read-only tier instead.
        pinned = _deploy_config(exemplar_web_terminals)
        rebound = _deploy_config(
            _rebind(_role_bound(exemplar_web_terminals), "readwrite-role", "readonly")
        )

        # Act
        rendered_pinned = render_web_terminals(pinned)
        rendered_rebound = render_web_terminals(rebound)

        # Assert
        assert rendered_rebound != rendered_pinned
        assert rendered_rebound[_COMPOSE] != rendered_pinned[_COMPOSE]
        # And the difference is in the deployment itself, not in the new lines.
        assert _without_session_roles(rendered_rebound) != rendered_pinned

    def test_the_rendered_altitude_lint_report_is_identical(
        self, exemplar_web_terminals: dict[str, Any]
    ) -> None:
        """Same findings, same severities, same wording, in the same order.

        The exemplar linted as a rendered deploy config reports a handful of
        findings (its persona projects are not on this disk), which is what
        makes the comparison worth making: two empty reports would match
        whatever the roster said.
        """
        # Arrange
        pinned = _deploy_config(exemplar_web_terminals)
        role_bound = _deploy_config(_role_bound(exemplar_web_terminals))

        # Act
        findings_pinned = lint_web_terminals(pinned)
        findings_role_bound = lint_web_terminals(role_bound)

        # Assert
        assert findings_pinned, "the comparison is only meaningful on a non-empty report"
        assert findings_role_bound == findings_pinned

    def test_the_rendered_altitude_lint_comparison_notices_a_role_bound_elsewhere(
        self, exemplar_web_terminals: dict[str, Any]
    ) -> None:
        """The control: a role naming a persona the catalog never declared is
        the existing unknown-persona finding, so the reports must differ."""
        # Arrange
        pinned = _deploy_config(exemplar_web_terminals)
        rebound = _deploy_config(
            _rebind(_role_bound(exemplar_web_terminals), "readwrite-role", "ghost")
        )

        # Act
        findings_rebound = lint_web_terminals(rebound)

        # Assert
        assert findings_rebound != lint_web_terminals(pinned)
        assert "web_terminals.unknown_persona_reference" in {f.code for f in findings_rebound}

    def test_the_profile_altitude_lint_report_is_identical(self) -> None:
        """``osprey profile validate`` / ``osprey build`` must read the twins the same.

        Run over a roster with a real finding in it — carol's tier served
        without a login — so the two reports are compared on their content
        rather than on their emptiness.
        """
        # Arrange
        pinned, role_bound = _profile_twins()
        for config in (pinned, role_bound):
            for entry in config[_PROFILE_WEB_KEY]["users"]:
                if entry["name"] == "carol":
                    entry["login"] = False

        # Act
        findings_pinned = lint_profile_config(pinned)
        findings_role_bound = lint_profile_config(role_bound)

        # Assert
        assert [f.code for f in findings_pinned] == [UNAUTHENTICATED_CODE]
        assert findings_role_bound == findings_pinned

    def test_env_provisioning_writes_a_byte_identical_env_users(self, tmp_path: Path) -> None:
        """``.env.users`` — the file every per-user container runs with.

        Which secrets it carries is decided by which persona projects the roster
        references, so the two twins agreeing here is the statement that a role
        provisions exactly the credentials its pin did. Both files are generated
        under the SAME project root, one after the other, so nothing about the
        environment can differ between them.
        """
        # Arrange
        root = _env_project_root(tmp_path)
        pinned = _env_deploy_config(_ENV_PINNED_USERS)
        role_bound = copy.deepcopy(pinned)
        web = role_bound["modules"]["web_terminals"]
        role_bound["modules"]["web_terminals"] = _role_bound(web)

        # Act
        generated_pinned = _generate_env_users(root, pinned)
        generated_role_bound = _generate_env_users(root, role_bound)

        # Assert: the roster's own personas are what put these in the file.
        assert "ANTHROPIC_API_KEY=admin-secret" in generated_pinned
        assert "ALS_APG_API_KEY=readonly-secret" in generated_pinned
        assert generated_role_bound == generated_pinned

    def test_env_provisioning_notices_a_role_bound_elsewhere(self, tmp_path: Path) -> None:
        """The control: the privileged persona goes unreferenced, and its
        provider secret must leave the file with it."""
        # Arrange
        root = _env_project_root(tmp_path)
        pinned = _env_deploy_config(_ENV_PINNED_USERS)
        rebound = copy.deepcopy(pinned)
        rebound["modules"]["web_terminals"] = _rebind(
            _role_bound(rebound["modules"]["web_terminals"]), "admin-role", "readonly"
        )

        # Act
        generated_pinned = _generate_env_users(root, pinned)
        generated_rebound = _generate_env_users(root, rebound)

        # Assert
        assert generated_rebound != generated_pinned
        assert "ANTHROPIC_API_KEY" not in generated_rebound

    def test_the_profile_card_prints_identical_lines(self, exemplar_deltas: dict[str, Any]) -> None:
        """The card ``osprey init`` prints, line for line.

        The card names each user's tier and whether their rights are
        approval-gated, so a role the card could not resolve would tell an
        operator the repo they just created grants something other than what it
        grants.
        """
        # Arrange
        pinned_config, role_bound_config = _profile_twins()
        pinned = _profile_for(pinned_config)
        role_bound = _profile_for(role_bound_config)

        # Act
        lines_pinned = format_profile_card(pinned, exemplar_deltas)
        lines_role_bound = format_profile_card(role_bound, exemplar_deltas)

        # Assert
        assert any("carol" in line and "admin" in line for line in lines_pinned)
        assert lines_role_bound == lines_pinned

    def test_the_profile_card_comparison_notices_a_role_bound_elsewhere(
        self, exemplar_deltas: dict[str, Any]
    ) -> None:
        """The control: carol's row must change when carol's role does."""
        # Arrange
        pinned_config, role_bound_config = _profile_twins()
        role_bound_config[_PROFILE_WEB_KEY] = _rebind(
            role_bound_config[_PROFILE_WEB_KEY], "admin-role", "readonly"
        )

        # Act
        lines_pinned = format_profile_card(_profile_for(pinned_config), exemplar_deltas)
        lines_rebound = format_profile_card(_profile_for(role_bound_config), exemplar_deltas)

        # Assert
        assert lines_rebound != lines_pinned
        assert any("carol" in line and "readonly" in line for line in lines_rebound)


def _profile_for(config: dict[str, Any]) -> BuildProfile:
    """The exemplar profile carrying *config* — everything else left as resolved."""
    resolved, _preset_dir = resolve_build_profile(None, "control-assistant", (), ())
    resolved.config = config
    return resolved


# ── The env-provisioning fixture ─────────────────────────────────────────────
#
# Two persona projects authenticating to DIFFERENT providers, so which personas
# the roster references decides which secrets `.env.users` carries. With one
# provider for both, the file would be the same whatever the roster bound and
# the comparison above would prove nothing.

_ENV_PERSONA_PROVIDERS = {"readonly": "als-apg", "admin": "anthropic"}

_ENV_PINNED_USERS = [
    {"name": "alice", "index": 0, "persona": "admin"},
    # No pin: inherits `default_persona`, in both spellings. The pre-roles
    # resolution rides along in every comparison here.
    {"name": "bob", "index": 1},
]


def _env_project_root(tmp_path: Path) -> Path:
    """A local-mode project root: one rendered project per persona, plus ``.env``.

    ``ensure_env_production`` reads the env chain off disk and nothing else, so
    the chain carries every provider secret in play; which of them reach the
    generated file is the roster's decision, which is what is under test.
    """
    root = tmp_path / "deploy"
    root.mkdir()
    for persona, provider in _ENV_PERSONA_PROVIDERS.items():
        project = root / f"{persona}-proj"
        project.mkdir()
        (project / "config.yml").write_text(
            f"project_name: {persona}-proj\nclaude_code:\n  provider: {provider}\n",
            encoding="utf-8",
        )
    (root / ".env").write_text(
        "ALS_APG_API_KEY=readonly-secret\n"
        "ANTHROPIC_API_KEY=admin-secret\n"
        "CBORG_API_KEY=deploy-secret\n",
        encoding="utf-8",
    )
    return root


def _env_deploy_config(users: list[dict[str, Any]]) -> dict[str, Any]:
    """A local-mode deploy config over the two-persona catalog."""
    return {
        "facility": {"timezone": "UTC"},
        "llm": {"provider": "cborg", "api_key_env_var": "CBORG_API_KEY"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "image_source": "local",
                "default_persona": "readonly",
                "personas": {
                    persona: {"project": f"{persona}-proj", "project_path": f"{persona}-proj"}
                    for persona in _ENV_PERSONA_PROVIDERS
                },
                "users": copy.deepcopy(users),
            }
        },
    }


def _generate_env_users(root: Path, config: dict[str, Any]) -> str:
    """Generate ``.env.users`` and hand back its text, leaving the root clean.

    The file is removed afterwards because the generator returns an existing one
    untouched — without this the second twin would be handed the first's file
    and the comparison would compare it with itself.
    """
    generated = env_production.ensure_env_production(config, root)
    text = generated.read_text(encoding="utf-8")
    generated.unlink()
    return text


# ---------------------------------------------------------------------------
# Regression 2 — a role is not a way around the privileged-terminal guard
# ---------------------------------------------------------------------------


class TestRoleBoundPrivilegeStillTripsTheGuard:
    """``login: false`` plus a role that reaches a deployment-editing persona.

    The exposure the guard exists for is a card on the landing page that opens
    straight into a terminal holding the setup tool and the Config panel. It is
    judged on the persona the entry RESOLVES to, so a binding the guard could
    not follow would be a way to ship exactly that terminal past a refusal the
    equivalent ``persona:`` pin cannot get past.
    """

    @staticmethod
    def _carol_open_via_role() -> dict[str, Any]:
        """The exemplar profile with carol reaching ``admin`` through a role,
        and opted out of the login wall."""
        _pinned, role_bound = _profile_twins()
        for entry in role_bound[_PROFILE_WEB_KEY]["users"]:
            if entry["name"] == "carol":
                entry["login"] = False
        return role_bound

    def test_lint_refuses_it_and_names_the_user_and_the_persona(self) -> None:
        """The lint belt behind ``osprey profile validate`` and ``osprey build``."""
        # Arrange
        config = self._carol_open_via_role()

        # Act
        findings = lint_profile_config(config)

        # Assert
        errors = [f for f in findings if f.severity == "error" and f.code == UNAUTHENTICATED_CODE]
        assert len(errors) == 1
        # The message speaks of the persona, not the role: the role is how the
        # entry got there, the persona is what it holds.
        assert "'carol'" in errors[0].message
        assert "'admin'" in errors[0].message
        assert "login: true" in errors[0].message

    def test_materialization_refuses_it_before_the_repo_exists(self) -> None:
        """``osprey init``'s own half of the guard, where the tiers have just resolved.

        The refusal at this altitude is what keeps the bad roster from reaching
        a repo at all, so a role that only the lint belt could follow would
        still let ``init`` write it out.
        """
        # Arrange
        profile = _profile_for(self._carol_open_via_role())

        # Act / Assert
        with pytest.raises(click.UsageError) as excinfo:
            _persona_profile_texts(profile, "Test", "", "control-assistant")
        message = str(excinfo.value)
        assert "'carol'" in message
        assert "'admin'" in message

    def test_the_same_role_behind_a_login_is_clean(self) -> None:
        """The negative control: the guard is about the open door, not the role.

        The exemplar ships carol on the privileged tier behind a password, which
        is a supported deployment; a rule that fired on the role itself would
        refuse it.
        """
        # Arrange
        _pinned, role_bound = _profile_twins()

        # Act
        findings = lint_profile_config(role_bound)

        # Assert
        assert [f.code for f in findings if f.severity == "error"] == []


# ---------------------------------------------------------------------------
# Regression 3 — `osprey users remove` leaves the survivors' bindings alone
# ---------------------------------------------------------------------------


_ROLE_ONLY_PROFILE = """\
name: demo
config:
  modules.web_terminals:
    enabled: true
    default_persona: readonly
    # The roster block comment, which the edit must keep.
    users:
      - name: alice
        index: 0
        role: operator
      - name: bob
        role: expert
      - carl
    authorization:
      roles:
        operator: {persona: readwrite}
        expert: {persona: admin}
"""


class TestUsersRemoveKeepsSurvivorRoleKeys:
    """The roster write-back behind ``osprey users remove``.

    :func:`~osprey.cli.users_cmd._drop_user_from_profile_roster` is the function
    that command edits ``profile.yml`` with — it freezes every entry's index,
    drops one entry and writes the file back. Rebuilding the survivors from the
    module's normalizer instead would drop what the normalizer does not keep, so
    these tests read the file back and assert on what SURVIVED rather than on
    the return value.
    """

    @staticmethod
    def _roster_after_removing(profile_path: Path, user: str) -> list[Any]:
        """Remove *user*, re-read the file from disk, hand back the roster."""
        import yaml

        _drop_user_from_profile_roster(profile_path, user)
        document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        return document["config"][_PROFILE_WEB_KEY]["users"]

    def test_survivors_keep_their_role_and_gain_no_persona(self, tmp_path: Path) -> None:
        """The regression: bob is still bound through ``expert``, and nothing
        wrote him a ``persona:`` — a rewrite that resolved roles on the way out
        would re-tier every survivor at the next build."""
        # Arrange
        profile_path = tmp_path / "profile.yml"
        profile_path.write_text(_ROLE_ONLY_PROFILE, encoding="utf-8")

        # Act
        roster = self._roster_after_removing(profile_path, "alice")

        # Assert
        assert [entry["name"] for entry in roster] == ["bob", "carl"]
        assert roster[0]["role"] == "expert"
        assert not any("persona" in entry for entry in roster)

    def test_the_removed_entry_is_gone_and_the_role_table_is_untouched(
        self, tmp_path: Path
    ) -> None:
        """A departed user's role declaration is not the departed user.

        The table is the deployment's authorization vocabulary, shared by every
        entry; removing one holder of a role must not withdraw the role.
        """
        # Arrange
        import yaml

        profile_path = tmp_path / "profile.yml"
        profile_path.write_text(_ROLE_ONLY_PROFILE, encoding="utf-8")

        # Act
        roster = self._roster_after_removing(profile_path, "alice")
        document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

        # Assert
        assert all(entry["name"] != "alice" for entry in roster)
        assert document["config"][_PROFILE_WEB_KEY]["authorization"]["roles"] == {
            "operator": {"persona": "readwrite"},
            "expert": {"persona": "admin"},
        }

    def test_indices_are_frozen_without_inventing_a_binding(self, tmp_path: Path) -> None:
        """Freezing ports must not be a way for a persona to appear.

        A bare-string entry is expanded into a mapping so an earlier removal
        cannot shift its ports. That expansion writes ``name`` and ``index`` —
        and, critically, nothing else: an expansion that filled in the
        ``default_persona`` would pin an entry that was inheriting it, and the
        pin would then survive a later change to the default.
        """
        # Arrange
        profile_path = tmp_path / "profile.yml"
        profile_path.write_text(_ROLE_ONLY_PROFILE, encoding="utf-8")

        # Act
        roster = self._roster_after_removing(profile_path, "alice")

        # Assert: bob and carl keep the ports they had at positions 1 and 2.
        assert [entry["index"] for entry in roster] == [1, 2]
        assert set(roster[1]) == {"name", "index"}

    def test_a_roster_the_user_is_not_on_is_left_alone(self, tmp_path: Path) -> None:
        """The no-op control: nothing is rewritten for a name nobody carries, so
        no survivor can lose a binding to a removal that never happened."""
        # Arrange
        profile_path = tmp_path / "profile.yml"
        profile_path.write_text(_ROLE_ONLY_PROFILE, encoding="utf-8")

        # Act
        edit = _drop_user_from_profile_roster(profile_path, "nobody")

        # Assert
        assert edit.changed is False
        assert profile_path.read_text(encoding="utf-8") == _ROLE_ONLY_PROFILE
