"""The build-time guards on deployment-editing privilege (setup tool + Config panel).

Three surfaces answer one question — can this persona edit the deployment it
runs in — from three different inputs, and these tests pin all three plus the
shared judgement underneath them:

* :func:`~osprey.deployment.web_terminals.personas.persona_privileges` and the
  two rules built on it (a privileged ``default_persona``, an unauthenticated
  privileged terminal);
* :func:`osprey.cli.profile_cmd._persona_profile_texts`, the refusal at
  ``osprey init`` time, where the persona presets have just been resolved;
* the lint belt, which is what ``osprey profile validate`` / ``osprey build``
  (a profile plus its ``personas/<name>.yml`` deltas) and ``osprey up`` (each
  persona's rendered ``config.yml``) reach.

The shipped ``control-assistant`` stack is exercised as itself throughout: its
roster is exactly the shape the guards exist for (a privileged ``admin`` tier
behind a login, an unauthenticated ``ariel`` tier that is not privileged), so a
guard that fired on it would be wrong and a guard that let a one-key edit to it
through would be useless.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.templates.claude_code import DENY_DEFAULTS
from osprey.deployment.web_terminals.lint import (
    lint_profile_config,
    lint_web_terminals,
    profile_config_errors,
    profile_config_warnings,
)
from osprey.deployment.web_terminals.personas import (
    ALL_PRIVILEGES,
    PRIVILEGE_CONFIG_PANEL,
    PRIVILEGE_SETUP_TOOL,
    auth_is_enforced,
    deployment_wide_privileged_exposure_problems,
    persona_privileges,
    privilege_phrase,
    privileged_default_persona_problem,
    unauthenticated_privileged_terminal_problems,
)

SETUP_TOOL = "mcp__osprey_workspace__setup_patch"

DEFAULT_PERSONA_CODE = "web_terminals.privileged_default_persona"
UNAUTHENTICATED_CODE = "web_terminals.unauthenticated_privileged_terminal"
DEPLOYMENT_WIDE_CODE = "web_terminals.unauthenticated_privileged_deployment"
UNKNOWN_PRIVILEGE_CODE = "web_terminals.persona_privileges_unknown"
NOT_RENDERED_CODE = "web_terminals.persona_project_path_not_rendered_yet"


def _codes(findings: list[Any], severity: str = "error") -> list[str]:
    return [finding.code for finding in findings if finding.severity == severity]


def _messages(findings: list[Any], code: str) -> list[str]:
    return [finding.message for finding in findings if finding.code == code]


# ── The shared judgement ─────────────────────────────────────────────────────


class TestPersonaPrivileges:
    """What :func:`persona_privileges` reads, and how it leans when unsure."""

    def test_base_tier_holds_neither(self):
        """The floor the bundled base sets: tool denied, panel switched off."""
        assert (
            persona_privileges(
                {
                    "claude_code.permissions.deny": [SETUP_TOOL],
                    "web.config_panel.enabled": False,
                }
            )
            == ()
        )

    def test_admin_tier_holds_both(self):
        """A tier that lifts the deny and turns the panel back on holds both."""
        assert persona_privileges(
            {
                "claude_code.permissions.deny": [SETUP_TOOL],
                "claude_code.permissions.remove_deny": [SETUP_TOOL],
                "web.config_panel.enabled": True,
            }
        ) == (PRIVILEGE_SETUP_TOOL, PRIVILEGE_CONFIG_PANEL)

    def test_nested_spelling_reads_the_same_as_dotted(self):
        """A privilege check must not depend on which equivalent spelling was used."""
        nested = {
            "claude_code": {"permissions": {"deny": [SETUP_TOOL]}},
            "web": {"config_panel": {"enabled": False}},
        }
        dotted = {
            "claude_code.permissions.deny": [SETUP_TOOL],
            "web.config_panel.enabled": False,
        }
        assert persona_privileges(nested) == persona_privileges(dotted) == ()

    def test_intermediate_spelling_reads_the_same(self):
        """`claude_code.permissions:` holding `deny` is the same key again."""
        assert (
            persona_privileges(
                {
                    "claude_code.permissions": {"deny": [SETUP_TOOL]},
                    "web.config_panel": {"enabled": False},
                }
            )
            == ()
        )

    def test_rendered_config_shape(self):
        """A rendered `config.yml` is nested and carries no `remove_deny`."""
        rendered = {
            "claude_code": {"permissions": {"deny": [SETUP_TOOL]}},
            "web": {"config_panel": {"enabled": False}},
        }
        assert persona_privileges(rendered) == ()

    def test_absent_config_panel_key_is_the_panel_privilege(self):
        """The template ships the panel ON, so saying nothing is holding it."""
        assert persona_privileges({"claude_code.permissions.deny": [SETUP_TOOL]}) == (
            PRIVILEGE_CONFIG_PANEL,
        )

    def test_absent_deny_is_the_setup_privilege(self):
        """A project that never gated the tool still has it."""
        assert persona_privileges({"web.config_panel.enabled": False}) == (PRIVILEGE_SETUP_TOOL,)

    def test_empty_layers_hold_both(self):
        """Nothing said about either key is the un-gated default on both."""
        assert persona_privileges() == (PRIVILEGE_SETUP_TOOL, PRIVILEGE_CONFIG_PANEL)

    def test_quoted_boolean_turns_the_panel_off(self):
        """The server honours `"false"`; reading it as truthy would refuse a safe config."""
        assert (
            persona_privileges(
                {
                    "claude_code.permissions.deny": [SETUP_TOOL],
                    "web.config_panel.enabled": "false",
                }
            )
            == ()
        )

    def test_unreadable_panel_value_leaves_the_panel_on(self):
        """The server discards it and keeps its default, which is ON — so this does too."""
        assert persona_privileges(
            {"claude_code.permissions.deny": [SETUP_TOOL], "web.config_panel.enabled": 3}
        ) == (PRIVILEGE_CONFIG_PANEL,)

    def test_delta_layered_over_host_lifts_the_floor(self):
        """The profile-time composition: the host denies, the delta beside it lifts."""
        host = {
            "claude_code.permissions.deny": [SETUP_TOOL],
            "web.config_panel.enabled": False,
        }
        delta = {
            "claude_code.permissions.remove_deny": [SETUP_TOOL],
            "web.config_panel.enabled": True,
        }
        assert persona_privileges(host, delta) == (
            PRIVILEGE_SETUP_TOOL,
            PRIVILEGE_CONFIG_PANEL,
        )

    def test_delta_that_lifts_nothing_keeps_the_floor(self):
        """A tier that only pins unrelated keys inherits the base's posture."""
        host = {
            "claude_code.permissions.deny": [SETUP_TOOL],
            "web.config_panel.enabled": False,
        }
        assert persona_privileges(host, {"control_system.writes_enabled": True}) == ()

    def test_deny_lists_union_across_layers(self):
        """A delta's own deny adds to the host's rather than replacing it."""
        host = {"claude_code.permissions.deny": ["Bash"], "web.config_panel.enabled": False}
        delta = {"claude_code.permissions.deny": [SETUP_TOOL]}
        assert persona_privileges(host, delta) == ()

    def test_wildcard_deny_is_not_lifted_by_an_exact_remove_deny(self):
        """Mirrors the render: `remove_deny` membership is exact, so the wildcard stands."""
        assert (
            persona_privileges(
                {
                    "claude_code.permissions.deny": ["mcp__osprey_workspace__*"],
                    "claude_code.permissions.remove_deny": [SETUP_TOOL],
                    "web.config_panel.enabled": False,
                }
            )
            == ()
        )

    def test_a_bare_string_deny_is_not_read_as_one_entry(self):
        """Mirrors the predicate: the render iterates the value, so a bare string
        denies one entry per CHARACTER and names no tool. Reading it as "one
        entry" here would compose a document the predicate then disagrees with —
        and leaning the other way reports MORE privilege, not less."""
        assert persona_privileges(
            {
                "claude_code.permissions.deny": SETUP_TOOL,
                "web.config_panel.enabled": False,
            }
        ) == (PRIVILEGE_SETUP_TOOL,)

    def test_non_mapping_layer_says_nothing(self):
        """An unreadable layer contributes no deny — and no false reassurance."""
        assert persona_privileges(None, "not a config") == (
            PRIVILEGE_SETUP_TOOL,
            PRIVILEGE_CONFIG_PANEL,
        )


class TestShippedPresetPrivileges:
    """The bundled tiers, judged as they actually resolve."""

    @pytest.mark.parametrize(
        "preset",
        [
            "control-assistant",
            "control-assistant-readonly",
            "control-assistant-readwrite",
            "control-assistant-ariel",
        ],
    )
    def test_unprivileged_tiers(self, preset: str):
        """Everything the base floors — including the login-less `ariel` tier."""
        resolved, _dir = resolve_build_profile(None, preset)
        assert persona_privileges(resolved.config) == ()

    def test_admin_tier_holds_both(self):
        resolved, _dir = resolve_build_profile(None, "control-assistant-admin")
        assert persona_privileges(resolved.config) == (
            PRIVILEGE_SETUP_TOOL,
            PRIVILEGE_CONFIG_PANEL,
        )


# ── The two rules ────────────────────────────────────────────────────────────


class TestPrivilegedDefaultPersona:
    def test_names_the_persona_what_it_holds_and_the_remedy(self):
        problem = privileged_default_persona_problem("admin", [PRIVILEGE_CONFIG_PANEL])
        assert problem is not None
        assert "'admin'" in problem
        assert PRIVILEGE_CONFIG_PANEL in problem
        assert "'readonly'" in problem
        assert "default_persona" in problem

    def test_unprivileged_default_is_fine(self):
        assert privileged_default_persona_problem("readonly", ()) is None

    @pytest.mark.parametrize("value", [None, "", 3, {"name": "admin"}])
    def test_missing_or_misshapen_default_is_not_this_checks_to_report(self, value: Any):
        assert privileged_default_persona_problem(value, [PRIVILEGE_CONFIG_PANEL]) is None


class TestUnauthenticatedPrivilegedTerminal:
    CAROL = {"name": "carol", "index": 3, "persona": "admin", "login": False}
    ALICE = {"name": "alice", "index": 0, "persona": "readwrite"}
    PRIVILEGES = {"admin": (PRIVILEGE_SETUP_TOOL, PRIVILEGE_CONFIG_PANEL)}

    def test_login_false_on_a_privileged_persona_is_named_with_its_remedy(self):
        problems = unauthenticated_privileged_terminal_problems(
            [self.ALICE, self.CAROL], self.PRIVILEGES
        )
        assert len(problems) == 1
        assert "'carol'" in problems[0]
        assert "login: false" in problems[0]
        assert "'admin'" in problems[0]
        assert PRIVILEGE_SETUP_TOOL in problems[0]
        assert "login: true" in problems[0]
        assert "'readonly'" in problems[0]

    def test_a_privileged_persona_behind_a_login_is_fine(self):
        entry = dict(self.CAROL)
        del entry["login"]
        assert unauthenticated_privileged_terminal_problems([entry], self.PRIVILEGES) == []

    def test_an_unprivileged_persona_may_be_served_openly(self):
        """The shipped `ariel` card: no login, and nothing it could edit."""
        ariel = {"name": "ariel", "index": 2, "persona": "ariel", "login": False}
        assert unauthenticated_privileged_terminal_problems([ariel], self.PRIVILEGES) == []

    def test_a_persona_whose_config_could_not_be_read_contributes_nothing(self):
        assert unauthenticated_privileged_terminal_problems([self.CAROL], {}) == []

    def test_an_entry_with_no_persona_in_effect_contributes_nothing(self):
        entry = {"name": "dave", "index": 4, "persona": None, "login": False}
        assert unauthenticated_privileged_terminal_problems([entry], self.PRIVILEGES) == []

    def test_every_offender_is_named_at_once(self):
        """An operator fixing a roster sees the whole list, not the second entry
        after fixing the first."""
        dave = {"name": "dave", "index": 4, "persona": "admin", "login": False}
        problems = unauthenticated_privileged_terminal_problems([self.CAROL, dave], self.PRIVILEGES)
        assert [problem for problem in problems if "'carol'" in problem]
        assert [problem for problem in problems if "'dave'" in problem]


class TestDeploymentWidePrivilegedExposure:
    """`auth.method: none`: no wall stands, so every privileged terminal is open."""

    CAROL = {"name": "carol", "index": 3, "persona": "admin"}
    ALICE = {"name": "alice", "index": 0, "persona": "readwrite"}
    PRIVILEGES = {"admin": (PRIVILEGE_SETUP_TOOL, PRIVILEGE_CONFIG_PANEL)}

    def test_every_privileged_entry_is_named_with_the_auth_remedy(self):
        problems = deployment_wide_privileged_exposure_problems(
            [self.ALICE, self.CAROL], self.PRIVILEGES
        )
        assert len(problems) == 1
        assert "'carol'" in problems[0]
        assert "auth.method" in problems[0]
        assert "'readonly'" in problems[0]

    def test_the_login_key_is_ignored_because_it_is_inert(self):
        """With no wall, `login: true` exempts nobody — the exposure is the same."""
        walled = dict(self.CAROL, login=True)
        assert len(deployment_wide_privileged_exposure_problems([walled], self.PRIVILEGES)) == 1

    def test_unprivileged_entries_are_not_named(self):
        assert deployment_wide_privileged_exposure_problems([self.ALICE], self.PRIVILEGES) == []


class TestAuthIsEnforced:
    @pytest.mark.parametrize(
        "web_terminals,expected",
        [
            ({}, False),
            # `none` is open and `token` is the magic-link posture: neither
            # stands a login wall in front of the roster.
            ({"auth": {"method": "none"}}, False),
            ({"auth": {"method": "token"}}, False),
            ({"auth": {"method": "password"}}, True),
            ({"auth": {"method": "oidc"}}, True),
            # An unknown method is lint's to reject; this guard does not pile on.
            ({"auth": {"method": "basic"}}, True),
        ],
    )
    def test_reads_the_stanza_the_way_the_nginx_seam_does(
        self, web_terminals: dict[str, Any], expected: bool
    ):
        assert auth_is_enforced(web_terminals) is expected


# ── The guard at `osprey init` time ──────────────────────────────────────────


def _shipped_host_config() -> dict[str, Any]:
    resolved, _dir = resolve_build_profile(None, "control-assistant")
    return copy.deepcopy(resolved.config)


def _persona_texts(config: dict[str, Any]):
    """Run `_persona_profile_texts` over a (possibly edited) host config."""
    from osprey.cli.profile_cmd import _persona_profile_texts

    resolved, _dir = resolve_build_profile(None, "control-assistant")
    resolved.config = config
    return _persona_profile_texts(resolved, "Test", "", "control-assistant")


class TestPersonaProfileTextsGuard:
    def test_shipped_catalog_materializes(self):
        """The stack ships safe: privileged tier behind a login, default unprivileged."""
        texts = _persona_texts(_shipped_host_config())
        assert set(texts) == {"admin", "ariel", "readonly", "readwrite"}

    def test_privileged_default_persona_refuses_with_the_remedy(self):
        import click

        config = _shipped_host_config()
        config["modules.web_terminals"]["default_persona"] = "admin"
        with pytest.raises(click.UsageError) as excinfo:
            _persona_texts(config)
        message = str(excinfo.value)
        assert "default_persona 'admin'" in message
        assert "'readonly'" in message

    def test_unauthenticated_privileged_entry_refuses_naming_the_user(self):
        import click

        config = _shipped_host_config()
        for user in config["modules.web_terminals"]["users"]:
            if user["name"] == "carol":
                user["login"] = False
        with pytest.raises(click.UsageError) as excinfo:
            _persona_texts(config)
        message = str(excinfo.value)
        assert "'carol'" in message
        assert "login: true" in message

    def test_an_unprivileged_persona_may_still_be_served_openly(self):
        """`ariel` keeps its login-less card — the guard is about privilege, not logins."""
        config = _shipped_host_config()
        texts = _persona_texts(config)
        assert "ariel" in texts

    def test_a_floorless_host_preset_refuses_the_open_entry_too(self):
        """The last place the pre-C1 reading survived. This guard used to be fed
        one baseline-RELATIVE map for both its rules, so on a host preset that
        floors neither surface every persona sat at the baseline, the map was
        empty for all of them, and `osprey init` emitted a repo serving an
        unauthenticated terminal that could edit the deployment. The login rule
        reads what the persona holds OUTRIGHT, which does not move when the
        floor goes away.

        The remedy has to move, though: with no floor there is no unprivileged
        tier to be re-pointed at, so it says how to create one."""
        import click

        config = _floorless(_shipped_host_config())
        for user in config["modules.web_terminals"]["users"]:
            if user["name"] == "carol":
                user["login"] = False
        with pytest.raises(click.UsageError) as excinfo:
            _persona_texts(config)
        message = str(excinfo.value)
        assert "'carol'" in message
        assert "floors neither surface" in message
        assert "the bundled stack's 'readonly'" not in message
        assert "login: true" in message
        # `osprey init` renders its refusals through rich, which eats `[...]`.
        assert "[" not in message

    def test_a_floorless_host_preset_behind_logins_still_materializes(self):
        """The negative control: dropping the floor is a migration, not an
        offence. Nothing here is served openly, so nothing is refused."""
        assert set(_persona_texts(_floorless(_shipped_host_config()))) == {
            "admin",
            "ariel",
            "readonly",
            "readwrite",
        }


# ── The lint belt at profile time ────────────────────────────────────────────


class TestProfileTimeBelt:
    """`osprey profile validate` / `osprey build`, which both reach `lint_profile_config`."""

    def test_shipped_preset_is_clean(self):
        findings = lint_profile_config(_shipped_host_config())
        assert DEFAULT_PERSONA_CODE not in _codes(findings)
        assert UNAUTHENTICATED_CODE not in _codes(findings)

    def test_privileged_default_persona_is_an_error(self):
        config = _shipped_host_config()
        config["modules.web_terminals"]["default_persona"] = "admin"
        findings = lint_profile_config(config)
        assert DEFAULT_PERSONA_CODE in _codes(findings)
        assert "'readonly'" in _messages(findings, DEFAULT_PERSONA_CODE)[0]

    def test_unauthenticated_privileged_entry_is_an_error_naming_the_user(self):
        config = _shipped_host_config()
        for user in config["modules.web_terminals"]["users"]:
            if user["name"] == "carol":
                user["login"] = False
        findings = lint_profile_config(config)
        assert UNAUTHENTICATED_CODE in _codes(findings)
        assert "'carol'" in _messages(findings, UNAUTHENTICATED_CODE)[0]

    def test_auth_none_is_reported_but_does_not_block(self):
        """The exposure is real and named; failing a build over the shipped
        default would reject deployments nobody exposed, so it is a WARN."""
        config = _shipped_host_config()
        config["modules.web_terminals"]["auth"] = {"method": "none"}
        findings = lint_profile_config(config)
        assert DEPLOYMENT_WIDE_CODE not in _codes(findings)
        assert DEPLOYMENT_WIDE_CODE in _codes(findings, "warn")
        messages = _messages(findings, DEPLOYMENT_WIDE_CODE)
        assert len(messages) == 1  # only carol's tier is privileged
        assert "'carol'" in messages[0]
        assert "auth.method" in messages[0]

    def test_auth_none_replaces_the_login_finding_rather_than_doubling_it(self):
        """One exposure, one message: `login: false` is inert with no wall."""
        config = _shipped_host_config()
        config["modules.web_terminals"]["auth"] = {"method": "none"}
        for user in config["modules.web_terminals"]["users"]:
            if user["name"] == "carol":
                user["login"] = False
        findings = lint_profile_config(config)
        assert UNAUTHENTICATED_CODE not in [f.code for f in findings]
        assert len(_messages(findings, DEPLOYMENT_WIDE_CODE)) == 1

    def test_an_unreferenced_privileged_persona_deploys_nothing(self):
        """A catalog entry nobody runs must not be able to fail a build."""
        config = _shipped_host_config()
        web_terminals = config["modules.web_terminals"]
        web_terminals["users"] = [
            user for user in web_terminals["users"] if user["name"] != "carol"
        ]
        findings = lint_profile_config(config)
        assert DEFAULT_PERSONA_CODE not in _codes(findings)
        assert UNAUTHENTICATED_CODE not in _codes(findings)


def _delta_repo(tmp_path: Path, *, admin_login: Any) -> dict[str, Any]:
    """A materialized-repo shape: a host `config:` bag plus `personas/*.yml` deltas.

    This is what `osprey init` leaves behind and what `osprey profile validate`
    is actually pointed at — the catalog names deltas, not presets, and the
    privileged keys live in the delta file rather than anywhere in the config
    being linted.
    """
    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    (personas_dir / "admin.yml").write_text(
        yaml.safe_dump(
            {
                "name": "Admin",
                "config": {
                    "claude_code.permissions.remove_deny": [SETUP_TOOL],
                    "web.config_panel.enabled": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (personas_dir / "readonly.yml").write_text(
        yaml.safe_dump({"name": "Readonly", "config": {"control_system.writes_enabled": False}}),
        encoding="utf-8",
    )
    carol: dict[str, Any] = {"name": "carol", "index": 1, "persona": "admin"}
    if admin_login is not None:
        carol["login"] = admin_login
    return {
        "facility.prefix": "ca",
        "claude_code.permissions.deny": [SETUP_TOOL],
        "web.config_panel.enabled": False,
        "modules.web_terminals": {
            "enabled": True,
            "image_source": "local",
            "default_persona": "readonly",
            "auth": {"method": "password"},
            "users": [{"name": "bob", "index": 0, "persona": "readonly"}, carol],
            "personas": {
                "readonly": {
                    "project": "ca-readonly",
                    "project_path": "build/ca-readonly",
                    "build_profile": "personas/readonly.yml",
                },
                "admin": {
                    "project": "ca-admin",
                    "project_path": "build/ca-admin",
                    "build_profile": "personas/admin.yml",
                },
            },
        },
    }


def _floorless(config: dict[str, Any]) -> dict[str, Any]:
    """Strip the base tier's deny floor and panel default from a profile config.

    THE repro behind the second review's first critical finding, and one an
    operator reaches by deleting two lines: with no floor every persona sits at
    the baseline, so a rule that asked "what does this persona hold BEYOND its
    baseline" answered "nothing" about a persona holding everything.
    """
    config = copy.deepcopy(config)
    config.pop("claude_code.permissions.deny", None)
    config.pop("web.config_panel.enabled", None)
    return config


class TestFloorlessDeploymentsAreNotExempt:
    """A profile that floors nothing must still not serve an open privileged terminal.

    The `login: false` rule is judged on what a persona ABSOLUTELY holds, so
    deleting the floor makes the finding worse rather than making it disappear.
    Its sibling rules stay baseline-relative — see `privileges_beyond_baseline`
    — and this class pins both sides of that asymmetry.
    """

    def test_login_false_is_refused_with_a_remedy_that_presupposes_no_tier(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        findings = lint_profile_config(_floorless(_delta_repo(tmp_path, admin_login=False)))
        assert UNAUTHENTICATED_CODE in _codes(findings)
        message = _messages(findings, UNAUTHENTICATED_CODE)[0]
        assert "'carol'" in message
        # The tier remedy would name a persona this deployment does not have.
        assert "the bundled stack's 'readonly'" not in message
        assert "floors neither surface" in message
        assert "claude_code.permissions.deny" in message
        assert SETUP_TOOL in message
        assert "web.config_panel.enabled: false" in message
        assert "claude_code.permissions.remove_deny" in message
        assert "login: true" in message
        # `osprey build` renders refusals through rich, which reads `[...]` as a
        # style tag: an earlier draft wrote the deny list as `deny: [<tool>]`
        # and the tool name vanished from the build's output.
        assert "[" not in message

    def test_the_same_deployment_with_everyone_authenticated_is_clean(
        self, tmp_path: Path, monkeypatch
    ):
        """The negative control, and the migration the relative reading exists
        for: a floorless deployment behind a login draws nothing at all — not
        the login finding, and not the `default_persona` finding either."""
        monkeypatch.chdir(tmp_path)
        findings = lint_profile_config(_floorless(_delta_repo(tmp_path, admin_login=None)))
        assert UNAUTHENTICATED_CODE not in _codes(findings)
        assert DEFAULT_PERSONA_CODE not in _codes(findings)
        assert UNKNOWN_PRIVILEGE_CODE not in _codes(findings)

    def test_a_floorless_privileged_default_persona_is_still_exempt(
        self, tmp_path: Path, monkeypatch
    ):
        """The asymmetry, stated as a test: an inherited default on a profile
        written before the floor existed has no better tier to be pointed at,
        so it stays silent where the typed `login: false` key does not."""
        monkeypatch.chdir(tmp_path)
        config = _floorless(_delta_repo(tmp_path, admin_login=None))
        config["modules.web_terminals"]["default_persona"] = "admin"
        assert DEFAULT_PERSONA_CODE not in _codes(lint_profile_config(config))

    def test_the_remedy_is_chosen_by_the_baseline_not_by_the_altitude(self):
        """The unit behind it: the same roster and the same privileges read one
        way against a floored base and another against a floorless one."""
        entries = [{"name": "carol", "persona": "admin", "login": False}]
        privileges = {"admin": ALL_PRIVILEGES}

        floored = unauthenticated_privileged_terminal_problems(entries, privileges)
        floorless = unauthenticated_privileged_terminal_problems(
            entries, privileges, baseline_privileges=ALL_PRIVILEGES
        )

        assert "the bundled stack's 'readonly'" in floored[0]
        assert "floors neither surface" not in floored[0]
        assert "floors neither surface" in floorless[0]
        # Both name the user and the exposure; only the way out differs.
        for message in (floored[0], floorless[0]):
            assert "'carol'" in message
            assert "Anyone who opens that terminal's page edits this deployment" in message

    def test_a_partly_floored_baseline_names_the_surface_it_leaves_open(self):
        """A PARTIAL floor is not a tier split either. With the panel unfloored
        every persona holds it, so "point it at a persona that holds neither"
        names a persona this deployment need not have — the same dishonest
        remedy the floorless case was fixed for. The honest one names the one
        surface still open and how to floor it."""
        problems = unauthenticated_privileged_terminal_problems(
            [{"name": "carol", "persona": "admin", "login": False}],
            {"admin": ALL_PRIVILEGES},
            baseline_privileges=[PRIVILEGE_CONFIG_PANEL],
        )
        assert "the bundled stack's 'readonly'" not in problems[0]
        assert f"does not floor {PRIVILEGE_CONFIG_PANEL} for its base tier" in problems[0]
        assert "web.config_panel.enabled: false" in problems[0]
        assert "web.config_panel.enabled: true" in problems[0]
        # The surface this deployment DOES floor is not mentioned: an operator
        # told to add a deny it already has would go looking for a second bug.
        assert "claude_code.permissions.deny" not in problems[0]
        assert "login: true" in problems[0]
        assert "[" not in problems[0]

    def test_a_baseline_with_only_the_setup_tool_names_that_one(self):
        """The mirror image, so the clause is built from the surface rather than
        from whichever one the sentence happened to be written around."""
        problems = unauthenticated_privileged_terminal_problems(
            [{"name": "carol", "persona": "admin", "login": False}],
            {"admin": ALL_PRIVILEGES},
            baseline_privileges=[PRIVILEGE_SETUP_TOOL],
        )
        assert f"does not floor {PRIVILEGE_SETUP_TOOL} for its base tier" in problems[0]
        assert "claude_code.permissions.deny" in problems[0]
        assert SETUP_TOOL in problems[0]
        assert "web.config_panel.enabled" not in problems[0]
        assert "[" not in problems[0]

    def test_a_fully_floored_baseline_keeps_the_tier_remedy(self):
        """The shape that HAS an unprivileged tier: nothing to floor, so the
        remedy points at the tier instead of naming keys."""
        problems = unauthenticated_privileged_terminal_problems(
            [{"name": "carol", "persona": "admin", "login": False}],
            {"admin": ALL_PRIVILEGES},
            baseline_privileges=[],
        )
        assert "the bundled stack's 'readonly'" in problems[0]
        assert "does not floor" not in problems[0]


class TestProfileTimeBeltReadsPersonaDeltas:
    def test_delta_that_lifts_the_floor_is_caught(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        findings = lint_profile_config(_delta_repo(tmp_path, admin_login=False))
        assert UNAUTHENTICATED_CODE in _codes(findings)
        assert "'carol'" in _messages(findings, UNAUTHENTICATED_CODE)[0]

    def test_the_same_delta_behind_a_login_is_fine(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        findings = lint_profile_config(_delta_repo(tmp_path, admin_login=None))
        assert UNAUTHENTICATED_CODE not in _codes(findings)

    def test_privileged_default_persona_from_a_delta_is_caught(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _delta_repo(tmp_path, admin_login=None)
        config["modules.web_terminals"]["default_persona"] = "admin"
        findings = lint_profile_config(config)
        assert DEFAULT_PERSONA_CODE in _codes(findings)

    def test_a_delta_file_that_is_not_there_is_refused_not_assumed_harmless(
        self, tmp_path: Path, monkeypatch
    ):
        """ "Cannot tell" is not "holds nothing" where the answer decides something."""
        monkeypatch.chdir(tmp_path)
        config = _delta_repo(tmp_path, admin_login=False)
        (tmp_path / "personas" / "admin.yml").unlink()
        findings = lint_profile_config(config)
        assert UNKNOWN_PRIVILEGE_CODE in _codes(findings)
        message = _messages(findings, UNKNOWN_PRIVILEGE_CODE)[0]
        assert "'admin'" in message
        assert "personas/admin.yml" in message
        assert "'carol'" in message

    def test_an_unreadable_persona_behind_a_login_is_not_refused(self, tmp_path: Path, monkeypatch):
        """A privileged persona behind a login is no exposure, so nor is an
        unreadable one — its real problem is reported by the check that owns it."""
        monkeypatch.chdir(tmp_path)
        config = _delta_repo(tmp_path, admin_login=None)
        (tmp_path / "personas" / "admin.yml").unlink()
        assert UNKNOWN_PRIVILEGE_CODE not in _codes(lint_profile_config(config))

    def test_an_unreadable_default_persona_is_refused(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _delta_repo(tmp_path, admin_login=None)
        config["modules.web_terminals"]["default_persona"] = "admin"
        (tmp_path / "personas" / "admin.yml").unlink()
        findings = lint_profile_config(config)
        assert UNKNOWN_PRIVILEGE_CODE in _codes(findings)
        assert "default_persona" in _messages(findings, UNKNOWN_PRIVILEGE_CODE)[0]

    def test_a_build_profile_outside_personas_is_refused_where_it_is_authored(
        self, tmp_path: Path, monkeypatch
    ):
        """The escape hatch: `../admin.yml` resolves as neither a delta nor
        a preset, so it used to read as unprivileged and build clean."""
        monkeypatch.chdir(tmp_path)
        config = _delta_repo(tmp_path, admin_login=False)
        (tmp_path / "personas" / "admin.yml").rename(tmp_path / "admin.yml")
        config["modules.web_terminals"]["personas"]["admin"]["build_profile"] = "../admin.yml"
        findings = lint_profile_config(config, profile_root=tmp_path)
        assert UNKNOWN_PRIVILEGE_CODE in _codes(findings)
        assert "'../admin.yml'" in _messages(findings, UNKNOWN_PRIVILEGE_CODE)[0]

    def test_a_catalog_entry_with_no_build_profile_is_refused(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _delta_repo(tmp_path, admin_login=False)
        del config["modules.web_terminals"]["personas"]["admin"]["build_profile"]
        assert UNKNOWN_PRIVILEGE_CODE in _codes(lint_profile_config(config))

    def test_a_deployment_with_no_floor_still_refuses_an_unreadable_open_terminal(
        self, tmp_path: Path, monkeypatch
    ):
        """The unknown rule's `login: false` half is NOT gated on the split.

        A floorless deployment hands both surfaces to every persona it has, so
        a persona nobody can read is one nobody can show holds less than
        everything — and this one is served to whoever opens its page. The
        earlier reading exempted exactly the deployment with the most to hide.
        """
        monkeypatch.chdir(tmp_path)
        config = _floorless(_delta_repo(tmp_path, admin_login=False))
        (tmp_path / "personas" / "admin.yml").unlink()
        findings = lint_profile_config(config)
        assert UNKNOWN_PRIVILEGE_CODE in _codes(findings)
        message = _messages(findings, UNKNOWN_PRIVILEGE_CODE)[0]
        assert "'carol'" in message
        assert "floors neither surface" in message

    def test_a_deployment_with_no_floor_has_no_unknown_DEFAULT_to_report(
        self, tmp_path: Path, monkeypatch
    ):
        """The other half keeps the gate. `default_persona` is inherited, not
        chosen, and a floorless deployment has no unprivileged tier to be
        re-pointed at — the same migration argument that makes the KNOWN
        default-persona rule baseline-relative."""
        monkeypatch.chdir(tmp_path)
        config = _floorless(_delta_repo(tmp_path, admin_login=None))
        config["modules.web_terminals"]["default_persona"] = "admin"
        (tmp_path / "personas" / "admin.yml").unlink()
        assert UNKNOWN_PRIVILEGE_CODE not in _codes(lint_profile_config(config))

    def test_auth_none_leaves_the_entry_half_of_the_unknown_rule_alone(
        self, tmp_path: Path, monkeypatch
    ):
        """With no wall standing no entry singled itself out, and the known
        version of this exposure is only advisory — so nor does the unknown one
        become an error."""
        monkeypatch.chdir(tmp_path)
        config = _delta_repo(tmp_path, admin_login=False)
        config["modules.web_terminals"]["auth"] = {"method": "none"}
        (tmp_path / "personas" / "admin.yml").unlink()
        assert UNKNOWN_PRIVILEGE_CODE not in _codes(lint_profile_config(config))


class TestProfileRootIsNotTheWorkingDirectory:
    """The seam the belt used to have: a delta resolved against the cwd.

    `cd data && osprey build`, `osprey profile validate <repo>` from the
    directory above, and `osprey build --repo <path>` each ran with a working
    directory that is not the profile's — and each read no delta at all, so the
    privileged roster the guard exists to refuse validated and built clean.
    """

    def test_a_run_from_a_subdirectory_still_refuses_the_roster(self, tmp_path: Path, monkeypatch):
        config = _delta_repo(tmp_path, admin_login=False)
        subdirectory = tmp_path / "data"
        subdirectory.mkdir()
        monkeypatch.chdir(subdirectory)
        findings = lint_profile_config(config, profile_root=tmp_path)
        assert UNAUTHENTICATED_CODE in _codes(findings)
        assert "'carol'" in _messages(findings, UNAUTHENTICATED_CODE)[0]

    def test_a_run_from_outside_the_repo_still_refuses_the_roster(
        self, tmp_path: Path, monkeypatch
    ):
        """The `--repo` shape: the cwd has no `personas/` directory at all."""
        repo = tmp_path / "repo"
        repo.mkdir()
        config = _delta_repo(repo, admin_login=False)
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.chdir(outside)
        assert UNAUTHENTICATED_CODE in _codes(lint_profile_config(config, profile_root=repo))

    def test_without_a_profile_root_the_check_sees_nothing(self, tmp_path: Path, monkeypatch):
        """Pinning the fallback so the omission is a caller bug, not a silent one."""
        config = _delta_repo(tmp_path, admin_login=False)
        subdirectory = tmp_path / "data"
        subdirectory.mkdir()
        monkeypatch.chdir(subdirectory)
        findings = lint_profile_config(config)
        assert UNAUTHENTICATED_CODE not in _codes(findings)
        # Not silent, though: it reports that it could not read the persona.
        assert UNKNOWN_PRIVILEGE_CODE in _codes(findings)


# ── The lint belt at deploy time ─────────────────────────────────────────────


def _rendered_repo(tmp_path: Path, *, admin_config: dict[str, Any], login: Any) -> dict[str, Any]:
    """A rendered project: each persona's own `build/<project>/config.yml` on disk."""
    for project, persona_config in (
        ("ca-admin", admin_config),
        (
            "ca-readonly",
            {
                "claude_code": {"permissions": {"deny": [SETUP_TOOL]}},
                "web": {"config_panel": {"enabled": False}},
            },
        ),
    ):
        project_dir = tmp_path / "build" / project
        project_dir.mkdir(parents=True)
        (project_dir / "config.yml").write_text(yaml.safe_dump(persona_config), encoding="utf-8")
    carol: dict[str, Any] = {"name": "carol", "index": 1, "persona": "admin"}
    if login is not None:
        carol["login"] = login
    return {
        "facility": {"prefix": "ca"},
        # The deploy config's own posture is the BASELINE every persona is judged
        # against, so the floor has to be here for there to be a tier split at
        # all — which is exactly what a rendered control-assistant project has.
        "claude_code": {"permissions": {"deny": [SETUP_TOOL]}},
        "web": {"config_panel": {"enabled": False}},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "image_source": "local",
                "default_persona": "readonly",
                "auth": {"method": "password"},
                "users": [{"name": "bob", "index": 0, "persona": "readonly"}, carol],
                "personas": {
                    "readonly": {
                        "project": "ca-readonly",
                        "project_path": "build/ca-readonly",
                        "build_profile": "personas/readonly.yml",
                    },
                    "admin": {
                        "project": "ca-admin",
                        "project_path": "build/ca-admin",
                        "build_profile": "personas/admin.yml",
                    },
                },
            }
        },
    }


def _ship_persona_settings(root: Path, *projects: str, lifted: tuple[str, ...] = ()) -> None:
    """Give each named rendered project the `.claude/settings.json` a build writes.

    `_rendered_repo` writes only the `config.yml` the privilege rules read, which is
    all those rules need. The open-mode egress gate reads this second artifact and
    fails closed on its absence, so an `auth.method: none` deployment has to ship it
    here too. `lifted` drops entries from the shipped deny list, the way a persona
    whose config carried `remove_deny` would render.
    """
    for project in projects:
        settings_dir = root / "build" / project / ".claude"
        settings_dir.mkdir(parents=True, exist_ok=True)
        deny = [entry for entry in DENY_DEFAULTS if entry not in lifted]
        (settings_dir / "settings.json").write_text(
            json.dumps({"permissions": {"deny": deny}}), encoding="utf-8"
        )


def _unrender(root: Path, project: str) -> None:
    """Remove a persona project's whole rendered directory.

    The `project_path` drift case, and the one the belt used to be fail-open
    on: a directory that is not there at all draws only a WARN
    (`persona_project_path_not_rendered_yet`), unlike an empty one, which is a
    hard `persona_missing_config_yml`. So this is the shape where "cannot read
    its privileges" reached the deploy path unaccompanied.
    """
    directory = root / "build" / project
    for path in sorted(directory.iterdir()):
        path.unlink()
    directory.rmdir()


def _registry_repo(tmp_path: Path, *, login: Any) -> dict[str, Any]:
    """A deployment in the DEFAULT image mode: images built in CI, nothing rendered here.

    `image_source: registry` is what `effective_image_source` returns for
    everything but the literal `local`, so this is the shape most fixtures in
    this file are not. The deploy host legitimately holds no rendered persona
    project — the images and their deltas were built elsewhere — so every
    persona this roster references is unreadable here.
    """
    carol: dict[str, Any] = {"name": "carol", "index": 1, "persona": "admin"}
    if login is not None:
        carol["login"] = login
    (tmp_path / "build").mkdir(exist_ok=True)
    return {
        "facility": {"prefix": "ca"},
        "claude_code": {"permissions": {"deny": [SETUP_TOOL]}},
        "web": {"config_panel": {"enabled": False}},
        "modules": {
            "web_terminals": {
                "enabled": True,
                # No `image_source`: registry is the default, and the premise of
                # registry mode is that no render happened on this host.
                "default_persona": "readonly",
                "auth": {"method": "password"},
                "users": [{"name": "bob", "index": 0, "persona": "readonly"}, carol],
                "personas": {
                    "readonly": {
                        "project": "ca-readonly",
                        "build_profile": "personas/readonly.yml",
                    },
                    "admin": {
                        "project": "ca-admin",
                        "build_profile": "personas/admin.yml",
                    },
                },
            }
        },
    }


PRIVILEGED_RENDER = {
    "claude_code": {"permissions": {"deny": ["Bash"]}},
    "web": {"config_panel": {"enabled": True}},
}
FLOORED_RENDER = {
    "claude_code": {"permissions": {"deny": [SETUP_TOOL]}},
    "web": {"config_panel": {"enabled": False}},
}


class TestRenderedProjectBelt:
    """`osprey up`, which lints the deploy config against what the build produced.

    An unauthenticated privileged terminal BLOCKS here as it does at profile
    altitude: it is the same open door either way, this is the only altitude a
    hand-edited `config.yml` is ever read at, and every surface that gates on
    the lint filters to errors — so a warning here reached nobody. A render that
    simply predates the base tier's deny floor is refused too, and told to
    rebuild.
    """

    def test_privileged_render_without_a_login_is_refused(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        findings = lint_web_terminals(config)
        assert UNAUTHENTICATED_CODE in _codes(findings)
        assert "'carol'" in _messages(findings, UNAUTHENTICATED_CODE)[0]

    def test_a_stale_render_is_refused_and_told_to_rebuild(self, tmp_path: Path, monkeypatch):
        """The remedy an operator can act on: the profile may be fine already."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        message = _messages(lint_web_terminals(config), UNAUTHENTICATED_CODE)[0]
        assert "osprey build" in message
        assert "predates" in message

    def test_the_rebuild_remedy_is_not_said_at_profile_altitude(self):
        """A profile has no render to be stale; the remedy there is an edit."""
        config = _shipped_host_config()
        for user in config["modules.web_terminals"]["users"]:
            if user["name"] == "carol":
                user["login"] = False
        message = _messages(lint_profile_config(config), UNAUTHENTICATED_CODE)[0]
        assert "osprey build" not in message

    def test_project_root_is_the_repo_the_caller_names_not_the_cwd(
        self, tmp_path: Path, monkeypatch
    ):
        """`osprey up` from anywhere reads the same persona configs."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        monkeypatch.chdir(elsewhere)
        assert UNAUTHENTICATED_CODE not in _codes(lint_web_terminals(config))
        findings = lint_web_terminals(config, project_root=tmp_path)
        assert UNAUTHENTICATED_CODE in _codes(findings)
        assert "'carol'" in _messages(findings, UNAUTHENTICATED_CODE)[0]

    def test_floored_render_without_a_login_is_silent(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=FLOORED_RENDER, login=False)
        findings = lint_web_terminals(config)
        # Both severities: the rule is an error here, so asserting only the
        # warnings would pass on a config it was screaming about.
        assert UNAUTHENTICATED_CODE not in _codes(findings)
        assert UNAUTHENTICATED_CODE not in _codes(findings, "warn")

    def test_privileged_render_behind_a_login_is_silent(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        findings = lint_web_terminals(config)
        assert UNAUTHENTICATED_CODE not in _codes(findings)
        assert UNAUTHENTICATED_CODE not in _codes(findings, "warn")

    def test_privileged_default_persona_in_a_render_is_reported(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        config["modules"]["web_terminals"]["default_persona"] = "admin"
        findings = lint_web_terminals(config)
        assert DEFAULT_PERSONA_CODE in _codes(findings, "warn")
        assert DEFAULT_PERSONA_CODE not in _codes(findings)

    def test_a_deployment_with_no_floor_is_refused_with_the_floor_remedy(
        self, tmp_path: Path, monkeypatch
    ):
        """A render whose two floor keys were deleted by hand. The deployment
        now hands both surfaces to every persona it has, so an open terminal
        there is the WORST version of this finding, not an exempt one — and the
        remedy switches to the one an operator with no tier can carry out."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        del config["claude_code"]
        del config["web"]
        findings = lint_web_terminals(config)
        assert UNAUTHENTICATED_CODE in _codes(findings)
        message = _messages(findings, UNAUTHENTICATED_CODE)[0]
        assert "'carol'" in message
        assert "floors neither surface" in message

    def test_an_unrendered_persona_is_refused_where_it_is_exposed(
        self, tmp_path: Path, monkeypatch
    ):
        """A persona referenced by a `login: false` entry whose project was
        never rendered — a `project_path` typo, a partial build.

        Its privileges cannot be read, and the only other signal is a WARN about
        a different question (`persona_project_path_not_rendered_yet`) that no
        error-filtering surface sees. Once `osprey up` consults this belt,
        reading the absent render as "holds nothing" is fail-open on the deploy
        path itself, so it is refused and told to rebuild.
        """
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        _unrender(tmp_path, "ca-admin")
        findings = lint_web_terminals(config)
        # Not guessed at: the privilege itself is still not asserted.
        assert UNAUTHENTICATED_CODE not in _codes(findings)
        assert UNAUTHENTICATED_CODE not in _codes(findings, "warn")
        assert UNKNOWN_PRIVILEGE_CODE in _codes(findings)
        message = _messages(findings, UNKNOWN_PRIVILEGE_CODE)[0]
        assert "'admin'" in message
        assert "'carol'" in message
        assert "build/ca-admin" in message
        assert "osprey build" in message
        # The plain warning is still said too: this rule adds the refusal, it
        # does not take over reporting an unrendered project.
        assert NOT_RENDERED_CODE in _codes(findings, "warn")

    def test_an_unrendered_persona_behind_a_login_stays_a_plain_warning(
        self, tmp_path: Path, monkeypatch
    ):
        """The narrowing: an unrendered persona nobody is exposed by is the
        render check's finding, not this one's. Otherwise every partial build
        would fail over personas that deploy nothing open."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        _unrender(tmp_path, "ca-admin")
        findings = lint_web_terminals(config)
        assert UNKNOWN_PRIVILEGE_CODE not in _codes(findings)
        assert NOT_RENDERED_CODE in _codes(findings, "warn")

    def test_the_unknown_refusal_offers_the_login_it_is_about(self, tmp_path: Path, monkeypatch):
        """Both other remedies presuppose a document that can be made to
        resolve — a render this deployment may build in CI rather than here. The
        entry's own `login: true` is the one an operator can always carry out,
        and it is the one that closes the door the finding is about, so the
        refusal is never a dead end."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        _unrender(tmp_path, "ca-admin")
        message = _messages(lint_web_terminals(config), UNKNOWN_PRIVILEGE_CODE)[0]
        assert "or set login: true for 'carol'" in message
        assert "[" not in message

    def test_the_unknown_refusal_says_both_reasons_when_both_apply(
        self, tmp_path: Path, monkeypatch
    ):
        """A persona can be at stake twice over — it is the `default_persona`
        AND an entry opted out of the login wall to reach it. The open door is
        the more serious half and the one whose remedy differs, so the finding
        says both rather than letting whichever ran first win."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        config["modules"]["web_terminals"]["default_persona"] = "admin"
        _unrender(tmp_path, "ca-admin")
        message = _messages(lint_web_terminals(config), UNKNOWN_PRIVILEGE_CODE)[0]
        assert "default_persona" in message
        assert "'carol'" in message
        assert "without a login" in message
        assert "or set login: true for 'carol'" in message

    def test_every_open_entry_on_one_unreadable_persona_is_named(self, tmp_path: Path, monkeypatch):
        """Two `login: false` entries, one unreadable persona. Naming only the
        first costs the operator a round trip: they set `login: true` for carol,
        re-run, and meet the same refusal for dave. The KNOWN half of this rule
        emits one message per entry and names them all — this half says both in
        one message, in the reason clause AND in the remedy."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        config["modules"]["web_terminals"]["users"].append(
            {"name": "dave", "index": 2, "persona": "admin", "login": False}
        )
        _unrender(tmp_path, "ca-admin")
        messages = _messages(lint_web_terminals(config), UNKNOWN_PRIVILEGE_CODE)
        assert len(messages) == 1
        assert "users 'carol' and 'dave' are served without a login" in messages[0]
        assert "or set login: true for 'carol' and 'dave'" in messages[0]
        assert "[" not in messages[0]

    def test_an_unreadable_persona_under_auth_none_is_a_warning_not_silence(
        self, tmp_path: Path, monkeypatch
    ):
        """With no wall standing no entry singled itself out, so nothing here is
        refused. It is not dropped either: under the identical posture a
        READABLE privileged persona draws its own warning, and this belt's whole
        premise is that "cannot tell" is never quieter than "known privileged".
        Skipping the advisory rung is how registry mode plus auth-off came to
        report nothing at all."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        config["modules"]["web_terminals"]["auth"] = {"method": "none"}
        _unrender(tmp_path, "ca-admin")
        findings = lint_web_terminals(config)

        assert UNKNOWN_PRIVILEGE_CODE not in _codes(findings)
        assert UNKNOWN_PRIVILEGE_CODE in _codes(findings, "warn")
        message = _messages(findings, UNKNOWN_PRIVILEGE_CODE)[0]
        assert "'carol'" in message
        assert "auth.method is not password or oidc" in message
        assert "turn authentication on" in message
        assert "[" not in message

    def test_the_registry_and_auth_none_cell_is_not_silent(self, tmp_path: Path, monkeypatch):
        """The exact cell the round-3 review found reporting nothing at any
        severity: both defaults at once, and a referenced persona nobody can
        read."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        config["modules"]["web_terminals"]["auth"] = {"method": "none"}
        del config["modules"]["web_terminals"]["image_source"]
        _unrender(tmp_path, "ca-admin")
        findings = lint_web_terminals(config)

        assert UNKNOWN_PRIVILEGE_CODE not in _codes(findings)
        assert UNKNOWN_PRIVILEGE_CODE in _codes(findings, "warn")

    def test_the_stance_names_the_surface_a_partial_floor_leaves_open(
        self, tmp_path: Path, monkeypatch
    ):
        """The posture sentence is built from the same baseline reading as the
        remedy that may follow it, so one message cannot claim a floor in one
        sentence and name it as missing in the next. Here the base denies the
        setup tool but leaves the Config panel on for every persona."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        del config["web"]
        _unrender(tmp_path, "ca-admin")
        message = _messages(lint_web_terminals(config), UNKNOWN_PRIVILEGE_CODE)[0]
        assert f"floors {PRIVILEGE_SETUP_TOOL} for its base tier" in message
        assert f"leaves {PRIVILEGE_CONFIG_PANEL} open to every persona" in message
        assert "floors both surfaces" not in message
        assert "floors neither surface" not in message

    def test_a_fully_floored_deployment_says_it_floors_both(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        _unrender(tmp_path, "ca-admin")
        message = _messages(lint_web_terminals(config), UNKNOWN_PRIVILEGE_CODE)[0]
        assert "floors both surfaces for its base tier" in message

    def test_an_unreadable_default_persona_is_refused_in_local_mode(
        self, tmp_path: Path, monkeypatch
    ):
        """`image_source: local` says the render was supposed to happen HERE, so
        a `default_persona` whose project is missing is drift — every roster
        entry that names no tier inherits a persona nobody can read — and it is
        refused."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        config["modules"]["web_terminals"]["default_persona"] = "admin"
        _unrender(tmp_path, "ca-admin")
        findings = lint_web_terminals(config)
        assert UNKNOWN_PRIVILEGE_CODE in _codes(findings)
        message = _messages(findings, UNKNOWN_PRIVILEGE_CODE)[0]
        assert "'admin'" in message
        assert "default_persona" in message

    def test_the_same_unreadable_default_persona_is_left_alone_in_registry_mode(
        self, tmp_path: Path, monkeypatch
    ):
        """The gate, as a one-key contrast with the test above. In registry mode
        the images and their deltas are built in CI and this host holds no
        render by design, so an unreadable default is that mode's normal state
        rather than drift, and no remedy exists that a pull-only host can carry
        out. The profile altitude, where the deltas live, owns that question."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        config["modules"]["web_terminals"]["default_persona"] = "admin"
        del config["modules"]["web_terminals"]["image_source"]
        _unrender(tmp_path, "ca-admin")
        assert UNKNOWN_PRIVILEGE_CODE not in _codes(lint_web_terminals(config))

    def test_the_login_half_is_not_gated_on_the_image_source(self, tmp_path: Path, monkeypatch):
        """The other half of the gate: an entry that typed itself public is
        refused in either mode, because its remedy is its own login key and
        every host holds that."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        del config["modules"]["web_terminals"]["image_source"]
        _unrender(tmp_path, "ca-admin")
        findings = lint_web_terminals(config)
        assert UNKNOWN_PRIVILEGE_CODE in _codes(findings)
        assert "'carol'" in _messages(findings, UNKNOWN_PRIVILEGE_CODE)[0]

    def test_an_unknown_default_persona_nobody_is_exposed_by_offers_no_login(
        self, tmp_path: Path, monkeypatch
    ):
        """The clause names a user, so it is said only where an entry named
        itself. An inherited default behind the wall has no `login: false` to
        turn off."""
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        config["modules"]["web_terminals"]["default_persona"] = "admin"
        _unrender(tmp_path, "ca-admin")
        message = _messages(lint_web_terminals(config), UNKNOWN_PRIVILEGE_CODE)[0]
        assert "default_persona" in message
        assert "login: true" not in message

    def test_a_disabled_module_is_not_linted_at_all(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        config["modules"]["web_terminals"]["enabled"] = False
        assert lint_web_terminals(config) == []


# ── The advisory channel ─────────────────────────────────────────────────────


class TestProfileConfigWarnings:
    """`auth.method: none` in front of a privileged terminal is deliberately not
    build-failing, so printing it is the only way it reaches an operator."""

    def test_the_deployment_wide_exposure_is_a_warning_not_an_error(self):
        config = _shipped_host_config()
        config["modules.web_terminals"]["auth"] = {"method": "none"}
        warnings = profile_config_warnings(config)
        assert any("auth.method" in message and "'carol'" in message for message in warnings)
        assert not any("auth.method" in message for message in profile_config_errors(config))

    def test_a_clean_profile_draws_no_advisory_about_privilege(self):
        assert not any(
            "auth.method" in message for message in profile_config_warnings(_shipped_host_config())
        )

    def test_warnings_read_the_profile_root_too(self, tmp_path: Path, monkeypatch):
        """Same seam as the errors: an advisory run from the wrong directory
        cannot find the delta, so it cannot name what the persona holds.

        It does not fall silent, though — that was the hole. Blind, the belt
        says it cannot tell; given the root, it names the exposure itself.
        """
        config = _delta_repo(tmp_path, admin_login=None)
        config["modules.web_terminals"]["auth"] = {"method": "none"}
        subdirectory = tmp_path / "data"
        subdirectory.mkdir()
        monkeypatch.chdir(subdirectory)

        blind = profile_config_warnings(config)
        assert blind
        assert all("could not be read" in message for message in blind)
        assert not any("anyone who can reach this deployment" in message for message in blind)

        seeing = profile_config_warnings(config, profile_root=tmp_path)
        assert any(
            "auth.method" in message and "anyone who can reach this deployment" in message
            for message in seeing
        )
        assert not any("could not be read" in message for message in seeing)


class TestPrivilegePhrase:
    def test_two_privileges_read_as_a_sentence(self):
        assert privilege_phrase(ALL_PRIVILEGES) == (
            f"{PRIVILEGE_SETUP_TOOL} and {PRIVILEGE_CONFIG_PANEL}"
        )

    def test_one_privilege_is_itself(self):
        assert privilege_phrase([PRIVILEGE_CONFIG_PANEL]) == PRIVILEGE_CONFIG_PANEL

    def test_nothing_to_name_is_the_empty_string_not_an_IndexError(self):
        """No caller reaches this with nothing, but a message helper that raised
        would turn a guard's own bug into a traceback from inside a build."""
        assert privilege_phrase(()) == ""


# ── The belt on the deploy path ──────────────────────────────────────────────


class TestTheUpPreflightGate:
    """`osprey up` consults this belt, which it never did before.

    Two `osprey` verbs read the rendered altitude — `osprey scaffold
    web-terminals lint` and the `render` pre-render gate — and neither is on the
    deploy path. So a hand-edit to `build/config.yml` (a file no build
    fingerprint covers: `profile.yml` is untouched, so the staleness check is
    silent) could set `login: false` on a privileged persona, fail the lint, and
    still pass `osprey up`'s preflight and start containers.

    The gate is deliberately NARROW: only the two open-door codes refuse a
    start. Every other lint error already gates the authoring verbs, and a start
    refused over the shape of a config an operator has since fixed by hand in
    the rendered file would be `up` second-guessing the verbs that own it.
    """

    @staticmethod
    def _problems(config: dict[str, Any], root: Path, monkeypatch) -> list[str]:
        from osprey.deployment.web_terminals import provision

        # The render probe has its own suite; inert here so these tests are
        # about the privilege gate rather than about any fixture's Dockerfiles.
        monkeypatch.setattr(provision, "verify_persona_renders", lambda *a, **k: None)
        return [
            problem
            for problem, _remedy in provision.web_terminal_preflight_problems(
                config, repo_root=root
            )
        ]

    @staticmethod
    def _advisories(config: dict[str, Any], root: Path) -> list[str]:
        from osprey.deployment.web_terminals import provision

        return provision.web_terminal_preflight_advisories(config, repo_root=root)

    def test_a_hand_edited_login_false_refuses_the_start(self, tmp_path: Path, monkeypatch):
        """THE repro: `profile.yml` untouched, one key flipped in the render."""
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        problems = self._problems(config, tmp_path, monkeypatch)
        assert any("'carol'" in problem and "without a login" in problem for problem in problems)

    def test_a_pre_floor_render_refuses_the_start_and_says_to_rebuild(
        self, tmp_path: Path, monkeypatch
    ):
        """A render that predates the base tier's deny floor: the deploy config
        itself floors neither surface, so every persona holds both. The remedy
        an operator can act on here is a rebuild, and the message says so."""
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        del config["claude_code"]
        del config["web"]
        problems = self._problems(config, tmp_path, monkeypatch)
        refusal = next(problem for problem in problems if "'carol'" in problem)
        assert "floors neither surface" in refusal
        assert "osprey build" in refusal

    def test_an_unreadable_persona_refuses_the_start_too(self, tmp_path: Path, monkeypatch):
        """The other open-door code. A persona whose project was never rendered
        cannot be shown to hold nothing, and this one is served to anyone."""
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        _unrender(tmp_path, "ca-admin")
        problems = self._problems(config, tmp_path, monkeypatch)
        assert any("could not be read" in problem and "'carol'" in problem for problem in problems)

    def test_registry_mode_refuses_the_open_terminal_and_offers_the_login(
        self, tmp_path: Path, monkeypatch
    ):
        """The DEFAULT image mode, which every other fixture here opts out of.

        A registry-mode host has no rendered persona project to read, so a
        `login: false` entry's persona is unreadable — and the gate is
        fail-CLOSED on purpose: "cannot tell" is not "harmless" on the one path
        where the answer becomes a running container. The refusal has to stay
        escapable, though, and `osprey build` is not the escape here (the images
        and their deltas are built in CI, which is registry mode's premise), so
        the message offers the login too.
        """
        config = _registry_repo(tmp_path, login=False)
        problems = self._problems(config, tmp_path, monkeypatch)
        refusal = next(
            problem
            for problem in problems
            if "could not be read" in problem and "'admin'" in problem
        )
        assert "'carol'" in refusal
        assert "or set login: true for 'carol'" in refusal
        assert "[" not in refusal

    def test_registry_mode_behind_a_login_starts(self, tmp_path: Path, monkeypatch):
        """The counterfactual, and the one that matters for the default mode: a
        pull-only host holds no persona render at all, so with every login on
        NOTHING here refuses the start.

        Not even the `default_persona`, which is unreadable on this host like
        every other persona is: in registry mode that is the normal state rather
        than drift, and there is no remedy an operator on a pull-only host can
        carry out. A gate that fired here would refuse every registry-mode
        deployment in the shipped default configuration."""
        config = _registry_repo(tmp_path, login=None)
        assert self._problems(config, tmp_path, monkeypatch) == []

    def test_a_floored_render_starts(self, tmp_path: Path, monkeypatch):
        """The negative control: the shape a clean `osprey build` produces
        passes preflight, so the gate cannot be green by refusing everything."""
        config = _rendered_repo(tmp_path, admin_config=FLOORED_RENDER, login=False)
        assert self._problems(config, tmp_path, monkeypatch) == []

    def test_a_privileged_persona_behind_a_login_starts(self, tmp_path: Path, monkeypatch):
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        assert self._problems(config, tmp_path, monkeypatch) == []

    def test_the_other_lint_errors_do_not_refuse_a_start(self, tmp_path: Path, monkeypatch):
        """The scope, pinned. This config carries a real lint ERROR of a
        different kind; the authoring verbs refuse it and `up` does not."""
        config = _rendered_repo(tmp_path, admin_config=FLOORED_RENDER, login=None)
        config["modules"]["web_terminals"]["personas"]["admin"]["extra_mounts"] = ["not a mount"]
        findings = lint_web_terminals(config, project_root=tmp_path)
        assert "web_terminals.persona_invalid_extra_mount" in _codes(findings)

        assert self._problems(config, tmp_path, monkeypatch) == []

    def test_a_privileged_default_persona_is_said_but_does_not_block(
        self, tmp_path: Path, monkeypatch
    ):
        """An authoring mistake, not an open door: the entries that inherit it
        are still behind the wall, and any that are not are named by the rule
        that does block. Refusing the start of a running stack over roster shape
        would stop a shift to fix a profile — so it is printed instead."""
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        config["modules"]["web_terminals"]["default_persona"] = "admin"

        assert self._problems(config, tmp_path, monkeypatch) == []
        assert any("default_persona" in message for message in self._advisories(config, tmp_path))

    def test_open_mode_is_said_but_does_not_block_a_contained_deployment(
        self, tmp_path: Path, monkeypatch
    ):
        """`auth.method: none` (open) is a legitimate loopback posture, and the
        privilege belt still only SAYS so: what a privileged terminal behind no wall
        costs is an advisory, not a refusal, or a start would be refused over a
        deployment nobody exposed.

        What does refuse an open start is the separate egress gate, and only when a
        persona can actually reach the deployment's own terminals — which is not the
        case here, where both rendered projects ship the shipped deny list."""
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        config["modules"]["web_terminals"]["auth"] = {"method": "none"}
        _ship_persona_settings(tmp_path, "ca-admin", "ca-readonly")

        assert self._problems(config, tmp_path, monkeypatch) == []
        assert any("auth.method" in message for message in self._advisories(config, tmp_path))

    def test_open_mode_blocks_when_a_persona_may_reach_the_host_network(
        self, tmp_path: Path, monkeypatch
    ):
        """The other half of the same posture, and the promise this gate reversed:
        open mode HAS a refusal. Nginx vouches for every terminal it proxies, so a
        persona whose shipped settings lift the shell is one prompt away from a
        neighbour's session — and that start is refused rather than advised."""
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        config["modules"]["web_terminals"]["auth"] = {"method": "none"}
        _ship_persona_settings(tmp_path, "ca-readonly")
        _ship_persona_settings(tmp_path, "ca-admin", lifted=("Bash",))

        problems = self._problems(config, tmp_path, monkeypatch)

        assert any("may still reach the host network" in problem for problem in problems)
        assert any("'admin'" in problem for problem in problems)

    def test_a_warn_with_a_blocking_code_is_printed_not_refused(self, tmp_path: Path, monkeypatch):
        """`persona_privileges_unknown` is in BOTH filter sets, because the same
        rule blocks when an entry opted out of a login and only advises when no
        wall stands at all. The two filters select on severity AND code, never
        on code alone — so this deployment starts, and says why.

        A filter that matched on code alone would refuse this start over a
        posture the design deliberately does not block on."""
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=None)
        # `token` rather than `none`: this test is about the lint filter sets, and
        # both postures stand no wall (`auth_is_enforced` reads `walled`), so they
        # select the same finding. Under `none` the deliberately unrendered project
        # would additionally trip the open-mode egress gate, which fails closed on a
        # settings artifact it cannot read — a real refusal, but a different one.
        config["modules"]["web_terminals"]["auth"] = {"method": "token"}
        _unrender(tmp_path, "ca-admin")

        assert self._problems(config, tmp_path, monkeypatch) == []
        assert any(
            "could not be read" in message and "'carol'" in message
            for message in self._advisories(config, tmp_path)
        )

    def test_a_clean_deployment_draws_no_advisory(self, tmp_path: Path):
        config = _rendered_repo(tmp_path, admin_config=FLOORED_RENDER, login=None)
        assert self._advisories(config, tmp_path) == []

    def test_the_gate_reads_the_repo_it_is_given_not_the_cwd(self, tmp_path: Path, monkeypatch):
        """`osprey up --repo <path>` from anywhere reads the same renders. A
        gate anchored on the working directory would pass every deploy run from
        outside the repo — which is the seam this belt already paid for once."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        monkeypatch.chdir(elsewhere)

        problems = self._problems(config, tmp_path, monkeypatch)

        assert any("'carol'" in problem for problem in problems)

    def test_the_gate_writes_nothing(self, tmp_path: Path, monkeypatch):
        """The pass this joins is ordered against provisioners that DO write, so
        a probe with a side effect would silently become a step in that order."""
        config = _rendered_repo(tmp_path, admin_config=PRIVILEGED_RENDER, login=False)
        before = sorted(path.name for path in tmp_path.iterdir())

        self._problems(config, tmp_path, monkeypatch)
        self._advisories(config, tmp_path)

        assert sorted(path.name for path in tmp_path.iterdir()) == before
