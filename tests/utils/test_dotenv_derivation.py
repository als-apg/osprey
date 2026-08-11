"""Tests for ``derive_project_env`` — the profile → project ``.env`` derivation.

The project ``.env`` is derived, never authored: the profile ``.env`` is the
source of truth for facility secrets, the render supplies structure and
build-computed values, and the existing project file contributes only what a
runtime writer (a scenario, a deploy mint, the bluesky substrate wiring) put
there. These tests pin the per-key classes and the two modes:

* **build** — full semantics; a key in none of the three classes is dropped, so
  a rebuilt project carries no facility state the profile cannot regenerate.
* **deploy** — overlay only; class-2 values update, missing keys append, and
  nothing is ever deleted.

The class-3 enumeration is duplicated in ``osprey.utils.dotenv`` (that module
sits at the bottom of the import graph), so a drift guard here cross-checks it
against the definitions that actually own those names.
"""

from __future__ import annotations

import logging

import pytest

from osprey.utils.dotenv import (
    BUILD_DERIVED_KEYS,
    RUNTIME_WRITER_KEYS,
    derive_project_env,
    parse_dotenv_file,
)


def values(text: str, tmp_path) -> dict[str, str]:
    """Parse derived ``.env`` text the way the build lifecycle reads it."""
    path = tmp_path / ".env.derived"
    path.write_text(text, encoding="utf-8")
    return parse_dotenv_file(path)


class TestBuildModeClasses:
    """Build mode — the three key classes and their precedence."""

    def test_build_derived_key_takes_rendered_value(self, tmp_path):
        derived = derive_project_env(
            "VA_LATTICE=from-profile\n",
            "VA_LATTICE=from-render\n",
            "VA_LATTICE=from-project\n",
            "build",
        )
        assert values(derived, tmp_path)["VA_LATTICE"] == "from-render"

    def test_build_derived_key_unwritten_when_render_drops_it(self, tmp_path):
        derived = derive_project_env(
            "VA_LATTICE=from-profile\n",
            "OTHER=x\n",
            "VA_LATTICE=stale\nVA_CHANNELS_FILE=stale.yml\n",
            "build",
        )
        parsed = values(derived, tmp_path)
        assert "VA_LATTICE" not in parsed
        assert "VA_CHANNELS_FILE" not in parsed

    def test_profile_value_wins_over_render(self, tmp_path):
        derived = derive_project_env(
            "ANTHROPIC_API_KEY=sk-profile\n",
            "ANTHROPIC_API_KEY=\n",
            "",
            "build",
        )
        assert values(derived, tmp_path)["ANTHROPIC_API_KEY"] == "sk-profile"

    def test_profile_value_wins_over_hand_edited_project(self, tmp_path):
        derived = derive_project_env(
            "FACILITY_URL=https://profile.example\n",
            "FACILITY_URL=\n",
            "FACILITY_URL=https://hand-edited.example\n",
            "build",
        )
        assert values(derived, tmp_path)["FACILITY_URL"] == "https://profile.example"

    def test_runtime_writer_key_preserved_over_profile_and_render(self, tmp_path):
        derived = derive_project_env(
            "ARIEL_DB_PASSWORD=profile-pw\n",
            "ARIEL_DB_PASSWORD=render-pw\n",
            "ARIEL_DB_PASSWORD=volume-pinned-pw\n",
            "build",
        )
        assert values(derived, tmp_path)["ARIEL_DB_PASSWORD"] == "volume-pinned-pw"

    def test_runtime_writer_key_falls_back_to_profile_when_project_lacks_it(self, tmp_path):
        derived = derive_project_env(
            "ARIEL_DB_PASSWORD=profile-pw\n",
            "ARIEL_DB_PASSWORD=render-pw\n",
            "",
            "build",
        )
        assert values(derived, tmp_path)["ARIEL_DB_PASSWORD"] == "profile-pw"

    def test_runtime_writer_key_preserved_even_when_the_render_omits_it(self, tmp_path):
        """Class 2 does not depend on the render mentioning the key.

        The profile-carried append and the runtime-preserved append both want
        this key. The project's value has to win either way: these are
        volume-pinned credentials, so taking the profile's copy would hand the
        rebuilt project a password the running containers do not accept.
        """
        derived = derive_project_env(
            "ZO_ROOT_USER_PASSWORD=profile-pw\n",
            "# the render emits no keys at all\n",
            "ZO_ROOT_USER_PASSWORD=volume-pinned-pw\n",
            "build",
        )
        parsed = values(derived, tmp_path)
        assert parsed["ZO_ROOT_USER_PASSWORD"] == "volume-pinned-pw"
        # And exactly once: the two append sections must not both emit it.
        assert derived.count("ZO_ROOT_USER_PASSWORD") == 1

    def test_an_empty_project_value_still_outranks_the_profile(self, tmp_path):
        """Class-2 ownership is decided by PRESENCE, and that is deliberate.

        An empty service token is not a blank waiting to be filled — it means
        "no token", which ``_ensure_service_tokens`` explicitly declines to
        mint over ("generating would silently override a deliberate value")
        and which makes the service fail closed; for an exposed deploy it
        refuses to bind rather than fail open. If the profile's value won here,
        a rebuild would re-arm a service the operator disarmed on purpose.
        """
        derived = derive_project_env(
            "BLUESKY_LAUNCH_TOKEN=profile-token\n",
            "# the render emits no keys at all\n",
            "BLUESKY_LAUNCH_TOKEN=\n",
            "build",
        )
        assert values(derived, tmp_path)["BLUESKY_LAUNCH_TOKEN"] == ""
        assert "profile-token" not in derived

    def test_project_owned_class2_key_does_not_suppress_other_carried_keys(self, tmp_path):
        """Leaving a class-2 key out of the carried section drops only that key."""
        derived = derive_project_env(
            "ZO_ROOT_USER_PASSWORD=profile-pw\nSITE_TOKEN=abc\n",
            "# no keys\n",
            "ZO_ROOT_USER_PASSWORD=volume-pinned-pw\n",
            "build",
        )
        parsed = values(derived, tmp_path)
        assert parsed["SITE_TOKEN"] == "abc"
        assert parsed["ZO_ROOT_USER_PASSWORD"] == "volume-pinned-pw"

    def test_degraded_mint_present_only_in_project_survives(self, tmp_path):
        """A deploy that could not write back to the profile minted into the
        project; the rebuild must not throw that token away."""
        derived = derive_project_env(
            "",
            "OTHER=x\n",
            "EVENT_DISPATCHER_TOKEN=minted-locally\nVA_BPM_ERRORS=BPM1:gain_x=1.2\n",
            "build",
        )
        parsed = values(derived, tmp_path)
        assert parsed["EVENT_DISPATCHER_TOKEN"] == "minted-locally"
        assert parsed["VA_BPM_ERRORS"] == "BPM1:gain_x=1.2"

    def test_unknown_project_only_key_dropped(self, tmp_path):
        derived = derive_project_env(
            "",
            "OTHER=x\n",
            "HAND_ADDED=surprise\n",
            "build",
        )
        assert "HAND_ADDED" not in values(derived, tmp_path)

    def test_profile_only_key_appended(self, tmp_path):
        derived = derive_project_env(
            "SITE_TOKEN=abc\n",
            "OTHER=x\n",
            "",
            "build",
        )
        parsed = values(derived, tmp_path)
        assert parsed["SITE_TOKEN"] == "abc"
        assert parsed["OTHER"] == "x"

    def test_build_derived_key_in_profile_never_latched_on(self, tmp_path):
        """Class 1 beats class 2: a profile carrying a build-derived key must
        not resurrect it once the render stops emitting it."""
        derived = derive_project_env(
            "VA_CHANNELS_FILE=/profile/channels.yml\n",
            "OTHER=x\n",
            "",
            "build",
        )
        assert "VA_CHANNELS_FILE" not in values(derived, tmp_path)


class TestBuildModeStructure:
    """Build mode — the render supplies the file's shape."""

    def test_render_comments_and_order_preserved(self):
        rendered = "# Provider keys\nANTHROPIC_API_KEY=\n\n# Facility\nFACILITY=als\n"
        derived = derive_project_env("ANTHROPIC_API_KEY=sk-1\n", rendered, "", "build")
        assert derived.splitlines()[:5] == [
            "# Provider keys",
            "ANTHROPIC_API_KEY=sk-1",
            "",
            "# Facility",
            "FACILITY=als",
        ]

    def test_appended_sections_are_bannered(self):
        derived = derive_project_env(
            "SITE_TOKEN=abc\n",
            "OTHER=x\n",
            "ARIEL_DB_PASSWORD=pinned\n",
            "build",
        )
        assert "# ── Carried from the profile .env ──" in derived
        assert "# ── Preserved from the project .env (runtime-written) ──" in derived
        lines = derived.splitlines()
        assert lines.index("SITE_TOKEN=abc") > lines.index("# ── Carried from the profile .env ──")
        assert lines.index("ARIEL_DB_PASSWORD=pinned") > lines.index(
            "# ── Preserved from the project .env (runtime-written) ──"
        )

    def test_no_banner_when_nothing_to_append(self):
        derived = derive_project_env("OTHER=y\n", "OTHER=x\n", "", "build")
        assert derived == "OTHER=y\n"

    def test_quoting_preserved_verbatim(self, tmp_path):
        derived = derive_project_env(
            'PASSPHRASE="a b # c"\n',
            "PASSPHRASE=\n",
            "",
            "build",
        )
        assert 'PASSPHRASE="a b # c"' in derived
        assert values(derived, tmp_path)["PASSPHRASE"] == "a b # c"

    def test_export_prefixed_line_matched_by_key(self, tmp_path):
        derived = derive_project_env(
            "SITE_TOKEN=from-profile\n",
            "export SITE_TOKEN=placeholder\n",
            "",
            "build",
        )
        parsed = values(derived, tmp_path)
        assert parsed["SITE_TOKEN"] == "from-profile"
        assert derived.count("SITE_TOKEN") == 1

    def test_empty_inputs_yield_empty_file(self):
        assert derive_project_env("", "", "", "build") == "\n"


class TestDeployMode:
    """Deploy mode — overlay only; never delete."""

    def test_unknown_project_key_kept(self, tmp_path):
        derived = derive_project_env("", "", "HAND_ADDED=surprise\n", "deploy")
        assert values(derived, tmp_path)["HAND_ADDED"] == "surprise"

    def test_profile_value_updates_in_place(self, tmp_path):
        derived = derive_project_env(
            "ANTHROPIC_API_KEY=sk-new\n",
            "",
            "# keys\nANTHROPIC_API_KEY=sk-old\nOTHER=x\n",
            "deploy",
        )
        assert derived.splitlines()[:3] == [
            "# keys",
            "ANTHROPIC_API_KEY=sk-new",  # gitleaks:allow - fixture value, not a secret
            "OTHER=x",
        ]
        assert values(derived, tmp_path)["OTHER"] == "x"

    def test_runtime_writer_key_not_overwritten_by_profile(self, tmp_path):
        derived = derive_project_env(
            "ARIEL_DB_PASSWORD=profile-pw\n",
            "",
            "ARIEL_DB_PASSWORD=volume-pinned-pw\n",
            "deploy",
        )
        assert values(derived, tmp_path)["ARIEL_DB_PASSWORD"] == "volume-pinned-pw"

    def test_missing_profile_key_appended(self, tmp_path):
        derived = derive_project_env("SITE_TOKEN=abc\n", "", "OTHER=x\n", "deploy")
        parsed = values(derived, tmp_path)
        assert parsed == {"OTHER": "x", "SITE_TOKEN": "abc"}

    def test_build_derived_key_not_unwritten(self, tmp_path):
        """The render is not authoritative during a deploy: an empty render
        must not prune the build's own keys."""
        derived = derive_project_env("", "", "VA_LATTICE=built.lat\n", "deploy")
        assert values(derived, tmp_path)["VA_LATTICE"] == "built.lat"

    def test_rendered_key_appended_when_absent(self, tmp_path):
        derived = derive_project_env("", "NEW_VAR=v\n", "OTHER=x\n", "deploy")
        assert values(derived, tmp_path)["NEW_VAR"] == "v"

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown derivation mode"):
            derive_project_env("", "", "", "overlay")


class TestClass2DivergenceWarning:
    """A class-2 key the profile and project disagree on is reported by name.

    The project's value is kept, which is right but silent: the operator's real
    problem is that the profile can no longer reproduce this project's secrets,
    and nothing else in the build says so.
    """

    def _derive(self, profile: str, existing: str, mode: str = "build") -> str:
        return derive_project_env(profile, "# the render emits no keys\n", existing, mode)

    def test_divergence_names_the_variable(self, caplog):
        with caplog.at_level(logging.WARNING, logger="build"):
            self._derive("ARIEL_DB_PASSWORD=profile-pw\n", "ARIEL_DB_PASSWORD=volume-pw\n")
        assert "ARIEL_DB_PASSWORD" in caplog.text

    def test_divergence_never_logs_either_value(self, caplog):
        """The variable is the information; both values are secrets."""
        with caplog.at_level(logging.WARNING, logger="build"):
            self._derive("ARIEL_DB_PASSWORD=profile-pw\n", "ARIEL_DB_PASSWORD=volume-pw\n")
        assert "profile-pw" not in caplog.text
        assert "volume-pw" not in caplog.text

    def test_agreement_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger="build"):
            self._derive("ARIEL_DB_PASSWORD=same-pw\n", "ARIEL_DB_PASSWORD=same-pw\n")
        assert "ARIEL_DB_PASSWORD" not in caplog.text

    def test_quoting_alone_is_not_a_divergence(self, caplog):
        """Compared as parsed values: the two files agree on the secret."""
        with caplog.at_level(logging.WARNING, logger="build"):
            self._derive('ARIEL_DB_PASSWORD="same pw"\n', "ARIEL_DB_PASSWORD='same pw'\n")
        assert "ARIEL_DB_PASSWORD" not in caplog.text

    def test_key_absent_from_the_project_is_not_a_divergence(self, caplog):
        """Nothing to disagree with — the profile's value is simply taken."""
        with caplog.at_level(logging.WARNING, logger="build"):
            derived = self._derive("ARIEL_DB_PASSWORD=profile-pw\n", "")
        assert "ARIEL_DB_PASSWORD" not in caplog.text
        assert "ARIEL_DB_PASSWORD=profile-pw" in derived

    def test_non_class2_key_divergence_is_silent(self, caplog):
        """A profile key winning over a hand edit is the ordinary case, not a
        conflict — warning on it would train the operator to ignore the log."""
        with caplog.at_level(logging.WARNING, logger="build"):
            self._derive("FACILITY_URL=from-profile\n", "FACILITY_URL=hand-edited\n")
        assert "FACILITY_URL" not in caplog.text

    def test_every_diverging_key_is_named(self, caplog):
        """One line per variable — an operator reconciling them needs all of them."""
        with caplog.at_level(logging.WARNING, logger="build"):
            self._derive(
                "ARIEL_DB_PASSWORD=profile-a\nZO_ROOT_USER_PASSWORD=profile-b\n",
                "ARIEL_DB_PASSWORD=project-a\nZO_ROOT_USER_PASSWORD=project-b\n",
            )
        assert "ARIEL_DB_PASSWORD" in caplog.text
        assert "ZO_ROOT_USER_PASSWORD" in caplog.text

    def test_only_the_diverging_key_is_named(self, caplog):
        """An agreeing sibling must not be swept into the warning."""
        with caplog.at_level(logging.WARNING, logger="build"):
            self._derive(
                "ARIEL_DB_PASSWORD=profile-a\nZO_ROOT_USER_PASSWORD=agreed\n",
                "ARIEL_DB_PASSWORD=project-a\nZO_ROOT_USER_PASSWORD=agreed\n",
            )
        assert "ARIEL_DB_PASSWORD" in caplog.text
        assert "ZO_ROOT_USER_PASSWORD" not in caplog.text

    def test_deploy_mode_does_not_warn(self, caplog):
        """The write-back has already reported the same fact on that path."""
        with caplog.at_level(logging.WARNING, logger="build"):
            self._derive(
                "ARIEL_DB_PASSWORD=profile-pw\n", "ARIEL_DB_PASSWORD=volume-pw\n", mode="deploy"
            )
        assert "ARIEL_DB_PASSWORD" not in caplog.text


class TestKeyClassEnumerations:
    """The class-1/class-3 enumerations against their owning definitions."""

    def test_classes_are_disjoint(self):
        assert not (BUILD_DERIVED_KEYS & RUNTIME_WRITER_KEYS)

    def test_runtime_writer_keys_cover_scenario_physics_vars(self):
        from osprey.simulation.apply import _PHYSICS_ENV_VARS

        assert set(_PHYSICS_ENV_VARS) <= RUNTIME_WRITER_KEYS

    def test_runtime_writer_keys_cover_bluesky_substrate_vars(self):
        from osprey.services.bluesky_bridge.substrate_devices import (
            DETECTORS_ENV,
            MOTORS_ENV,
            SUBSTRATE_ENV,
        )

        assert {SUBSTRATE_ENV, MOTORS_ENV, DETECTORS_ENV} <= RUNTIME_WRITER_KEYS

    def test_runtime_writer_keys_cover_minted_service_tokens(self):
        from osprey.deployment.container_lifecycle import _SERVICE_TOKEN_VARS

        minted = {var for vars_ in _SERVICE_TOKEN_VARS.values() for var in vars_}
        assert minted <= RUNTIME_WRITER_KEYS

    def test_runtime_writer_keys_hold_nothing_beyond_the_three_writers(self):
        """The drift guard in the other direction: no key without an owner.

        The subset checks above catch a writer whose key was never enumerated
        here. They cannot catch the reverse — a name added to the duplicated
        enumeration that no writer actually produces, which quietly promotes an
        ordinary rendered key into one an existing project value always wins
        over. The three groups the enumeration's own comments claim are the
        whole of it, so assert exactly that.
        """
        from osprey.deployment.container_lifecycle import _SERVICE_TOKEN_VARS
        from osprey.services.bluesky_bridge.substrate_devices import (
            DETECTORS_ENV,
            MOTORS_ENV,
            SUBSTRATE_ENV,
        )
        from osprey.simulation.apply import _PHYSICS_ENV_VARS

        owned = (
            set(_PHYSICS_ENV_VARS)
            | {SUBSTRATE_ENV, MOTORS_ENV, DETECTORS_ENV}
            | {var for vars_ in _SERVICE_TOKEN_VARS.values() for var in vars_}
        )
        assert RUNTIME_WRITER_KEYS == owned

    def test_the_minted_archiver_password_is_runtime_written(self):
        """MONGO_ROOT_PASSWORD is pinned by the volume that adopted it.

        Named explicitly rather than left to the set check above because what
        the archiver's Mongo volume holds is not reproducible at all: a rebuild
        that re-derived this key from the profile would mint a value the
        existing volume rejects, and recovering means discarding every seeded
        and recorded sample in it.
        """
        assert "MONGO_ROOT_PASSWORD" in RUNTIME_WRITER_KEYS
