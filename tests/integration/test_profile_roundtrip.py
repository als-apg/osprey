"""The profile-is-the-source-of-truth roundtrip, driven through the real CLI.

This is the executable contract for the feature's cross-cutting requirement:

    After ``profile new`` -> ``build`` -> edit the profile -> ``build --force``,
    the project reflects EVERY edit and holds no facility-authored state the
    profile cannot regenerate.

Everything else here exists to make that sentence checkable end to end, so the
suite is organized as one scripted roundtrip plus the semantics that hang off
it:

1. **The roundtrip** (the :func:`roundtrip` fixture and the two classes that
   read it) — materialize a profile, build it, then edit one artifact in *every*
   convention category plus ``triggers.yml``, a persona delta, and the profile
   ``.env``; rebuild with ``--force`` and assert each edit landed. Then delete
   the project outright and rebuild: what comes back is the measure of what the
   profile can regenerate.
2. **FR-9 env classes** — build-derived keys un-write, runtime-written keys
   survive, profile keys win, and a hand edit the profile cannot account for is
   dropped.
3. **The persona stack** — a ``personas/<name>.yml`` delta anchored at its root,
   its exclusions (list artifact, convention artifact, and a convention artifact
   shadowing a framework render, which the exclusion restores), its inherited
   secrets, root-edit staleness, and the ways a persona reference fails.
4. **Standalone builds** — a bare temp-file profile is NOT a persona delta and
   materializes no profile directory beside its project.
5. **Preset provenance** — a moved-on preset advises; a *different* preset on
   reuse refuses.

Cost control: ``osprey build`` is the slow step, so the scripted roundtrip runs
once per module (``--skip-deps --skip-lifecycle``, which keeps it network-free)
and the read-only assertions share it. Tests that mutate take a copy.

Environment: ``osprey profile new`` seeds the profile ``.env`` from the shell's
exported provider keys, so every invocation here runs under
:func:`_sanitized_provider_env` — the developer's real keys must not reach a
fixture, and the sentinel value below is what the assertions track through the
pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.build_cmd import build
from osprey.cli.profile_cmd import profile as profile_group
from osprey.models.provider_registry import PROVIDER_API_KEYS
from osprey.utils.dotenv import RUNTIME_WRITER_KEYS, parse_dotenv_file, parse_dotenv_text

# The provider key value the whole suite tracks. `hello-world` selects the
# anthropic provider, so this is the one key `profile new` seeds.
PROVIDER_VAR = "ANTHROPIC_API_KEY"
PROFILE_KEY_VALUE = "sk-profile-owned-value"

# A secret the operator adds to the profile `.env` by hand — the plain case of
# "a value set in the profile reaches every project built from it".
ARCHIVER_VAR = "FACILITY_ARCHIVER_URL"
ARCHIVER_VALUE = "http://archiver.facility.example:17668"

PRESET = "hello-world"
PROJECT_NAME = "facility"

#: The facility repo `osprey profile new` creates, and the profile nested inside
#: it. The command is given the repo; everything the roundtrip asserts on lives
#: in the profile directory it writes there.
REPO_DIRNAME = "facility-repo"
PROFILE_RELPATH = f"{REPO_DIRNAME}/profile"

# The roster user whose per-user web-terminal context the roundtrip exercises.
# `hello-world` ships no web_terminals block, so the profile edit turns one on:
# the per-user convention category is roster-derived, and with an empty roster
# the build skips it entirely (which is the persona case, asserted separately).
ROSTER_USER = "opsuser"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@contextmanager
def _sanitized_provider_env() -> Iterator[None]:
    """Run with every provider API key cleared and ours exported.

    ``osprey profile new`` seeds the profile ``.env`` from ``os.environ``
    (FR-1), and the CLI bulk-loads ``.env`` files back into ``os.environ``
    elsewhere, so an unsanitized run would both leak the developer's real keys
    into a fixture and make the seeded set depend on whose machine ran the
    suite. ``monkeypatch`` is not used because the module-scoped fixture below
    is set up before pytest's function-scoped environment guards exist.
    """
    saved = dict(os.environ)
    try:
        for var in PROVIDER_API_KEYS.values():
            if var:
                os.environ.pop(var, None)
        os.environ[PROVIDER_VAR] = PROFILE_KEY_VALUE
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _invoke(cli: click.Command, args: list[str]) -> Result:
    """Invoke a CLI command in-process under a sanitized environment."""
    with _sanitized_provider_env():
        return CliRunner().invoke(cli, args)


def _assert_ok(result: Result, what: str) -> None:
    if result.exit_code != 0:
        raise AssertionError(f"{what} failed (exit {result.exit_code}):\n{result.output}")


def _profile_new(target: Path, *extra: str) -> Result:
    """Create the facility repo holding *target*, the profile directory."""
    return _invoke(profile_group, ["new", str(target.parent), "--preset", PRESET, *extra])


def _build(project_name: str, profile_path: Path, output_dir: Path, *extra: str) -> Result:
    return _invoke(
        build,
        [
            project_name,
            str(profile_path),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(output_dir),
            *extra,
        ],
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _profile_yaml(profile_path: Path) -> dict:
    return yaml.safe_load(profile_path.read_text(encoding="utf-8"))


def _config_yaml(project_dir: Path) -> dict:
    return yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))


def _user_owned(project_dir: Path) -> set[str]:
    scaffold = _config_yaml(project_dir).get("scaffold") or {}
    return set(scaffold.get("user_owned") or [])


# ---------------------------------------------------------------------------
# The profile edits — one artifact per convention category
# ---------------------------------------------------------------------------

# ``rules/safety.md`` deliberately collides with a framework-selected rule
# (``hello-world`` lists ``safety``): a profile file must WIN over the framework
# render, and a persona excluding it must get the framework version back.
SHADOWED_RULE = "rules/safety"
SHADOWED_RULE_BODY = "# Facility safety rule\n\nThis text is the profile's, not the framework's.\n"

#: ``(profile-relative source, project-relative destination, body)`` for one
#: artifact in every convention category. Directory-shaped categories name a
#: file inside the directory that is copied as a unit.
CONVENTION_EDITS: tuple[tuple[str, str, str], ...] = (
    ("rules/safety.md", ".claude/rules/safety.md", SHADOWED_RULE_BODY),
    ("rules/facility-ops.md", ".claude/rules/facility-ops.md", "# Ops rule\n"),
    ("skills/orbit-check/SKILL.md", ".claude/skills/orbit-check/SKILL.md", "# Orbit check\n"),
    ("agents/orbit-writer.md", ".claude/agents/orbit-writer.md", "# Orbit writer agent\n"),
    ("commands/osprey/scan.md", ".claude/commands/osprey/scan.md", "# Scan command\n"),
    ("output-styles/terse.md", ".claude/output-styles/terse.md", "# Terse style\n"),
    ("hooks/facility_probe.py", ".claude/hooks/facility_probe.py", "#!/usr/bin/env python3\n"),
    (
        f"web-terminal-context/{ROSTER_USER}/context.md",
        f"docker/web-terminal-context/{ROSTER_USER}/context.md",
        "# Ops user context\n",
    ),
    ("mcp_servers/facility_rpc/server.py", "_mcp_servers/facility_rpc/server.py", "# rpc server\n"),
    (
        "services/facility_gw/docker-compose.yml",
        "services/facility_gw/docker-compose.yml",
        "services: {}\n",
    ),
    ("project/docs/runbook.md", "docs/runbook.md", "# Facility runbook\n"),
)

#: ``scaffold.user_owned`` names the edits above must produce. Ownership follows
#: the DESTINATION, so the three landing outside ``.claude/`` and outside
#: ``services/`` (per-user context, the MCP server, the mirrored doc) are not
#: ownable and are absent by design.
EXPECTED_OWNED = {
    "rules/safety",
    "rules/facility-ops",
    "skills/orbit-check",
    "agents/orbit-writer",
    "commands/osprey/scan",
    "output-styles/terse",
    "hooks/facility_probe.py",
    "services/facility_gw",
}

PERSONA_NAME = "reader"

PERSONA_DELTA = """\
name: Facility Reader
exclude:
  rules:
    # Qualified: a convention artifact of the profile's own, omitted from this
    # persona.
    - rules/facility-ops
    # Qualified, and the artifact SHADOWS a framework-selected rule. Omitting
    # the profile's copy restores the framework render — the point of excluding
    # a shadow is to get the built-in version back.
    - rules/safety
    # Bare: unselects the built-in library artifact of that name outright.
    - timezone
config:
  modules.web_terminals.enabled: false
"""

TRIGGERS_BODY = """\
dispatcher:
  max_concurrent_runs: 1
triggers:
  facility_alarm:
    type: webhook
    prompt: Investigate the facility alarm.
"""

# FR-9 fixtures, written into the PROJECT `.env` before the `--force` rebuild —
# the file `_clear_rendered_project_dir` preserves and the derivation reads as
# "what a runtime writer left here".
STALE_BUILD_DERIVED = ("VA_CHANNELS_FILE", "/gone/channel_manifest.json")
SIM_FAULT = ("VA_BPM_ERRORS", "0.35")
DEGRADED_MINT = ("EVENT_DISPATCHER_TOKEN", "minted-into-the-project-only")
HAND_EDIT = ("FACILITY_HAND_EDIT", "not-in-the-profile")


def _apply_profile_edits(profile_dir: Path) -> None:
    """Make every edit the roundtrip's contract is stated over.

    Convention artifacts, ``triggers.yml`` + the ``dispatch:`` key that names
    it, a persona delta, a roster so the per-user context category has
    something to derive from, and two ``.env`` values: the seeded provider key
    (whose value must beat a later hand edit in the project) and a new facility
    secret.
    """
    for source_rel, _dest_rel, body in CONVENTION_EDITS:
        _write(profile_dir / source_rel, body)

    _write(profile_dir / "triggers.yml", TRIGGERS_BODY)
    _write(profile_dir / "personas" / f"{PERSONA_NAME}.yml", PERSONA_DELTA)

    raw = _profile_yaml(profile_dir / "profile.yml")
    raw["dispatch"] = {"triggers": "triggers.yml"}
    config = raw.setdefault("config", {})
    config["modules.web_terminals.enabled"] = True
    config["modules.web_terminals.users"] = [ROSTER_USER]
    # The two values a roster cannot be deployed without, and which `osprey
    # build` therefore refuses a profile for: the container-name prefix
    # (`<prefix>-web-<user>`) and the per-user web port family's base, which has
    # no registry default because it is facility-chosen.
    config["facility.prefix"] = "fac"
    config["modules.web_terminals.web_base_port"] = 9091
    (profile_dir / "profile.yml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    # Written as an operator would: the profile `.env` is a file they own.
    (profile_dir / ".env").write_text(
        f"{PROVIDER_VAR}={PROFILE_KEY_VALUE}\n{ARCHIVER_VAR}={ARCHIVER_VALUE}\n",
        encoding="utf-8",
    )


def _seed_project_runtime_env(project_dir: Path) -> None:
    """Append the FR-9 fixture keys to the built project's own ``.env``."""
    env_path = project_dir / ".env"
    extra = "\n".join(
        f"{key}={value}"
        for key, value in (STALE_BUILD_DERIVED, SIM_FAULT, DEGRADED_MINT, HAND_EDIT)
    )
    # The provider key gets a divergent value here so the rebuild has to choose
    # between the project's copy and the profile's.
    extra += f"\n{PROVIDER_VAR}=hand-edited-in-the-project\n"
    env_path.write_text(env_path.read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8")


# ---------------------------------------------------------------------------
# The scripted roundtrip
# ---------------------------------------------------------------------------


@dataclass
class Roundtrip:
    """One completed ``new -> build -> edit -> build --force`` cycle."""

    base: Path
    profile_dir: Path
    project_dir: Path
    #: ``.claude/rules/safety.md`` as the framework rendered it, captured from
    #: the FIRST build — the text a persona excluding the profile's shadow of
    #: that rule must get back.
    framework_safety_rule: str = ""
    #: Rendered-project entries present before any profile edit.
    baseline_rules: set[str] = field(default_factory=set)

    @property
    def profile_path(self) -> Path:
        return self.profile_dir / "profile.yml"

    def project_env(self) -> dict[str, str]:
        return parse_dotenv_file(self.project_dir / ".env")


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory: pytest.TempPathFactory) -> Roundtrip:
    """Run the whole scripted roundtrip once; read-only tests share it."""
    base = tmp_path_factory.mktemp("roundtrip")
    profile_dir = base / PROFILE_RELPATH
    project_dir = base / PROJECT_NAME

    _assert_ok(_profile_new(profile_dir), "osprey profile new")

    result = _build(PROJECT_NAME, profile_dir / "profile.yml", base)
    _assert_ok(result, "first osprey build")

    trip = Roundtrip(base=base, profile_dir=profile_dir, project_dir=project_dir)
    trip.framework_safety_rule = (project_dir / ".claude" / "rules" / "safety.md").read_text(
        encoding="utf-8"
    )
    trip.baseline_rules = {p.name for p in (project_dir / ".claude" / "rules").iterdir()}

    _apply_profile_edits(profile_dir)
    _seed_project_runtime_env(project_dir)

    _assert_ok(_build(PROJECT_NAME, profile_dir / "profile.yml", base, "--force"), "build --force")
    return trip


@pytest.fixture
def workspace(roundtrip: Roundtrip, tmp_path: Path) -> Roundtrip:
    """A private copy of the completed roundtrip, for tests that mutate."""
    base = tmp_path / "copy"
    shutil.copytree(roundtrip.base, base, symlinks=True)
    return Roundtrip(
        base=base,
        profile_dir=base / PROFILE_RELPATH,
        project_dir=base / PROJECT_NAME,
        framework_safety_rule=roundtrip.framework_safety_rule,
        baseline_rules=set(roundtrip.baseline_rules),
    )


# ---------------------------------------------------------------------------
# 1. The roundtrip: every edit lands
# ---------------------------------------------------------------------------


class TestEveryEditReachesTheProject:
    """Contract point 1a — the profile edits and nothing but the profile."""

    @pytest.mark.parametrize(
        ("source_rel", "dest_rel", "body"),
        CONVENTION_EDITS,
        ids=[edit[0] for edit in CONVENTION_EDITS],
    )
    def test_convention_artifact_lands_at_its_destination(
        self, roundtrip: Roundtrip, source_rel: str, dest_rel: str, body: str
    ) -> None:
        landed = roundtrip.project_dir / dest_rel
        assert landed.is_file(), f"{source_rel} did not reach {dest_rel}"
        assert landed.read_text(encoding="utf-8") == body

    def test_profile_file_wins_over_the_framework_render_it_shadows(
        self, roundtrip: Roundtrip
    ) -> None:
        """``rules/safety.md`` collides with a framework-selected rule."""
        rendered = (roundtrip.project_dir / ".claude" / "rules" / "safety.md").read_text(
            encoding="utf-8"
        )
        assert rendered == SHADOWED_RULE_BODY
        assert rendered != roundtrip.framework_safety_rule

    def test_shadowing_artifacts_are_registered_as_user_owned(self, roundtrip: Roundtrip) -> None:
        """Ownership derives from the destination, uniformly across classes.

        Registration is what makes the next render skip the file and prune
        leave it alone, so it is the durable half of "the profile wins".
        """
        assert EXPECTED_OWNED <= _user_owned(roundtrip.project_dir)

    def test_categories_landing_outside_the_ownership_system_are_not_registered(
        self, roundtrip: Roundtrip
    ) -> None:
        """Landing and being *owned* are different things.

        Per-user context is roster-derived, MCP servers register through
        ``claude_code.servers``, and a mirror file outside ``.claude/`` has no
        render that could contest it — so all three arrive without entering the
        ownership system.
        """
        for path in (
            f"docker/web-terminal-context/{ROSTER_USER}/context.md",
            "_mcp_servers/facility_rpc/server.py",
            "docs/runbook.md",
        ):
            assert (roundtrip.project_dir / path).is_file(), f"{path} never landed"

        owned = _user_owned(roundtrip.project_dir)
        assert not {name for name in owned if name.startswith("web-terminal-context/")}
        assert not {name for name in owned if name.startswith("_mcp_servers/")}
        assert "docs/runbook.md" not in owned

    def test_shadowing_a_framework_artifact_is_announced(
        self, workspace: Roundtrip, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silently replacing a framework artifact is how an operator loses
        track of which version is running, so every shadow says so."""
        with caplog.at_level(logging.INFO):
            _assert_ok(
                _build(PROJECT_NAME, workspace.profile_path, workspace.base, "--force"),
                "rebuild for the shadow notice",
            )
        assert f"profile overrides framework rule '{SHADOWED_RULE}'" in caplog.text

    def test_triggers_come_from_the_profile(self, roundtrip: Roundtrip) -> None:
        """FR-3: the profile owns ``triggers.yml``; the build copies it in."""
        shipped = yaml.safe_load(
            (roundtrip.project_dir / "triggers.yml").read_text(encoding="utf-8")
        )
        assert "facility_alarm" in shipped["triggers"]

    def test_profile_env_value_reaches_the_project(self, roundtrip: Roundtrip) -> None:
        assert roundtrip.project_env()[ARCHIVER_VAR] == ARCHIVER_VALUE

    def test_force_never_touches_the_profile(self, roundtrip: Roundtrip) -> None:
        """``--force`` replaces the project; a profile is replaced only by
        ``profile new --force``."""
        for source_rel, _dest, body in CONVENTION_EDITS:
            assert (roundtrip.profile_dir / source_rel).read_text(encoding="utf-8") == body
        assert parse_dotenv_file(roundtrip.profile_dir / ".env")[ARCHIVER_VAR] == ARCHIVER_VALUE


class TestNoUnregenerableFacilityState:
    """Contract point 1b — what the profile cannot reproduce must not be there."""

    def test_project_env_holds_only_accountable_keys(self, roundtrip: Roundtrip) -> None:
        """Every project ``.env`` key traces to the profile or to a runtime writer.

        The hand edit is the control: a key that is neither is facility state
        the profile cannot regenerate, and build-mode derivation drops it. The
        render contributes no third source here — with no virtual accelerator
        configured, ``env.j2`` emits only comments.
        """
        profile_keys = set(parse_dotenv_file(roundtrip.profile_dir / ".env"))
        accountable = profile_keys | RUNTIME_WRITER_KEYS
        unaccounted = set(roundtrip.project_env()) - accountable
        assert not unaccounted, f"project .env carries unregenerable keys: {sorted(unaccounted)}"
        assert HAND_EDIT[0] not in roundtrip.project_env()

    def test_the_project_carries_one_secrets_surface(self, roundtrip: Roundtrip) -> None:
        """FR-4: ``.env`` + ``.env.example``, and no third env file."""
        env_files = {p.name for p in roundtrip.project_dir.glob(".env*")}
        assert env_files == {".env", ".env.example"}

    def test_the_project_mirror_refuses_a_build_owned_path(
        self, workspace: Roundtrip, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The mirror is the escape hatch for files the build does not own.

        Letting it write one the build DOES own would put two writers on the
        same artifact; the refusal names the channel that carries the change
        instead, which is the only way an operator finds it.
        """
        _write(workspace.profile_dir / "project" / "config.yml", "control_system: {}\n")
        with caplog.at_level(logging.ERROR):
            result = _build(PROJECT_NAME, workspace.profile_path, workspace.base, "--force")
        assert result.exit_code != 0
        assert "config.yml" in caplog.text
        assert "`config:` block" in caplog.text

    def test_deleting_the_project_loses_nothing_the_profile_owns(
        self, workspace: Roundtrip
    ) -> None:
        """The direct measure of regenerability: rebuild from nothing but the profile."""
        shutil.rmtree(workspace.project_dir)
        _assert_ok(
            _build(PROJECT_NAME, workspace.profile_path, workspace.base), "rebuild after delete"
        )

        for _source_rel, dest_rel, body in CONVENTION_EDITS:
            rebuilt = workspace.project_dir / dest_rel
            assert rebuilt.is_file(), f"{dest_rel} did not come back"
            assert rebuilt.read_text(encoding="utf-8") == body
        assert (workspace.project_dir / "triggers.yml").is_file()

        env = parse_dotenv_file(workspace.project_dir / ".env")
        assert env[ARCHIVER_VAR] == ARCHIVER_VALUE
        assert env[PROVIDER_VAR] == PROFILE_KEY_VALUE
        assert EXPECTED_OWNED <= _user_owned(workspace.project_dir)

        # The documented exception: FR-9 class-2 keys live only in the project,
        # so deleting it loses them. That is why deploy writes them back to the
        # profile, and why the degraded path warns when it cannot.
        assert SIM_FAULT[0] not in env
        assert DEGRADED_MINT[0] not in env


# ---------------------------------------------------------------------------
# 2. FR-9 env derivation classes
# ---------------------------------------------------------------------------


class TestEnvDerivationClasses:
    """Contract point 2 — each FR-9 class asserted through a real rebuild."""

    def test_profile_key_wins_over_a_hand_edit_in_the_project(self, roundtrip: Roundtrip) -> None:
        """Class 3: the profile is the source of truth for a secret's value."""
        assert roundtrip.project_env()[PROVIDER_VAR] == PROFILE_KEY_VALUE

    def test_build_derived_key_is_un_written_when_the_render_drops_it(
        self, roundtrip: Roundtrip
    ) -> None:
        """Class 1: a build that can no longer generate a VA manifest must not
        leave the project pointing at one."""
        assert STALE_BUILD_DERIVED[0] not in roundtrip.project_env()

    def test_sim_fault_written_by_a_runtime_writer_survives(self, roundtrip: Roundtrip) -> None:
        """Class 2: the active scenario's physics vars are live state."""
        assert roundtrip.project_env()[SIM_FAULT[0]] == SIM_FAULT[1]

    def test_degraded_topology_mint_present_only_in_the_project_survives(
        self, roundtrip: Roundtrip
    ) -> None:
        """Class 2, the hard half: a token the deploy could not write back to
        the profile is pinned by the containers already trusting it."""
        assert roundtrip.project_env()[DEGRADED_MINT[0]] == DEGRADED_MINT[1]

    def test_project_value_beats_a_divergent_profile_copy_of_a_class2_key(
        self, workspace: Roundtrip, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The volume-pinned case, end to end.

        A profile copied between deployments (or hand-edited) can carry a
        different value for a runtime-written credential than the project's
        containers were initialized with. The rebuild must keep the project's,
        and must say the two disagree — the profile can no longer reproduce
        this project's secrets, and nothing else in the build reports that.
        """
        env_path = workspace.profile_dir / ".env"
        env_path.write_text(
            env_path.read_text(encoding="utf-8") + "ZO_ROOT_USER_PASSWORD=from-another-stack\n",
            encoding="utf-8",
        )
        project_env = workspace.project_dir / ".env"
        project_env.write_text(
            project_env.read_text(encoding="utf-8") + "\nZO_ROOT_USER_PASSWORD=volume-pinned\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING, logger="build"):
            _assert_ok(
                _build(PROJECT_NAME, workspace.profile_path, workspace.base, "--force"),
                "rebuild with a divergent class-2 key in the profile",
            )

        assert parse_dotenv_file(project_env)["ZO_ROOT_USER_PASSWORD"] == "volume-pinned"
        assert "ZO_ROOT_USER_PASSWORD" in caplog.text
        assert "from-another-stack" not in caplog.text
        assert "volume-pinned" not in caplog.text

    def test_a_profile_copy_of_a_build_derived_key_is_ignored_too(
        self, workspace: Roundtrip
    ) -> None:
        """Class 1 beats class 3: the build owns these keys outright, so
        carrying one in the profile does not latch it on either."""
        env_path = workspace.profile_dir / ".env"
        env_path.write_text(
            env_path.read_text(encoding="utf-8") + "VA_LATTICE=/profile/lattice.mat\n",
            encoding="utf-8",
        )
        _assert_ok(
            _build(PROJECT_NAME, workspace.profile_path, workspace.base, "--force"),
            "rebuild with a build-derived key in the profile",
        )
        assert "VA_LATTICE" not in parse_dotenv_file(workspace.project_dir / ".env")


# ---------------------------------------------------------------------------
# 3. The persona stack
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def persona_project(roundtrip: Roundtrip) -> Path:
    """Build the delta the scripted edits wrote at ``personas/reader.yml``.

    Nothing is added to the profile here: the delta is one of the roundtrip's
    edits, so what this builds is the same profile every other test reads.
    """
    delta = roundtrip.profile_dir / "personas" / f"{PERSONA_NAME}.yml"
    project = f"{PROJECT_NAME}-{PERSONA_NAME}"
    _assert_ok(_build(project, delta, roundtrip.base), "persona build")
    return roundtrip.base / project


class TestPersonaStack:
    """Contract point 3 — deltas anchor at the root and subtract from it."""

    def test_delta_inherits_the_roots_convention_artifacts(self, persona_project: Path) -> None:
        assert (persona_project / ".claude" / "agents" / "orbit-writer.md").is_file()
        assert (persona_project / ".claude" / "skills" / "orbit-check" / "SKILL.md").is_file()

    def test_delta_anchors_data_at_the_profile_root(self, persona_project: Path) -> None:
        """``data:`` resolves against the root, never against ``personas/``."""
        assert (persona_project / "data" / "channel_limits.json").is_file()

    def test_persona_env_carries_the_parent_profiles_keys(self, persona_project: Path) -> None:
        env = parse_dotenv_file(persona_project / ".env")
        assert env[PROVIDER_VAR] == PROFILE_KEY_VALUE
        assert env[ARCHIVER_VAR] == ARCHIVER_VALUE

    def test_excluding_a_convention_artifact_omits_it(
        self, persona_project: Path, roundtrip: Roundtrip
    ) -> None:
        assert (roundtrip.project_dir / ".claude" / "rules" / "facility-ops.md").is_file()
        assert not (persona_project / ".claude" / "rules" / "facility-ops.md").exists()

    def test_excluding_a_list_artifact_unselects_the_builtin(
        self, persona_project: Path, roundtrip: Roundtrip
    ) -> None:
        """The bare spelling reaches the built-in library, not the profile."""
        assert "timezone.md" in roundtrip.baseline_rules, "fixture drift: nothing was excluded"
        assert not (persona_project / ".claude" / "rules" / "timezone.md").exists()

    def test_excluding_a_shadow_restores_the_framework_render(
        self, persona_project: Path, roundtrip: Roundtrip
    ) -> None:
        """The two ``exclude:`` namespaces are disjoint, and this is why it
        matters: omitting the profile's ``rules/safety.md`` must bring back the
        framework's own ``safety`` rule, not delete the rule entirely."""
        restored = persona_project / ".claude" / "rules" / "safety.md"
        assert restored.is_file(), "excluding the shadow deleted the framework rule too"
        assert restored.read_text(encoding="utf-8") == roundtrip.framework_safety_rule

    def test_persona_ownership_derives_from_the_post_exclude_set(
        self, persona_project: Path
    ) -> None:
        owned = _user_owned(persona_project)
        assert "rules/facility-ops" not in owned
        assert "rules/safety" not in owned
        assert "agents/orbit-writer" in owned

    def test_persona_build_skips_per_user_context(
        self, persona_project: Path, roundtrip: Roundtrip
    ) -> None:
        """A persona disables ``modules.web_terminals``, so it resolves an empty
        roster and the roster-derived category is skipped entirely."""
        rel = Path("docker") / "web-terminal-context" / ROSTER_USER / "context.md"
        assert (roundtrip.project_dir / rel).is_file()
        assert not (persona_project / rel).exists()

    def test_a_root_edit_makes_the_persona_project_stale(
        self, workspace: Roundtrip, tmp_path: Path
    ) -> None:
        """Persona staleness resolves the same implicit merge the build does,
        so editing the root moves every persona project's hash."""
        from osprey.deployment.staleness import staleness_reasons

        delta = workspace.profile_dir / "personas" / f"{PERSONA_NAME}.yml"
        output = workspace.base / "staleness-probe"
        _assert_ok(_build(PERSONA_NAME, delta, output), "persona build")
        persona_dir = output / PERSONA_NAME
        assert staleness_reasons(persona_dir) == []

        raw = _profile_yaml(workspace.profile_path)
        raw["config"]["control_system.type"] = "epics"
        workspace.profile_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

        reasons = staleness_reasons(persona_dir)
        assert any("has changed" in reason for reason in reasons), reasons


class TestPersonaReferenceErrors:
    """Contract point 3, error half — every unusable reference is actionable."""

    def test_extends_inside_personas_is_rejected(
        self, workspace: Roundtrip, caplog: pytest.LogCaptureFixture
    ) -> None:
        delta = _write(
            workspace.profile_dir / "personas" / "bad.yml",
            f"name: Bad\nextends: {PRESET}\n",
        )
        with caplog.at_level(logging.ERROR):
            result = _build("bad-persona", delta, workspace.base)
        assert result.exit_code != 0
        assert "extends" in caplog.text
        assert "personas/" in caplog.text

    def test_a_personas_file_without_a_root_is_rejected(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A delta cannot be built alone: it is missing everything the root supplies."""
        orphan = _write(tmp_path / "personas" / "lonely.yml", "name: Lonely\n")
        with caplog.at_level(logging.ERROR):
            result = _build("orphan", orphan, tmp_path / "out")
        assert result.exit_code != 0
        assert "profile root is missing" in caplog.text

    def test_a_catalog_entry_naming_a_missing_delta_is_rejected(self, workspace: Roundtrip) -> None:
        from osprey.deployment.web_terminals.persona_images import auto_render_missing_personas

        with pytest.raises(ValueError, match="no file exists at"):
            auto_render_missing_personas(
                _catalog_config("personas/ghost.yml", workspace.base),
                _resolved_users(),
                dict(os.environ),
                workspace.project_dir,
            )

    def test_a_catalog_entry_pointing_outside_personas_is_rejected(
        self, workspace: Roundtrip
    ) -> None:
        """A persona is a delta over THIS deployment's profile or it is nothing:
        anywhere else would build it without that profile's data, secrets and
        conventions, or over somebody else's."""
        _write(workspace.base / "elsewhere" / "ops.yml", "name: Elsewhere\n")
        from osprey.deployment.web_terminals.persona_images import auto_render_missing_personas

        with pytest.raises(ValueError, match="not a direct child"):
            auto_render_missing_personas(
                _catalog_config("../elsewhere/ops.yml", workspace.base),
                _resolved_users(),
                dict(os.environ),
                workspace.project_dir,
            )

    def test_an_off_chain_persona_preset_is_refused_at_profile_new(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A persona preset that is not a delta over the host preset would emit
        a persona file carrying its own base, which the host cannot supply.

        No bundled preset renders ``control-assistant``'s app template from
        outside its extends chain, so the off-chain case is simulated for the
        readonly persona: its raw ``extends`` is repointed away from the host
        and the chain predicate pinned false.
        """
        from osprey.cli import build_profile_presets

        real_load = build_profile_presets._load_preset_raw
        real_reaches = build_profile_presets._preset_extends_chain_reaches

        def out_of_chain_raw(name: str):
            raw, path = real_load(name)
            if name == "control-assistant-readonly":
                raw = {**raw, "extends": "hello-world"}
            return raw, path

        def out_of_chain_for_readonly(child: str, ancestor: str) -> bool:
            if child == "control-assistant-readonly":
                return False
            return real_reaches(child, ancestor)

        monkeypatch.setattr(build_profile_presets, "_load_preset_raw", out_of_chain_raw)
        monkeypatch.setattr(
            build_profile_presets, "_preset_extends_chain_reaches", out_of_chain_for_readonly
        )

        result = _invoke(
            profile_group,
            [
                "new",
                str(tmp_path / "off-chain-profile"),
                "--preset",
                "control-assistant",
            ],
        )
        assert result.exit_code != 0
        assert "does not extend" in result.output
        assert not (tmp_path / "off-chain-profile").exists(), (
            "materialization must fail before creating the directory"
        )


def _catalog_config(build_profile: str, base: Path) -> dict:
    """A deploy config whose persona catalog names ``build_profile``."""
    return {
        "modules": {
            "web_terminals": {
                "personas": {
                    PERSONA_NAME: {
                        "project": f"{PROJECT_NAME}-{PERSONA_NAME}",
                        # Must not exist: auto-render only resolves the delta
                        # for a persona whose project directory is absent.
                        "project_path": str(base / "unrendered-persona"),
                        "build_profile": build_profile,
                    }
                }
            }
        }
    }


def _resolved_users() -> list[dict]:
    return [{"name": ROSTER_USER, "persona": PERSONA_NAME, "project": f"{PROJECT_NAME}-reader"}]


# ---------------------------------------------------------------------------
# 4. Bare temp-file profiles stay standalone
# ---------------------------------------------------------------------------


BARE_PROFILE = """\
name: Bare
app_template: hello_world
provider: anthropic
model: haiku
config:
  control_system.type: mock
"""


def test_a_bare_temp_file_profile_builds_standalone(tmp_path: Path) -> None:
    """Root discovery's narrow trigger: only a file inside ``personas/`` beside
    a ``profile.yml`` is a delta. Everything else — including the bare temp-file
    profiles tests build — anchors at its own parent and needs no profile
    directory, no ``.env``, and no data tree."""
    bare = _write(tmp_path / "scratch" / "bare.yml", BARE_PROFILE)
    output = tmp_path / "out"
    _assert_ok(_build("bare", bare, output), "bare temp-file build")

    assert (output / "bare" / "config.yml").is_file()
    # No profile directory is materialized beside it: only `--preset` does that.
    assert not (output / "bare-profile").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out", "scratch"]
    assert sorted(p.name for p in (tmp_path / "scratch").iterdir()) == ["bare.yml"]


# ---------------------------------------------------------------------------
# 5. Preset provenance on reuse
# ---------------------------------------------------------------------------


def _preset_build(project: str, output: Path, preset: str, *extra: str) -> Result:
    return _invoke(
        build,
        [
            project,
            "--preset",
            preset,
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(output),
            *extra,
        ],
    )


@pytest.fixture(scope="module")
def preset_built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ``--preset`` build, which materializes ``<name>-profile/`` beside it."""
    base = tmp_path_factory.mktemp("preset_reuse")
    _assert_ok(_preset_build("drift", base, PRESET), "first --preset build")
    assert (base / "drift-profile" / "profile.yml").is_file()
    return base


class TestPresetProvenanceOnReuse:
    """Contract point 5 — a reused profile is built verbatim, and says so."""

    def test_a_moved_on_preset_advises_and_still_builds(
        self, preset_built: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        base = tmp_path / "drifted"
        shutil.copytree(preset_built, base, symlinks=True)
        profile_path = base / "drift-profile" / "profile.yml"

        # Stand in for the installed preset having moved on since materialization.
        profile_path.write_text(
            profile_path.read_text(encoding="utf-8").replace(
                "preset_hash: sha256:", "preset_hash: sha256:0000"
            ),
            encoding="utf-8",
        )
        marker = _write(base / "drift-profile" / "rules" / "local.md", "# local\n")

        with caplog.at_level(logging.WARNING, logger="build"):
            result = _preset_build("drift", base, PRESET, "--force")
        _assert_ok(result, "reuse build after preset drift")

        assert "has changed since" in caplog.text
        assert "nothing from the preset is re-applied" in caplog.text
        # Advisory, not a re-materialization: the local edit is still there and
        # reached the project.
        assert marker.is_file()
        assert (base / "drift" / ".claude" / "rules" / "local.md").is_file()

    def test_a_different_preset_on_reuse_is_refused(
        self, preset_built: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        base = tmp_path / "mismatch"
        shutil.copytree(preset_built, base, symlinks=True)

        with caplog.at_level(logging.ERROR):
            result = _preset_build("drift", base, "control-assistant", "--force")

        assert result.exit_code != 0
        assert "materialized from preset" in caplog.text


def test_the_manifest_records_the_profile_that_was_built(roundtrip: Roundtrip) -> None:
    """Everything downstream — staleness, deploy write-back, persona
    auto-render — finds the profile through this record."""
    manifest = json.loads(
        (roundtrip.project_dir / ".osprey-manifest.json").read_text(encoding="utf-8")
    )
    recorded = Path(manifest["build_args"]["profile_path_abs"])
    assert recorded == roundtrip.profile_path


def test_the_project_env_is_owner_only(roundtrip: Roundtrip) -> None:
    """The derived file carries the facility's secrets, so it is born 0600."""
    assert (roundtrip.project_dir / ".env").stat().st_mode & 0o777 == 0o600


def _documented_vars(env_example: Path) -> set[str]:
    """Variable names an ``.env.example`` documents, commented lines included."""
    return set(
        parse_dotenv_text(
            "\n".join(
                line.lstrip("#") for line in env_example.read_text(encoding="utf-8").splitlines()
            )
        )
    )


def test_profile_env_example_is_the_one_secrets_list(roundtrip: Roundtrip) -> None:
    """FR-1/FR-4: the profile's ``.env.example`` documents the variable set, and
    the project's copy renders from the same template — so the file that holds
    the values and the file that documents them cannot name different ones."""
    documented = _documented_vars(roundtrip.profile_dir / ".env.example")
    assert PROVIDER_VAR in documented
    assert documented == _documented_vars(roundtrip.project_dir / ".env.example")
