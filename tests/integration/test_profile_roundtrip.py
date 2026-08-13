"""The profile-is-the-source-of-truth roundtrip, driven through the real CLI.

This is the executable contract for the feature's cross-cutting requirement:

    After ``init`` -> ``build`` -> edit the source zone -> ``build``, the render
    reflects EVERY edit and holds no facility-authored state the source cannot
    regenerate.

Everything else here exists to make that sentence checkable end to end, so the
suite is organized as one scripted roundtrip plus the semantics that hang off
it:

1. **Every edit reaches the render** (the :func:`roundtrip` fixture and
   :class:`TestEveryEditReachesTheProject`) — ``osprey init`` a deployment repo,
   build it, then edit one artifact in *every* convention category plus
   ``triggers.yml``, a persona delta, and the repo ``.env``; rebuild and assert
   each edit landed, and that the rebuild never wrote back into the source.
2. **Nothing unregenerable accumulates** (:class:`TestNoUnregenerableFacilityState`)
   — the render carries no secrets surface at all, the mirror refuses a
   build-owned path, and deleting ``build/`` outright loses nothing the source
   zone owns. The render carries no ``.env`` at all, so there is no merge of the
   profile's ``.env`` into a copy inside the render and no rules governing one;
   that contract is asserted directly, as the *absence* of a secrets surface.
3. **The persona stack** — a ``personas/<name>.yml`` delta anchored at its root,
   its exclusions (list artifact, convention artifact, and a convention artifact
   shadowing a framework render, which the exclusion restores), its inherited
   secrets, root-edit staleness, and the ways a persona reference fails.
4. **Standalone builds** — a directory holding one ``profile.yml`` IS a
   deployment repo: it renders into its own ``build/`` and materializes nothing
   beside itself.

Cost control: ``osprey build`` is the slow step, so the scripted roundtrip runs
once per module (``--skip-deps --skip-lifecycle``, which keeps it network-free)
and the read-only assertions share it. Tests that mutate take a copy.

Environment: ``osprey init`` seeds the repo ``.env`` from the shell's exported
provider keys, so every invocation here runs under
:func:`_sanitized_provider_env` — the developer's real keys must not reach a
fixture, and the sentinel value below is what the assertions track through the
pipeline.
"""

from __future__ import annotations

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
from osprey.cli.init_cmd import init
from osprey.models.provider_registry import PROVIDER_API_KEYS
from osprey.utils.dotenv import parse_dotenv_file

# The provider key value the whole suite tracks. `hello-world` selects the
# anthropic provider, so this is the one key `osprey init` seeds.
PROVIDER_VAR = "ANTHROPIC_API_KEY"
PROFILE_KEY_VALUE = "sk-profile-owned-value"

# A secret the operator adds to the profile `.env` by hand — the plain case of
# "a value set in the profile reaches every project built from it".
ARCHIVER_VAR = "FACILITY_ARCHIVER_URL"
ARCHIVER_VALUE = "http://archiver.facility.example:17668"

PRESET = "hello-world"

#: The deployment repo `osprey init` creates. Its root IS the source zone the
#: roundtrip edits, and `build/` inside it is the render it asserts on — one
#: directory, not the two sibling trees the legacy shape had.
REPO_DIRNAME = "facility-repo"

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

    ``osprey init`` seeds the repo ``.env`` from ``os.environ``
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


def _init(repo: Path, *extra: str) -> Result:
    """Materialize the deployment repo at *repo*."""
    return _invoke(init, [str(repo), "--preset", PRESET, "--no-git", *extra])


def _build(repo: Path, *extra: str) -> Result:
    """Re-render *repo*'s build zone.

    There is no ``--force``: a build wipes and re-renders ``build/`` every time,
    which is what makes the second half of this roundtrip a plain rebuild rather
    than an overwrite with a preserve-list.
    """
    return _invoke(
        build,
        ["--repo", str(repo), "--skip-deps", "--skip-lifecycle", *extra],
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

# The render used to receive a derived `.env` of its own, and the fixtures that
# once lived here seeded it so a rebuild's per-key derivation rules could be
# observed: a stale build-derived pointer, a simulation fault a runtime writer
# left, a token minted into the project alone, and a hand edit the profile
# cannot reproduce. There is no such file now — the repo root's `.env` is the
# deployment's whole secret store — so the contract those fixtures served is
# stated directly instead: the render carries no secrets surface at all.


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


# ---------------------------------------------------------------------------
# The scripted roundtrip
# ---------------------------------------------------------------------------


@dataclass
class Roundtrip:
    """One completed ``init -> build -> edit -> build`` cycle."""

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

    def repo_env(self) -> dict[str, str]:
        """The deployment's ``.env`` — at the repo root, beside profile.yml.

        The only one. It is what an operator edits, what every compose
        invocation is pointed at, and what ``osprey up`` mints into. The render
        receives no copy of it.
        """
        return parse_dotenv_file(self.profile_dir / ".env")


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory: pytest.TempPathFactory) -> Roundtrip:
    """Run the whole scripted roundtrip once; read-only tests share it."""
    base = tmp_path_factory.mktemp("roundtrip")
    profile_dir = base / REPO_DIRNAME
    project_dir = profile_dir / "build"

    _assert_ok(_init(profile_dir), "osprey init")

    result = _build(profile_dir)
    _assert_ok(result, "first osprey build")

    trip = Roundtrip(base=base, profile_dir=profile_dir, project_dir=project_dir)
    trip.framework_safety_rule = (project_dir / ".claude" / "rules" / "safety.md").read_text(
        encoding="utf-8"
    )
    trip.baseline_rules = {p.name for p in (project_dir / ".claude" / "rules").iterdir()}

    _apply_profile_edits(profile_dir)

    _assert_ok(_build(profile_dir), "second osprey build")
    return trip


@pytest.fixture
def workspace(roundtrip: Roundtrip, tmp_path: Path) -> Roundtrip:
    """A private copy of the completed roundtrip, for tests that mutate."""
    base = tmp_path / "copy"
    shutil.copytree(roundtrip.base, base, symlinks=True)
    return Roundtrip(
        base=base,
        profile_dir=base / REPO_DIRNAME,
        project_dir=base / REPO_DIRNAME / "build",
        framework_safety_rule=roundtrip.framework_safety_rule,
        baseline_rules=set(roundtrip.baseline_rules),
    )


# ---------------------------------------------------------------------------
# 1. Every edit reaches the render
# ---------------------------------------------------------------------------


class TestEveryEditReachesTheProject:
    """The source zone's edits, and nothing but them, reach the render."""

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
                _build(workspace.profile_dir),
                "rebuild for the shadow notice",
            )
        assert f"profile overrides framework rule '{SHADOWED_RULE}'" in caplog.text

    def test_triggers_come_from_the_profile(self, roundtrip: Roundtrip) -> None:
        """FR-3: the profile owns ``triggers.yml``; the build copies it in."""
        shipped = yaml.safe_load(
            (roundtrip.project_dir / "triggers.yml").read_text(encoding="utf-8")
        )
        assert "facility_alarm" in shipped["triggers"]

    def test_the_repos_env_holds_the_value_the_edits_wrote(self, roundtrip: Roundtrip) -> None:
        """The repo's own `.env` is the deployment's secret store, and it sits
        beside profile.yml rather than inside the render — which is what makes
        `rm -rf build/` safe to run."""
        assert roundtrip.repo_env()[ARCHIVER_VAR] == ARCHIVER_VALUE

    def test_a_rebuild_never_touches_the_source(self, roundtrip: Roundtrip) -> None:
        """A build replaces the render and reads the source. The source zone is
        replaced only by ``osprey init --force``."""
        for source_rel, _dest, body in CONVENTION_EDITS:
            assert (roundtrip.profile_dir / source_rel).read_text(encoding="utf-8") == body
        assert parse_dotenv_file(roundtrip.profile_dir / ".env")[ARCHIVER_VAR] == ARCHIVER_VALUE


# ---------------------------------------------------------------------------
# 2. Nothing unregenerable accumulates
# ---------------------------------------------------------------------------


class TestNoUnregenerableFacilityState:
    """What the source zone cannot reproduce must not be in the render."""

    def test_the_render_carries_no_secrets_surface_at_all(self, roundtrip: Roundtrip) -> None:
        """The render holds ``.env.example`` and nothing else env-shaped.

        This is the strongest form of "the profile can reproduce the render":
        unregenerable facility state cannot be *dropped* from a file that was
        never written. The build used to derive a project ``.env`` from the
        profile's, which meant a per-key rule deciding what survived a rebuild
        and a second copy of every secret sitting inside the disposable zone.

        ``.env.example`` stays because it carries no values — it is the
        documented list of what the repo's own ``.env`` may hold.
        """
        env_files = {p.name for p in roundtrip.project_dir.glob(".env*")}
        assert env_files == {".env.example"}
        # And the secrets are exactly one level up, where nothing wipes them.
        assert roundtrip.repo_env()[PROVIDER_VAR] == PROFILE_KEY_VALUE

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
            result = _build(workspace.profile_dir)
        assert result.exit_code != 0
        assert "config.yml" in caplog.text
        assert "`config:` block" in caplog.text

    def test_deleting_the_project_loses_nothing_the_profile_owns(
        self, workspace: Roundtrip
    ) -> None:
        """The direct measure of regenerability: rebuild from nothing but the profile."""
        shutil.rmtree(workspace.project_dir)
        _assert_ok(_build(workspace.profile_dir), "rebuild after delete")

        for _source_rel, dest_rel, body in CONVENTION_EDITS:
            rebuilt = workspace.project_dir / dest_rel
            assert rebuilt.is_file(), f"{dest_rel} did not come back"
            assert rebuilt.read_text(encoding="utf-8") == body
        assert (workspace.project_dir / "triggers.yml").is_file()

        # The `.env` is not in the render at all, so deleting the render cannot
        # touch it — which is the strongest form of this contract point.
        env = parse_dotenv_file(workspace.profile_dir / ".env")
        assert env[ARCHIVER_VAR] == ARCHIVER_VALUE
        assert env[PROVIDER_VAR] == PROFILE_KEY_VALUE
        assert EXPECTED_OWNED <= _user_owned(workspace.project_dir)


# ---------------------------------------------------------------------------
# 3. The persona stack
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def persona_project(roundtrip: Roundtrip) -> Path:
    """Render the delta the scripted edits wrote at ``personas/reader.yml``.

    Nothing is added to the profile here: the delta is one of the roundtrip's
    own edits, so what this renders is the same profile every other test reads.
    """
    project = roundtrip.profile_dir / "build" / f"{REPO_DIRNAME}-{PERSONA_NAME}"
    assert project.is_dir(), (
        f"the roundtrip's build should have rendered {project.name} from the "
        f"delta at personas/{PERSONA_NAME}.yml; build/ holds "
        f"{sorted(p.name for p in (roundtrip.profile_dir / 'build').iterdir())}"
    )
    return project


class TestPersonaStack:
    """Contract point 3 — deltas anchor at the root and subtract from it."""

    def test_delta_inherits_the_roots_convention_artifacts(self, persona_project: Path) -> None:
        assert (persona_project / ".claude" / "agents" / "orbit-writer.md").is_file()
        assert (persona_project / ".claude" / "skills" / "orbit-check" / "SKILL.md").is_file()

    def test_delta_anchors_data_at_the_profile_root(self, persona_project: Path) -> None:
        """``data:`` resolves against the root, never against ``personas/``."""
        assert (persona_project / "data" / "channel_limits.json").is_file()

    def test_a_persona_render_carries_no_secrets_of_its_own(
        self, persona_project: Path, roundtrip: Roundtrip
    ) -> None:
        """A persona project is a container build context sitting inside
        ``build/``, and ``build/`` is documented as safe to ``rm -rf``. Copying
        the repo's secrets into it would put the facility's keys somewhere that
        promise makes disposable — so the persona gets none, and reads the
        repo's own ``.env`` at deploy time like every other service.
        """
        assert not (persona_project / ".env").exists()
        # The keys are still exactly where they belong, one level up.
        assert roundtrip.repo_env()[PROVIDER_VAR] == PROFILE_KEY_VALUE

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

    def test_a_root_edit_moves_the_repos_one_drift_verdict(
        self, workspace: Roundtrip, tmp_path: Path
    ) -> None:
        """One repo, one drift verdict, personas folded into it.

        Editing the root has to register as drift, and it registers in exactly
        one place: ``build/.osprey-manifest.json``. A persona render carries its
        own manifest for what runs inside its container, but deliberately no
        per-key digests — the repo's fingerprint already folds every delta in,
        and a second set nothing consults would be a second hash mechanism.

        Built fresh rather than taken from the shared ``workspace`` copy: a
        manifest records the profile it was rendered from by ABSOLUTE path, so
        a copied repo reads the original's profile and reports clean however
        much the copy is edited.
        """
        from osprey.deployment.staleness import staleness_reasons

        repo = tmp_path / "drift-probe"
        _assert_ok(_init(repo), "osprey init")
        _assert_ok(_build(repo), "osprey build")
        assert staleness_reasons(repo / "build") == []

        profile = repo / "profile.yml"
        profile.write_text(
            profile.read_text(encoding="utf-8") + "\ndeploy_services: false\n", encoding="utf-8"
        )

        reasons = staleness_reasons(repo / "build")
        assert any("has changed" in reason for reason in reasons), reasons


class TestPersonaReferenceErrors:
    """Contract point 3, error half — every unusable reference is actionable."""

    def test_extends_inside_personas_is_rejected(
        self, workspace: Roundtrip, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write(
            workspace.profile_dir / "personas" / "bad.yml",
            f"name: Bad\nextends: {PRESET}\n",
        )
        with caplog.at_level(logging.ERROR):
            result = _build(workspace.profile_dir)
        assert result.exit_code != 0
        assert "extends" in caplog.text
        assert "personas/" in caplog.text

    def test_a_personas_file_without_a_root_is_rejected(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A ``personas/`` directory with no ``profile.yml`` above it is not a
        deployment repo: the deltas in it have nothing to merge over, so there
        is nothing to render and the refusal has to say which piece is missing.
        """
        _write(tmp_path / "personas" / "lonely.yml", "name: Lonely\n")
        with caplog.at_level(logging.ERROR):
            result = _build(tmp_path)
        assert result.exit_code != 0

    def test_a_catalog_entry_naming_a_missing_delta_is_rejected(self, workspace: Roundtrip) -> None:
        from osprey.deployment.web_terminals.persona_images import verify_persona_renders

        with pytest.raises(ValueError, match="no file exists at"):
            verify_persona_renders(
                _catalog_config("personas/ghost.yml", workspace.base),
                _resolved_users(),
                workspace.profile_dir,
            )

    def test_a_catalog_entry_pointing_outside_personas_is_rejected(
        self, workspace: Roundtrip
    ) -> None:
        """A persona is a delta over THIS deployment's profile or it is nothing:
        anywhere else would build it without that profile's data, secrets and
        conventions, or over somebody else's."""
        _write(workspace.base / "elsewhere" / "ops.yml", "name: Elsewhere\n")
        from osprey.deployment.web_terminals.persona_images import verify_persona_renders

        with pytest.raises(ValueError, match="not a direct child"):
            verify_persona_renders(
                _catalog_config("../elsewhere/ops.yml", workspace.base),
                _resolved_users(),
                workspace.profile_dir,
            )

    def test_an_off_chain_persona_preset_is_refused_at_init(
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
            init,
            [
                str(tmp_path / "off-chain-repo"),
                "--preset",
                "control-assistant",
                "--no-git",
            ],
        )
        assert result.exit_code != 0
        assert "does not extend" in result.output
        assert not (tmp_path / "off-chain-repo" / "profile.yml").exists(), (
            "materialization must fail before writing the source zone"
        )


def _catalog_config(build_profile: str, base: Path) -> dict:
    """A deploy config whose persona catalog names ``build_profile``."""
    return {
        "modules": {
            "web_terminals": {
                "personas": {
                    PERSONA_NAME: {
                        "project": f"{REPO_DIRNAME}-{PERSONA_NAME}",
                        # Must not exist: the delta is only resolved for a
                        # persona whose project directory is absent.
                        "project_path": str(base / "unrendered-persona"),
                        "build_profile": build_profile,
                    }
                }
            }
        }
    }


def _resolved_users() -> list[dict]:
    return [{"name": ROSTER_USER, "persona": PERSONA_NAME, "project": f"{REPO_DIRNAME}-reader"}]


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


def test_a_minimal_profile_directory_builds_standalone(tmp_path: Path) -> None:
    """A directory holding one ``profile.yml`` IS a deployment repo.

    Root discovery's narrow trigger: only a file inside ``personas/`` beside a
    ``profile.yml`` is a delta. Everything else — including a hand-written
    scratch profile like this one — is simply the source zone of the repo it
    sits in, and needs no data tree and no ``.env`` to render.
    """
    repo = tmp_path / "scratch"
    _write(repo / "profile.yml", BARE_PROFILE)

    _assert_ok(_build(repo), "minimal repo build")

    assert (repo / "build" / "config.yml").is_file()
    # The render lands INSIDE the repo. Nothing is materialized beside it.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scratch"]
    assert "build" in {p.name for p in repo.iterdir()}
