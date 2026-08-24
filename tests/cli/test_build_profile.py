"""Tests for the ``bluesky_web:`` and ``environment:`` blocks of the
build-profile schema.

Covers the :class:`BlueskyWebConfig` dataclass, the ``BuildProfile.validate``
exemption that lets the non-builtin ``bluesky`` web_panels id validate
without a pre-existing ``web.panels.bluesky.url`` when a ``bluesky_web``
block is present
(its url is derived post-build by ``_inject_bluesky_web``), and a
regression guard that the shipped control-assistant preset/profile still
validates — the gate task 3.3 (tutorial-config) re-runs after adding the
bluesky panel to that preset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import build_profile as bp
from osprey.cli.build_cmd import build
from osprey.cli.build_profile import (
    BlueskyWebConfig,
    BuildProfile,
    EnvironmentConfig,
    _parse_profile,
    load_profile,
)
from osprey.errors import BuildProfileError


def test_no_bluesky_web_validates(tmp_path: Path) -> None:
    """A profile with no bluesky_web block validates without raising."""
    BuildProfile(name="x").validate(tmp_path)


def test_bluesky_web_ids_validate_without_url_when_bluesky_web_present(
    tmp_path: Path,
) -> None:
    """The bluesky-panel ids need no manual web.panels.<id>.url override
    when a bluesky_web block is present — the urls are derived post-build by
    _inject_bluesky_web, which runs after this validator."""
    profile = BuildProfile(
        name="x",
        web_panels=["bluesky"],
        bluesky_web=BlueskyWebConfig(),
    )
    profile.validate(tmp_path)  # must not raise


def test_bluesky_web_ids_without_bluesky_web_still_require_url(tmp_path: Path) -> None:
    """The escape hatch is narrow: the bluesky-panel ids with no bluesky_web
    block and no url override are still rejected (nothing would derive their
    URL)."""
    profile = BuildProfile(name="x", web_panels=["plan"])
    with pytest.raises(BuildProfileError, match="plan"):
        profile.validate(tmp_path)


def test_unbacked_custom_panel_still_requires_url_even_with_bluesky_web(
    tmp_path: Path,
) -> None:
    """The bluesky_web escape hatch applies only to the three known ids — any
    other url-less custom panel is still rejected even when a bluesky_web
    block is present."""
    profile = BuildProfile(
        name="x",
        web_panels=["grafana"],
        bluesky_web=BlueskyWebConfig(),
    )
    with pytest.raises(BuildProfileError, match="grafana"):
        profile.validate(tmp_path)


def test_bluesky_web_port_overflow_raises(tmp_path: Path) -> None:
    """An out-of-range bluesky_web.port fails validation."""
    profile = BuildProfile(name="x", bluesky_web=BlueskyWebConfig(port=70000))
    with pytest.raises(BuildProfileError, match="bluesky_web.port"):
        profile.validate(tmp_path)


def test_bluesky_web_default_port() -> None:
    """BlueskyWebConfig defaults to port 8095, matching the compose template
    and the sidecar's default uvicorn bind."""
    assert BlueskyWebConfig().port == 8095


def test_bluesky_web_not_a_mapping_raises() -> None:
    """A non-mapping 'bluesky_web' block raises during parsing."""
    with pytest.raises(BuildProfileError, match="bluesky_web"):
        _parse_profile({"name": "x", "bluesky_web": "not-a-mapping"})


def test_bluesky_web_is_known_key() -> None:
    """'bluesky_web' is a recognized top-level profile key (no unknown-key warning)."""
    assert "bluesky_web" in bp._KNOWN_PROFILE_KEYS


def test_bluesky_web_parse_round_trip() -> None:
    """A bluesky_web block parses its port field through _parse_profile."""
    raw = {"name": "x", "bluesky_web": {"port": 9100}}
    profile = _parse_profile(raw)
    assert profile.bluesky_web is not None
    assert profile.bluesky_web.port == 9100


def test_bluesky_web_parse_defaults_when_empty_mapping() -> None:
    """An empty bluesky_web mapping (`bluesky_web: {}`) parses to defaults."""
    raw = {"name": "x", "bluesky_web": {}}
    profile = _parse_profile(raw)
    assert profile.bluesky_web is not None
    assert profile.bluesky_web.port == 8095


# ── panel_presets ("Layouts") ────────────────────────────────────────────────


def test_panel_presets_builtin_members_validate(tmp_path: Path) -> None:
    """A preset whose members are built-in panel ids validates without raising."""
    profile = BuildProfile(name="x", panel_presets={"Machine setup": ["artifacts", "ariel"]})
    profile.validate(tmp_path)  # must not raise


def test_panel_presets_unknown_member_raises(tmp_path: Path) -> None:
    """A preset member that is not a known panel id fails validation."""
    profile = BuildProfile(name="x", panel_presets={"Setup": ["artifacts", "ghost"]})
    with pytest.raises(BuildProfileError, match="ghost"):
        profile.validate(tmp_path)


def test_panel_presets_url_backed_member_validates(tmp_path: Path) -> None:
    """A member backed by a web.panels.<id>.url override is a known id."""
    profile = BuildProfile(
        name="x",
        panel_presets={"Dash": ["grafana"]},
        config={"web.panels.grafana.url": "http://grafana.local:3000"},
    )
    profile.validate(tmp_path)  # must not raise


def test_panel_presets_web_panels_member_validates(tmp_path: Path) -> None:
    """A member declared in web_panels is a known id."""
    profile = BuildProfile(name="x", web_panels=["ariel"], panel_presets={"L": ["ariel"]})
    profile.validate(tmp_path)  # must not raise


def test_panel_presets_non_list_value_raises(tmp_path: Path) -> None:
    """A preset whose value is not a list of ids is rejected."""
    profile = BuildProfile(name="x", panel_presets={"Bad": "artifacts"})  # type: ignore[dict-item]
    with pytest.raises(BuildProfileError, match="must be a list"):
        profile.validate(tmp_path)


def test_is_known_panel_id_shared_predicate(tmp_path: Path) -> None:
    """_is_known_panel_id covers built-in / web_panels / url-backed for both callers."""
    profile = BuildProfile(
        name="x",
        web_panels=["ariel"],
        config={"web.panels.grafana.url": "http://x:3000"},
    )
    assert profile._is_known_panel_id("artifacts")  # built-in
    assert profile._is_known_panel_id("ariel")  # web_panels entry
    assert profile._is_known_panel_id("grafana")  # url-backed custom
    assert not profile._is_known_panel_id("nope")


def test_control_assistant_profile_validates() -> None:
    """The shipped control-assistant preset/profile validates cleanly.

    This is a regression guard task 3.3 (tutorial-config) re-runs after
    adding the bluesky_web block + the bluesky-panel web_panels ids to
    this preset — it must keep validating once that wiring lands.
    """
    presets_dir = bp._presets_dir()
    raw = yaml.safe_load((presets_dir / "control-assistant.yml").read_text(encoding="utf-8"))
    profile = _parse_profile(raw)

    profile.validate(presets_dir)  # raises BuildProfileError on any issue


# ── Task 3.3: turn-key VA-backed plan stack render ───────────────────────────
#
# The control-assistant preset now bakes the bluesky/virtual_accelerator/
# bluesky_web injector blocks in directly (no --set/--override flags needed),
# so `osprey build` on the bare preset renders the full plan stack + the
# BLUESKY panel turn-key. These tests build a control-assistant deployment
# repo in-process (CliRunner, --skip-deps --skip-lifecycle -- Docker-free,
# mirroring tests/cli/test_va_default_config.py's scaffolded_project fixture and
# tests/e2e/_orm_stack.py's build_via_cli_runner) and assert on the rendered
# build/config.yml.


@pytest.fixture(scope="module")
def turnkey_plan_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the bare control-assistant preset (no overrides) into a tmp repo.

    The exemplar repo (:func:`build_exemplar_repo`) is control-assistant's
    profile.yml written out verbatim, so materializing it and building it is
    the bare preset with no overrides. Module-scoped: the build is the slow
    part (template render + service template copies) and every test in this
    section only reads the resulting config.yml, so one build is shared
    across assertions.
    """
    from tests.fixtures.lifecycle_repo import build_exemplar_repo

    tmp_path = tmp_path_factory.mktemp("turnkey-plan")
    repo = build_exemplar_repo(tmp_path / "turnkey-plan")
    runner = CliRunner()
    result = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    project_dir = repo / "build"
    assert (project_dir / "config.yml").exists()
    return project_dir


@pytest.fixture(scope="module")
def turnkey_plan_config(turnkey_plan_project: Path) -> dict:
    return yaml.safe_load((turnkey_plan_project / "config.yml").read_text(encoding="utf-8"))


class TestControlAssistantTurnkeyPlanServices:
    """The three injected services render with the preset's baked-in ports."""

    def test_control_assistant_bluesky_service_rendered(self, turnkey_plan_config: dict) -> None:
        bluesky = turnkey_plan_config["services"]["bluesky"]
        assert bluesky["port"] == 8090
        assert bluesky["tiled_enabled"] is True
        assert bluesky["tiled_port"] == 8091

    def test_control_assistant_virtual_accelerator_service_rendered(
        self, turnkey_plan_config: dict
    ) -> None:
        va = turnkey_plan_config["services"]["virtual_accelerator"]
        assert va["port"] == 5064

    def test_control_assistant_bluesky_web_service_rendered(
        self, turnkey_plan_config: dict
    ) -> None:
        bluesky_web = turnkey_plan_config["services"]["bluesky_web"]
        assert bluesky_web["port"] == 8095

    def test_control_assistant_deployed_services_includes_all_three(
        self, turnkey_plan_config: dict
    ) -> None:
        deployed = turnkey_plan_config["deployed_services"]
        assert "bluesky" in deployed
        assert "virtual_accelerator" in deployed
        assert "bluesky_web" in deployed


class TestControlAssistantTurnkeyPlanPanels:
    """The one bluesky-panel web.panels entry is registered with a url."""

    def test_control_assistant_bluesky_panel(self, turnkey_plan_config: dict) -> None:
        panel = turnkey_plan_config["web"]["panels"]["bluesky"]
        assert panel["path"] == "/bluesky/"
        assert panel["url"]


class TestControlAssistantTurnkeyPlanControlSystem:
    """The preset's config overrides land: VA-by-default + subprocess execution.

    The VA soft-IOC ships and is deployed unconditionally as part of the
    turn-key plan stack, and control_system.type defaults to
    "virtual_accelerator" so a fresh tutorial project drives it end to end
    out of the box -- flipping the one config line to "mock" is the
    documented fallback for environments with no containers to depend on
    (covered by tests/cli/test_va_default_config.py).
    """

    def test_control_assistant_control_system_type_is_virtual_accelerator(
        self, turnkey_plan_config: dict
    ) -> None:
        assert turnkey_plan_config["control_system"]["type"] == "virtual_accelerator"

    def test_control_assistant_execution_method_is_subprocess(
        self, turnkey_plan_config: dict
    ) -> None:
        """Agent Python runs as a host subprocess — the preset pins nothing, so
        this is the template default the built project inherits."""
        assert turnkey_plan_config["execution"]["execution_method"] == "subprocess"

    def test_control_assistant_bluesky_mcp_server_enabled(self, turnkey_plan_config: dict) -> None:
        assert turnkey_plan_config["claude_code"]["servers"]["bluesky"]["enabled"] is True


def test_control_assistant_turnkey_plan_preset_validates(turnkey_plan_project: Path) -> None:
    """BuildProfile.validate() passes for the preset as shipped (bare, no
    overrides) -- the non-builtin bluesky-panel ids are accepted because
    the preset's own bluesky_web block is present."""
    presets_dir = bp._presets_dir()
    raw = yaml.safe_load((presets_dir / "control-assistant.yml").read_text(encoding="utf-8"))
    profile = _parse_profile(raw)
    profile.validate(presets_dir)  # raises BuildProfileError on any issue


# ── Task 3.1: the ``environment:`` block ─────────────────────────────────────
#
# ``environment:`` declares the Python environment agent-authored code executes
# in: a base interpreter (bare or a venv's), extra packages, and — only for a
# venv base — distributions to leave out of the freeze. There is no mode flag:
# venv-ness is detected from a ``pyvenv.cfg`` beside the interpreter's dir.


@pytest.fixture
def bare_interpreter(tmp_path: Path) -> Path:
    """An existing, executable interpreter with no venv marker around it."""
    python = tmp_path / "opt" / "python3.11" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    return python


@pytest.fixture
def venv_interpreter(tmp_path: Path) -> Path:
    """An executable interpreter inside a directory marked as a venv."""
    venv = tmp_path / "facility-venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    return python


class TestEnvironmentSchema:
    """Field presence/absence and the parse-time shape contract."""

    def test_environment_absent_is_all_defaults(self, tmp_path: Path) -> None:
        """A profile with no environment block still exposes a usable config —
        consumers never need a None check."""
        profile = _parse_profile({"name": "x"})
        assert profile.environment.python is None
        assert profile.environment.packages == []
        assert profile.environment.inherit_exclude == []
        profile.validate(tmp_path)  # must not raise

    def test_environment_is_known_key(self) -> None:
        """'environment' is a recognized top-level profile key."""
        assert "environment" in bp._KNOWN_PROFILE_KEYS

    def test_environment_parse_round_trip(self, venv_interpreter: Path) -> None:
        """All three fields round-trip through the raw-profile loader."""
        raw = {
            "name": "x",
            "environment": {
                "python": str(venv_interpreter),
                "packages": ["numpy>=1.26", "at"],
                "inherit_exclude": ["osprey-framework"],
            },
        }
        profile = _parse_profile(raw)
        assert profile.environment.python == str(venv_interpreter)
        assert profile.environment.packages == ["numpy>=1.26", "at"]
        assert profile.environment.inherit_exclude == ["osprey-framework"]

    def test_environment_null_block_parses_to_defaults(self) -> None:
        """`environment:` with no value (YAML null) is the same as omitting it."""
        profile = _parse_profile({"name": "x", "environment": None})
        assert profile.environment == EnvironmentConfig()

    def test_environment_empty_mapping_parses_to_defaults(self) -> None:
        profile = _parse_profile({"name": "x", "environment": {}})
        assert profile.environment == EnvironmentConfig()

    def test_environment_not_a_mapping_raises(self) -> None:
        with pytest.raises(BuildProfileError, match="'environment' must be a mapping"):
            _parse_profile({"name": "x", "environment": "not-a-mapping"})

    def test_environment_unknown_key_raises(self) -> None:
        """A typo inside the block is rejected outright rather than dropped —
        a silently ignored 'package:' would leave the environment incomplete.

        The message is the SHARED closed-block shape (``_reject_unknown_block_keys``),
        the same one the top-level and ``bluesky:`` checks emit: the offender
        named, its nearest spelling suggested, and the full valid set listed.
        Asserting the suggestion is what keeps this block on the shared helper —
        the older bespoke wording ("must be one of [...]") carried no suggestion,
        so a drift back to it fails here rather than passing quietly.
        """
        with pytest.raises(BuildProfileError) as excinfo:
            _parse_profile({"name": "x", "environment": {"package": ["numpy"]}})

        message = str(excinfo.value)
        assert "Unknown environment key(s)" in message
        assert "'package'" in message
        assert "did you mean 'packages'?" in message
        assert "valid keys are:" in message

    def test_environment_python_non_string_raises(self) -> None:
        with pytest.raises(BuildProfileError, match="environment.python must be a string"):
            _parse_profile({"name": "x", "environment": {"python": 3.11}})

    @pytest.mark.parametrize("key", ["packages", "inherit_exclude"])
    def test_environment_list_field_non_list_raises(self, key: str) -> None:
        with pytest.raises(BuildProfileError, match=f"environment.{key} must be a list"):
            _parse_profile({"name": "x", "environment": {key: "numpy"}})

    def test_environment_coexists_with_dependencies(self, bare_interpreter: Path) -> None:
        """environment.packages does not replace or disturb `dependencies`."""
        profile = _parse_profile(
            {
                "name": "x",
                "dependencies": ["pandas"],
                "environment": {"python": str(bare_interpreter), "packages": ["numpy"]},
            }
        )
        assert profile.dependencies == ["pandas"]
        assert profile.environment.packages == ["numpy"]

    def test_environment_loads_from_yaml_file(self, tmp_path: Path, venv_interpreter: Path) -> None:
        """Full round trip through load_profile (YAML → parse → validate)."""
        profile_path = tmp_path / "facility.yml"
        profile_path.write_text(
            yaml.safe_dump(
                {
                    "name": "facility",
                    "environment": {
                        "python": str(venv_interpreter),
                        "packages": ["numpy"],
                        "inherit_exclude": ["pip"],
                    },
                }
            ),
            encoding="utf-8",
        )
        profile = load_profile(profile_path)
        assert profile.environment.python == str(venv_interpreter)
        assert profile.environment.packages == ["numpy"]
        assert profile.environment.inherit_exclude == ["pip"]

    def test_environment_merges_through_extends(
        self, tmp_path: Path, venv_interpreter: Path
    ) -> None:
        """A child profile inherits the base's environment block and adds to it."""
        base = tmp_path / "base.yml"
        base.write_text(
            yaml.safe_dump(
                {
                    "name": "base",
                    "environment": {"python": str(venv_interpreter), "packages": ["numpy"]},
                }
            ),
            encoding="utf-8",
        )
        child = tmp_path / "child.yml"
        child.write_text(
            yaml.safe_dump(
                {"name": "child", "extends": "base.yml", "environment": {"packages": ["at"]}}
            ),
            encoding="utf-8",
        )
        profile = load_profile(child)
        assert profile.environment.python == str(venv_interpreter)
        assert profile.environment.packages == ["numpy", "at"]


class TestEnvironmentAccessors:
    """resolved_python() / venv_base() — the shape downstream consumers use."""

    def test_resolved_python_none_when_unset(self) -> None:
        assert EnvironmentConfig().resolved_python() is None

    def test_resolved_python_expands_user(self) -> None:
        resolved = EnvironmentConfig(python="~/venvs/als/bin/python").resolved_python()
        assert resolved is not None
        assert "~" not in str(resolved)
        assert resolved.is_absolute()

    def test_venv_base_none_when_unset(self) -> None:
        assert EnvironmentConfig().venv_base() is None

    def test_venv_base_none_for_bare_interpreter(self, bare_interpreter: Path) -> None:
        assert EnvironmentConfig(python=str(bare_interpreter)).venv_base() is None

    def test_venv_base_returns_venv_root(self, venv_interpreter: Path) -> None:
        """`<venv>/bin/python` resolves to `<venv>` — the root task 3.3 freezes."""
        cfg = EnvironmentConfig(python=str(venv_interpreter))
        assert cfg.venv_base() == venv_interpreter.parent.parent

    def test_venv_base_detects_interpreter_in_venv_root(self, tmp_path: Path) -> None:
        """An interpreter sitting directly in the venv root is still detected."""
        venv = tmp_path / "flat-venv"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
        python = venv / "python"
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o755)
        assert EnvironmentConfig(python=str(python)).venv_base() == venv


class TestEnvironmentValidation:
    """The rules BuildProfile.validate enforces on the block."""

    def test_bare_interpreter_base_validates(self, tmp_path: Path, bare_interpreter: Path) -> None:
        profile = BuildProfile(
            name="x",
            environment=EnvironmentConfig(python=str(bare_interpreter), packages=["numpy"]),
        )
        profile.validate(tmp_path)  # must not raise

    def test_venv_base_validates(self, tmp_path: Path, venv_interpreter: Path) -> None:
        profile = BuildProfile(
            name="x", environment=EnvironmentConfig(python=str(venv_interpreter))
        )
        profile.validate(tmp_path)  # must not raise

    def test_missing_python_raises(self, tmp_path: Path) -> None:
        profile = BuildProfile(name="x", environment=EnvironmentConfig(python="/nope/bin/python"))
        with pytest.raises(BuildProfileError, match="environment.python not found"):
            profile.validate(tmp_path)

    def test_non_executable_python_raises(self, tmp_path: Path) -> None:
        python = tmp_path / "python"
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o644)
        profile = BuildProfile(name="x", environment=EnvironmentConfig(python=str(python)))
        with pytest.raises(BuildProfileError, match="is not an executable file"):
            profile.validate(tmp_path)

    def test_directory_python_raises(self, tmp_path: Path) -> None:
        """A directory is traversable-executable but is not an interpreter."""
        profile = BuildProfile(name="x", environment=EnvironmentConfig(python=str(tmp_path)))
        with pytest.raises(BuildProfileError, match="is not an executable file"):
            profile.validate(tmp_path)

    def test_relative_python_raises(self, tmp_path: Path) -> None:
        """environment.python names a host interpreter, not a profile-relative
        asset — consumers hold only the profile, so it must be absolute."""
        profile = BuildProfile(name="x", environment=EnvironmentConfig(python="venv/bin/python"))
        with pytest.raises(BuildProfileError, match="must be an absolute path"):
            profile.validate(tmp_path)

    def test_empty_python_string_raises(self, tmp_path: Path) -> None:
        profile = BuildProfile(name="x", environment=EnvironmentConfig(python="   "))
        with pytest.raises(BuildProfileError, match="environment.python must be a non-empty"):
            profile.validate(tmp_path)

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_empty_package_entry_raises(self, tmp_path: Path, bad: str) -> None:
        profile = BuildProfile(name="x", environment=EnvironmentConfig(packages=["numpy", bad]))
        with pytest.raises(BuildProfileError, match="environment.packages entries must be"):
            profile.validate(tmp_path)

    def test_non_string_package_entry_raises(self, tmp_path: Path) -> None:
        profile = BuildProfile(name="x", environment=EnvironmentConfig(packages=[3]))  # type: ignore[list-item]
        with pytest.raises(BuildProfileError, match="environment.packages entries must be"):
            profile.validate(tmp_path)

    def test_empty_inherit_exclude_entry_raises(
        self, tmp_path: Path, venv_interpreter: Path
    ) -> None:
        profile = BuildProfile(
            name="x",
            environment=EnvironmentConfig(python=str(venv_interpreter), inherit_exclude=[""]),
        )
        with pytest.raises(BuildProfileError, match="environment.inherit_exclude entries must be"):
            profile.validate(tmp_path)

    def test_non_list_packages_reported_not_crashed(self, tmp_path: Path) -> None:
        """A directly-constructed profile that bypasses the parser still gets a
        legible error rather than iterating a string character by character."""
        profile = BuildProfile(name="x", environment=EnvironmentConfig(packages="numpy"))  # type: ignore[arg-type]
        with pytest.raises(BuildProfileError, match="environment.packages must be a list"):
            profile.validate(tmp_path)

    def test_inherit_exclude_without_any_base_raises(self, tmp_path: Path) -> None:
        """No environment.python at all: nothing to exclude from."""
        profile = BuildProfile(name="x", environment=EnvironmentConfig(inherit_exclude=["pip"]))
        with pytest.raises(BuildProfileError, match="inherit_exclude requires environment.python"):
            profile.validate(tmp_path)

    def test_inherit_exclude_with_bare_interpreter_raises(
        self, tmp_path: Path, bare_interpreter: Path
    ) -> None:
        """A bare interpreter carries no installed set to freeze, so an
        inherit_exclude would be a silent no-op."""
        profile = BuildProfile(
            name="x",
            environment=EnvironmentConfig(python=str(bare_interpreter), inherit_exclude=["pip"]),
        )
        with pytest.raises(BuildProfileError, match="inherit_exclude requires a venv base"):
            profile.validate(tmp_path)

    def test_inherit_exclude_with_venv_base_validates(
        self, tmp_path: Path, venv_interpreter: Path
    ) -> None:
        profile = BuildProfile(
            name="x",
            environment=EnvironmentConfig(
                python=str(venv_interpreter),
                packages=["numpy"],
                inherit_exclude=["osprey-framework", "pip"],
            ),
        )
        profile.validate(tmp_path)  # must not raise

    def test_inherit_exclude_quiet_when_python_already_bad(self, tmp_path: Path) -> None:
        """One root cause, one error: a missing interpreter is not also reported
        as a missing venv base."""
        profile = BuildProfile(
            name="x",
            environment=EnvironmentConfig(python="/nope/bin/python", inherit_exclude=["pip"]),
        )
        with pytest.raises(BuildProfileError) as excinfo:
            profile.validate(tmp_path)
        assert "environment.python not found" in str(excinfo.value)
        assert "inherit_exclude" not in str(excinfo.value)


# ── config: claude_code.permissions — refused shapes and spellings ────
#
# The build composes these lists in three places (the settings.json render,
# `build_cmd._profile_setup_patch_capable`, and
# `profile_conventions.is_setup_patch_capable`), and a value that is not a list
# of strings makes them disagree in BOTH directions at once. So the profile is
# refused rather than guessed at, and each reader may then assume a list.


SETUP_PATCH = "mcp__osprey_workspace__setup_patch"


def _profile_with_config(config: dict) -> dict:
    return {"name": "x", "config": config}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(SETUP_PATCH, "bare string", id="bare_string"),
        pytest.param({"tool": SETUP_PATCH}, "dict", id="mapping"),
        pytest.param(7, "int", id="int"),
        pytest.param(None, "NoneType", id="null"),
        pytest.param([SETUP_PATCH, 7], "7", id="list_with_an_int"),
        pytest.param([SETUP_PATCH, ""], "''", id="list_with_an_empty_string"),
    ],
)
def test_a_misshapen_permission_list_is_refused(value: object, expected: str) -> None:
    """Every non-list spelling is named, with the key that carries it."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile(_profile_with_config({"claude_code.permissions.deny": value}))

    message = str(excinfo.value)
    assert "claude_code.permissions.deny" in message
    assert expected in message
    # And it says what shape was wanted, not merely that this one is wrong.
    assert "list of non-empty tool-matcher strings" in message


def test_the_bare_string_refusal_says_why_the_render_cannot_take_it() -> None:
    """The reason is the render's, not a style rule: it ITERATES the value."""
    with pytest.raises(BuildProfileError, match="one entry per character"):
        _parse_profile(_profile_with_config({"claude_code.permissions.deny": SETUP_PATCH}))


@pytest.mark.parametrize("key", ["allow", "ask", "deny", "remove_ask", "remove_deny"])
def test_every_permissions_list_key_is_checked(key: str) -> None:
    """All five, not only the two the setup-capability check reads — each one is
    rendered by iterating it, so each one has the same failure mode."""
    with pytest.raises(BuildProfileError, match=f"claude_code.permissions.{key}"):
        _parse_profile(_profile_with_config({f"claude_code.permissions.{key}": "Bash"}))


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"claude_code": {"permissions": {"deny": SETUP_PATCH}}}, id="nested_block"),
        pytest.param(
            {"claude_code.permissions": {"deny": SETUP_PATCH}}, id="dotted_permissions_mapping"
        ),
    ],
)
def test_the_nested_spellings_are_refused_equally(config: dict) -> None:
    """`config_update_fields` takes all three spellings to the same rendered
    leaf, so a check that saw only the flattest one would wave the others
    through to the render."""
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile(_profile_with_config(config))

    message = str(excinfo.value)
    assert "claude_code.permissions.deny" in message
    assert "nested under" in message


def test_a_well_shaped_permission_list_parses() -> None:
    """The refusal is about shape alone — a list of strings is untouched."""
    profile = _parse_profile(
        _profile_with_config(
            {
                "claude_code.permissions.deny": [SETUP_PATCH, "Bash"],
                "claude_code.permissions.remove_deny": [],
                "claude_code.permissions.allow": ["Read"],
            }
        )
    )
    assert profile.config["claude_code.permissions.deny"] == [SETUP_PATCH, "Bash"]


def test_a_config_block_spelling_claude_code_both_ways_is_refused() -> None:
    """The privilege bug this closes.

    `config_update_fields` sets each dotted path verbatim, so a bare
    `claude_code` key applied after a dotted one REPLACES the whole subtree: this
    profile renders a config.yml carrying the deny and NO remove_deny — its
    settings.json denies the tool — while `_profile_setup_patch_capable` unions
    both spellings, reads the lift, and chowns build/config.yml to the agent.
    """
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile(
            _profile_with_config(
                {
                    "claude_code.permissions.remove_deny": [SETUP_PATCH],
                    "claude_code": {"permissions": {"deny": [SETUP_PATCH]}},
                }
            )
        )

    message = str(excinfo.value)
    # Both keys named: an operator must be able to see which two lines collide.
    assert "'claude_code'" in message
    assert "'claude_code.permissions.remove_deny'" in message
    assert "discarded silently" in message


def test_the_mixed_spelling_refusal_names_every_dotted_key() -> None:
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile(
            _profile_with_config(
                {
                    "claude_code.provider": "anthropic",
                    "claude_code.permissions.deny": [SETUP_PATCH],
                    "claude_code": {"default_model": "opus"},
                }
            )
        )

    message = str(excinfo.value)
    assert "'claude_code.permissions.deny'" in message
    assert "'claude_code.provider'" in message


def test_a_middle_split_key_beside_a_dotted_one_is_refused() -> None:
    """The split point the original refusal did not cover.

    `claude_code.permissions:` holding a `deny` list is neither the bare nested
    mapping nor a leaf key, so the old rule (bare key + any dotted key) let it
    through — while `config_update_fields` renders whichever of the two comes
    last, measured. Same hazard, same refusal.
    """
    with pytest.raises(BuildProfileError) as excinfo:
        _parse_profile(
            _profile_with_config(
                {
                    "claude_code.permissions": {"deny": [SETUP_PATCH]},
                    "claude_code.permissions.deny": ["Bash"],
                }
            )
        )

    message = str(excinfo.value)
    assert "'claude_code.permissions'" in message
    assert "'claude_code.permissions.deny'" in message
    assert "discarded silently" in message


def test_sibling_claude_code_keys_are_not_a_collision() -> None:
    """Only a PREFIX pair is ambiguous — two leaves side by side render both."""
    profile = _parse_profile(
        _profile_with_config(
            {
                "claude_code.permissions.deny": [SETUP_PATCH],
                "claude_code.permissions.allow": ["Read"],
            }
        )
    )
    assert profile.config["claude_code.permissions.deny"] == [SETUP_PATCH]


def test_either_claude_code_spelling_alone_is_fine() -> None:
    """Only the MIX is ambiguous. Both spellings remain individually valid —
    the presets write dotted keys, and nothing forbids a nested block."""
    dotted = _parse_profile(_profile_with_config({"claude_code.provider": "anthropic"}))
    assert dotted.config["claude_code.provider"] == "anthropic"
    nested = _parse_profile(
        _profile_with_config({"claude_code": {"permissions": {"deny": [SETUP_PATCH]}}})
    )
    assert nested.config["claude_code"]["permissions"]["deny"] == [SETUP_PATCH]


@pytest.mark.parametrize(
    "preset",
    [
        "control-assistant",
        "control-assistant-readonly",
        "control-assistant-readwrite",
        "control-assistant-ariel",
        "control-assistant-admin",
        "hello-world",
    ],
)
def test_the_shipped_presets_pass_both_refusals(preset: str) -> None:
    """The refusals must not cost anything the project already ships. Resolved
    through the real path, so an `extends` parent's spelling counts too."""
    profile, _ = bp.resolve_build_profile(None, preset=preset)
    assert profile.name
