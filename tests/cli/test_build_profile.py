"""Tests for the ``bluesky_panels:`` and ``environment:`` blocks of the
build-profile schema.

Covers the :class:`BlueskyPanelsConfig` dataclass, the ``BuildProfile.validate``
exemption that lets the non-builtin scan-panel web_panels ids
(``plan``, ``results``) validate without a
pre-existing ``web.panels.<id>.url`` when a ``bluesky_panels`` block is present
(their urls are derived post-build by ``_inject_bluesky_panels``), and a
regression guard that the shipped control-assistant preset/profile still
validates — the gate task 3.3 (tutorial-config) re-runs after adding the
scan panels to that preset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import build_profile as bp
from osprey.cli.build_cmd import build
from osprey.cli.build_profile import (
    BlueskyPanelsConfig,
    BuildProfile,
    EnvironmentConfig,
    _parse_profile,
    load_profile,
)
from osprey.errors import BuildProfileError


def test_no_bluesky_panels_validates(tmp_path: Path) -> None:
    """A profile with no bluesky_panels block validates without raising."""
    BuildProfile(name="x").validate(tmp_path)


def test_bluesky_panels_ids_validate_without_url_when_bluesky_panels_present(
    tmp_path: Path,
) -> None:
    """The scan-panel ids need no manual web.panels.<id>.url override
    when a bluesky_panels block is present — the urls are derived post-build by
    _inject_bluesky_panels, which runs after this validator."""
    profile = BuildProfile(
        name="x",
        web_panels=["plan", "results"],
        bluesky_panels=BlueskyPanelsConfig(),
    )
    profile.validate(tmp_path)  # must not raise


def test_bluesky_panels_ids_without_bluesky_panels_still_require_url(tmp_path: Path) -> None:
    """The escape hatch is narrow: the scan-panel ids with no bluesky_panels
    block and no url override are still rejected (nothing would derive their
    URL)."""
    profile = BuildProfile(name="x", web_panels=["plan"])
    with pytest.raises(BuildProfileError, match="plan"):
        profile.validate(tmp_path)


def test_unbacked_custom_panel_still_requires_url_even_with_bluesky_panels(
    tmp_path: Path,
) -> None:
    """The bluesky_panels escape hatch applies only to the three known ids — any
    other url-less custom panel is still rejected even when a bluesky_panels
    block is present."""
    profile = BuildProfile(
        name="x",
        web_panels=["grafana"],
        bluesky_panels=BlueskyPanelsConfig(),
    )
    with pytest.raises(BuildProfileError, match="grafana"):
        profile.validate(tmp_path)


def test_bluesky_panels_port_overflow_raises(tmp_path: Path) -> None:
    """An out-of-range bluesky_panels.port fails validation."""
    profile = BuildProfile(name="x", bluesky_panels=BlueskyPanelsConfig(port=70000))
    with pytest.raises(BuildProfileError, match="bluesky_panels.port"):
        profile.validate(tmp_path)


def test_bluesky_panels_default_port() -> None:
    """BlueskyPanelsConfig defaults to port 8095, matching the compose template
    and the sidecar's default uvicorn bind."""
    assert BlueskyPanelsConfig().port == 8095


def test_bluesky_panels_not_a_mapping_raises() -> None:
    """A non-mapping 'bluesky_panels' block raises during parsing."""
    with pytest.raises(BuildProfileError, match="bluesky_panels"):
        _parse_profile({"name": "x", "bluesky_panels": "not-a-mapping"})


def test_bluesky_panels_is_known_key() -> None:
    """'bluesky_panels' is a recognized top-level profile key (no unknown-key warning)."""
    assert "bluesky_panels" in bp._KNOWN_PROFILE_KEYS


def test_bluesky_panels_parse_round_trip() -> None:
    """A bluesky_panels block parses its port field through _parse_profile."""
    raw = {"name": "x", "bluesky_panels": {"port": 9100}}
    profile = _parse_profile(raw)
    assert profile.bluesky_panels is not None
    assert profile.bluesky_panels.port == 9100


def test_bluesky_panels_parse_defaults_when_empty_mapping() -> None:
    """An empty bluesky_panels mapping (`bluesky_panels: {}`) parses to defaults."""
    raw = {"name": "x", "bluesky_panels": {}}
    profile = _parse_profile(raw)
    assert profile.bluesky_panels is not None
    assert profile.bluesky_panels.port == 8095


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
    adding the bluesky_panels block + the scan-panel web_panels ids to
    this preset — it must keep validating once that wiring lands.
    """
    presets_dir = bp._presets_dir()
    raw = yaml.safe_load((presets_dir / "control-assistant.yml").read_text(encoding="utf-8"))
    profile = _parse_profile(raw)

    profile.validate(presets_dir)  # raises BuildProfileError on any issue


# ── Task 3.3: turn-key VA-backed scan stack render ───────────────────────────
#
# The control-assistant preset now bakes the bluesky/virtual_accelerator/
# bluesky_panels injector blocks in directly (no --set/--override flags needed),
# so `osprey build` on the bare preset renders the full scan stack + the
# three scan panels turn-key. These tests build the preset in-process
# (CliRunner, --skip-deps --skip-lifecycle -- Docker-free, mirroring
# tests/cli/test_va_default_config.py's scaffolded_project fixture and
# tests/e2e/_orm_stack.py's build_via_cli_runner) and assert on the rendered
# project's config.yml.


@pytest.fixture(scope="module")
def turnkey_scan_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the bare control-assistant preset (no overrides) into a tmp dir.

    Module-scoped: the build is the slow part (template render + service
    template copies) and every test in this section only reads the resulting
    config.yml, so one build is shared across assertions.
    """
    tmp_path = tmp_path_factory.mktemp("turnkey-scan")
    runner = CliRunner()
    result = runner.invoke(
        build,
        [
            "turnkey-scan",
            "--preset",
            "control-assistant",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    project_dir = tmp_path / "turnkey-scan"
    assert (project_dir / "config.yml").exists()
    return project_dir


@pytest.fixture(scope="module")
def turnkey_scan_config(turnkey_scan_project: Path) -> dict:
    return yaml.safe_load((turnkey_scan_project / "config.yml").read_text(encoding="utf-8"))


class TestControlAssistantTurnkeyScanServices:
    """The three injected services render with the preset's baked-in ports."""

    def test_control_assistant_bluesky_service_rendered(self, turnkey_scan_config: dict) -> None:
        bluesky = turnkey_scan_config["services"]["bluesky"]
        assert bluesky["port"] == 8090
        assert bluesky["tiled_enabled"] is True
        assert bluesky["tiled_port"] == 8091

    def test_control_assistant_virtual_accelerator_service_rendered(
        self, turnkey_scan_config: dict
    ) -> None:
        va = turnkey_scan_config["services"]["virtual_accelerator"]
        assert va["port"] == 5064

    def test_control_assistant_bluesky_panels_service_rendered(
        self, turnkey_scan_config: dict
    ) -> None:
        bluesky_panels = turnkey_scan_config["services"]["bluesky_panels"]
        assert bluesky_panels["port"] == 8095

    def test_control_assistant_deployed_services_includes_all_three(
        self, turnkey_scan_config: dict
    ) -> None:
        deployed = turnkey_scan_config["deployed_services"]
        assert "bluesky" in deployed
        assert "virtual_accelerator" in deployed
        assert "bluesky_panels" in deployed


class TestControlAssistantTurnkeyScanPanels:
    """The three scan-panel web.panels entries are registered with a url."""

    def test_control_assistant_scan_plan_panel(self, turnkey_scan_config: dict) -> None:
        panel = turnkey_scan_config["web"]["panels"]["plan"]
        assert panel["path"] == "/plan/"
        assert panel["url"]

    def test_control_assistant_scan_results_panel(self, turnkey_scan_config: dict) -> None:
        panel = turnkey_scan_config["web"]["panels"]["results"]
        assert panel["path"] == "/results/"
        assert panel["url"]


class TestControlAssistantTurnkeyScanControlSystem:
    """The preset's config overrides land: mock-by-default + subprocess execution.

    The VA soft-IOC ships and is deployed as part of the turn-key scan stack,
    but control_system.type stays "mock" so a fresh tutorial project remains
    safe out of the box -- flipping the one config line to
    "virtual_accelerator" is what engages the deployed VA end-to-end (covered
    by tests/cli/test_va_default_config.py).
    """

    def test_control_assistant_control_system_type_is_mock(self, turnkey_scan_config: dict) -> None:
        assert turnkey_scan_config["control_system"]["type"] == "mock"

    def test_control_assistant_execution_method_is_subprocess(
        self, turnkey_scan_config: dict
    ) -> None:
        """Agent Python runs as a host subprocess — the preset pins nothing, so
        this is the template default the built project inherits."""
        assert turnkey_scan_config["execution"]["execution_method"] == "subprocess"

    def test_control_assistant_scan_mcp_server_enabled(self, turnkey_scan_config: dict) -> None:
        assert turnkey_scan_config["claude_code"]["servers"]["bluesky"]["enabled"] is True


def test_control_assistant_turnkey_scan_preset_validates(turnkey_scan_project: Path) -> None:
    """BuildProfile.validate() passes for the preset as shipped (bare, no
    overrides) -- the three non-builtin scan-panel ids are accepted because
    the preset's own bluesky_panels block is present."""
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
        a silently ignored 'package:' would leave the environment incomplete."""
        with pytest.raises(BuildProfileError, match="Unknown environment key"):
            _parse_profile({"name": "x", "environment": {"package": ["numpy"]}})

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
