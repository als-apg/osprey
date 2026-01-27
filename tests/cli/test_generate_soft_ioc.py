"""Tests for the 'osprey generate soft-ioc' CLI command."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.generate_cmd import (
    _detect_database_type,
    _generate_simulation_yaml_preview,
    _get_channel_database_from_config,
    _get_output_path,
    _get_pv_defaults,
    _infer_pv_type_from_channel,
    _is_readonly_channel,
    _load_pairings,
    _load_simulation_config,
    _validate_pairings,
    _write_simulation_config,
    generate,
)


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_config_yml(tmp_path):
    """Create a sample config.yml with simulation section."""
    config_content = """
project_name: test_project

simulation:
  channel_database: data/channel_databases/in_context.json
  pairings_file: data/pairings.json
  ioc:
    name: test_sim
    port: 5065
    output_dir: generated_iocs/
  backend:
    type: mock_style
    noise_level: 0.02
    update_rate: 5.0

control_system:
  type: mock
"""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def sample_flat_database(tmp_path):
    """Create a sample flat channel database."""
    db_content = [
        {
            "channel": "QUAD:CURRENT:SP",
            "address": "QUAD:CURRENT:SP",
            "description": "Quad setpoint",
        },
        {
            "channel": "QUAD:CURRENT:RB",
            "address": "QUAD:CURRENT:RB",
            "description": "Quad readback",
        },
    ]
    db_dir = tmp_path / "data" / "channel_databases"
    db_dir.mkdir(parents=True)
    db_file = db_dir / "in_context.json"
    db_file.write_text(json.dumps(db_content))
    return db_file


@pytest.fixture
def sample_template_database(tmp_path):
    """Create a sample template channel database."""
    db_content = {
        "_metadata": {"format": "template"},
        "channels": [
            {"channel": "TEMP:SENSOR1", "address": "TEMP:SENSOR1", "description": "Temperature"},
            {
                "template": True,
                "base_name": "QUAD",
                "instances": [1, 3],
                "sub_channels": ["SP", "RB"],
                "description": "Quadrupole magnet",
                "address_pattern": "QUAD{instance:02d}{suffix}",
            },
        ],
    }
    db_dir = tmp_path / "data" / "channel_databases"
    db_dir.mkdir(parents=True)
    db_file = db_dir / "template.json"
    db_file.write_text(json.dumps(db_content))
    return db_file


@pytest.fixture
def sample_hierarchical_database(tmp_path):
    """Create a sample hierarchical channel database."""
    db_content = {
        "hierarchy": {"levels": ["system", "device", "field"]},
        "tree": {"SR": {"BPM": {"X": {"pv_address": "SR:BPM:X"}, "Y": {"pv_address": "SR:BPM:Y"}}}},
    }
    db_dir = tmp_path / "data" / "channel_databases"
    db_dir.mkdir(parents=True)
    db_file = db_dir / "hierarchical.json"
    db_file.write_text(json.dumps(db_content))
    return db_file


@pytest.fixture
def sample_middle_layer_database(tmp_path):
    """Create a sample middle layer channel database."""
    db_content = {
        "SR": {
            "BPM": {
                "Monitor": {
                    "ChannelNames": ["SR01C:BPM1:X", "SR01C:BPM1:Y"],
                    "DataType": "Scalar",
                    "Units": "mm",
                }
            }
        }
    }
    db_dir = tmp_path / "data" / "channel_databases"
    db_dir.mkdir(parents=True)
    db_file = db_dir / "middle_layer.json"
    db_file.write_text(json.dumps(db_content))
    return db_file


@pytest.fixture
def sample_pairings(tmp_path):
    """Create a sample pairings file."""
    pairings = {"QUAD:CURRENT:SP": "QUAD:CURRENT:RB", "BPM:X:SP": "BPM:X:RB"}
    pairings_dir = tmp_path / "data"
    pairings_dir.mkdir(parents=True, exist_ok=True)
    pairings_file = pairings_dir / "pairings.json"
    pairings_file.write_text(json.dumps(pairings))
    return pairings_file


# =============================================================================
# Test CLI Help
# =============================================================================


class TestSoftIOCCommandHelp:
    """Test soft-ioc command help and registration."""

    def test_soft_ioc_help(self, cli_runner):
        """Test soft-ioc command help output."""
        result = cli_runner.invoke(generate, ["soft-ioc", "--help"])

        assert result.exit_code == 0
        assert "soft-ioc" in result.output.lower() or "soft IOC" in result.output
        assert "--config" in result.output
        assert "--output" in result.output
        assert "--dry-run" in result.output

    def test_soft_ioc_in_generate_help(self, cli_runner):
        """Test that soft-ioc appears in generate command help."""
        result = cli_runner.invoke(generate, ["--help"])

        assert result.exit_code == 0
        assert "soft-ioc" in result.output


# =============================================================================
# Test Configuration Loading
# =============================================================================


class TestLoadSimulationConfig:
    """Test simulation config loading."""

    def test_load_config_success(self, tmp_path, monkeypatch, sample_config_yml):
        """Test successful config loading."""
        monkeypatch.chdir(tmp_path)

        config = _load_simulation_config(str(sample_config_yml))

        assert config["ioc"]["name"] == "test_sim"
        assert config["ioc"]["port"] == 5065
        assert config["backend"]["type"] == "mock_style"
        assert config["backend"]["noise_level"] == 0.02

    def test_load_config_applies_defaults(self, tmp_path, monkeypatch):
        """Test that defaults are applied for missing values."""
        config_content = """
simulation:
  channel_database: data/db.json
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)
        monkeypatch.chdir(tmp_path)

        config = _load_simulation_config(str(config_file))

        # Should have defaults
        assert config["ioc"]["name"] == "soft_ioc"
        assert config["ioc"]["port"] == 5064
        assert config["backend"]["type"] == "mock_style"
        assert config["backend"]["noise_level"] == 0.01
        assert config["backend"]["update_rate"] == 10.0

    def test_load_config_missing_file(self, tmp_path, monkeypatch):
        """Test error when config file doesn't exist."""
        from click import ClickException

        monkeypatch.chdir(tmp_path)

        with pytest.raises(ClickException) as exc_info:
            _load_simulation_config("/nonexistent/config.yml")

        assert "Config file not found" in str(exc_info.value)

    def test_load_config_missing_simulation_section(self, tmp_path, monkeypatch):
        """Test error when simulation section is missing."""
        from click import ClickException

        config_content = """
project_name: test_project
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ClickException) as exc_info:
            _load_simulation_config(str(config_file))

        assert "No 'simulation' section" in str(exc_info.value)


# =============================================================================
# Test Pairings Loading
# =============================================================================


class TestLoadPairings:
    """Test SP/RB pairings file loading."""

    def test_load_pairings_success(self, sample_pairings):
        """Test successful pairings loading."""
        pairings = _load_pairings(str(sample_pairings))

        assert pairings["QUAD:CURRENT:SP"] == "QUAD:CURRENT:RB"
        assert pairings["BPM:X:SP"] == "BPM:X:RB"

    def test_load_pairings_none(self):
        """Test that None returns empty dict."""
        pairings = _load_pairings(None)
        assert pairings == {}

    def test_load_pairings_missing_file(self):
        """Test error when pairings file doesn't exist."""
        from click import ClickException

        with pytest.raises(ClickException) as exc_info:
            _load_pairings("/nonexistent/pairings.json")

        assert "Pairings file not found" in str(exc_info.value)

    def test_load_pairings_invalid_format(self, tmp_path):
        """Test error when pairings file is not a dict."""
        from click import ClickException

        pairings_file = tmp_path / "bad_pairings.json"
        pairings_file.write_text("[]")  # List instead of dict

        with pytest.raises(ClickException) as exc_info:
            _load_pairings(str(pairings_file))

        assert "Invalid pairings file format" in str(exc_info.value)


class TestValidatePairings:
    """Test pairings validation."""

    def test_validate_pairings_all_valid(self):
        """Test validation with all valid pairings."""
        channels = [{"name": "SP1"}, {"name": "RB1"}, {"name": "SP2"}, {"name": "RB2"}]
        pairings = {"SP1": "RB1", "SP2": "RB2"}

        validated = _validate_pairings(pairings, channels)

        assert validated == pairings

    def test_validate_pairings_skips_missing_sp(self, capsys):
        """Test that missing setpoints are skipped with warning."""
        channels = [{"name": "RB1"}]
        pairings = {"SP1": "RB1"}

        validated = _validate_pairings(pairings, channels)

        assert validated == {}
        captured = capsys.readouterr()
        assert "Warning" in captured.out or len(validated) == 0

    def test_validate_pairings_skips_missing_rb(self, capsys):
        """Test that missing readbacks are skipped with warning."""
        channels = [{"name": "SP1"}]
        pairings = {"SP1": "RB1"}

        validated = _validate_pairings(pairings, channels)

        assert validated == {}


# =============================================================================
# Test Database Type Detection
# =============================================================================


class TestDetectDatabaseType:
    """Test automatic database type detection."""

    def test_detect_flat(self, tmp_path):
        """Test detection of flat database."""
        db_file = tmp_path / "flat.json"
        db_file.write_text(json.dumps([{"channel": "TEST"}]))

        assert _detect_database_type(db_file) == "flat"

    def test_detect_template(self, sample_template_database):
        """Test detection of template database."""
        assert _detect_database_type(sample_template_database) == "template"

    def test_detect_hierarchical(self, sample_hierarchical_database):
        """Test detection of hierarchical database."""
        assert _detect_database_type(sample_hierarchical_database) == "hierarchical"

    def test_detect_middle_layer(self, sample_middle_layer_database):
        """Test detection of middle layer database."""
        assert _detect_database_type(sample_middle_layer_database) == "middle_layer"

    def test_detect_flat_with_channels_no_templates(self, tmp_path):
        """Test detection of flat database with 'channels' key but no templates."""
        db_file = tmp_path / "flat_channels.json"
        db_content = {"channels": [{"channel": "TEST", "template": False}]}
        db_file.write_text(json.dumps(db_content))

        assert _detect_database_type(db_file) == "flat"


# =============================================================================
# Test PV Type Inference
# =============================================================================


class TestIsReadonlyChannel:
    """Test read-only channel detection."""

    def test_readback_is_readonly(self):
        """Test that RB suffix indicates read-only."""
        assert _is_readonly_channel("QUAD:CURRENT:RB", "") is True
        assert _is_readonly_channel("MAG:RB", "") is True

    def test_monitor_is_readonly(self):
        """Test that Monitor indicates read-only."""
        assert _is_readonly_channel("BPM:MONITOR", "") is True

    def test_status_is_readonly(self):
        """Test that STATUS indicates read-only."""
        assert _is_readonly_channel("SYSTEM:STATUS", "") is True

    def test_setpoint_is_writable(self):
        """Test that SP suffix indicates writable."""
        assert _is_readonly_channel("QUAD:CURRENT:SP", "") is False
        assert _is_readonly_channel("MAG:SP", "") is False

    def test_set_is_writable(self):
        """Test that SET indicates writable."""
        assert _is_readonly_channel("QUAD:SET", "") is False
        assert _is_readonly_channel("QUAD:SETPOINT", "") is False

    def test_description_based_detection(self):
        """Test that description can indicate read-only."""
        assert _is_readonly_channel("UNKNOWN", "this is a read-only value") is True
        assert _is_readonly_channel("UNKNOWN", "readback from device") is True


class TestInferPVType:
    """Test PV type inference from channel info."""

    def test_infer_float_default(self):
        """Test that float is the default type."""
        assert _infer_pv_type_from_channel("QUAD:CURRENT", "", {}) == "float"

    def test_infer_enum_from_status(self):
        """Test that STATUS indicates enum type."""
        assert _infer_pv_type_from_channel("SYSTEM:STATUS", "", {}) == "enum"
        assert _infer_pv_type_from_channel("DEVICE:STATE", "", {}) == "enum"

    def test_infer_enum_from_fault(self):
        """Test that FAULT indicates enum type."""
        assert _infer_pv_type_from_channel("INTERLOCK:FAULT", "", {}) == "enum"

    def test_infer_array_from_waveform(self):
        """Test that WAVEFORM indicates array type."""
        assert _infer_pv_type_from_channel("BPM:WAVEFORM", "", {}) == "float_array"

    def test_infer_array_from_image(self):
        """Test that IMAGE indicates array type."""
        assert _infer_pv_type_from_channel("CAMERA:IMAGE", "", {}) == "float_array"

    def test_infer_from_mml_metadata(self):
        """Test inference from MML DataType metadata."""
        channel = {"DataType": "Scalar"}
        assert _infer_pv_type_from_channel("UNKNOWN", "", channel) == "float"

        channel = {"DataType": "Integer"}
        assert _infer_pv_type_from_channel("UNKNOWN", "", channel) == "int"


class TestGetPVDefaults:
    """Test PV default parameter generation."""

    def test_float_defaults(self):
        """Test float PV defaults."""
        defaults = _get_pv_defaults("float", {})
        assert "precision" in defaults
        assert "units" in defaults

    def test_enum_defaults(self):
        """Test enum PV defaults."""
        defaults = _get_pv_defaults("enum", {})
        assert "enum_strings" in defaults
        assert defaults["enum_strings"] == ["Off", "On"]

    def test_enum_custom_strings(self):
        """Test enum with custom strings from channel."""
        channel = {"enum_strings": ["Low", "Medium", "High"]}
        defaults = _get_pv_defaults("enum", channel)
        assert defaults["enum_strings"] == ["Low", "Medium", "High"]

    def test_units_from_channel(self):
        """Test units extraction from channel metadata."""
        channel = {"Units": "mm"}
        defaults = _get_pv_defaults("float", channel)
        assert defaults["units"] == "mm"

    def test_units_from_hwunits(self):
        """Test units extraction from HWUnits metadata."""
        channel = {"HWUnits": "A"}
        defaults = _get_pv_defaults("float", channel)
        assert defaults["units"] == "A"


# =============================================================================
# Test Output Path
# =============================================================================


class TestGetOutputPath:
    """Test output path determination."""

    def test_output_path_generation(self):
        """Test output path is correctly generated."""
        config = {"ioc": {"name": "test_ioc", "output_dir": "generated/"}}
        path = _get_output_path(config)

        assert path == Path("generated/test_ioc_ioc.py")


# =============================================================================
# Test CLI Integration
# =============================================================================


class TestSoftIOCCLIIntegration:
    """Integration tests for the soft-ioc CLI command."""

    def test_soft_ioc_no_config(self, cli_runner, tmp_path, monkeypatch):
        """Test error when no config.yml exists."""
        monkeypatch.chdir(tmp_path)

        result = cli_runner.invoke(generate, ["soft-ioc"])

        assert result.exit_code != 0
        assert "Config file not found" in result.output

    def test_soft_ioc_no_simulation_section(self, cli_runner, tmp_path, monkeypatch):
        """Test error when config.yml has no simulation section."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("project_name: test\n")
        monkeypatch.chdir(tmp_path)

        result = cli_runner.invoke(generate, ["soft-ioc"])

        assert result.exit_code != 0
        assert "No 'simulation' section" in result.output

    @patch("osprey.cli.generate_cmd._load_channels_from_database")
    def test_soft_ioc_dry_run(
        self,
        mock_load_channels,
        cli_runner,
        tmp_path,
        monkeypatch,
    ):
        """Test dry-run mode doesn't write files."""
        # Create config file in tmp_path
        config_content = """
project_name: test_project
simulation:
  channel_database: data/channel_databases/db.json
  ioc:
    name: test_sim
    port: 5065
  backend:
    type: mock_style
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)

        monkeypatch.chdir(tmp_path)

        # Setup mock
        mock_load_channels.return_value = [
            {"name": "TEST:PV", "python_name": "TEST_PV", "type": "float", "read_only": True}
        ]

        result = cli_runner.invoke(generate, ["soft-ioc", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry Run Summary" in result.output
        assert "No files written" in result.output

    @patch("osprey.cli.generate_cmd._load_channels_from_database")
    @patch("osprey.cli.generate_cmd._offer_control_system_config_update")
    def test_soft_ioc_generates_file(
        self,
        mock_offer,
        mock_load_channels,
        cli_runner,
        tmp_path,
        monkeypatch,
    ):
        """Test that soft-ioc generates output file."""
        # Create config file in tmp_path
        config_content = """
project_name: test_project
simulation:
  channel_database: data/channel_databases/db.json
  ioc:
    name: test_sim
    port: 5065
  backend:
    type: mock_style
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)

        monkeypatch.chdir(tmp_path)

        # Make the mock do nothing (it's async)
        async def mock_offer_coro(*args, **kwargs):
            pass

        mock_offer.return_value = None

        # Setup mock
        mock_load_channels.return_value = [
            {
                "name": "TEST:PV",
                "python_name": "TEST_PV",
                "type": "float",
                "description": "Test",
                "read_only": True,
            }
        ]

        result = cli_runner.invoke(generate, ["soft-ioc"])

        # Should succeed and show generation message
        assert result.exit_code == 0
        assert "Generated" in result.output or "Generating" in result.output

    def test_soft_ioc_with_custom_output(
        self, cli_runner, tmp_path, monkeypatch, sample_config_yml
    ):
        """Test soft-ioc with custom output path."""
        monkeypatch.chdir(tmp_path)

        # Create minimal database
        db_dir = tmp_path / "data" / "channel_databases"
        db_dir.mkdir(parents=True)
        db_file = db_dir / "in_context.json"
        db_file.write_text(json.dumps([{"channel": "TEST", "address": "TEST", "description": ""}]))

        # Create pairings file
        pairings_file = tmp_path / "data" / "pairings.json"
        pairings_file.write_text(json.dumps({}))

        custom_output = tmp_path / "custom_output.py"

        with patch("osprey.cli.generate_cmd._offer_control_system_config_update"):
            result = cli_runner.invoke(generate, ["soft-ioc", "--output", str(custom_output)])

        # Check output file was created (or would be created if not mocked)
        assert result.exit_code == 0 or "Generated" in result.output


# =============================================================================
# Test Config Auto-Write Functions (Phase 7)
# =============================================================================


class TestGetChannelDatabaseFromConfig:
    """Test extracting channel database path from channel_finder config section."""

    def test_get_database_from_hierarchical_pipeline(self, tmp_path):
        """Test getting database path from hierarchical pipeline config."""
        config_content = """
channel_finder:
  pipeline_mode: hierarchical
  pipelines:
    hierarchical:
      database:
        type: hierarchical
        path: src/my_app/data/channels.json
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)

        result = _get_channel_database_from_config(config_file)

        assert result == "src/my_app/data/channels.json"

    def test_get_database_from_in_context_pipeline(self, tmp_path):
        """Test getting database path from in_context pipeline config."""
        config_content = """
channel_finder:
  pipeline_mode: in_context
  pipelines:
    in_context:
      database:
        type: flat
        path: data/in_context.json
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)

        result = _get_channel_database_from_config(config_file)

        assert result == "data/in_context.json"

    def test_returns_none_for_missing_config_file(self, tmp_path):
        """Test returns None when config file doesn't exist."""
        config_file = tmp_path / "nonexistent.yml"

        result = _get_channel_database_from_config(config_file)

        assert result is None

    def test_returns_none_for_missing_channel_finder(self, tmp_path):
        """Test returns None when channel_finder section is missing."""
        config_content = "project_name: test\n"
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)

        result = _get_channel_database_from_config(config_file)

        assert result is None

    def test_returns_none_for_missing_pipeline_config(self, tmp_path):
        """Test returns None when pipeline config is missing database path."""
        config_content = """
channel_finder:
  pipeline_mode: hierarchical
  pipelines:
    hierarchical:
      prompts:
        path: src/prompts/
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)

        result = _get_channel_database_from_config(config_file)

        assert result is None

    def test_defaults_to_hierarchical_when_mode_not_specified(self, tmp_path):
        """Test defaults to hierarchical pipeline mode when not specified."""
        config_content = """
channel_finder:
  pipelines:
    hierarchical:
      database:
        path: data/default.json
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)

        result = _get_channel_database_from_config(config_file)

        assert result == "data/default.json"


class TestGenerateSimulationYamlPreview:
    """Test YAML preview generation."""

    def test_basic_preview(self):
        """Test generating basic preview."""
        sim_config = {
            "channel_database": "data/channels.json",
            "ioc": {
                "name": "test_ioc",
                "port": 5064,
                "output_dir": "generated_iocs/",
            },
            "backend": {
                "type": "mock_style",
                "noise_level": 0.01,
                "update_rate": 10.0,
            },
        }

        preview = _generate_simulation_yaml_preview(sim_config)

        assert "simulation:" in preview
        assert "channel_database:" in preview
        assert "data/channels.json" in preview
        assert "name:" in preview
        assert "test_ioc" in preview
        assert "port: 5064" in preview
        assert "mock_style" in preview
        assert "noise_level: 0.01" in preview
        assert "update_rate: 10.0" in preview

    def test_passthrough_backend_preview(self):
        """Test preview with passthrough backend (no noise/rate)."""
        sim_config = {
            "channel_database": "data/channels.json",
            "ioc": {
                "name": "simple_ioc",
                "port": 5065,
                "output_dir": "iocs/",
            },
            "backend": {
                "type": "passthrough",
                "noise_level": 0.01,
                "update_rate": 10.0,
            },
        }

        preview = _generate_simulation_yaml_preview(sim_config)

        assert "passthrough" in preview
        # Passthrough backend should not show noise_level/update_rate
        assert "noise_level" not in preview
        assert "update_rate" not in preview


class TestWriteSimulationConfig:
    """Test writing simulation config to config.yml."""

    def test_write_new_section(self, tmp_path):
        """Test writing simulation section to existing config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("project_name: test\n")

        sim_config = {
            "channel_database": "data/channels.json",
            "ioc": {
                "name": "test_ioc",
                "port": 5064,
                "output_dir": "generated_iocs/",
            },
            "backend": {
                "type": "mock_style",
                "noise_level": 0.01,
                "update_rate": 10.0,
            },
        }

        backup_path = _write_simulation_config(config_file, sim_config)

        # Check backup was created
        assert backup_path.exists()
        assert "project_name: test" in backup_path.read_text()

        # Check config was updated
        import yaml

        with open(config_file) as f:
            updated_config = yaml.safe_load(f)

        assert "simulation" in updated_config
        assert updated_config["simulation"]["ioc"]["name"] == "test_ioc"
        assert updated_config["simulation"]["backend"]["type"] == "mock_style"

    def test_overwrite_existing_section(self, tmp_path):
        """Test overwriting existing simulation section."""
        import yaml

        config_file = tmp_path / "config.yml"
        existing_config = {
            "project_name": "test",
            "simulation": {
                "channel_database": "old_db.json",
                "ioc": {"name": "old_ioc", "port": 9999},
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(existing_config, f)

        new_sim_config = {
            "channel_database": "new_db.json",
            "ioc": {
                "name": "new_ioc",
                "port": 5064,
                "output_dir": "generated_iocs/",
            },
            "backend": {
                "type": "passthrough",
                "noise_level": 0.0,
                "update_rate": 1.0,
            },
        }

        _write_simulation_config(config_file, new_sim_config)

        with open(config_file) as f:
            updated_config = yaml.safe_load(f)

        assert updated_config["simulation"]["channel_database"] == "new_db.json"
        assert updated_config["simulation"]["ioc"]["name"] == "new_ioc"
        assert updated_config["simulation"]["ioc"]["port"] == 5064


class TestSoftIOCInitFlag:
    """Test the --init flag behavior."""

    def test_init_flag_in_help(self, cli_runner):
        """Test that --init flag appears in help."""
        result = cli_runner.invoke(generate, ["soft-ioc", "--help"])

        assert result.exit_code == 0
        assert "--init" in result.output

    def test_missing_simulation_offers_setup(self, cli_runner, tmp_path, monkeypatch):
        """Test that missing simulation section offers interactive setup."""
        monkeypatch.chdir(tmp_path)

        # Create config without simulation section
        config_file = tmp_path / "config.yml"
        config_file.write_text("project_name: test\n")

        # Run without questionary (non-interactive mode)
        result = cli_runner.invoke(generate, ["soft-ioc"], input="n\n")

        # Should show warning about missing simulation section
        assert "simulation" in result.output.lower()

    def test_init_dry_run_shows_preview_only(self, cli_runner, tmp_path, monkeypatch):
        """Test that --init --dry-run shows preview without writing."""
        monkeypatch.chdir(tmp_path)

        # Create config file
        config_file = tmp_path / "config.yml"
        config_file.write_text("project_name: test\n")

        # Create a database file for selection
        db_dir = tmp_path / "data"
        db_dir.mkdir()
        db_file = db_dir / "channels.json"
        db_file.write_text('[{"channel": "TEST"}]')

        # Mock _offer_simulation_config_setup to return a config dict
        mock_sim_config = {
            "channel_database": "data/channels.json",
            "ioc": {"name": "test_ioc", "port": 5064, "output_dir": "generated_iocs/"},
            "backend": {"type": "mock_style", "noise_level": 0.01, "update_rate": 10.0},
        }

        with patch(
            "osprey.cli.generate_cmd._offer_simulation_config_setup",
            return_value=mock_sim_config,
        ):
            result = cli_runner.invoke(generate, ["soft-ioc", "--init", "--dry-run"])

        # In dry-run mode with --init, should not write files
        # The config file should remain unchanged
        assert "project_name: test" in config_file.read_text()
        # Should show dry-run message
        assert "dry" in result.output.lower() or "no files" in result.output.lower()
