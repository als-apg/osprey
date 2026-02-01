"""Tests for IOC backend implementations and runtime loader."""

import sys
import textwrap
from unittest.mock import MagicMock

import pytest

from osprey.generators.ioc_backends import (
    ChainedBackend,
    MockStyleBackend,
    PassthroughBackend,
    _instantiate_backend,
    load_backends_from_config,
)

# =============================================================================
# Test PassthroughBackend
# =============================================================================


class TestPassthroughBackend:
    """Tests for PassthroughBackend."""

    def test_initialize_returns_empty_dict(self):
        """Test initialize returns empty dict."""
        backend = PassthroughBackend()
        pv_defs = [{"name": "PV1", "type": "float"}, {"name": "PV2", "type": "float"}]

        result = backend.initialize(pv_defs)

        assert result == {}

    def test_on_write_returns_empty_dict(self):
        """Test on_write returns empty dict (handles write, no cascading updates)."""
        backend = PassthroughBackend()

        result = backend.on_write("PV1", 42.0)

        assert result == {}

    def test_step_returns_empty_dict(self):
        """Test step returns empty dict (no time evolution)."""
        backend = PassthroughBackend()

        result = backend.step(0.1)

        assert result == {}


# =============================================================================
# Test MockStyleBackend
# =============================================================================


class TestMockStyleBackend:
    """Tests for MockStyleBackend."""

    def test_initialize_sets_values_based_on_pv_names(self):
        """Test initialize sets initial values based on PV name keywords."""
        backend = MockStyleBackend()
        pv_defs = [
            {"name": "BPM:X", "type": "float"},
            {"name": "DCCT:CURRENT", "type": "float"},
            {"name": "MAG:VOLTAGE", "type": "float"},
            {"name": "VAC:PRESSURE", "type": "float"},
            {"name": "TEMP:READING", "type": "float"},
        ]

        result = backend.initialize(pv_defs)

        # BPM should be near zero (random offset)
        assert "BPM:X" in result
        assert -0.2 < result["BPM:X"] < 0.2

        # DCCT should be 500 (beam current)
        assert result["DCCT:CURRENT"] == 500.0

        # Voltage should be 5000
        assert result["MAG:VOLTAGE"] == 5000.0

        # Pressure should be 1e-9
        assert result["VAC:PRESSURE"] == 1e-9

        # Temperature should be 25
        assert result["TEMP:READING"] == 25.0

    def test_initialize_default_value(self):
        """Test initialize uses default value for unknown PV names."""
        backend = MockStyleBackend()
        pv_defs = [{"name": "UNKNOWN:PV", "type": "float"}]

        result = backend.initialize(pv_defs)

        assert result["UNKNOWN:PV"] == 100.0

    def test_on_write_updates_paired_readback(self):
        """Test on_write updates paired readback PV."""
        pairings = {"MAG:SP": "MAG:RB"}
        backend = MockStyleBackend(noise_level=0.0, pairings=pairings)

        result = backend.on_write("MAG:SP", 50.0)

        # With no noise, RB should equal SP
        assert "MAG:RB" in result
        assert result["MAG:RB"] == 50.0

    def test_on_write_with_noise(self):
        """Test on_write applies noise to readback."""
        pairings = {"MAG:SP": "MAG:RB"}
        backend = MockStyleBackend(noise_level=0.1, pairings=pairings)

        # Run multiple times to check noise is applied
        values = []
        for _ in range(10):
            result = backend.on_write("MAG:SP", 100.0)
            values.append(result["MAG:RB"])

        # Values should not all be identical (noise applied)
        # At least some variance expected with 10% noise
        assert not all(v == values[0] for v in values), "Expected noise variance"

    def test_on_write_no_pairing(self):
        """Test on_write returns empty dict for unpaired PV."""
        backend = MockStyleBackend(pairings={})

        result = backend.on_write("MAG:SP", 50.0)

        assert result == {}

    def test_step_evolves_bpm_values(self):
        """Test step evolves BPM values over time."""
        backend = MockStyleBackend()
        backend.initialize([{"name": "BPM:X", "type": "float"}])

        initial = backend._state["BPM:X"]
        updates = backend.step(1.0)

        # BPM should have drifted
        assert "BPM:X" in updates
        assert updates["BPM:X"] != initial

    def test_step_evolves_pressure_values(self):
        """Test step evolves pressure values over time."""
        backend = MockStyleBackend()
        backend.initialize([{"name": "VAC:PRESSURE", "type": "float"}])

        initial = backend._state["VAC:PRESSURE"]
        updates = backend.step(1.0)

        # Pressure should be updated (may increase or decrease due to noise)
        assert "VAC:PRESSURE" in updates
        # Value should be in reasonable range (not wildly different)
        assert updates["VAC:PRESSURE"] > 0
        assert updates["VAC:PRESSURE"] < initial * 10  # Not more than 10x

    def test_reproducible_rng_per_pv(self):
        """Test that each PV has reproducible random behavior."""
        backend1 = MockStyleBackend()
        backend2 = MockStyleBackend()

        pv_defs = [{"name": "BPM:X", "type": "float"}]

        result1 = backend1.initialize(pv_defs)
        result2 = backend2.initialize(pv_defs)

        # Same PV name should produce same initial value (reproducible RNG)
        assert result1["BPM:X"] == result2["BPM:X"]


# =============================================================================
# Test ChainedBackend
# =============================================================================


class TestChainedBackend:
    """Tests for ChainedBackend."""

    def test_initialize_merges_results(self):
        """Test initialize merges results from all backends (last wins)."""
        backend1 = MagicMock()
        backend1.initialize.return_value = {"PV1": 1.0, "PV2": 2.0}

        backend2 = MagicMock()
        backend2.initialize.return_value = {"PV2": 20.0, "PV3": 3.0}

        chained = ChainedBackend([backend1, backend2])
        pv_defs = [{"name": "PV1"}, {"name": "PV2"}, {"name": "PV3"}]

        result = chained.initialize(pv_defs)

        # Last backend wins on conflicts
        assert result == {"PV1": 1.0, "PV2": 20.0, "PV3": 3.0}

    def test_on_write_last_to_first_first_non_none_wins(self):
        """Test on_write iterates last to first, first non-None wins."""
        backend1 = MagicMock()
        backend1.on_write.return_value = {"base": True}

        backend2 = MagicMock()
        backend2.on_write.return_value = None  # Delegates

        backend3 = MagicMock()
        backend3.on_write.return_value = {"overlay": True}

        chained = ChainedBackend([backend1, backend2, backend3])

        result = chained.on_write("PV1", 42.0)

        # backend3 (last) should be checked first and returns non-None
        assert result == {"overlay": True}
        backend3.on_write.assert_called_once_with("PV1", 42.0)
        backend2.on_write.assert_not_called()  # Not reached
        backend1.on_write.assert_not_called()  # Not reached

    def test_on_write_delegates_to_earlier_backend(self):
        """Test on_write delegates when later backends return None."""
        backend1 = MagicMock()
        backend1.on_write.return_value = {"base": True}

        backend2 = MagicMock()
        backend2.on_write.return_value = None  # Delegates

        chained = ChainedBackend([backend1, backend2])

        result = chained.on_write("PV1", 42.0)

        # backend2 returns None, so backend1's result is used
        assert result == {"base": True}
        backend2.on_write.assert_called_once()
        backend1.on_write.assert_called_once()

    def test_on_write_returns_empty_if_all_delegate(self):
        """Test on_write returns empty dict if all backends return None."""
        backend1 = MagicMock()
        backend1.on_write.return_value = None

        backend2 = MagicMock()
        backend2.on_write.return_value = None

        chained = ChainedBackend([backend1, backend2])

        result = chained.on_write("PV1", 42.0)

        assert result == {}

    def test_step_merges_all_results(self):
        """Test step runs all backends and merges results (last wins)."""
        backend1 = MagicMock()
        backend1.step.return_value = {"PV1": 1.0, "PV2": 2.0}

        backend2 = MagicMock()
        backend2.step.return_value = {"PV2": 20.0, "PV3": 3.0}

        chained = ChainedBackend([backend1, backend2])

        result = chained.step(0.1)

        # Last backend wins on conflicts
        assert result == {"PV1": 1.0, "PV2": 20.0, "PV3": 3.0}
        backend1.step.assert_called_once_with(0.1)
        backend2.step.assert_called_once_with(0.1)


# =============================================================================
# Test _instantiate_backend
# =============================================================================


class TestInstantiateBackend:
    """Tests for _instantiate_backend helper function."""

    def test_instantiate_mock_style(self):
        """Test instantiating mock_style backend."""
        config = {"type": "mock_style", "noise_level": 0.05}
        pairings = {"SP": "RB"}

        backend = _instantiate_backend(config, pairings)

        assert isinstance(backend, MockStyleBackend)
        assert backend.noise_level == 0.05
        assert backend.pairings == pairings

    def test_instantiate_mock_style_defaults(self):
        """Test mock_style uses default noise_level."""
        config = {"type": "mock_style"}

        backend = _instantiate_backend(config, None)

        assert isinstance(backend, MockStyleBackend)
        assert backend.noise_level == 0.01

    def test_instantiate_passthrough(self):
        """Test instantiating passthrough backend."""
        config = {"type": "passthrough"}

        backend = _instantiate_backend(config, None)

        assert isinstance(backend, PassthroughBackend)

    def test_instantiate_custom_backend(self, tmp_path):
        """Test instantiating custom backend via dynamic import."""
        # Create a temporary module with a custom backend
        module_code = textwrap.dedent("""
            class CustomBackend:
                def __init__(self, tau=1.0):
                    self.tau = tau

                def initialize(self, pv_defs):
                    return {}

                def on_write(self, pv_name, value):
                    return {}

                def step(self, dt):
                    return {}
        """)

        module_file = tmp_path / "custom_backend.py"
        module_file.write_text(module_code)

        # Add to sys.path temporarily
        sys.path.insert(0, str(tmp_path))
        try:
            config = {
                "module_path": "custom_backend",
                "class_name": "CustomBackend",
                "params": {"tau": 2.5},
            }

            backend = _instantiate_backend(config, None)

            assert backend.tau == 2.5
        finally:
            sys.path.remove(str(tmp_path))
            # Clean up imported module
            if "custom_backend" in sys.modules:
                del sys.modules["custom_backend"]

    def test_instantiate_custom_without_params(self, tmp_path):
        """Test instantiating custom backend without params."""
        module_code = textwrap.dedent("""
            class SimpleBackend:
                def __init__(self):
                    self.initialized = True

                def initialize(self, pv_defs):
                    return {}

                def on_write(self, pv_name, value):
                    return {}

                def step(self, dt):
                    return {}
        """)

        module_file = tmp_path / "simple_backend.py"
        module_file.write_text(module_code)

        sys.path.insert(0, str(tmp_path))
        try:
            config = {
                "module_path": "simple_backend",
                "class_name": "SimpleBackend",
            }

            backend = _instantiate_backend(config, None)

            assert backend.initialized is True
        finally:
            sys.path.remove(str(tmp_path))
            if "simple_backend" in sys.modules:
                del sys.modules["simple_backend"]

    def test_instantiate_unknown_config_raises(self):
        """Test unknown config raises ValueError."""
        config = {"unknown": "config"}

        with pytest.raises(ValueError, match="Unknown backend config"):
            _instantiate_backend(config, None)

    def test_instantiate_from_file_path(self, tmp_path):
        """Test instantiating custom backend via file_path."""
        module_code = textwrap.dedent("""
            class FilePathBackend:
                def __init__(self, tau=1.0):
                    self.tau = tau

                def initialize(self, pv_defs):
                    return {}

                def on_write(self, pv_name, value):
                    return {}

                def step(self, dt):
                    return {}
        """)

        backend_file = tmp_path / "my_backend.py"
        backend_file.write_text(module_code)

        config = {
            "file_path": "my_backend.py",
            "class_name": "FilePathBackend",
            "params": {"tau": 3.5},
        }

        backend = _instantiate_backend(config, None, config_dir=tmp_path)

        assert backend.tau == 3.5

        # Cleanup
        if "my_backend" in sys.modules:
            del sys.modules["my_backend"]

    def test_instantiate_from_file_path_absolute(self, tmp_path):
        """Test instantiating custom backend via absolute file_path."""
        module_code = textwrap.dedent("""
            class AbsolutePathBackend:
                def __init__(self):
                    self.loaded = True

                def initialize(self, pv_defs):
                    return {}

                def on_write(self, pv_name, value):
                    return {}

                def step(self, dt):
                    return {}
        """)

        backend_file = tmp_path / "absolute_backend.py"
        backend_file.write_text(module_code)

        config = {
            "file_path": str(backend_file),  # Absolute path
            "class_name": "AbsolutePathBackend",
        }

        # config_dir doesn't matter for absolute paths
        backend = _instantiate_backend(config, None, config_dir=None)

        assert backend.loaded is True

        # Cleanup
        if "absolute_backend" in sys.modules:
            del sys.modules["absolute_backend"]

    def test_instantiate_from_file_path_not_found(self, tmp_path):
        """Test file_path raises FileNotFoundError for missing file."""
        config = {
            "file_path": "nonexistent.py",
            "class_name": "SomeBackend",
        }

        with pytest.raises(FileNotFoundError, match="Backend file not found"):
            _instantiate_backend(config, None, config_dir=tmp_path)


# =============================================================================
# Test load_backends_from_config
# =============================================================================


class TestLoadBackendsFromConfig:
    """Tests for load_backends_from_config function."""

    def test_empty_config_returns_mock_style(self):
        """Test empty config defaults to mock_style backend."""
        backend = load_backends_from_config({})

        assert isinstance(backend, MockStyleBackend)

    def test_single_backend_returns_directly(self):
        """Test single backend returns directly (not wrapped in ChainedBackend)."""
        config = {
            "base": {"type": "passthrough"},
            "overlays": [],
        }

        backend = load_backends_from_config(config)

        assert isinstance(backend, PassthroughBackend)

    def test_multiple_backends_returns_chained(self):
        """Test multiple backends returns ChainedBackend."""
        config = {
            "base": {"type": "mock_style", "noise_level": 0.01},
            "overlays": [{"type": "passthrough"}],
        }

        backend = load_backends_from_config(config)

        assert isinstance(backend, ChainedBackend)
        assert len(backend.backends) == 2
        assert isinstance(backend.backends[0], MockStyleBackend)
        assert isinstance(backend.backends[1], PassthroughBackend)

    def test_pairings_passed_to_mock_style(self):
        """Test pairings are passed to MockStyleBackend."""
        config = {"base": {"type": "mock_style"}}
        pairings = {"SP1": "RB1"}

        backend = load_backends_from_config(config, pairings)

        assert isinstance(backend, MockStyleBackend)
        assert backend.pairings == pairings

    def test_multiple_overlays(self):
        """Test multiple overlays are chained correctly."""
        config = {
            "base": {"type": "mock_style"},
            "overlays": [
                {"type": "passthrough"},
                {"type": "passthrough"},
            ],
        }

        backend = load_backends_from_config(config)

        assert isinstance(backend, ChainedBackend)
        assert len(backend.backends) == 3


# =============================================================================
# Test SimulationBackend Protocol Compliance
# =============================================================================


class TestSimulationBackendProtocol:
    """Tests that backend classes implement SimulationBackend protocol."""

    def test_passthrough_implements_protocol(self):
        """Test PassthroughBackend implements SimulationBackend."""
        from osprey.generators.backend_protocol import SimulationBackend

        backend = PassthroughBackend()
        assert isinstance(backend, SimulationBackend)

    def test_mock_style_implements_protocol(self):
        """Test MockStyleBackend implements SimulationBackend."""
        from osprey.generators.backend_protocol import SimulationBackend

        backend = MockStyleBackend()
        assert isinstance(backend, SimulationBackend)

    def test_chained_implements_protocol(self):
        """Test ChainedBackend implements SimulationBackend."""
        from osprey.generators.backend_protocol import SimulationBackend

        backend = ChainedBackend([PassthroughBackend()])
        assert isinstance(backend, SimulationBackend)


# =============================================================================
# Integration Tests
# =============================================================================


class TestBackendIntegration:
    """Integration tests for backend behaviors."""

    def test_mock_style_sp_rb_workflow(self):
        """Test complete SP->RB workflow with MockStyleBackend."""
        pairings = {
            "MAG:Q1:CURRENT:SP": "MAG:Q1:CURRENT:RB",
            "MAG:Q2:CURRENT:SP": "MAG:Q2:CURRENT:RB",
        }
        backend = MockStyleBackend(noise_level=0.0, pairings=pairings)

        # Initialize
        pv_defs = [
            {"name": "MAG:Q1:CURRENT:SP", "type": "float"},
            {"name": "MAG:Q1:CURRENT:RB", "type": "float"},
            {"name": "MAG:Q2:CURRENT:SP", "type": "float"},
            {"name": "MAG:Q2:CURRENT:RB", "type": "float"},
        ]
        initial = backend.initialize(pv_defs)

        # All PVs should have initial values (current = 150)
        assert initial["MAG:Q1:CURRENT:SP"] == 150.0
        assert initial["MAG:Q2:CURRENT:SP"] == 150.0

        # Write to SP, check RB updates
        updates = backend.on_write("MAG:Q1:CURRENT:SP", 200.0)
        assert updates["MAG:Q1:CURRENT:RB"] == 200.0

        # Q2 should not be affected
        assert "MAG:Q2:CURRENT:RB" not in updates

    def test_chained_backend_delegation(self):
        """Test ChainedBackend delegation behavior."""
        # Base backend handles all writes
        base = MockStyleBackend(noise_level=0.0, pairings={"SP": "RB"})

        # Overlay that only handles specific PV
        class SelectiveBackend:
            def __init__(self):
                self.handled_pvs = {"SPECIAL:SP"}

            def initialize(self, pv_defs):
                return {}

            def on_write(self, pv_name, value):
                if pv_name in self.handled_pvs:
                    return {"SPECIAL:SP": value * 2, "SPECIAL:RB": value * 2}
                return None  # Delegate

            def step(self, dt):
                return {}

        overlay = SelectiveBackend()
        chained = ChainedBackend([base, overlay])

        # Regular write should be handled by base
        result = chained.on_write("SP", 100.0)
        assert result == {"RB": 100.0}

        # Special write should be handled by overlay
        result = chained.on_write("SPECIAL:SP", 50.0)
        assert result == {"SPECIAL:SP": 100.0, "SPECIAL:RB": 100.0}
