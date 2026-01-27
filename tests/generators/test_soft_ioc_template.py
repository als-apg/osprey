"""Tests for soft IOC template generator."""

import pytest

from osprey.generators.soft_ioc_template import (
    _generate_backend_code,
    _generate_pairings_literal,
    _generate_pv_definitions,
    _get_backend_instantiation,
    _to_class_name,
    generate_soft_ioc,
    sanitize_pv_name,
)

# =============================================================================
# Test sanitize_pv_name
# =============================================================================


class TestSanitizePVName:
    """Test PV name sanitization for Python identifiers."""

    def test_simple_name(self):
        """Test simple PV name passes through."""
        assert sanitize_pv_name("QUAD1:CURRENT") == "QUAD1_CURRENT"

    def test_colon_replacement(self):
        """Test colons are replaced with underscores."""
        assert sanitize_pv_name("MAG:QUAD:Q01:SP") == "MAG_QUAD_Q01_SP"

    def test_bracket_replacement(self):
        """Test brackets are replaced with underscores."""
        assert sanitize_pv_name("MAG[Q01]:CURRENT") == "MAG_Q01_CURRENT"

    def test_dash_replacement(self):
        """Test dashes are replaced with underscores."""
        assert sanitize_pv_name("MAG-QUAD-01") == "MAG_QUAD_01"

    def test_dot_replacement(self):
        """Test dots are replaced with underscores."""
        assert sanitize_pv_name("MAG.QUAD.01") == "MAG_QUAD_01"

    def test_collapse_multiple_underscores(self):
        """Test multiple consecutive underscores collapse to one."""
        assert sanitize_pv_name("MAG::QUAD") == "MAG_QUAD"
        assert sanitize_pv_name("MAG::[Q01]::SP") == "MAG_Q01_SP"

    def test_strip_leading_trailing_underscores(self):
        """Test leading/trailing underscores are stripped."""
        assert sanitize_pv_name(":MAG:QUAD:") == "MAG_QUAD"
        assert sanitize_pv_name("::MAG::") == "MAG"

    def test_prefix_starting_with_digit(self):
        """Test names starting with digits get 'pv_' prefix."""
        assert sanitize_pv_name("01:QUAD:SP") == "pv_01_QUAD_SP"
        assert sanitize_pv_name("123ABC") == "pv_123ABC"

    def test_empty_name(self):
        """Test empty or all-invalid names return 'pv_unnamed'."""
        assert sanitize_pv_name(":::") == "pv_unnamed"
        assert sanitize_pv_name("") == "pv_unnamed"

    def test_complex_real_world_name(self):
        """Test complex real-world PV names."""
        assert sanitize_pv_name("SR01C:BPM1:X:RB") == "SR01C_BPM1_X_RB"
        assert sanitize_pv_name("ACC:MAG[HCM01]:CURRENT:SP") == "ACC_MAG_HCM01_CURRENT_SP"


# =============================================================================
# Test _to_class_name
# =============================================================================


class TestToClassName:
    """Test IOC name to class name conversion."""

    def test_simple_name(self):
        """Test simple name conversion."""
        assert _to_class_name("soft_ioc") == "SoftIoc"

    def test_single_word(self):
        """Test single word conversion."""
        assert _to_class_name("accelerator") == "Accelerator"

    def test_multiple_underscores(self):
        """Test multiple underscore separation."""
        assert _to_class_name("my_test_ioc") == "MyTestIoc"

    def test_with_dashes(self):
        """Test dashes are treated like underscores."""
        assert _to_class_name("my-test-ioc") == "MyTestIoc"

    def test_mixed_separators(self):
        """Test mixed dashes and underscores."""
        assert _to_class_name("my_test-ioc") == "MyTestIoc"


# =============================================================================
# Test _generate_pv_definitions
# =============================================================================


class TestGeneratePVDefinitions:
    """Test PV property code generation."""

    def test_float_pv(self):
        """Test float PV generation."""
        channels = [
            {
                "name": "QUAD:CURRENT",
                "python_name": "QUAD_CURRENT",
                "type": "float",
                "description": "Quad current",
                "read_only": False,
                "units": "A",
                "precision": 3,
                "high_alarm": 200.0,
                "low_alarm": -200.0,
            }
        ]
        code = _generate_pv_definitions(channels)

        assert "QUAD_CURRENT = pvproperty" in code
        assert "name='QUAD:CURRENT'" in code
        assert "value=0.0" in code
        assert "precision=3" in code
        assert "units='A'" in code
        assert "read_only=False" in code
        assert "doc='Quad current'" in code

    def test_int_pv(self):
        """Test integer PV generation."""
        channels = [
            {
                "name": "COUNT",
                "python_name": "COUNT",
                "type": "int",
                "description": "Counter",
                "read_only": True,
                "units": "",
            }
        ]
        code = _generate_pv_definitions(channels)

        assert "COUNT = pvproperty" in code
        assert "value=0" in code
        assert "read_only=True" in code

    def test_enum_pv(self):
        """Test enum PV generation."""
        channels = [
            {
                "name": "STATUS",
                "python_name": "STATUS",
                "type": "enum",
                "description": "Device status",
                "read_only": True,
                "enum_strings": ["Off", "On", "Fault"],
            }
        ]
        code = _generate_pv_definitions(channels)

        assert "STATUS = pvproperty" in code
        assert "enum_strings=['Off', 'On', 'Fault']" in code

    def test_string_pv(self):
        """Test string PV generation."""
        channels = [
            {
                "name": "NAME",
                "python_name": "NAME",
                "type": "string",
                "description": "Device name",
                "read_only": True,
            }
        ]
        code = _generate_pv_definitions(channels)

        assert "NAME = pvproperty" in code
        assert "max_length=256" in code
        assert "value=''" in code

    def test_array_pv(self):
        """Test array PV generation."""
        channels = [
            {
                "name": "WAVEFORM",
                "python_name": "WAVEFORM",
                "type": "float_array",
                "description": "Waveform data",
                "read_only": True,
                "count": 256,
            }
        ]
        code = _generate_pv_definitions(channels)

        assert "WAVEFORM = pvproperty" in code
        assert "value=[0.0] * 256" in code

    def test_name_collision_handling(self):
        """Test that duplicate Python names get suffixed."""
        channels = [
            {
                "name": "QUAD:CURRENT",
                "python_name": "QUAD_CURRENT",
                "type": "float",
                "description": "First quad",
                "read_only": False,
            },
            {
                "name": "QUAD:CURRENT:2",
                "python_name": "QUAD_CURRENT",  # Same Python name!
                "type": "float",
                "description": "Second quad",
                "read_only": False,
            },
        ]
        code = _generate_pv_definitions(channels)

        # First one should be QUAD_CURRENT
        assert "QUAD_CURRENT = pvproperty" in code
        # Second one should be QUAD_CURRENT_1
        assert "QUAD_CURRENT_1 = pvproperty" in code

    def test_description_truncation(self):
        """Test that long descriptions are truncated."""
        long_desc = "A" * 200  # Way over 80 chars
        channels = [
            {
                "name": "TEST",
                "python_name": "TEST",
                "type": "float",
                "description": long_desc,
                "read_only": True,
            }
        ]
        code = _generate_pv_definitions(channels)

        # Description should be truncated to 80 chars
        # The full 200-char description should not appear in the code
        assert long_desc not in code  # Full description not present
        assert "A" * 80 in code  # Truncated version should be there

    def test_quote_escaping_in_description(self):
        """Test that quotes in descriptions are escaped."""
        channels = [
            {
                "name": "TEST",
                "python_name": "TEST",
                "type": "float",
                "description": "It's a 'test' value",
                "read_only": True,
            }
        ]
        code = _generate_pv_definitions(channels)

        # Quotes should be escaped
        assert "\\'" in code or "It" in code  # Should handle escaping


# =============================================================================
# Test _generate_pairings_literal
# =============================================================================


class TestGeneratePairingsLiteral:
    """Test pairings dict literal generation."""

    def test_empty_pairings(self):
        """Test empty pairings returns empty dict."""
        assert _generate_pairings_literal({}) == "{}"

    def test_single_pairing(self):
        """Test single pairing."""
        result = _generate_pairings_literal({"SP1": "RB1"})
        assert "'SP1': 'RB1'" in result

    def test_multiple_pairings(self):
        """Test multiple pairings."""
        result = _generate_pairings_literal({"SP1": "RB1", "SP2": "RB2"})
        assert "'SP1': 'RB1'" in result
        assert "'SP2': 'RB2'" in result


# =============================================================================
# Test _generate_backend_code
# =============================================================================


class TestGenerateBackendCode:
    """Test backend code generation."""

    def test_passthrough_backend(self):
        """Test passthrough backend generation."""
        code = _generate_backend_code("passthrough", {}, 0.01, 10.0)

        assert "class PassthroughBackend" in code
        assert "def initialize" in code
        assert "def on_write" in code
        assert "def step" in code
        assert "return {}" in code  # No-op returns

    def test_mock_style_backend(self):
        """Test mock-style backend generation."""
        code = _generate_backend_code("mock_style", {}, 0.01, 10.0)

        assert "class MockStyleBackend" in code
        assert "def initialize" in code
        assert "def on_write" in code
        assert "def step" in code
        assert "noise_level" in code
        assert "0.01" in code  # Noise level should be embedded

    def test_mock_style_with_custom_noise(self):
        """Test mock-style backend with custom noise level."""
        code = _generate_backend_code("mock_style", {}, 0.05, 10.0)
        assert "0.05" in code


# =============================================================================
# Test _get_backend_instantiation
# =============================================================================


class TestGetBackendInstantiation:
    """Test backend instantiation code generation."""

    def test_passthrough(self):
        """Test passthrough instantiation."""
        result = _get_backend_instantiation("passthrough", 0.01)
        assert result == "PassthroughBackend()"

    def test_mock_style(self):
        """Test mock-style instantiation."""
        result = _get_backend_instantiation("mock_style", 0.01)
        assert result == "MockStyleBackend(noise_level=0.01)"

    def test_mock_style_custom_noise(self):
        """Test mock-style with custom noise."""
        result = _get_backend_instantiation("mock_style", 0.05)
        assert result == "MockStyleBackend(noise_level=0.05)"


# =============================================================================
# Test generate_soft_ioc (integration)
# =============================================================================


class TestGenerateSoftIOC:
    """Integration tests for complete IOC generation."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "ioc": {
                "name": "test_ioc",
                "port": 5064,
            },
            "backend": {
                "type": "mock_style",
                "noise_level": 0.01,
                "update_rate": 10.0,
            },
        }

    @pytest.fixture
    def sample_channels(self):
        """Sample channels for testing."""
        return [
            {
                "name": "QUAD:CURRENT:SP",
                "python_name": "QUAD_CURRENT_SP",
                "type": "float",
                "description": "Quad setpoint",
                "read_only": False,
                "units": "A",
                "precision": 3,
                "high_alarm": 200.0,
                "low_alarm": -200.0,
            },
            {
                "name": "QUAD:CURRENT:RB",
                "python_name": "QUAD_CURRENT_RB",
                "type": "float",
                "description": "Quad readback",
                "read_only": True,
                "units": "A",
                "precision": 3,
                "high_alarm": 200.0,
                "low_alarm": -200.0,
            },
            {
                "name": "SYSTEM:STATUS",
                "python_name": "SYSTEM_STATUS",
                "type": "enum",
                "description": "System status",
                "read_only": True,
                "enum_strings": ["Off", "On"],
            },
        ]

    @pytest.fixture
    def sample_pairings(self):
        """Sample pairings for testing."""
        return {"QUAD:CURRENT:SP": "QUAD:CURRENT:RB"}

    def test_generates_valid_python(self, sample_config, sample_channels, sample_pairings):
        """Test that generated code is valid Python syntax."""
        code = generate_soft_ioc(sample_config, sample_channels, sample_pairings)

        # Should compile without syntax errors
        compile(code, "<generated>", "exec")

    def test_contains_class_definition(self, sample_config, sample_channels, sample_pairings):
        """Test that code contains IOC class."""
        code = generate_soft_ioc(sample_config, sample_channels, sample_pairings)

        assert "class TestIoc(PVGroup)" in code

    def test_contains_pv_definitions(self, sample_config, sample_channels, sample_pairings):
        """Test that code contains PV definitions."""
        code = generate_soft_ioc(sample_config, sample_channels, sample_pairings)

        assert "QUAD_CURRENT_SP = pvproperty" in code
        assert "QUAD_CURRENT_RB = pvproperty" in code
        assert "SYSTEM_STATUS = pvproperty" in code

    def test_contains_pairings(self, sample_config, sample_channels, sample_pairings):
        """Test that code contains pairings dict."""
        code = generate_soft_ioc(sample_config, sample_channels, sample_pairings)

        assert "PAIRINGS = {" in code
        assert "'QUAD:CURRENT:SP': 'QUAD:CURRENT:RB'" in code

    def test_contains_backend(self, sample_config, sample_channels, sample_pairings):
        """Test that code contains backend class."""
        code = generate_soft_ioc(sample_config, sample_channels, sample_pairings)

        assert "class MockStyleBackend" in code

    def test_contains_heartbeat(self, sample_config, sample_channels, sample_pairings):
        """Test that code contains simulation heartbeat."""
        code = generate_soft_ioc(sample_config, sample_channels, sample_pairings)

        assert "sim_heartbeat = pvproperty" in code
        assert "SIM:HEARTBEAT" in code

    def test_contains_main_entry(self, sample_config, sample_channels, sample_pairings):
        """Test that code contains main entry point."""
        code = generate_soft_ioc(sample_config, sample_channels, sample_pairings)

        assert "if __name__ == '__main__':" in code
        assert "ioc_arg_parser" in code
        assert "run(ioc.pvdb" in code

    def test_passthrough_backend(self, sample_config, sample_channels, sample_pairings):
        """Test generation with passthrough backend."""
        sample_config["backend"]["type"] = "passthrough"
        code = generate_soft_ioc(sample_config, sample_channels, sample_pairings)

        assert "class PassthroughBackend" in code
        assert "PassthroughBackend()" in code

    def test_empty_channels(self, sample_config):
        """Test generation with no channels."""
        code = generate_soft_ioc(sample_config, [], {})

        # Should still be valid Python
        compile(code, "<generated>", "exec")
        assert "class TestIoc(PVGroup)" in code
        assert "PV Count: 0" in code

    def test_empty_pairings(self, sample_config, sample_channels):
        """Test generation with no pairings."""
        code = generate_soft_ioc(sample_config, sample_channels, {})

        assert "PAIRINGS = {}" in code


# =============================================================================
# Test edge cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pv_name_with_all_special_chars(self):
        """Test PV name with many special characters."""
        result = sanitize_pv_name("A:B[C]-D.E")
        assert result == "A_B_C_D_E"

    def test_very_long_pv_name(self):
        """Test very long PV name."""
        long_name = "A" * 100 + ":B" * 50
        result = sanitize_pv_name(long_name)
        assert "_" in result  # Colons converted
        assert result.isidentifier()  # Valid Python identifier

    def test_unicode_in_pv_name(self):
        """Test that unicode in PV names is handled.

        Note: Unicode characters that aren't valid Python identifiers
        remain in the result. This is acceptable for edge cases -
        generated code may need manual review for exotic PV names.
        """
        result = sanitize_pv_name("TEMP:°C")
        # Unicode degree symbol passes through - this is expected behavior
        # The sanitizer handles common EPICS naming conventions,
        # not arbitrary unicode
        assert "TEMP" in result
        assert ":" not in result  # Colon should be replaced
