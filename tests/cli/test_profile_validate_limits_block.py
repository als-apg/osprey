"""The per-type limits block, as the build and ``osprey profile validate`` read it.

``control_system.connector.<type>.limits_checking`` overrides the
deployment-wide pair as a whole block, so a profile that writes one leaf, or
writes the block somewhere the render will not put it, has stated a posture
nothing will ever answer with. Both are refused by name rather than ignored.

The refusals are about the RENDER, not about spelling taste: a ``config:``
entry's top-level key is the only part the emitter splits on dots
(:func:`osprey.utils.config_writer.config_update_fields`), so every dot below
it lands in a literal key name. That is what separates the two custom-type
spellings tested here — one map key holding the dotted type renders correctly,
the same type flattened into the top-level key does not.
"""

from __future__ import annotations

from typing import Any

from osprey.cli.build_profile_deploy import limits_block_errors

CUSTOM_TYPE = "mypkg.TangoConnector"


def _complete(*, enabled: bool = True, allow_unlisted: bool = False) -> dict[str, Any]:
    """The two leaves of one per-type block, as a mapping."""
    return {"enabled": enabled, "allow_unlisted_channels": allow_unlisted}


def test_a_complete_flat_built_in_block_passes() -> None:
    config = {
        "control_system.connector.virtual_accelerator.limits_checking.enabled": True,
        "control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels": (
            True
        ),
    }
    assert limits_block_errors(config) == []


def test_a_flat_built_in_block_missing_a_leaf_names_the_leaf() -> None:
    config = {"control_system.connector.virtual_accelerator.limits_checking.enabled": True}
    errors = limits_block_errors(config)
    assert len(errors) == 1
    assert "allow_unlisted_channels" in errors[0]
    assert "virtual_accelerator" in errors[0]


def test_a_flat_built_in_block_missing_enabled_names_enabled() -> None:
    config = {
        "control_system.connector.epics.limits_checking.allow_unlisted_channels": False,
    }
    errors = limits_block_errors(config)
    assert len(errors) == 1
    assert "'enabled'" in errors[0]
    assert "epics" in errors[0]


def test_a_flat_dotted_custom_type_is_refused_even_when_both_leaves_are_stated() -> None:
    """The dots in the type are indistinguishable from path separators once flattened."""
    config = {
        f"control_system.connector.{CUSTOM_TYPE}.limits_checking.enabled": True,
        f"control_system.connector.{CUSTOM_TYPE}.limits_checking.allow_unlisted_channels": True,
    }
    errors = limits_block_errors(config)
    assert len(errors) == 2
    joined = "\n".join(errors)
    assert f"control_system.connector.{CUSTOM_TYPE}.limits_checking.enabled" in joined
    assert (
        f"control_system.connector.{CUSTOM_TYPE}.limits_checking.allow_unlisted_channels" in joined
    )


def test_a_dotted_leaf_map_key_is_refused() -> None:
    """``virtual_accelerator: {"limits_checking.enabled": …}`` renders one literal key."""
    config = {
        "control_system.connector.virtual_accelerator": {"limits_checking.enabled": True},
    }
    errors = limits_block_errors(config)
    assert errors
    assert any("limits_checking.enabled" in error for error in errors)


def test_a_complete_dotted_prefix_custom_type_passes() -> None:
    """A dotted type spelled as its own map key is the spelling that renders."""
    config = {"control_system.connector": {CUSTOM_TYPE: {"limits_checking": _complete()}}}
    assert limits_block_errors(config) == []


def test_a_dotted_prefix_custom_type_missing_a_leaf_names_the_leaf() -> None:
    config = {
        "control_system.connector": {CUSTOM_TYPE: {"limits_checking": {"enabled": True}}},
    }
    errors = limits_block_errors(config)
    assert len(errors) == 1
    assert "allow_unlisted_channels" in errors[0]
    assert CUSTOM_TYPE in errors[0]


def test_a_fully_nested_block_passes() -> None:
    config = {
        "control_system": {
            "connector": {"virtual_accelerator": {"limits_checking": _complete()}},
        }
    }
    assert limits_block_errors(config) == []


def test_control_system_addressed_at_two_depths_names_both_lines() -> None:
    config = {
        "control_system": {
            "connector": {"virtual_accelerator": {"limits_checking": _complete()}},
        },
        "control_system.limits_checking.enabled": True,
    }
    errors = limits_block_errors(config)
    assert len(errors) == 1
    assert "'control_system'" in errors[0]
    assert "'control_system.limits_checking.enabled'" in errors[0]


def test_a_deeper_pair_below_control_system_passes() -> None:
    """Two dotted keys at different depths merge at render, so they are not refused.

    ``config_update_fields`` writes each entry through ``_set_dotted_anchored``
    with ``create_only=True``, which walks into whatever an existing
    intermediate key holds. Only the bare top-level key, having no dots to
    walk, replaces a subtree.
    """
    config = {
        "control_system.connector": {"virtual_accelerator": {"limits_checking": _complete()}},
        "control_system.connector.epics.limits_checking.enabled": True,
        "control_system.connector.epics.limits_checking.allow_unlisted_channels": False,
    }
    assert limits_block_errors(config) == []


def test_a_config_with_no_limits_keys_passes() -> None:
    config = {
        "control_system.type": "live_standin",
        "control_system.writes_enabled": False,
        "control_system.connector.virtual_accelerator.writes_enabled": True,
    }
    assert limits_block_errors(config) == []


def test_gateway_keys_are_untouched() -> None:
    config = {
        "control_system.connector.epics.gateways.read.address": "gw-read.example.org",
        "control_system.connector.epics.gateways.write.address": "gw-write.example.org",
        "control_system.connector.epics.limits_checking.enabled": True,
        "control_system.connector.epics.limits_checking.allow_unlisted_channels": False,
    }
    assert limits_block_errors(config) == []


def test_the_deployment_wide_block_is_untouched() -> None:
    """Only ``control_system.connector.*`` paths are per-type; the pair above is not."""
    config = {
        "control_system.limits_checking.enabled": True,
        "control_system.limits_checking.database_path": "limits.yml",
    }
    assert limits_block_errors(config) == []


def test_a_non_mapping_config_has_nothing_to_refuse() -> None:
    assert limits_block_errors({}) == []
    assert limits_block_errors(None) == []  # type: ignore[arg-type]


def test_a_stray_block_for_a_type_the_deployment_does_not_run_is_still_checked() -> None:
    """The refusal is about the block's shape, not about whether the type is selected."""
    config = {
        "control_system.type": "mock",
        "control_system.connector.live_standin.limits_checking.enabled": True,
    }
    errors = limits_block_errors(config)
    assert len(errors) == 1
    assert "live_standin" in errors[0]
    assert "allow_unlisted_channels" in errors[0]
