"""Facility-identity resolution: `facility.name` is canonical.

`facility.name` is the key every other reader already consults — the three
channel-finder server contexts and the web-terminal landing render. These tests
pin the build path onto the same key, with the older top-level `facility_name`
kept working as a fallback, and the project name as the last resort.
"""

from pathlib import Path

import pytest
import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.utils.facility import resolve_facility_name

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# resolve_facility_name — the shared resolution order
# ---------------------------------------------------------------------------


def test_canonical_key_wins():
    config = {"facility": {"name": "Canonical Light Source"}, "facility_name": "Legacy Name"}
    assert resolve_facility_name(config, "proj") == "Canonical Light Source"


def test_legacy_key_is_the_fallback():
    assert resolve_facility_name({"facility_name": "Legacy Name"}, "proj") == ("Legacy Name")


def test_legacy_key_used_when_facility_block_carries_only_prefix():
    config = {"facility": {"prefix": "ca"}, "facility_name": "Legacy Name"}
    assert resolve_facility_name(config, "proj") == "Legacy Name"


def test_neither_key_falls_back_to_the_supplied_default():
    assert resolve_facility_name({}, "my-project") == "my-project"


@pytest.mark.parametrize(
    "config",
    [
        {"facility": {"name": ""}, "facility_name": "Legacy Name"},
        {"facility": {"name": None}, "facility_name": "Legacy Name"},
    ],
)
def test_empty_canonical_value_falls_through_to_legacy(config):
    """A blank name would reach the prompts as a hole in the sentence."""
    assert resolve_facility_name(config, "proj") == "Legacy Name"


def test_empty_values_at_both_levels_fall_through_to_the_default():
    config = {"facility": {"name": ""}, "facility_name": ""}
    assert resolve_facility_name(config, "my-project") == "my-project"


def test_non_mapping_facility_value_is_tolerated():
    """A hand-edited `facility: something` must not crash the build."""
    assert (
        resolve_facility_name({"facility": "oops", "facility_name": "Legacy Name"}, "proj")
        == "Legacy Name"
    )


# ---------------------------------------------------------------------------
# Build path: build_claude_code_context feeds `{{ facility_name }}`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"project_name": "demo", "facility": {"name": "Canonical LS"}}, "Canonical LS"),
        ({"project_name": "demo", "facility_name": "Legacy LS"}, "Legacy LS"),
        ({"project_name": "demo"}, "demo"),
    ],
    ids=["facility.name", "legacy facility_name", "neither -> project name"],
)
def test_claude_code_context_resolves_facility_name(tmp_path, config, expected):
    manager = TemplateManager()
    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, tmp_path, config
    )
    assert ctx["facility_name"] == expected


# ---------------------------------------------------------------------------
# End-to-end: the name reaches the rendered agent prompts
# ---------------------------------------------------------------------------


# The channel-finder agent prompt interpolates the name into this sentence. The
# assertions anchor on the whole sentence rather than the bare name: the built
# project also seeds a static `.claude/rules/facility.md` that happens to spell
# out the shipped example name, so a bare substring search passes even when the
# interpolated value is wrong.
_PROMPT_ARTIFACT = Path(".claude/agents/channel-finder.md")
_PROMPT_SENTENCE = "You are exploring the {} control system database."


def _channel_finder_prompt(project_dir: Path) -> str:
    return (project_dir / _PROMPT_ARTIFACT).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def channel_finder_project(tmp_path_factory) -> Path:
    """A channel-finder build — the path where the config value used to be shadowed."""
    out_dir = tmp_path_factory.mktemp("cf_facility")
    return TemplateManager().create_project(
        project_name="cf-facility",
        output_dir=out_dir,
        data_bundle="channel_finder_standalone",
        context={"channel_finder_mode": "in_context", "default_provider": "anthropic"},
    )


def test_channel_finder_build_uses_the_config_name_not_the_project_name(channel_finder_project):
    """The bundle ships a facility name; the prompt must carry it, not `cf-facility`."""
    config = yaml.safe_load((channel_finder_project / "config.yml").read_text(encoding="utf-8"))
    # The bundle ships the canonical `facility.name`; read it the way every
    # production reader does so this stays about the value reaching the prompt
    # rather than about which of the two spellings the template happens to use.
    shipped = config["facility"]["name"]
    assert shipped == resolve_facility_name(config, "cf-facility")
    assert shipped and shipped != "cf-facility"

    prompt = _channel_finder_prompt(channel_finder_project)
    assert _PROMPT_SENTENCE.format(shipped) in prompt
    assert _PROMPT_SENTENCE.format("cf-facility") not in prompt


@pytest.mark.parametrize(
    ("facility_block", "expected"),
    [
        ({"facility": {"name": "Regenerated LS"}}, "Regenerated LS"),
        ({"facility_name": "Regenerated Legacy LS"}, "Regenerated Legacy LS"),
    ],
    ids=["facility.name", "legacy facility_name"],
)
def test_regenerated_prompts_pick_up_the_edited_facility_name(
    tmp_path_factory, facility_block, expected
):
    """Editing config.yml and regenerating rewrites the prompts through both keys."""
    out_dir = tmp_path_factory.mktemp("regen_facility")
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="regen-facility",
        output_dir=out_dir,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical", "default_provider": "anthropic"},
    )

    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.pop("facility_name", None)
    config.pop("facility", None)
    config.update(facility_block)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manager.regenerate_claude_code(project_dir)
    prompt = _channel_finder_prompt(project_dir)
    assert _PROMPT_SENTENCE.format(expected) in prompt
    assert _PROMPT_SENTENCE.format("regen-facility") not in prompt
