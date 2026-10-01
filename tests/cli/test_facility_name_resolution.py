"""Facility-name resolution.

`resolve_facility_name` reads `facility.name`, then the top-level
`facility_name`, then its default. The build path reads neither key: the agent
context and the prompts rendered from it carry the name the render root
reports, which is its facility file's identity, or the project name where the
render holds no facility file.
"""

import json
from pathlib import Path

import pytest
import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.utils.facility import facility_identity, resolve_facility_name


def _bundle_data_root(bundle: str = "control_assistant") -> Path:
    """The tree these fixtures hand the render as the profile's ``data:``.

    A build copies the tree its profile's ``data:`` key names, and that key is
    required — nothing falls back to a packaged tree any more. These fixtures
    render straight from a bundle rather than from a profile, so they name the
    tree that bundle packages, which is the content the render used to reach
    for on its own.
    """
    return Path(TemplateManager().template_root) / "apps" / bundle / "data"


def _create_project(manager: TemplateManager, **kwargs) -> Path:
    """``create_project`` plus the three steps a real build takes next.

    A build renders the framework template, overlays the resolved profile's
    ``config:`` block onto the result, stamps ``.osprey-manifest.json``, and
    regenerates ``.claude/`` from the finished config. The template carries
    only derived and profile-field-derived keys, so a fixture that stops after
    the render holds half a config — the declarative half is the preset's, and
    the artifacts rendered before it landed do not know about the deployment's
    control system, services or servers. These fixtures render from a bundle
    rather than from a profile, so they overlay the preset ``osprey init``
    pairs with that bundle.
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.utils.config_writer import config_update_fields

    bundle = kwargs.setdefault("data_bundle", "control_assistant")
    preset = bundle.replace("_", "-")
    kwargs.setdefault("data_root", _bundle_data_root(bundle))
    project = manager.create_project(**kwargs)
    profile, _preset_dir = resolve_build_profile(None, preset=preset)
    config_update_fields(project / "config.yml", profile.config)
    manager.generate_manifest(
        project, kwargs["project_name"], preset, {}, artifacts=kwargs.get("artifacts")
    )
    # The build's last render, and the one that ships: `create_project` wrote
    # `.claude/` from a config.yml that did not yet carry the preset's block.
    manager.regenerate_claude_code(project)
    return project


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


def _write_facility_file(render_root: Path, identity: dict) -> None:
    document = {"schema": "osprey.facility.facility/1", "identity": identity}
    (render_root / "facility.json").write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize(
    "config",
    [
        {"project_name": "demo", "facility": {"name": "Canonical LS"}},
        {"project_name": "demo", "facility_name": "Legacy LS"},
        {"project_name": "demo"},
    ],
    ids=["facility.name", "legacy facility_name", "neither"],
)
def test_claude_code_context_without_a_facility_file_carries_the_project_name(tmp_path, config):
    manager = TemplateManager()
    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, tmp_path, config
    )
    assert ctx["facility_name"] == facility_identity(tmp_path, "demo")["name"] == "demo"


def test_claude_code_context_carries_the_facility_file_name(tmp_path):
    _write_facility_file(tmp_path, {"code": "cls", "name": "Canonical LS"})
    manager = TemplateManager()
    ctx = claude_code.build_claude_code_context(
        manager.template_root,
        manager.jinja_env,
        tmp_path,
        {"project_name": "demo", "facility": {"name": "Configured LS"}},
    )
    assert ctx["facility_name"] == "Canonical LS"


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
    """A channel-finder render with no facility file."""
    out_dir = tmp_path_factory.mktemp("cf_facility")
    return _create_project(
        TemplateManager(),
        project_name="cf-facility",
        output_dir=out_dir,
        data_bundle="channel_finder_standalone",
        context={"channel_finder_mode": "in_context", "default_provider": "anthropic"},
    )


def test_channel_finder_build_uses_the_name_the_render_root_reports(channel_finder_project):
    """The prompt carries the render root's name, whatever the config's keys say."""
    config = yaml.safe_load((channel_finder_project / "config.yml").read_text(encoding="utf-8"))
    configured = config["facility"]["name"]
    reported = facility_identity(channel_finder_project, "cf-facility")["name"]
    assert configured and configured != reported

    prompt = _channel_finder_prompt(channel_finder_project)
    assert _PROMPT_SENTENCE.format(reported) in prompt
    assert _PROMPT_SENTENCE.format(configured) not in prompt


def test_regenerated_prompts_pick_up_the_facility_file_name(tmp_path_factory):
    """Regenerating a render root that holds a facility file rewrites the prompts with its name."""
    out_dir = tmp_path_factory.mktemp("regen_facility")
    manager = TemplateManager()
    project_dir = _create_project(
        manager,
        project_name="regen-facility",
        output_dir=out_dir,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical", "default_provider": "anthropic"},
    )
    assert _PROMPT_SENTENCE.format("regen-facility") in _channel_finder_prompt(project_dir)

    _write_facility_file(project_dir, {"code": "rls", "name": "Regenerated LS"})

    manager.regenerate_claude_code(project_dir)
    prompt = _channel_finder_prompt(project_dir)
    assert _PROMPT_SENTENCE.format("Regenerated LS") in prompt
    assert _PROMPT_SENTENCE.format("regen-facility") not in prompt
