"""The build's render projection: ``modules.web_terminals.renders.<name>``.

``osprey build`` writes one entry per login it renders into the root render's
``build/config.yml`` — ``{kind, project, project_path}`` derived from the
file's location and ``project_name``, plus ``extra_mounts``, ``seed_base``
and ``image`` copied from the file's top level when it sets them. The retired
render catalog appears nowhere in the rendered config. The profile lint
judges the same projection before any render exists, and a profile that spells
it is refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.cli.build_profile_deploy import build_lint_view
from osprey.cli.build_profile_emit import effective_web_terminals, emits_layer_profiles
from osprey.cli.build_profile_model import render_key_errors
from osprey.deployment.web_terminals.render_images import _referenced_renders
from osprey.deployment.web_terminals.render_naming import (
    derived_renders,
    render_projection,
)
from osprey.deployment.web_terminals.roster import resolve_cards, served_renders
from osprey.utils.workspace import RETIRED_LAYER_DIRS
from osprey_connectors.errors import BuildProfileError
from tests.cli._lifecycle_build import invoke_build, rendered_config
from tests.fixtures.lifecycle_repo import build_exemplar_repo, preserved_environ

PROJECT = "als-exemplar"
TIERS = ("admin", "readonly", "readwrite")
APPS = ("channel-finder", "knowledge", "logbook")
ENABLED = {"modules.web_terminals.enabled": True}

#: Top-level knobs written into one tier file, to be carried verbatim.
READWRITE_KNOBS = {
    "extra_mounts": ["/srv/beamline:/srv/beamline:ro"],
    "seed_base": False,
    "image": "registry.example.org/ops/readwrite:2026.10",
}


def _with_readwrite_knobs(repo: Path) -> Path:
    tier = repo / "tiers" / "readwrite.yml"
    tier.write_text(
        tier.read_text(encoding="utf-8") + "\n" + yaml.safe_dump(READWRITE_KNOBS),
        encoding="utf-8",
    )
    return repo


def _rendered_web_terminals(repo: Path) -> dict:
    config = rendered_config(repo)
    return config["modules"]["web_terminals"]


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    """The exemplar, built once for the module; plus every view the lint judged.

    The build runs end to end with the real profile lint, from a working
    directory outside the repo (``--repo``), so the projection must come from
    the repo the build was pointed at, never from the cwd. The lint is wrapped,
    not replaced: every view it judges is recorded and its verdict stands.
    """
    import osprey.deployment.web_terminals.lint as lint

    judged: list[dict] = []
    real_profile_config_errors = lint.profile_config_errors

    def record(view: dict, *args, **kwargs) -> list[str]:
        judged.append(dict(view))
        return real_profile_config_errors(view, *args, **kwargs)

    outside = tmp_path_factory.mktemp("renders")
    repo = build_exemplar_repo(outside / "deployments" / PROJECT)
    with pytest.MonkeyPatch.context() as patch, preserved_environ():
        patch.setattr(lint, "profile_config_errors", record)
        patch.chdir(outside)
        result = invoke_build(repo)
    assert result.exit_code == 0, result.output
    return repo, judged


def test_build_writes_one_projection_entry_per_login(built_repo):
    repo, _judged = built_repo
    renders = _rendered_web_terminals(repo)["renders"]

    assert set(renders) == {*TIERS, *APPS}
    for name in (*TIERS, *APPS):
        assert renders[name]["kind"] == "tier"
    for name, entry in renders.items():
        assert entry["project"] == f"{PROJECT}-{name}"
        assert entry["project_path"] == f"build/{PROJECT}-{name}"
        assert (repo / entry["project_path"] / "config.yml").is_file()


def test_projection_carries_the_layer_files_top_level_knobs(tmp_path: Path):
    repo = _with_readwrite_knobs(build_exemplar_repo(tmp_path / PROJECT))

    renders = render_projection(repo, PROJECT)

    assert {key: renders["readwrite"][key] for key in READWRITE_KNOBS} == READWRITE_KNOBS
    # A file that sets none of them carries only the derived three.
    assert set(renders["readonly"]) == {"kind", "project", "project_path"}
    assert set(renders["logbook"]) == {"kind", "project", "project_path"}


def test_built_entries_carry_only_the_derived_keys_when_no_file_sets_a_knob(built_repo):
    repo, _judged = built_repo
    renders = _rendered_web_terminals(repo)["renders"]

    for name, entry in renders.items():
        assert set(entry) == {"kind", "project", "project_path"}, name


def test_login_renders_carry_no_projection_of_their_own(built_repo):
    repo, _judged = built_repo
    for name in (*TIERS, *APPS):
        render = yaml.safe_load(
            (repo / "build" / f"{PROJECT}-{name}" / "config.yml").read_text(encoding="utf-8")
        )
        web_terminals = (render.get("modules") or {}).get("web_terminals") or {}
        assert "renders" not in web_terminals, name


def test_the_lint_judges_the_projection_the_build_writes(built_repo):
    repo, judged = built_repo
    renders = _rendered_web_terminals(repo)["renders"]

    assert judged, "the build ran no web-terminal lint"
    view = judged[0]
    projected = {
        key.removeprefix("modules.web_terminals.renders."): value
        for key, value in view.items()
        if key.startswith("modules.web_terminals.renders.")
    }
    assert projected == renders


def test_rendered_projection_feeds_the_deploy_readers(built_repo):
    repo, _judged = built_repo
    config = rendered_config(repo)

    assert set(effective_web_terminals(config)["renders"]) == {*TIERS, *APPS}
    assert emits_layer_profiles(config)

    # The secret/telemetry walk: every tier a person runs, plus every shared login.
    referenced = {
        name for name, render in served_renders(config, strict=True).items() if render is not None
    }
    assert {"readonly", "readwrite", "admin"} <= referenced
    assert set(APPS) <= referenced


def test_render_projection_lists_every_login_file(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "tiers").mkdir(parents=True)
    (repo / "apps" / "knob").mkdir(parents=True)
    (repo / "profile.yml").write_text("name: x\n")
    (repo / "tiers" / "ops.yml").write_text("name: ops\nseed_base: false\n")
    (repo / "tiers" / "logbook.yml").write_text("name: logbook\nimage: reg/logbook:1\n")
    (repo / "tiers" / "broken.yml").write_text(": : not yaml : [\n")
    (repo / "apps" / "knob" / "app.yml").write_text("name: knob\n")

    projection = render_projection(repo, "site")

    assert projection == {
        "broken": {"kind": "tier", "project": "site-broken", "project_path": "build/site-broken"},
        "logbook": {
            "kind": "tier",
            "project": "site-logbook",
            "project_path": "build/site-logbook",
            "image": "reg/logbook:1",
        },
        "ops": {
            "kind": "tier",
            "project": "site-ops",
            "project_path": "build/site-ops",
            "seed_base": False,
        },
    }


def test_derived_renders_is_empty_for_a_render_without_a_terminal_stack(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "tiers").mkdir(parents=True)
    (repo / "tiers" / "ops.yml").write_text("name: ops\n")

    assert derived_renders({"modules.web_terminals.enabled": False}, "site", repo) == {}
    assert derived_renders({}, "site", repo) == {}
    assert derived_renders(ENABLED, "site", None) == {}
    assert derived_renders(ENABLED, "site", repo) == {
        "modules.web_terminals.renders.ops": {
            "kind": "tier",
            "project": "site-ops",
            "project_path": "build/site-ops",
        }
    }


def test_derived_renders_projects_for_a_dispatch_tier_without_a_terminal_stack(tmp_path: Path):
    """The dispatch workers run their tier with no terminal beside them, and the
    worker's image is named from this projection."""
    repo = tmp_path / "repo"
    (repo / "tiers").mkdir(parents=True)
    (repo / "tiers" / "ops.yml").write_text("name: ops\n")
    disabled = {"modules.web_terminals.enabled": False}

    assert derived_renders(disabled, "site", repo, dispatch_tier="ops") == {
        "modules.web_terminals.renders.ops": {
            "kind": "tier",
            "project": "site-ops",
            "project_path": "build/site-ops",
        }
    }
    assert derived_renders(disabled, "site", repo, dispatch_tier=None) == {}


def test_lint_view_without_a_profile_root_projects_nothing():
    view = build_lint_view(None, dict(ENABLED), "site")

    assert view["project_name"] == "site"
    assert not any(key.startswith("modules.web_terminals.renders") for key in view)


@pytest.mark.parametrize(
    "config",
    [
        {"modules.web_terminals.renders.ops.project": "x"},
        {"modules.web_terminals.renders": {"ops": {"kind": "tier"}}},
        {"modules": {"web_terminals": {"renders": {}}}},
    ],
)
def test_a_profile_spelling_the_projection_is_refused(config):
    errors = render_key_errors(config)

    assert len(errors) == 1
    assert "modules.web_terminals.renders" in errors[0]
    assert "tiers/" in errors[0] and "apps/" not in errors[0]


def test_a_profile_without_the_projection_passes():
    assert render_key_errors({"modules.web_terminals.enabled": True}) == []


def test_image_build_units_read_project_path_and_source_file_from_the_projection():
    # The served set decides membership (two people on `ops`, the shared
    # `logbook`); a resolved entry for a render nobody serves builds nothing.
    config = {
        "modules": {
            "web_terminals": {
                "users": [
                    {"name": "a", "index": 0, "tier": "ops"},
                    {"name": "b", "index": 1, "tier": "ops"},
                ],
                "shared": ["logbook"],
                "renders": {
                    "ops": {"kind": "tier", "project": "s-ops", "project_path": "build/s-ops"},
                    "logbook": {
                        "kind": "tier",
                        "project": "s-logbook",
                        "project_path": "build/s-logbook",
                    },
                },
            }
        }
    }
    resolved = [
        {"render": "ops", "project": "s-ops", "image": "s-ops:local"},
        {"render": "ops", "project": "s-ops", "image": "s-ops:local"},
        {"render": "logbook", "project": "s-logbook", "image": "s-logbook:local"},
        {"render": "gone", "project": "s-gone", "image": "s-gone:local"},
    ]

    units = _referenced_renders(config, resolved)

    assert units == [
        {
            "render": "ops",
            "project": "s-ops",
            "project_path": "build/s-ops",
            "build_profile": "tiers/ops.yml",
            "image": "s-ops:local",
        },
        {
            "render": "logbook",
            "project": "s-logbook",
            "project_path": "build/s-logbook",
            "build_profile": "tiers/logbook.yml",
            "image": "s-logbook:local",
        },
    ]


@pytest.mark.parametrize("other_dir", [*RETIRED_LAYER_DIRS])
def test_two_layer_files_with_one_name_are_refused_not_hidden(tmp_path: Path, other_dir: str):
    repo = tmp_path / "repo"
    (repo / "tiers").mkdir(parents=True)
    (repo / other_dir).mkdir()
    (repo / "profile.yml").write_text("name: x\n")
    (repo / "tiers" / "ops.yml").write_text("name: ops\n")
    (repo / other_dir / "ops.yml").write_text("name: ops\n")

    with pytest.raises(BuildProfileError, match="build/<project>-ops"):
        render_projection(repo, "site")


def _web_terminals_with_renders(**extra) -> dict:
    return {
        "enabled": True,
        "users": [{"name": "alice", "index": 1}],
        "renders": {
            "ops": {"kind": "tier", "project": "s-ops", "project_path": "build/s-ops"},
            "logbook": {"kind": "tier", "project": "s-logbook", "project_path": "build/s-logbook"},
        },
        "shared": ["logbook"],
        **extra,
    }


def test_people_resolve_against_tier_renders_only():
    resolved = resolve_cards(_web_terminals_with_renders(default_tier="ops"), {}, strict=True)

    (alice,) = [entry for entry in resolved if entry["name"] == "alice"]
    assert alice["render"] == "ops"
    assert alice["project"] == "s-ops"


def test_a_person_given_a_shared_login_as_tier_is_refused():
    # `logbook` is listed in `shared`: one file has one audience, so it is
    # never a person's access level as well.
    web_terminals = _web_terminals_with_renders(default_tier="logbook")

    with pytest.raises(ValueError, match="tier or a shared card"):
        resolve_cards(web_terminals, {}, strict=True)
    lenient = resolve_cards(web_terminals, {}, strict=False)
    (alice,) = [entry for entry in lenient if entry["name"] == "alice"]
    assert alice["project"] is None


def test_a_tier_file_carrying_its_top_level_knobs_builds(tmp_path: Path):
    """B27: `extra_mounts`, `seed_base` and `image` at the top of a tier file —
    where `osprey init --revalidate` moves them from the retired catalog — are
    keys a tier file takes, so the build accepts them."""
    repo = _with_readwrite_knobs(build_exemplar_repo(tmp_path / PROJECT))
    with preserved_environ():
        result = invoke_build(repo)
    assert result.exit_code == 0, result.output
    assert (
        render_projection(repo, PROJECT)["readwrite"]["extra_mounts"]
        == (READWRITE_KNOBS["extra_mounts"])
    )
