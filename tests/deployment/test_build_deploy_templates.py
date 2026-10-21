"""Multi-user render parity, held across the deploy split.

A facility with a render catalog needs one web-terminal image per render.
Building them in the pipeline would put a per-render `osprey build` render and
a per-render `build-web-terminal-<render>` job in the CI template, guarded
only by string assertions against the shipped template text.

The pipeline builds no web-terminal images at all — the deploy host does, from
the render catalog, when `osprey up` runs. This module holds both halves of
that split against the exemplar profile, which declares two tiers: the
rendered pipeline must build no web-terminal image, and every referenced
render must still come out as its own host-side build unit. Losing either
half loses multi-user support without any test going red.

The builder's own behavior (build args, dev wheels, the render check) is covered in
``tests/deployment/web_terminals/test_render_images.py``; what is tested here
is only that the renders reach it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_deploy import DeployConfig, parse_deploy_block
from osprey.cli.deploy_scaffold_templates import CI_TEMPLATES, build_ci_context, render
from osprey.deployment.web_terminals.render_images import _referenced_renders
from osprey.deployment.web_terminals.render_naming import render_project

EXEMPLAR_DIR = Path(__file__).parent / "goldens" / "exemplar-profile"

#: Matches the goldens' frozen provenance token; nothing here reads it.
FROZEN_VERSION = "OSPREY_VERSION"

PROJECT_NAME = "demo-facility"


def _exemplar_tiers() -> list[str]:
    """The exemplar's tier names, one per file in its ``tiers/`` directory."""
    return sorted(path.stem for path in (EXEMPLAR_DIR / "tiers").glob("*.yml"))


@pytest.fixture(scope="module")
def exemplar() -> dict[str, Any]:
    return yaml.safe_load((EXEMPLAR_DIR / "profile.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def exemplar_deploy(exemplar: dict[str, Any]) -> DeployConfig:
    deploy = parse_deploy_block(exemplar)
    assert deploy is not None, "the exemplar profile must declare a deploy block"
    return deploy


@pytest.fixture(scope="module")
def web_terminals(exemplar: dict[str, Any]) -> dict[str, Any]:
    """The exemplar's ``modules.web_terminals`` subtree, as the host reads it."""
    return exemplar["config"]["modules.web_terminals"]


@pytest.fixture(scope="module")
def rendered_ci(exemplar: dict[str, Any], exemplar_deploy: DeployConfig) -> str:
    context = build_ci_context(
        exemplar, exemplar_deploy, EXEMPLAR_DIR, PROJECT_NAME, FROZEN_VERSION
    )
    return render(CI_TEMPLATES[exemplar_deploy.ci], context)


def test_exemplar_declares_more_than_one_render(web_terminals: dict[str, Any]) -> None:
    """Guards the rest of this module against becoming vacuous.

    Every assertion below is about a facility whose people sit on more than one
    tier. A profile edit that dropped the tier files or collapsed the roster
    onto one tier would leave them all passing on nothing.
    """
    assert _exemplar_tiers() == ["readonly", "readwrite"]
    assert {u["tier"] for u in web_terminals["users"]} == {"readonly", "readwrite"}


def test_pipeline_builds_no_web_terminal_image(rendered_ci: str) -> None:
    """No job in the pipeline builds a web-terminal image, per-render or not.

    The images stage covers the facility's own service directories only. A
    render image is the deploy host's to build, against the host's runtime and
    the render project `osprey build` already wrote — a pipeline that built one
    would be pushing a second, divergent source of the same tag.
    """
    jobs = {
        name: body for name, body in yaml.safe_load(rendered_ci).items() if isinstance(body, dict)
    }
    # A concrete image job carries no `stage:` of its own — it inherits one by
    # extending the hidden `.service-image` template. Selecting on `stage`
    # alone would scan only that template and skip every job that does the
    # actual building.
    hidden = {name for name, body in jobs.items() if body.get("stage") == "images"}
    image_jobs = {
        name: body for name, body in jobs.items() if name in hidden or body.get("extends") in hidden
    }
    assert hidden < set(image_jobs), (
        "the exemplar must render at least one concrete image job, not just the "
        f"template it extends (found {sorted(image_jobs)})"
    )

    forbidden = ["web-terminal", *_exemplar_tiers()]
    for name, body in image_jobs.items():
        # The job's whole body, not just its name: a job called `image:foo`
        # whose script builds a render image would pass a name-only check.
        text = yaml.safe_dump(body)
        for token in forbidden:
            assert token not in name, f"image job {name!r} names {token!r}"
            assert token not in text, f"image job {name!r} references {token!r}"


def test_every_referenced_render_is_a_host_side_build_unit(
    web_terminals: dict[str, Any],
) -> None:
    """Each render a user references resolves to exactly one build unit.

    This is the half that replaced the pipeline's per-render job block: the
    catalog entry a user points at is what the host builds, one image per
    distinct render regardless of how many users share it.
    """
    # The host reads the rendered config, where the build has projected each
    # tier file in `tiers/` into a catalog entry carrying its render name and
    # directory.
    catalog = {
        name: {
            "kind": "tier",
            "project": render_project(PROJECT_NAME, name)[0],
            "project_path": render_project(PROJECT_NAME, name)[1],
        }
        for name in _exemplar_tiers()
    }
    rendered = {**web_terminals, "renders": catalog}

    resolved_users = [
        {"render": user["tier"], "project": catalog[user["tier"]]["project"]}
        for user in web_terminals["users"]
    ]
    # A third user sharing a render must not add a second build of it.
    resolved_users.append({"render": "readonly", "project": catalog["readonly"]["project"]})

    units = _referenced_renders({"modules": {"web_terminals": rendered}}, resolved_users)

    assert [u["render"] for u in units] == ["readonly", "readwrite"]
    for unit in units:
        entry = catalog[unit["render"]]
        assert unit["project"] == entry["project"]
        assert unit["project_path"] == entry["project_path"]
        # A tier render is built from its own file in the repo's `tiers/`.
        assert unit["build_profile"] == f"tiers/{unit['render']}.yml"
        assert (EXEMPLAR_DIR / unit["build_profile"]).is_file()
