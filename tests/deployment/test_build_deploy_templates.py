"""Multi-user persona parity, held across the deploy split.

A facility with a persona catalog needs one web-terminal image per persona.
That used to be a pipeline concern: the CI template carried a per-persona
`osprey build` render and a per-persona `build-web-terminal-<persona>` job, and
the guards on it were string assertions against the shipped template text.

The pipeline no longer builds web-terminal images at all — the deploy host
does, from the persona catalog, when `osprey deploy up` runs. This module holds
both halves of that relocation to the exemplar profile, which declares two
personas: the rendered pipeline must build no web-terminal image, and every
referenced persona must still come out as its own host-side build unit. Losing
either half loses multi-user support without any test going red.

The builder's own behavior (build args, dev wheels, auto-render) is covered in
``tests/deployment/web_terminals/test_persona_images.py``; what is tested here
is only that the personas reach it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_deploy import DeployConfig, parse_deploy_block
from osprey.cli.deploy_scaffold_templates import CI_TEMPLATES, build_ci_context, render
from osprey.deployment.web_terminals.persona_images import _referenced_personas

EXEMPLAR_DIR = Path(__file__).parent / "goldens" / "exemplar-profile"

#: Matches the goldens' frozen provenance token; nothing here reads it.
FROZEN_VERSION = "OSPREY_VERSION"

PROJECT_NAME = "demo-facility"


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


def test_exemplar_declares_more_than_one_persona(web_terminals: dict[str, Any]) -> None:
    """Guards the rest of this module against becoming vacuous.

    Every assertion below is about a multi-persona facility. A profile edit
    that dropped the catalog would leave them all passing on nothing.
    """
    assert set(web_terminals["personas"]) == {"readonly", "readwrite"}
    assert {u["persona"] for u in web_terminals["users"]} == {"readonly", "readwrite"}


def test_pipeline_builds_no_web_terminal_image(
    rendered_ci: str, web_terminals: dict[str, Any]
) -> None:
    """No job in the pipeline builds a web-terminal image, per-persona or not.

    The images stage covers the facility's own service directories only. A
    persona image is the deploy host's to build, against the host's runtime and
    the persona project it renders on demand — a pipeline that built one would
    be pushing a second, divergent source of the same tag.
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

    forbidden = ["web-terminal", *web_terminals["personas"]]
    for name, body in image_jobs.items():
        # The job's whole body, not just its name: a job called `image:foo`
        # whose script builds a persona image would pass a name-only check.
        text = yaml.safe_dump(body)
        for token in forbidden:
            assert token not in name, f"image job {name!r} names {token!r}"
            assert token not in text, f"image job {name!r} references {token!r}"


def test_every_referenced_persona_is_a_host_side_build_unit(
    web_terminals: dict[str, Any],
) -> None:
    """Each persona a user references resolves to exactly one build unit.

    This is the half that replaced the pipeline's per-persona job block: the
    catalog entry a user points at is what the host builds, one image per
    distinct persona regardless of how many users share it.
    """
    catalog = web_terminals["personas"]
    resolved_users = [
        {"persona": user["persona"], "project": catalog[user["persona"]]["project"]}
        for user in web_terminals["users"]
    ]
    # A third user sharing a persona must not add a second build of it.
    resolved_users.append({"persona": "readonly", "project": catalog["readonly"]["project"]})

    units = _referenced_personas({"modules": {"web_terminals": web_terminals}}, resolved_users)

    assert [u["persona"] for u in units] == ["readonly", "readwrite"]
    for unit in units:
        entry = catalog[unit["persona"]]
        assert unit["project"] == entry["project"]
        assert unit["project_path"] == entry["project_path"]
        assert unit["build_profile"] == entry["build_profile"]
