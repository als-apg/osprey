"""String-content guards on shipped `osprey-build-deploy` skill templates.

Some templates under `templates/skills/osprey-build-deploy/templates/` are
**agent-rendered prose**, not Jinja: an LLM reads the `# IF`/`# FOR` pseudo-loop
comments at facility-scaffolding time and emits the real, concrete file. There
is no Python producer to exercise, so the only guard available is a string
assertion on the shipped template text itself (PLAN Task 4.1) — this mirrors
how the plain-Jinja templates in this same skill get producer-based tests
(e.g. `tests/deployment/test_compose_generator.py`), just without a render
step in between.

This module covers `.gitlab-ci.yml`'s per-persona build-job block (PLAN
Task 4.1, multi-user-support P4): each non-default entry in
`modules.web_terminals.personas` needs its own `osprey build` render plus its
own suffixed `web-terminal-<persona>` image, built/pushed at SHA and re-tagged
to `:latest` by the `release` job — all without touching the existing
un-suffixed default `build-web-terminal`/`web-terminal` job, which predates
personas entirely.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import pytest

TEMPLATE_PATH = "templates/skills/osprey-build-deploy/templates/core/.gitlab-ci.yml"


@pytest.fixture()
def gitlab_ci_template() -> str:
    with as_file(files("osprey").joinpath(TEMPLATE_PATH)) as path:
        return Path(path).read_text(encoding="utf-8")


def test_default_web_terminal_job_untouched(gitlab_ci_template: str) -> None:
    """The pre-existing un-suffixed default job must survive unmodified."""
    assert "build-web-terminal:\n" in gitlab_ci_template
    assert "-t $REGISTRY/web-terminal:$CI_COMMIT_SHORT_SHA .\n" in gitlab_ci_template
    assert "docker push $REGISTRY/web-terminal:$CI_COMMIT_SHORT_SHA\n" in gitlab_ci_template


def test_persona_render_loop_in_osprey_build_stage(gitlab_ci_template: str) -> None:
    """Each non-default persona gets its own `osprey build <project> <build_profile>`."""
    assert (
        "# FOR each in modules.web_terminals.personas where each.name != default_persona"
        in gitlab_ci_template
    )
    assert "osprey build ${each.project} ${each.build_profile}" in gitlab_ci_template
    assert "--runtime-root /app/${each.project}" in gitlab_ci_template
    assert "artifacts/personas/${each.name}" in gitlab_ci_template
    assert "cp -r build-output/${each.project} artifacts/personas/${each.name}/${each.project}" in (
        gitlab_ci_template
    )


def test_persona_docker_build_job_block(gitlab_ci_template: str) -> None:
    """Each non-default persona gets its own suffixed build/push job, built from
    its own rendered project via build-context-scoped args (never the runtime
    OSPREY_PROJECT_DIR name — see docker-compose.yml.j2/sdk_runner.py/dispatch_api.py
    for that name's actual, unrelated in-container meaning)."""
    assert "build-web-terminal-${each.name}:" in gitlab_ci_template
    assert (
        "--build-arg OSPREY_PROJECT_SRC=artifacts/personas/${each.name}/${each.project}"
        in gitlab_ci_template
    )
    assert "--build-arg OSPREY_PROJECT_NAME=${each.project}" in gitlab_ci_template
    # The runtime OSPREY_PROJECT_DIR name (docker-compose.yml.j2/sdk_runner.py/
    # dispatch_api.py) must never be reused as a build-arg here — only the
    # explanatory comment is allowed to name it, never a `--build-arg` line.
    assert "--build-arg OSPREY_PROJECT_DIR=" not in gitlab_ci_template
    assert "-t $REGISTRY/web-terminal-${each.name}:$CI_COMMIT_SHORT_SHA .\n" in gitlab_ci_template
    assert (
        "docker push $REGISTRY/web-terminal-${each.name}:$CI_COMMIT_SHORT_SHA\n"
        in gitlab_ci_template
    )
    # Reuses the shared docker-build-template anchor like every other build-* job.
    assert "build-web-terminal-${each.name}:\n  <<: *docker-build-template" in gitlab_ci_template


def test_release_job_covers_persona_images(gitlab_ci_template: str) -> None:
    """`release` needs each persona build job and re-tags each image to :latest."""
    release_section = gitlab_ci_template.split("release:\n", 1)[1]

    assert "- build-web-terminal-${each.name}" in release_section
    # The un-suffixed default job stays listed alongside the persona loop.
    assert "- build-web-terminal\n" in release_section

    assert 'IMAGES="$IMAGES web-terminal-${each.name}"' in release_section
    # The un-suffixed default image stays in the re-tag list too.
    assert 'IMAGES="$IMAGES web-terminal"\n' in release_section


def test_persona_loop_guarded_by_web_terminals_module(gitlab_ci_template: str) -> None:
    """Persona job blocks stay nested inside `# IF MODULE web_terminals.enabled`."""
    # web_terminals.enabled guards the persona loop at all four sites it
    # appears (osprey-build render, docker-build job, release `needs:`,
    # release `IMAGES` script) — one `# IF` per site, one `# FOR` per site.
    assert gitlab_ci_template.count("# IF MODULE web_terminals.enabled") == 4
    assert (
        gitlab_ci_template.count(
            "# FOR each in modules.web_terminals.personas where each.name != default_persona"
        )
        == 4
    )
