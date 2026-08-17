"""The collect-all precondition pass: probe everything cheap, refuse once.

A start has several preconditions answerable from the deployment's own files
before anything is built. Raised one at a time they cost the operator a whole
deploy attempt per finding. These tests pin the three properties that make the
pass worth having and safe to run where it runs:

* it reports every finding in ONE refusal, in the order the deploy would have
  met them;
* it sits BELOW every step that writes what it reads, so it never refuses what
  the deploy is about to provide;
* it sits ABOVE the first minutes-long step.

The probes' own answers are pinned by the suites of the checks they mirror
(``test_provision.py``, ``test_artifacts.py``, ``test_env_production.py``);
what is tested here is the collecting, the placement, and the frame.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from osprey.deployment import container_lifecycle
from osprey.deployment.errors import DeploymentPreconditionError, UnmetPreconditionsError
from osprey.deployment.runtime_helper import ComposeProvider
from osprey.deployment.web_terminals import provision

# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------


def test_the_frame_enumerates_every_finding_under_a_counted_summary():
    exc = UnmetPreconditionsError(
        [
            ("the render is missing", "run `osprey build`"),
            ("the token conflicts with a shell", ""),
        ]
    )

    assert exc.summary == "2 unmet preconditions on this deployment"
    assert exc.reason == (
        "1. the render is missing\n   run `osprey build`\n\n2. the token conflicts with a shell"
    )


def test_a_multi_line_finding_stays_one_numbered_item():
    """Continuation lines indent under the number, or a finding that wraps
    reads as two separate problems in the rendered block."""
    exc = UnmetPreconditionsError([("first line\nsecond line", "")])

    assert exc.reason == "1. first line\n   second line"


def test_a_single_finding_gets_the_same_structured_frame():
    """N >= 1, deliberately: a start that refuses over one precondition reads
    exactly like one that refuses over four, so the common case is not also the
    unfamiliar one."""
    exc = UnmetPreconditionsError([("the render is missing", "run `osprey build`")])

    assert exc.summary == "1 unmet precondition on this deployment"
    assert exc.reason == "1. the render is missing\n   run `osprey build`"


def test_the_frame_carries_no_top_level_remedy():
    """With several things to fix there is no single `→` line that is honest,
    so each finding carries its own and the aggregate carries none -- the
    convention `gate_start_from_build` already follows for its two exits."""
    exc = UnmetPreconditionsError([("a", "fix a"), ("b", "fix b")])

    assert exc.remedy == ""


def test_the_aggregate_reaches_the_one_precondition_handler():
    """It has to be a DeploymentPreconditionError, or the start verbs' single
    `except` clause renders it as a crash instead of a refusal."""
    exc = UnmetPreconditionsError([("a", "")])

    assert isinstance(exc, DeploymentPreconditionError)
    assert str(exc) == f"{exc.summary}: {exc.reason}"


def test_the_cli_renders_the_aggregate_through_the_shared_refusal_shape(capsys):
    """No new handler: the aggregate lands on the one renderer every unmet
    precondition already goes through, and its three fields fill the same three
    slots."""
    import click

    from osprey.cli import deploy_cmd

    exc = UnmetPreconditionsError([("the render is missing", "run `osprey build`"), ("b", "")])

    with pytest.raises(click.Abort):
        deploy_cmd._abort_unmet_precondition(exc, nothing_done="Nothing was deployed.")

    rendered = capsys.readouterr().err
    assert "2 unmet preconditions on this deployment" in rendered
    assert "1. the render is missing" in rendered
    assert "2. b" in rendered
    assert "Nothing was deployed." in rendered
    # No `→` line: every remedy here belongs to one finding, and a trailing
    # arrow would name one of them as THE way through.
    assert "→" not in rendered


# ---------------------------------------------------------------------------
# Collecting: three problems, one refusal
# ---------------------------------------------------------------------------


def _persona_project(root: Path, name: str, *, writes: bool, denies_bash: bool) -> str:
    """Write a persona project under *root*; return its relative project_path."""
    project_dir = root / "profiles" / name
    (project_dir / ".claude").mkdir(parents=True)
    (project_dir / "config.yml").write_text(
        yaml.safe_dump({"project_name": name, "control_system": {"writes_enabled": writes}}),
        encoding="utf-8",
    )
    deny = ["Bash", "Edit"] if denies_bash else ["Edit"]
    (project_dir / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"deny": deny}}), encoding="utf-8"
    )
    return f"profiles/{name}"


def _three_problem_config(root: Path) -> dict:
    """One deployment carrying one of each collectable problem.

    Two personas, because the render problem and the shell/token conflict
    cannot both be true of the same one: a persona with no render contributes
    nothing to the launch-token set, so the conflict needs a persona whose
    render is there and wrong.
    """
    return {
        "deployed_services": [],
        "facility": {"prefix": "test"},
        "registry": {"url": "registry.example.org/test"},
        "claude_code": {"telemetry": {"openobserve": {"password": "${OBS_PASSWORD}"}}},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "image_source": "local",
                "auth": {"method": "none"},
                "default_persona": "readwrite",
                "users": [
                    {"name": "alice", "index": 0, "persona": "unrendered"},
                    {"name": "bob", "index": 1, "persona": "readwrite"},
                ],
                "personas": {
                    "unrendered": {"project": "unrendered", "project_path": "build/nowhere"},
                    "readwrite": {
                        "project": "rw",
                        "project_path": _persona_project(
                            root, "readwrite", writes=True, denies_bash=False
                        ),
                    },
                },
            }
        },
    }


def test_three_unmet_preconditions_are_reported_in_one_refusal(tmp_path):
    """The whole point of the pass. Serially, this deployment costs three
    deploy attempts to diagnose; here it costs one."""
    # A chain that is present but does not carry the telemetry password the
    # config demands -- the third problem.
    (tmp_path / ".env").write_text("FOO=bar\n", encoding="utf-8")

    with pytest.raises(UnmetPreconditionsError) as excinfo:
        container_lifecycle._collect_unmet_preconditions(
            _three_problem_config(tmp_path),
            tmp_path,
            {},
            dev_mode=False,
            web_terminals_enabled=True,
        )

    exc = excinfo.value
    assert exc.summary == "3 unmet preconditions on this deployment"
    problems = [problem for problem, _remedy in exc.findings]
    assert "unrendered" in problems[0]
    assert "readwrite" in problems[1]
    assert "OBS_PASSWORD" in problems[2]


def test_a_clean_deployment_collects_nothing(tmp_path, monkeypatch):
    """The negative control: the same shape with each problem fixed collects
    nothing, so the pass cannot be green by refusing everything.

    The render probe is inert here — a complete persona render is a fixture in
    its own right and `test_persona_images.py` owns that question. What this
    asserts is that the two probes reading files the fixture DOES write both
    answer "no problem" when the files are right."""
    monkeypatch.setattr(
        provision, "verify_persona_renders", lambda config, resolved_users, repo_root=None: None
    )
    (tmp_path / ".env").write_text("OBS_PASSWORD=set\n", encoding="utf-8")
    config = _three_problem_config(tmp_path)
    web_terminals = config["modules"]["web_terminals"]
    # One persona, rendered, denying the shell.
    web_terminals["default_persona"] = "safe"
    web_terminals["users"] = [{"name": "bob", "index": 0, "persona": "safe"}]
    web_terminals["personas"] = {
        "safe": {
            "project": "safe",
            "project_path": _persona_project(tmp_path, "safe", writes=True, denies_bash=True),
        }
    }

    container_lifecycle._collect_unmet_preconditions(
        config, tmp_path, {}, dev_mode=False, web_terminals_enabled=True
    )


def test_a_deploy_without_web_terminals_asks_none_of_the_web_questions(tmp_path):
    """The three web probes read artifacts only the web tier has. A plain
    services deploy has none of them, and must not be asked about them."""
    container_lifecycle._collect_unmet_preconditions(
        _three_problem_config(tmp_path),
        tmp_path,
        {},
        dev_mode=False,
        web_terminals_enabled=False,
    )


# ---------------------------------------------------------------------------
# The unreleased-pin probe: only about a build that will happen
# ---------------------------------------------------------------------------


def _worker_config() -> dict:
    return {"deployed_services": ["dispatch_worker"], "project_name": "proj"}


def test_the_pin_is_probed_only_for_a_build_this_deploy_will_make(monkeypatch):
    """The refusal belongs to the project-image build. A deployment that
    builds nothing must not be sent to fix a pin that was never going to be
    used -- both no-op cases of that build answer "nothing to report"."""
    monkeypatch.delenv("OSPREY_PIP_SPEC", raising=False)
    monkeypatch.setattr("osprey.version.is_release", lambda: False)

    # No dispatch worker at all.
    assert container_lifecycle._unreleased_pin_problem({"deployed_services": []}, {}, False) is None
    # A worker pinned to a prebuilt image.
    assert (
        container_lifecycle._unreleased_pin_problem(
            _worker_config(), {"OSPREY_WORKER_IMAGE": "registry.example.org/prebuilt:1"}, False
        )
        is None
    )


def test_a_definite_unreleased_pin_is_reported_with_its_own_remedy(monkeypatch):
    monkeypatch.delenv("OSPREY_PIP_SPEC", raising=False)
    monkeypatch.setattr("osprey.version.is_release", lambda: False)

    finding = container_lifecycle._unreleased_pin_problem(_worker_config(), {}, False)

    assert finding is not None
    problem, remedy = finding
    assert "Cannot pin osprey-framework" in problem
    assert "--dev" in remedy


def test_dev_mode_is_left_to_the_build_to_answer(monkeypatch):
    """Under --dev the pin is inert when the wheel stages and live when it does
    not, and which happens is not knowable without staging it. Pre-judging it
    here would refuse the very workflow the refusal recommends."""
    monkeypatch.delenv("OSPREY_PIP_SPEC", raising=False)
    monkeypatch.setattr("osprey.version.is_release", lambda: False)

    assert container_lifecycle._unreleased_pin_problem(_worker_config(), {}, True) is None


def test_an_operator_pin_answers_the_question_outright(monkeypatch):
    monkeypatch.setenv("OSPREY_PIP_SPEC", "git+https://example.invalid/osprey@abc123")
    monkeypatch.setattr("osprey.version.is_release", lambda: False)

    assert container_lifecycle._unreleased_pin_problem(_worker_config(), {}, False) is None


def test_the_build_target_is_the_gate_the_build_itself_uses(monkeypatch):
    """5a's extraction has to answer the same question `_build_project_image`
    asks, or the probe and the build disagree about whether a build happens."""
    assert container_lifecycle._project_image_build_target({"deployed_services": []}, {}) is None
    assert (
        container_lifecycle._project_image_build_target(
            _worker_config(), {"OSPREY_WORKER_IMAGE": "prebuilt:1"}
        )
        is None
    )
    assert container_lifecycle._project_image_build_target(_worker_config(), {}) == "proj:local"


# ---------------------------------------------------------------------------
# Placement: below every provisioner that writes what the pass reads
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed_start(monkeypatch):
    """Reduce ``_start_stack`` to the pass and the step after it.

    Everything that touches a host or a container runtime is inert; the two
    collaborators these tests re-patch are the ones whose ordering against the
    pass is the subject.
    """
    reached: list[str] = []
    for name in (
        "_check_shared_disk_preflight",
        "_preflight_host_ports",
        "_preflight_archiver_pymongo",
        "_preflight_stale_store_volumes",
        "_ensure_bluesky_substrate_env",
        "_ensure_bluesky_control_plane_keys",
        "_ensure_bluesky_document_plane_certs",
        "_preflight_env_chain_drift",
        "_preflight_env_shadowing",
        "preflight_web_terminals",
        "deploy_up_web_terminals",
        "log_endpoint_summary",
    ):
        monkeypatch.setattr(container_lifecycle, name, lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_build_project_image",
        lambda *a, **k: reached.append("build_image"),
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0),
    )
    # The persona-render probe has its own suite; kept inert so these tests are
    # about placement rather than about any repo's rendered state.
    monkeypatch.setattr(
        provision, "verify_persona_renders", lambda config, resolved_users, repo_root=None: None
    )
    monkeypatch.setattr(
        container_lifecycle, "_compose_provider", lambda config: ComposeProvider.DOCKER_V2
    )
    return reached


def _mint_dependent_config() -> dict:
    """A deployment whose only unmet precondition is one the mint resolves."""
    return {
        "deployed_services": ["osprey-mcp"],
        "facility": {"prefix": "test"},
        "claude_code": {"telemetry": {"openobserve": {"password": "${OBS_PASSWORD}"}}},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": "build/ops-app"}},
            }
        },
    }


def test_a_fresh_repo_with_no_env_file_is_not_refused_over_what_the_mint_writes(
    tmp_path, stubbed_start, monkeypatch
):
    """THE property the pass's position exists for.

    A fresh repo has committed defaults and no ``.env`` at all; the deploy's own
    token mint writes that file, and the pass runs below it. So the variable
    this config demands is there by the time the pass looks, and the start is
    not refused over a file the deploy just created.

    The counterfactual is the test below: the same repo and the same config,
    with nothing minted, IS refused. A pass hoisted above the mint would take
    that second outcome on the first repo too, and refuse every fresh deploy.
    """
    (tmp_path / ".env.shared").write_text("SHARED_DEFAULT=1\n", encoding="utf-8")
    monkeypatch.setattr(
        container_lifecycle,
        "_ensure_service_tokens",
        lambda *a, **k: (tmp_path / ".env").write_text("OBS_PASSWORD=minted\n", encoding="utf-8"),
    )

    container_lifecycle._start_stack(_mint_dependent_config(), [], tmp_path, detached=True)

    assert stubbed_start == ["build_image"]


def test_the_same_repo_is_refused_when_nothing_provides_the_variable(
    tmp_path, stubbed_start, monkeypatch
):
    """The counterfactual that gives the test above its meaning: with the mint
    writing nothing, the identical start refuses. So the first test is passing
    because of WHERE the pass sits, not because the probe is inert."""
    (tmp_path / ".env.shared").write_text("SHARED_DEFAULT=1\n", encoding="utf-8")
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)

    with pytest.raises(UnmetPreconditionsError, match="OBS_PASSWORD"):
        container_lifecycle._start_stack(_mint_dependent_config(), [], tmp_path, detached=True)

    assert stubbed_start == []  # nothing was built


def test_a_pre_rename_secrets_file_is_migrated_before_the_pass_reads_it(
    tmp_path, stubbed_start, monkeypatch
):
    """The pass's other upper constraint. ``migrate_users_env`` carries a
    pre-rename ``.env.production`` onto ``.env.users``; the pass reads that file
    one step later, so a repo whose secrets are sitting there under the old name
    is not refused for not having them."""
    (tmp_path / ".env.shared").write_text("SHARED_DEFAULT=1\n", encoding="utf-8")
    (tmp_path / ".env.production").write_text("OBS_PASSWORD=carried\n", encoding="utf-8")
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)

    container_lifecycle._start_stack(_mint_dependent_config(), [], tmp_path, detached=True)

    assert stubbed_start == ["build_image"]
    assert (tmp_path / ".env.users").is_file()
