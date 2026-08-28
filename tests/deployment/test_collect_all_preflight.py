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
    """Write a persona project under *root*; return its relative project_path.

    The bluesky server is spelled out because it is opt-in in the registry: a
    project that omits the key runs no server, holds no launch token, and so
    cannot raise the shell/token conflict this file's collection depends on.
    """
    project_dir = root / "profiles" / name
    (project_dir / ".claude").mkdir(parents=True)
    (project_dir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "project_name": name,
                "control_system": {"writes_enabled": writes},
                "claude_code": {"servers": {"bluesky": {"enabled": True}}},
            }
        ),
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
        # Master switch on: an inert block is asked for no credential at all, so
        # leaving it out would make the telemetry problem below disappear.
        "claude_code": {
            "telemetry": {"enabled": True, "openobserve": {"password": "${OBS_PASSWORD}"}}
        },
        "modules": {
            "web_terminals": {
                "enabled": True,
                "image_source": "local",
                # `token` — today's default posture, and what this fixture has always
                # meant: no login wall, each terminal reached through its own magic
                # link. `none` now means OPEN, which carries a collectable problem of
                # its own (personas that can still reach the host network), and this
                # deployment is about the other three.
                "auth": {"method": "token"},
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
# The required-env probe: what the profile says the agent cannot run without
# ---------------------------------------------------------------------------


def _write_profile(root: Path, document: dict) -> None:
    (root / "profile.yml").write_text(yaml.safe_dump(document), encoding="utf-8")


def _required(root: Path, *names: str, deploy: dict | None = None) -> None:
    document: dict = {"name": "facility", "env": {"required": list(names)}}
    if deploy is not None:
        document["deploy"] = deploy
    _write_profile(root, document)


def test_a_declared_name_no_source_provides_is_reported(tmp_path):
    """The probe's reason for existing: the profile says the agent cannot run
    without this variable, and nothing the stack reads carries it. Serially
    this is a deploy that starts and then fails at the first request."""
    _required(tmp_path, "FACILITY_API_TOKEN")

    problems = container_lifecycle._required_env_problems(tmp_path, {})

    assert len(problems) == 1
    problem, remedy = problems[0]
    assert "FACILITY_API_TOKEN" in problem
    assert "env.required" in problem
    # The remedy names the file that lists it and the copy that creates a `.env`.
    assert ".env.example" in remedy
    assert "cp .env.example .env" in remedy


@pytest.mark.parametrize("filename", [".env", ".env.shared"])
def test_either_chain_file_provides_the_value(tmp_path, filename):
    """Existence is the question, so either half of the chain answers it — the
    shared default no less than the host's own file."""
    _required(tmp_path, "FACILITY_API_TOKEN")
    (tmp_path / filename).write_text("FACILITY_API_TOKEN=set\n", encoding="utf-8")

    assert container_lifecycle._required_env_problems(tmp_path, {}) == []


def test_an_exported_value_provides_it_with_no_file_at_all(tmp_path):
    """A shell export reaches compose without being in any file, so a repo with
    no chain at all is not refused when the environment carries the value."""
    _required(tmp_path, "FACILITY_API_TOKEN")

    assert container_lifecycle._required_env_problems(tmp_path, {"FACILITY_API_TOKEN": "set"}) == []


def test_the_chain_is_read_from_the_repo_not_from_the_process(tmp_path):
    """Under `osprey up --repo` the CLI's own entry-time `.env` load is a
    different directory's chain, so the probe reads the deployment repo's files
    rather than trusting the environment it inherited."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _required(repo, "FACILITY_API_TOKEN")
    (repo / ".env").write_text("FACILITY_API_TOKEN=set\n", encoding="utf-8")

    assert container_lifecycle._required_env_problems(repo, {}) == []


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_value_is_a_missing_value(tmp_path, value):
    """`.env.example` renders every required name as a bare `NAME=`, so the
    copied-but-never-filled-in file is precisely the shape this probe exists to
    catch. Counting its blanks as answers would silence it for the operator who
    most needs it."""
    _required(tmp_path, "FACILITY_API_TOKEN")
    (tmp_path / ".env").write_text(f"FACILITY_API_TOKEN={value}\n", encoding="utf-8")

    problems = container_lifecycle._required_env_problems(tmp_path, {"OTHER": "x"})

    assert len(problems) == 1
    assert "FACILITY_API_TOKEN" in problems[0][0]


def test_an_empty_local_value_is_still_answered_by_the_environment(tmp_path):
    """Empty means "no value here", not "no value anywhere" — the union is over
    sources, so an export still satisfies a name blanked out in the file."""
    _required(tmp_path, "FACILITY_API_TOKEN")
    (tmp_path / ".env").write_text("FACILITY_API_TOKEN=\n", encoding="utf-8")

    assert container_lifecycle._required_env_problems(tmp_path, {"FACILITY_API_TOKEN": "set"}) == []


def test_every_missing_name_is_its_own_finding_in_declaration_order(tmp_path):
    """One finding per name, in the order of the file the operator opens to fix
    them, rather than one finding listing three names."""
    _required(tmp_path, "ZULU_TOKEN", "ALPHA_TOKEN", "MIDDLE_TOKEN")
    (tmp_path / ".env").write_text("MIDDLE_TOKEN=set\n", encoding="utf-8")

    problems = container_lifecycle._required_env_problems(tmp_path, {})

    assert [problem.split()[0] for problem, _remedy in problems] == ["ZULU_TOKEN", "ALPHA_TOKEN"]


def test_ci_only_credentials_never_refuse_a_host_that_builds_its_own_images(tmp_path):
    """The registry and external-project tokens are read where images are pushed
    and pulled from a registry. A developer host that builds locally has no use
    for either, and refusing its start over them would make every such deploy
    unstartable — even though the profile declares them like everything else,
    because the deploy block has no env channel of its own."""
    _required(
        tmp_path,
        "FACILITY_REGISTRY_TOKEN",
        "PARTNER_PULL_TOKEN",
        "FACILITY_API_TOKEN",
        deploy={
            "ci": "gitlab",
            "registry": {
                "url": "git.example.org:5050/physics/facility",
                "token_env_var": "FACILITY_REGISTRY_TOKEN",
            },
            "external_projects": [
                {
                    "name": "partner",
                    "url": "git.example.org:5050/partner",
                    "image": "partner:1",
                    "token_env_var": "PARTNER_PULL_TOKEN",
                }
            ],
        },
    )

    problems = container_lifecycle._required_env_problems(tmp_path, {})

    assert [problem.split()[0] for problem, _remedy in problems] == ["FACILITY_API_TOKEN"]


def test_a_machine_minted_name_is_never_the_operators_to_supply(tmp_path):
    """The exemplar profile declares its dispatch tokens under `env.required`,
    and `osprey up` mints them itself. With the service deployed the mint above
    the pass has already written the value; without it, nothing in the stack
    reads the name at all. Either way a refusal would send the operator to
    invent a secret the deploy owns."""
    minted = sorted(container_lifecycle._deploy_written_env_vars())
    assert "EVENT_DISPATCHER_TOKEN" in minted  # the census the exclusion reuses
    _required(tmp_path, *minted, "FACILITY_API_TOKEN")

    problems = container_lifecycle._required_env_problems(tmp_path, {})

    assert [problem.split()[0] for problem, _remedy in problems] == ["FACILITY_API_TOKEN"]


def test_a_deploy_block_without_tokens_excludes_nothing(tmp_path):
    """The negative control for the exclusion: a profile that names no
    credential excludes no name, so the carve-out cannot be silently swallowing
    the whole declaration."""
    _required(
        tmp_path,
        "FACILITY_API_TOKEN",
        deploy={"ci": "gitlab", "registry": {"url": "git.example.org:5050/physics/facility"}},
    )

    assert len(container_lifecycle._required_env_problems(tmp_path, {})) == 1


@pytest.mark.parametrize(
    "document",
    [
        None,  # no profile.yml at all
        {"name": "facility"},  # no env block
        {"name": "facility", "env": {"pinned": ["A"]}},  # no required key
        {"name": "facility", "env": {"required": "FACILITY_API_TOKEN"}},  # scalar, not a list
        {"name": "facility", "env": {"required": [None, ""]}},  # unusable entries
        {"name": "facility", "env": "required"},  # env is not a mapping
    ],
)
def test_every_nothing_declared_shape_reports_nothing(tmp_path, document):
    """A profile that declares nothing, or declares it unusably, leaves the
    deploy exactly where it was before the probe existed."""
    if document is not None:
        _write_profile(tmp_path, document)

    assert container_lifecycle._required_env_problems(tmp_path, {}) == []


def test_an_unparseable_profile_is_not_this_probes_refusal(tmp_path):
    """A broken profile is the staleness check's finding and `osprey build`'s
    failure. Raising it from here would report it as a missing variable."""
    (tmp_path / "profile.yml").write_text("env: [unclosed\n", encoding="utf-8")

    assert container_lifecycle._required_env_problems(tmp_path, {}) == []


def test_a_name_declared_twice_is_reported_once(tmp_path):
    _required(tmp_path, "FACILITY_API_TOKEN", "FACILITY_API_TOKEN")

    assert len(container_lifecycle._required_env_problems(tmp_path, {})) == 1


def test_the_probe_writes_nothing(tmp_path):
    """Pure per the pass's contract: the probes run between provisioners that
    ARE ordered against each other, so a side effect here would silently be a
    step in that order."""
    _required(tmp_path, "FACILITY_API_TOKEN")
    before = sorted(p.name for p in tmp_path.iterdir())

    container_lifecycle._required_env_problems(tmp_path, {})

    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_the_missing_variable_is_collected_with_the_other_findings(tmp_path):
    """Through the pass rather than the probe: a deployment missing a required
    variable AND carrying another problem is described once, not twice."""
    _required(tmp_path, "FACILITY_API_TOKEN")

    with pytest.raises(UnmetPreconditionsError) as excinfo:
        container_lifecycle._collect_unmet_preconditions(
            _three_problem_config(tmp_path),
            tmp_path,
            {},
            dev_mode=False,
            web_terminals_enabled=True,
        )

    problems = [problem for problem, _remedy in excinfo.value.findings]
    assert any("FACILITY_API_TOKEN" in problem for problem in problems)
    # The web findings still come first, and the chain question before the
    # build's own pin: the order a deploy would have met them in.
    assert "unrendered" in problems[0]


# ---------------------------------------------------------------------------
# The profile the deploy reads is the one THIS host built
# ---------------------------------------------------------------------------
# One repo, several `profiles/<name>.yml` overlays, and `.env.variant` naming
# the one this host builds. `osprey build` merges that overlay over
# `profile.yml` before resolving anything, so a declaration it contributes is
# part of what the built stack IS. Reading the tracked file alone at `osprey up`
# would leave every such declaration unenforced — the build renders a project
# whose profile requires a variable, and the deploy goes on believing nothing
# was required.


def _variant(root: Path, name: str, document: dict | None = None, *, select: bool = True) -> None:
    """Write an overlay under ``profiles/`` and (by default) select it."""
    if document is not None:
        variants = root / "profiles"
        variants.mkdir(exist_ok=True)
        (variants / f"{name}.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    if select:
        (root / ".env.variant").write_text(f"OSPREY_PROFILE_VARIANT={name}\n", encoding="utf-8")


def test_a_required_name_the_selected_variant_adds_is_enforced(tmp_path):
    """The finding this exists for: a host whose variant declares the variable."""
    _write_profile(tmp_path, {"name": "facility"})
    _variant(tmp_path, "teststand", {"env": {"required": ["FACILITY_API_TOKEN"]}})

    problems = container_lifecycle._required_env_problems(tmp_path, {})

    assert [problem.split()[0] for problem, _remedy in problems] == ["FACILITY_API_TOKEN"]


def test_the_variants_declarations_are_added_to_the_tracked_ones(tmp_path):
    """The build merges profile layers; a list does not replace, it unions.

    A deploy that took the overlay as the whole answer would stop enforcing
    everything the tracked profile declares the moment a host picked a variant.
    """
    _required(tmp_path, "TRACKED_TOKEN")
    _variant(tmp_path, "teststand", {"env": {"required": ["VARIANT_TOKEN"]}})

    problems = container_lifecycle._required_env_problems(tmp_path, {})

    assert sorted(problem.split()[0] for problem, _remedy in problems) == [
        "TRACKED_TOKEN",
        "VARIANT_TOKEN",
    ]


def test_a_pin_the_selected_variant_adds_is_read_back(tmp_path):
    """`env.pinned` travels the same path, and its consumers refuse on it."""
    _write_profile(tmp_path, {"name": "facility", "env": {"pinned": ["TRACKED_URL"]}})
    _variant(tmp_path, "teststand", {"env": {"pinned": ["VARIANT_URL"]}})

    assert container_lifecycle.pinned_env_keys(tmp_path) == {"TRACKED_URL", "VARIANT_URL"}


def test_an_unselected_variant_contributes_nothing(tmp_path):
    """The overlays are tracked; the choice between them is not. A repo that
    carries three of them and selects none builds the tracked profile."""
    _write_profile(tmp_path, {"name": "facility"})
    _variant(tmp_path, "teststand", {"env": {"required": ["VARIANT_TOKEN"]}}, select=False)

    assert container_lifecycle._required_env_problems(tmp_path, {}) == []


def test_a_variant_this_repo_does_not_define_is_not_this_probes_refusal(tmp_path):
    """`osprey build` refuses an unknown variant, listing the ones that work.

    Raising it from a deploy probe would abort the start over a file the deploy
    does not build from — the same reason an unparseable profile is absorbed.
    """
    _required(tmp_path, "TRACKED_TOKEN")
    _variant(tmp_path, "teststand", {"env": {"required": ["VARIANT_TOKEN"]}}, select=False)
    (tmp_path / ".env.variant").write_text("OSPREY_PROFILE_VARIANT=controlroom\n", encoding="utf-8")

    problems = container_lifecycle._required_env_problems(tmp_path, {})

    assert [problem.split()[0] for problem, _remedy in problems] == ["TRACKED_TOKEN"]


def test_a_variant_named_by_a_repo_with_no_variants_leaves_the_profile_standing(tmp_path):
    """What a host looks like after the variants are removed from the repo."""
    _required(tmp_path, "TRACKED_TOKEN")
    (tmp_path / ".env.variant").write_text("OSPREY_PROFILE_VARIANT=teststand\n", encoding="utf-8")

    problems = container_lifecycle._required_env_problems(tmp_path, {})

    assert [problem.split()[0] for problem, _remedy in problems] == ["TRACKED_TOKEN"]


@pytest.mark.parametrize("overlay", ["env: [unclosed\n", "a string, not a mapping\n", ""])
def test_an_unusable_overlay_leaves_the_tracked_document_standing(tmp_path, overlay):
    """Same tolerant rule the tracked file gets, applied to the layer above it."""
    _required(tmp_path, "TRACKED_TOKEN")
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "teststand.yml").write_text(overlay, encoding="utf-8")
    (tmp_path / ".env.variant").write_text("OSPREY_PROFILE_VARIANT=teststand\n", encoding="utf-8")

    problems = container_lifecycle._required_env_problems(tmp_path, {})

    assert [problem.split()[0] for problem, _remedy in problems] == ["TRACKED_TOKEN"]


def test_the_variant_read_writes_nothing(tmp_path):
    """The pass's purity contract covers the layer this reads too."""
    _required(tmp_path, "TRACKED_TOKEN")
    _variant(tmp_path, "teststand", {"env": {"required": ["VARIANT_TOKEN"]}})
    before = sorted(p.name for p in tmp_path.iterdir())

    container_lifecycle._required_env_problems(tmp_path, {})

    assert sorted(p.name for p in tmp_path.iterdir()) == before


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
        # Master switch on: an inert block is asked for no credential at all, so
        # leaving it out would make the telemetry problem below disappear.
        "claude_code": {
            "telemetry": {"enabled": True, "openobserve": {"password": "${OBS_PASSWORD}"}}
        },
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


def test_a_required_name_written_one_step_above_is_not_refused(
    tmp_path, stubbed_start, monkeypatch
):
    """The required-env probe inherits the pass's position: it reads the chain
    as it stands after every provisioner above it, so a required name a step of
    this same deploy writes into `.env` is not reported as missing on a fresh
    repo that has never been deployed.

    The counterfactual is below: with nothing writing it, the identical start
    refuses on the same name."""
    _required(tmp_path, "OBS_PASSWORD")
    monkeypatch.setattr(
        container_lifecycle,
        "_ensure_service_tokens",
        lambda *a, **k: (tmp_path / ".env").write_text("OBS_PASSWORD=minted\n", encoding="utf-8"),
    )

    container_lifecycle._start_stack(_mint_dependent_config(), [], tmp_path, detached=True)

    assert stubbed_start == ["build_image"]


def test_a_required_name_nothing_provides_refuses_before_the_build(
    tmp_path, stubbed_start, monkeypatch
):
    """The counterfactual, and the probe's whole point on the deploy path: the
    refusal lands in seconds, before the minutes-long image build."""
    _required(tmp_path, "FACILITY_API_TOKEN")
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)

    with pytest.raises(UnmetPreconditionsError, match="FACILITY_API_TOKEN"):
        container_lifecycle._start_stack(_mint_dependent_config(), [], tmp_path, detached=True)

    assert stubbed_start == []  # nothing was built


# ---------------------------------------------------------------------------
# The privilege belt on the deploy path
# ---------------------------------------------------------------------------
# `osprey up` used to read no lint at all: the two verbs that read the rendered
# altitude (`osprey scaffold web-terminals lint` and the `render` pre-render
# gate) are both authoring verbs. So a hand-edit to `build/config.yml` -- a file
# no build fingerprint covers, since `profile.yml` is untouched -- could serve a
# deployment-editing terminal with no login, fail the lint, and start anyway.
# The pass carries the two open-door findings now; what is tested here is that
# the refusal lands before anything is built or started.


def _open_terminal_config(root: Path) -> dict:
    """A rendered deployment serving a privileged persona with no login.

    The base floors both surfaces and the `admin` render lifts neither key back
    -- which is what a render that predates the floor looks like, and what a
    hand-edited one looks like too. `carol` opts out of the login wall.
    """
    for project, document in (
        ("admin-app", {"project_name": "admin-app"}),
        (
            "readonly-app",
            {
                "project_name": "readonly-app",
                "claude_code": {"permissions": {"deny": ["mcp__osprey_workspace__setup_patch"]}},
                "web": {"config_panel": {"enabled": False}},
            },
        ),
    ):
        project_dir = root / "build" / project
        project_dir.mkdir(parents=True)
        (project_dir / "config.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return {
        "deployed_services": [],
        "facility": {"prefix": "test"},
        "claude_code": {"permissions": {"deny": ["mcp__osprey_workspace__setup_patch"]}},
        "web": {"config_panel": {"enabled": False}},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "image_source": "local",
                "auth": {"method": "password"},
                "default_persona": "readonly",
                "users": [
                    {"name": "bob", "index": 0, "persona": "readonly"},
                    {"name": "carol", "index": 1, "persona": "admin", "login": False},
                ],
                "personas": {
                    "readonly": {"project": "readonly-app", "project_path": "build/readonly-app"},
                    "admin": {"project": "admin-app", "project_path": "build/admin-app"},
                },
            }
        },
    }


def test_an_open_privileged_terminal_refuses_the_start_before_anything_runs(
    tmp_path, stubbed_start, monkeypatch
):
    """The deploy-path half of the belt. Nothing is built, and no container
    runtime is asked anything -- the refusal is the first thing that happens."""
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)

    with pytest.raises(UnmetPreconditionsError, match="carol") as excinfo:
        container_lifecycle._start_stack(
            _open_terminal_config(tmp_path), [], tmp_path, detached=True
        )

    problem = next(problem for problem, _remedy in excinfo.value.findings if "'carol'" in problem)
    assert "without a login" in problem
    assert stubbed_start == []  # nothing was built, nothing was started


def test_the_same_deployment_behind_a_login_starts(tmp_path, stubbed_start, monkeypatch):
    """The counterfactual that gives the test above its meaning: one key back to
    its default and the identical start proceeds to the image build."""
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    config = _open_terminal_config(tmp_path)
    del config["modules"]["web_terminals"]["users"][1]["login"]

    container_lifecycle._start_stack(config, [], tmp_path, detached=True)

    assert stubbed_start == ["build_image"]


def test_a_privileged_default_persona_is_printed_and_the_start_proceeds(
    tmp_path, stubbed_start, monkeypatch, capsys
):
    """The advisory half. It is a real exposure and it is said out loud, but
    refusing the start of a running stack over the shape of its roster would
    stop a shift to fix a profile."""
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    config = _open_terminal_config(tmp_path)
    del config["modules"]["web_terminals"]["users"][1]["login"]
    config["modules"]["web_terminals"]["default_persona"] = "admin"

    container_lifecycle._start_stack(config, [], tmp_path, detached=True)

    assert stubbed_start == ["build_image"]
    assert "default_persona" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The open-mode egress refusal, as the collect-all pass reports it
# ---------------------------------------------------------------------------
# The pass ASKS the gate's predicate rather than raising on it, so its finding is
# built from the same per-persona answer the gate would have raised with. Asking
# only for the offender NAMES would report every deployment against the whole
# four-entry egress set -- and the operator whose persona lifted exactly one of
# them would go looking for four.


def test_the_collect_all_preflight_reports_the_open_mode_refusal(
    tmp_path, stubbed_start, monkeypatch
):
    """The refusal lands in the collect-all frame, and it names the ONE entry
    this deployment is missing rather than the whole egress set."""
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    config = _open_mode_terminal_config(tmp_path, lift="WebFetch")

    with pytest.raises(UnmetPreconditionsError) as excinfo:
        container_lifecycle._start_stack(config, [], tmp_path, detached=True)

    problem = next(
        problem
        for problem, _remedy in excinfo.value.findings
        if "may still reach the host network" in problem
    )
    headline = problem.split("\n")[0]
    assert "'WebFetch'" in headline
    # The three it DOES deny stay out of the headline: the report names what to
    # restore, not what the posture requires in general.
    assert "'Bash'" not in headline
    assert stubbed_start == []  # nothing was built, nothing was started


def _open_mode_terminal_config(root: Path, *, lift: str) -> dict:
    """The rendered deployment above, opened up, with *lift* missing from one
    persona's shipped deny list and every other persona clean."""
    from osprey.cli.templates.claude_code import DENY_DEFAULTS

    config = _open_terminal_config(root)
    del config["modules"]["web_terminals"]["users"][1]["login"]
    config["modules"]["web_terminals"]["auth"] = {"method": "none"}
    for project, deny in (
        ("readonly-app", list(DENY_DEFAULTS)),
        ("admin-app", [entry for entry in DENY_DEFAULTS if entry != lift]),
    ):
        settings_dir = root / "build" / project / ".claude"
        settings_dir.mkdir(parents=True)
        (settings_dir / "settings.json").write_text(
            json.dumps({"permissions": {"deny": deny}}), encoding="utf-8"
        )
    return config
