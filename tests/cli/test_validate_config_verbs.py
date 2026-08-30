"""``osprey validate`` and ``osprey config`` — the two read-only repo verbs.

Both were reached, before this, by knowing which group they hid under and which
of four project-discovery rules that group used. They are top-level verbs now,
sharing one discovery rule: walk up from where you are standing to the nearest
``profile.yml``. These tests pin what that buys — the same answer from the repo
root and from three directories down — and the parts that had to survive the
move: the accumulated error report, the deploy-config lint, and the exit codes
a CI job gates on.

``config`` is three views of one subject, and the tests below hold each to the
file it claims to be showing: the source manifest verbatim (comments are why
anyone opens it), the build output as built, and the framework template that
needs no deployment at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.config_cmd import config
from osprey.cli.main import cli
from osprey.cli.validate_cmd import validate
from osprey.port_layout import DEFAULT_PORT_BASE, default_port

#: The frozen reference deployment — the shape `osprey init` is being built to
#: emit. Used here so the zero-argument path is exercised against a real
#: profile and not only against the two-line minimum.
EXEMPLAR = Path(__file__).resolve().parents[1] / "deployment" / "goldens" / "exemplar-profile"

#: A roster that stands the persona stack up, which is what makes the web-stack
#: lint apply at all: a config with no catalog is never asked about
#: ``registry.url``.
WEB_TERMINALS: dict[str, Any] = {
    "enabled": True,
    "users": [{"name": "operator", "index": 0, "persona": "readwrite"}],
    "default_persona": "readwrite",
    "personas": {"readwrite": {"build_profile": "hello-world", "project": "demo-readwrite"}},
}

#: A source manifest with a comment in it. The comment is the assertion: the
#: default view prints the file, not a YAML round-trip of the file.
PROFILE_WITH_COMMENTS = """\
# The facility this deployment serves.
name: Demo Facility
data_bundle: hello_world  # the bundled sample data
"""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A deployment repo: a directory with a valid ``profile.yml`` at its root."""
    root = tmp_path / "als-assistant"
    root.mkdir()
    (root / "profile.yml").write_text(PROFILE_WITH_COMMENTS, encoding="utf-8")
    return root


def _write_profile(root: Path, body: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "profile.yml"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# validate: discovery
# ---------------------------------------------------------------------------


def test_validate_takes_no_argument_at_the_repo_root(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal spelling. The repo you are standing in is the thing checked,
    so nobody has to name a path they are already inside of."""
    monkeypatch.chdir(repo)

    result = runner.invoke(validate, [])

    assert result.exit_code == 0, result.output
    assert "Profile is valid" in result.output
    assert "Demo Facility" in result.output


def test_validate_from_a_nested_directory_checks_the_same_repo(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git's rule, and the reason it is worth copying: a verb typed three
    directories down acts on the deployment, not on the directory."""
    nested = repo / "data" / "channels"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    result = runner.invoke(validate, [])

    assert result.exit_code == 0, result.output
    assert str(repo / "profile.yml") in result.output.replace("\n", "")


def test_validate_repo_flag_moves_the_starting_directory(
    runner: CliRunner, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--repo`` substitutes where the walk begins; it does not skip the walk.
    Pointing it at a subdirectory therefore resolves to the enclosing repo,
    exactly as standing in that subdirectory would."""
    nested = repo / "personas"
    nested.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.chdir(outside)

    result = runner.invoke(validate, ["--repo", str(nested)])

    assert result.exit_code == 0, result.output
    assert "Profile is valid" in result.output


def test_validate_outside_any_repo_is_an_operator_error_not_a_verdict(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 1, not 2. A CI job gating on this verb must be able to tell "your
    profile is wrong" from "you ran me in the wrong directory"."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(validate, [])

    assert result.exit_code == 1, result.output
    assert "No OSPREY deployment repo found" in result.output
    assert "osprey init" in result.output


def test_validate_reports_the_exemplar_deployment_valid(runner: CliRunner) -> None:
    """The reference deployment is the specification the verb is derived from,
    so it validates through the same zero-argument path an operator uses."""
    result = runner.invoke(validate, ["--repo", str(EXEMPLAR)])

    assert result.exit_code == 0, result.output
    assert "Profile is valid" in result.output


# ---------------------------------------------------------------------------
# validate: the retained positional
# ---------------------------------------------------------------------------


def test_validate_checks_a_named_profile_file(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positional survives for the profile that is not a repo root — a
    persona delta, which is a profile in its own right and can be wrong on its
    own terms."""
    delta = repo / "personas" / "reader.yml"
    delta.parent.mkdir()
    delta.write_text("name: Reader\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    result = runner.invoke(validate, [str(delta)])

    assert result.exit_code == 0, result.output
    assert "reader.yml" in result.output


def test_validate_names_a_directory_that_holds_no_profile(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A directory argument is accepted as shorthand for its profile.yml, so
    the failure has to say which file was missing and where one comes from."""
    empty = tmp_path / "not-a-repo"
    empty.mkdir()

    result = runner.invoke(validate, [str(empty)])

    assert result.exit_code == 2, result.output
    assert "profile.yml" in result.output
    assert "osprey init" in result.output


def test_validate_refuses_repo_and_a_path_together(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two ways to name one subject, and no rule for which wins. Refused rather
    than silently ignoring the loser, which is how an operator ends up
    validating a profile they did not mean."""
    monkeypatch.chdir(repo)

    result = runner.invoke(validate, ["--repo", str(repo), str(repo / "profile.yml")])

    assert result.exit_code == 2, result.output
    assert "--repo and TARGET" in result.output


# ---------------------------------------------------------------------------
# validate: what it checks, and what it exits with
# ---------------------------------------------------------------------------


def test_validate_exits_2_on_an_invalid_profile(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CI gate. Exit 2 with the problem named — the code the emitted
    pipeline has always keyed on, unchanged by the move."""
    root = tmp_path / "broken"
    root.mkdir()
    (root / "profile.yml").write_text(
        "name: Bad\ndata_bundle: hello_world\nconfig: []\n", encoding="utf-8"
    )
    monkeypatch.chdir(root)

    result = runner.invoke(validate, [])

    assert result.exit_code == 2, result.output
    assert "config must be a mapping" in result.output


def test_validate_still_runs_the_deploy_config_lint(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The web-stack lint is a second check past profile validity — a profile
    can be structurally fine and still describe a stack that cannot deploy. It
    moved with the verb; without this test the relocation could quietly drop
    it and every one of the tests above would still pass."""
    root = tmp_path / "unbuildable"
    _write_profile(
        root,
        {
            "name": "Demo Facility",
            "data_bundle": "hello_world",
            "config": {"facility.prefix": "demo", "modules.web_terminals": dict(WEB_TERMINALS)},
        },
    )
    monkeypatch.chdir(root)

    result = runner.invoke(validate, [])

    assert result.exit_code == 2, result.output
    assert "registry.url is not set" in result.output


def test_validate_refuses_a_per_type_limits_block_missing_a_leaf(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-type ``limits_checking`` block overrides the deployment-wide pair
    as a whole, so a block stating one leaf has no posture to answer with and
    every reader falls back to refusing unlisted channels. The verb an operator
    runs after a profile edit has to say so — by name — rather than report a
    deployment valid that will quietly ignore what the profile stated."""
    root = tmp_path / "half-a-block"
    _write_profile(
        root,
        {
            "name": "Demo Facility",
            "data_bundle": "hello_world",
            "config": {
                "control_system.connector.virtual_accelerator.limits_checking.enabled": True
            },
        },
    )
    monkeypatch.chdir(root)

    result = runner.invoke(validate, [])

    assert result.exit_code == 2, result.output
    assert "allow_unlisted_channels" in result.output
    assert "virtual_accelerator" in result.output


def test_validate_passes_a_complete_per_type_limits_block(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the same check: a block stating both leaves is the
    supported way to relax a simulator without relaxing the machine, so it must
    reach the verdict untouched. Pinned beside the refusal because a lint that
    refuses everything passes the test above on its own."""
    root = tmp_path / "whole-block"
    _write_profile(
        root,
        {
            "name": "Demo Facility",
            "data_bundle": "hello_world",
            "config": {
                "control_system.connector.virtual_accelerator.limits_checking.enabled": True,
                "control_system.connector.virtual_accelerator.limits_checking."
                "allow_unlisted_channels": True,
            },
        },
    )
    monkeypatch.chdir(root)

    result = runner.invoke(validate, [])

    assert result.exit_code == 0, result.output
    assert "Profile is valid" in result.output


def test_validate_points_at_build_not_at_a_project_name(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next-step line is the verb's own hand-off, so it names the surface
    that exists now: one repo, rendered by a bare ``osprey build``."""
    monkeypatch.chdir(repo)

    result = runner.invoke(validate, [])

    assert "osprey build" in result.output
    assert "PROJECT_NAME" not in result.output


# ---------------------------------------------------------------------------
# config: the source view
# ---------------------------------------------------------------------------


def test_config_shows_the_source_manifest_verbatim(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Comments and all. The source manifest is hand-written, and its comments
    are half of what a reader opens it for — a YAML round-trip would drop them
    and print a file nobody has on disk."""
    monkeypatch.chdir(repo)

    result = runner.invoke(config, [])

    assert result.exit_code == 0, result.output
    assert result.output == PROFILE_WITH_COMMENTS + "\n"


def test_config_from_a_nested_directory_finds_the_same_repo(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One discovery rule means ``config`` and ``validate`` cannot disagree
    about which deployment the operator is standing in."""
    nested = repo / "var" / "audit"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    result = runner.invoke(config, [])

    assert result.exit_code == 0, result.output
    assert "Demo Facility" in result.output


def test_config_outside_any_repo_says_so(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(config, [])

    assert result.exit_code == 1, result.output
    assert "No OSPREY deployment repo found" in result.output


# ---------------------------------------------------------------------------
# config: the rendered view
# ---------------------------------------------------------------------------


def test_config_rendered_reads_the_build_output(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the containers actually read. Distinct from the source view on
    purpose: when a deployment behaves unlike its manifest suggests, the
    difference between these two files is the answer."""
    built = repo / "build"
    built.mkdir()
    (built / "config.yml").write_text("project:\n  name: als_assistant\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    result = runner.invoke(config, ["--rendered"])

    assert result.exit_code == 0, result.output
    assert "als_assistant" in result.output
    assert "Demo Facility" not in result.output


def test_config_rendered_before_a_build_says_to_build(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build/`` is disposable, so its absence is an ordinary state — a fresh
    clone, or the morning after a reset — and the message is an instruction,
    not a diagnosis."""
    monkeypatch.chdir(repo)

    result = runner.invoke(config, ["--rendered"])

    assert result.exit_code == 1, result.output
    assert "osprey build" in result.output


# ---------------------------------------------------------------------------
# config: the defaults view
# ---------------------------------------------------------------------------


def test_config_defaults_needs_no_deployment_repo(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one repo-free mode of a repo-scoped verb: the framework template is
    a property of the installation, so "what keys exist?" is answerable from
    anywhere, including from a directory where nothing is deployed yet."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(config, ["--defaults"])

    assert result.exit_code == 0, result.output
    assert "No OSPREY deployment repo found" not in result.output
    assert "project_name:" in result.output


def test_config_defaults_renders_parseable_yaml(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The view is a rendered Jinja template, so a template edit can produce
    something that prints fine and parses as nothing. Loading it here is what
    makes that a test failure instead of a support question."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(config, ["--defaults"])

    loaded = yaml.safe_load(result.output)

    assert isinstance(loaded, dict), result.output
    assert loaded["project_name"] == "example_project"
    assert "control_system" in loaded


def test_config_defaults_shows_the_layout_at_the_default_base(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This view has no deployment to resolve a port base from, so it is the
    one place the layout's own default is the right answer — and the ports it
    prints have to BE that default, not an empty value or some other base."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(config, ["--defaults"])
    loaded = yaml.safe_load(result.output)

    assert loaded["services"]["openobserve"]["port"] == default_port(
        "openobserve", base=DEFAULT_PORT_BASE
    )


def test_config_refuses_rendered_and_defaults_together(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different files, one output stream. Naming both is a question with no
    answer, so it is a usage error rather than a silent precedence rule."""
    monkeypatch.chdir(repo)

    result = runner.invoke(config, ["--rendered", "--defaults"])

    assert result.exit_code == 2, result.output
    assert "--rendered and --defaults" in result.output


def test_config_output_pipes_cleanly(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``osprey config --defaults > defaults.yml`` has to write the file it
    showed. Off a terminal the content goes out unchanged — no header, no
    rewrapping to the console width, nothing to strip back off."""
    monkeypatch.chdir(repo)

    source = runner.invoke(config, [])
    defaults = runner.invoke(config, ["--defaults"])

    for result in (source, defaults):
        assert "Source profile" not in result.output
        assert "Framework default" not in result.output
    assert yaml.safe_load(source.output)["name"] == "Demo Facility"


# ---------------------------------------------------------------------------
# Both verbs are reachable as themselves
# ---------------------------------------------------------------------------


def test_both_verbs_are_registered_as_top_level_commands() -> None:
    """The relocation is only real once the CLI resolves the names. Checked
    through the lazy registry rather than by running the group, which is what
    an operator's ``osprey validate`` goes through."""
    assert cli.get_command(None, "validate") is validate
    assert cli.get_command(None, "config") is config
