"""``osprey set`` — the one sanctioned CLI write into ``profile.yml``.

The verb has one job and a short list of ways it must refuse. What is pinned
here is what makes it trustworthy as the *only* writer:

- the write lands in the repo's ``profile.yml``, in place, with the surrounding
  comments intact — a profile is a hand-edited document, and a writer that
  reflows it is one an operator stops using;
- ``build/config.yml`` is never touched, however the key is spelled: a setting
  that reached only the render would be undone by the next build;
- a command line that cannot be honored in full changes nothing at all, so a
  typo in the third pair cannot leave the first two applied;
- the shorthands absorbed from the retired ``config set-*`` commands land as
  the same literal dotted keys a hand-editor would write;
- standing outside a deployment repo, and writing ``extends``, are refusals.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.set_cmd import set as set_command
from osprey.deployment import staleness

pytestmark = pytest.mark.unit


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _profile_text(repo: Path) -> str:
    return (repo / "profile.yml").read_text(encoding="utf-8")


def _comment_lines(text: str) -> list[str]:
    """Every comment in the document, stripped of the indentation around it."""
    return [line.strip() for line in text.splitlines() if line.lstrip().startswith("#")]


def _invoke(runner: CliRunner, repo: Path, *args: str):
    """Run ``osprey set`` with ``--repo`` rather than a chdir.

    The flag substitutes the directory the walk starts from, so it exercises
    the same resolution an operator standing in the repo gets — without the
    process-global working-directory change an in-process runner would leak
    into every test after it.
    """
    return runner.invoke(set_command, ["--repo", str(repo), *args], catch_exceptions=False)


def _stamp_build(repo: Path) -> Path:
    """Stamp a build manifest carrying the profile's CURRENT fingerprint.

    Enough of a ``build/`` for the drift check to have something to compare
    against: a set that changes a key then reads as drift, which is exactly
    what the hint after the write has to say.
    """
    from osprey.cli.templates.manifest import get_framework_release_version

    fingerprint = staleness.profile_fingerprint(repo / "profile.yml")
    assert fingerprint is not None, "exemplar profile must resolve"
    manifest = staleness.build_manifest_path(repo)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.2.0",
                "creation": {
                    "osprey_version": get_framework_release_version(),
                    "template": "control_assistant",
                    "preset_hash": fingerprint.profile_hash,
                    staleness.KEY_DIGESTS_MANIFEST_KEY: fingerprint.key_digests,
                },
                "build_args": {},
            }
        ),
        encoding="utf-8",
    )
    return manifest


# --- the write itself -------------------------------------------------------


def test_top_level_key_written_in_place_with_comments_intact(runner, lifecycle_repo):
    """A shorthand key is replaced where it sits, comment and all."""
    before = _profile_text(lifecycle_repo)
    assert "model: haiku" in before

    result = _invoke(runner, lifecycle_repo, "model=sonnet")

    assert result.exit_code == 0, result.output
    after = _profile_text(lifecycle_repo)
    assert "model: sonnet" in after
    assert "model: haiku" not in after
    # The inline comment on that very line, and the block comment above it.
    assert "# tier (haiku/sonnet/opus)" in after
    assert "# Default LLM provider and model." in after
    # Every other line of prose survives the round trip. Counted rather than
    # sampled: a writer that drops the file's comments would still pass any
    # handful of spot checks on the ones near the key it touched.
    assert _comment_lines(after) == _comment_lines(before)


def test_one_key_produces_a_one_line_diff(runner, lifecycle_repo):
    """The write touches the key's line and nothing else in the document.

    profile.yml is tracked in git, so the diff a set produces is what a
    facility reviews. A round-trip writer that re-indents the file's block
    sequences keeps every comment and still turns a one-word change into an
    80-line diff — comment preservation alone does not cover this, which is
    why the property pinned here is the diff itself.
    """
    before = _profile_text(lifecycle_repo).splitlines()

    result = _invoke(runner, lifecycle_repo, "model=sonnet")

    assert result.exit_code == 0, result.output
    after = _profile_text(lifecycle_repo).splitlines()
    changed = [
        line
        for line in difflib.unified_diff(before, after, lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    # Asserted by shape rather than against the exemplar's exact wording: what
    # must hold is "one line out, one line in", whatever that line says.
    assert len(changed) == 2, "\n".join(changed)
    removed, added = changed
    assert removed.startswith("-model: ")
    assert added.startswith("+model: sonnet")
    # The trailing comment rides along on the same line it started on.
    assert removed.partition("#")[2] == added.partition("#")[2]


def test_dotted_config_key_replaces_the_literal_entry(runner, lifecycle_repo):
    """``config.`` keys address the profile's dotted entries, not a subtree."""
    result = _invoke(runner, lifecycle_repo, "config.control_system.type=epics")

    assert result.exit_code == 0, result.output
    after = _profile_text(lifecycle_repo)
    assert "control_system.type: epics" in after
    assert "control_system.type: mock" not in after
    # The rest of the config: block is untouched — including the nested
    # web-terminals module a whole-subtree write would have flattened away.
    assert "claude_code.servers.health.enabled: true" in after
    assert "modules.web_terminals:" in after
    assert "- name: alice" in after


def test_multiple_pairs_all_written(runner, lifecycle_repo):
    result = _invoke(
        runner,
        lifecycle_repo,
        "provider=cborg",
        "channel_finder_mode=in_context",
        "config.facility.name=Test Ring",
    )

    assert result.exit_code == 0, result.output
    after = _profile_text(lifecycle_repo)
    assert "provider: cborg" in after
    assert "channel_finder_mode: in_context" in after
    assert "facility.name: Test Ring" in after


def test_values_are_read_as_yaml(runner, lifecycle_repo):
    """Booleans and numbers land as themselves, not as quoted strings."""
    result = _invoke(
        runner,
        lifecycle_repo,
        "config.control_system.writes_enabled=true",
        "config.web.panels.events.health_timeout=30",
    )

    assert result.exit_code == 0, result.output
    after = _profile_text(lifecycle_repo)
    assert "control_system.writes_enabled: true" in after
    assert "web.panels.events.health_timeout: 30" in after


def test_build_config_is_never_touched(runner, lifecycle_repo):
    """Goal 6: the generated config.yml is not what `set` edits."""
    rendered = lifecycle_repo / "build" / "config.yml"
    rendered.parent.mkdir(parents=True, exist_ok=True)
    rendered.write_text("control_system:\n  type: mock\n", encoding="utf-8")

    result = _invoke(runner, lifecycle_repo, "connector=epics")

    assert result.exit_code == 0, result.output
    assert rendered.read_text(encoding="utf-8") == "control_system:\n  type: mock\n"


# --- absorbed shorthands ----------------------------------------------------


def test_connector_shorthand_writes_the_control_system_key(runner, lifecycle_repo):
    """Absorbs `config set-control-system` — into the PROFILE, not the render."""
    result = _invoke(runner, lifecycle_repo, "connector=virtual_accelerator")

    assert result.exit_code == 0, result.output
    assert "control_system.type: virtual_accelerator" in _profile_text(lifecycle_repo)
    # Reported under the key it actually wrote, so the fold is visible.
    assert "config.control_system.type" in result.output


def test_unknown_connector_is_refused_and_writes_nothing(runner, lifecycle_repo):
    before = _profile_text(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, "connector=epcis")

    assert result.exit_code != 0
    assert "Unknown connector" in result.stderr
    assert _profile_text(lifecycle_repo) == before


def test_epics_gateway_shorthand_writes_the_facility_addresses(runner, lifecycle_repo):
    """Absorbs `config set-epics-gateway --facility als`."""
    result = _invoke(runner, lifecycle_repo, "epics_gateway=als")

    assert result.exit_code == 0, result.output
    after = _profile_text(lifecycle_repo)
    prefix = "control_system.connector.epics.gateways"
    assert f"{prefix}.read_only.address: cagw-alsdmz.als.lbl.gov" in after
    assert f"{prefix}.read_only.port: 5064" in after
    assert f"{prefix}.write_access.port: 5084" in after
    # Booleans survive the JSON→YAML round trip as booleans.
    assert f"{prefix}.read_only.use_name_server: false" in after


def test_unknown_facility_names_the_known_ones(runner, lifecycle_repo):
    before = _profile_text(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, "epics_gateway=nsls")

    assert result.exit_code != 0
    assert "Unknown facility" in result.stderr
    assert "als" in result.stderr
    assert _profile_text(lifecycle_repo) == before


def test_tier_is_a_settable_key(runner, lifecycle_repo):
    """Absorbs `build --tier`: the tier is profile content like anything else."""
    result = _invoke(runner, lifecycle_repo, "tier=1", "channel_finder_mode=in_context")

    assert result.exit_code == 0, result.output
    after = _profile_text(lifecycle_repo)
    assert "tier: 1" in after
    assert "channel_finder_mode: in_context" in after


# --- unrecognized keys ------------------------------------------------------


def test_a_key_the_schema_does_not_know_is_written_but_called_out(runner, lifecycle_repo):
    """The profile schema is closed, so this key breaks the next build.

    Learning that from `osprey build` — which reports it against ``profile.yml``
    — leaves the operator looking for a hand edit they never made. The warning
    names it here, at the command that wrote it, and still reports the write
    honestly: the value IS in the file.
    """
    result = _invoke(runner, lifecycle_repo, "modle=sonnet")

    assert result.exit_code == 0, result.output
    assert "modle: sonnet" in _profile_text(lifecycle_repo)
    assert "Not a profile key: modle" in result.output


def test_recognized_and_config_prefixed_keys_are_never_called_out(runner, lifecycle_repo):
    """The rendered config's key space is open — only the profile's is closed.

    Paired with the test above so a warning that fired on everything (or on
    nothing) cannot read as a pass in either direction.
    """
    result = _invoke(
        runner, lifecycle_repo, "model=sonnet", "config.facility.name=Somewhere", "tier=2"
    )

    assert result.exit_code == 0, result.output
    assert "Not a profile key" not in result.output


def test_the_shorthands_are_expanded_before_the_key_is_judged(runner, lifecycle_repo):
    """`epics_gateway` is a CLI-only spelling that never reaches the profile.

    It is not a schema key, so judging the raw command line rather than the
    expansion would warn about the one key this command exists to expand.
    """
    result = _invoke(runner, lifecycle_repo, "epics_gateway=als")

    assert result.exit_code == 0, result.output
    assert "Not a profile key" not in result.output


# --- refusals ---------------------------------------------------------------


def test_outside_a_repo_is_refused(runner, tmp_path, monkeypatch):
    """No profile.yml at or above cwd — the walk fails, and nothing is written."""
    elsewhere = tmp_path / "not-a-repo"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = runner.invoke(set_command, ["model=sonnet"], catch_exceptions=False)

    assert result.exit_code != 0
    assert "No OSPREY deployment repo found" in result.stderr
    assert "profile.yml" in result.stderr


def test_extends_is_refused(runner, lifecycle_repo):
    """The same refusal materialization makes: a repo profile inherits nothing."""
    before = _profile_text(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, "extends=../other/profile.yml")

    assert result.exit_code != 0
    assert "Cannot override 'extends'" in result.stderr
    assert _profile_text(lifecycle_repo) == before


def test_pair_without_equals_is_refused(runner, lifecycle_repo):
    before = _profile_text(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, "model")

    assert result.exit_code != 0
    assert "KEY=VALUE" in result.stderr
    assert _profile_text(lifecycle_repo) == before


def test_one_bad_pair_writes_none_of_them(runner, lifecycle_repo):
    """All-or-nothing: the merge fails before the document is rewritten."""
    before = _profile_text(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, "model=sonnet", "connector=nonsense")

    assert result.exit_code != 0
    assert _profile_text(lifecycle_repo) == before


def test_no_pairs_is_a_usage_error(runner, lifecycle_repo):
    result = _invoke(runner, lifecycle_repo)

    assert result.exit_code != 0
    assert "Nothing to set" in result.stderr


# --- discovery and the drift hint -------------------------------------------


def test_resolves_the_repo_from_a_subdirectory(runner, lifecycle_repo, monkeypatch):
    """The walk-up rule: any subdirectory is inside the deployment."""
    monkeypatch.chdir(lifecycle_repo / "data" / "channel_databases")

    result = runner.invoke(set_command, ["model=opus"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "model: opus" in _profile_text(lifecycle_repo)


def test_drift_hint_reports_no_build(runner, lifecycle_repo):
    """A repo that has never been built says so rather than claiming sync."""
    result = _invoke(runner, lifecycle_repo, "model=sonnet")

    assert result.exit_code == 0, result.output
    assert "build: none" in result.output


def test_drift_hint_reports_the_build_this_edit_invalidated(runner, lifecycle_repo):
    _stamp_build(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, "model=sonnet")

    assert result.exit_code == 0, result.output
    assert "OUT OF DATE" in result.output


def test_written_keys_are_reported(runner, lifecycle_repo):
    result = _invoke(runner, lifecycle_repo, "model=sonnet", "config.facility.name=Ring")

    assert result.exit_code == 0, result.output
    assert str(lifecycle_repo / "profile.yml") in result.output
    assert "model" in result.output
    assert "config.facility.name" in result.output


# --- the verb is reachable --------------------------------------------------


def test_registered_on_the_top_level_cli():
    """`osprey set` resolves through the lazy group and is listed in --help."""
    from click import Context

    from osprey.cli.main import cli

    with Context(cli) as ctx:
        assert cli.get_command(ctx, "set") is set_command
        assert "set" in cli.list_commands(ctx)
