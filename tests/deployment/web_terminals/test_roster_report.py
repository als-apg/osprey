"""``osprey users list``: the joined roster report and its command."""

from __future__ import annotations

import json
import textwrap

import pytest
from click.testing import CliRunner

from osprey.cli.users_cmd import users
from osprey.deployment.web_terminals import roster_report
from osprey.deployment.web_terminals.roster_report import build_roster_report, last_sign_ins
from osprey_connectors.config import load_project_config

CONFIG = textwrap.dedent(
    """
    project_name: demo
    modules:
      web_terminals:
        enabled: true
        image_source: local
        default_tier: viewer
        auth:
          method: oidc
          oidc:
            claim: email
        renders:
          viewer: {kind: tier, project: demo-viewer, project_path: build/demo-viewer}
          editor: {kind: tier, project: demo-editor, project_path: build/demo-editor}
          spare: {kind: tier, project: demo-spare, project_path: build/demo-spare}
          runner: {kind: tier, project: demo-runner, project_path: build/demo-runner}
          search: {kind: app, project: demo-search, project_path: build/demo-search}
        users:
          - {name: desk, index: 0, tier: viewer, display_name: Desk, access: [roster]}
          - {name: alice, index: 1, tier: editor, oidc_subject: alice@example.org,
             control_identity: alice}
          - {name: bob, index: 2, tier: viewer, oidc_subject: bob@other.org}
          - {name: lab, index: 3, tier: viewer, display_name: Lab,
             access: [self, "user:bob@other.org", "user:eve@else.org", "domain:partner.org"]}
          - {name: ops, index: 4, tier: viewer, display_name: Ops, access: ["tier:editor"]}
        shared: [search]
    services:
      dispatch_worker: {tier: runner}
    """
)


@pytest.fixture
def repo(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (tmp_path / "profile.yml").write_text("name: demo\n")
    (build / "config.yml").write_text(CONFIG)
    for render, writes in (("viewer", "never"), ("editor", "allowed"), ("search", "never")):
        (build / f"demo-{render}").mkdir()
        (build / f"demo-{render}" / "config.yml").write_text(
            f"control_system:\n  writes: {writes}\n"
        )
    return tmp_path


def _report(repo):
    config = load_project_config(str(repo / "build" / "config.yml"))
    return build_roster_report(config, repo, project_name="demo", app_titles={"search": "Search"})


def _card(report, name):
    return next(card for card in report["cards"] if card["name"] == name)


def _person(report, name):
    return next(person for person in report["people"] if person["name"] == name)


def test_a_shared_login_is_a_card_and_not_a_person(repo):
    report = _report(repo)
    assert [person["name"] for person in report["people"]] == ["alice", "bob"]
    assert _card(report, "desk")["shared"] is True


def test_every_login_opens_its_own_card_and_the_whole_roster_ones(repo):
    report = _report(repo)
    assert set(_person(report, "alice")["opens"]) == {"desk", "alice", "ops", "search"}


def test_a_user_principal_opens_the_card_to_that_identity_alone(repo):
    report = _report(repo)
    assert "lab" in _person(report, "bob")["opens"]
    assert "lab" not in _person(report, "alice")["opens"]


def test_principals_reaching_past_the_roster_are_named(repo):
    assert _report(repo)["off_roster"] == {"lab": ["domain:partner.org", "user:eve@else.org"]}


def test_a_card_reports_its_own_renders_write_posture(repo):
    report = _report(repo)
    assert _card(report, "alice")["writes"] == "allowed"
    assert _card(report, "search")["writes"] == "never"


def test_a_render_that_cannot_be_read_has_no_posture_rather_than_a_guess(repo):
    (repo / "build" / "demo-editor" / "config.yml").unlink()
    assert _card(_report(repo), "alice")["writes"] is None


def test_a_tier_is_unused_only_when_neither_a_card_nor_dispatch_runs_it(repo):
    assert _report(repo)["unused_tiers"] == ["spare"]


@pytest.mark.parametrize("method", ["none", "token"])
def test_the_grid_is_replaced_where_nothing_enforces_access(repo, method):
    path = repo / "build" / "config.yml"
    path.write_text(CONFIG.replace("method: oidc", f"method: {method}"))
    output = CliRunner().invoke(users, ["list", "--repo", str(repo)]).output
    assert "WHO CAN OPEN WHAT" not in output
    assert "not enforced" in output


def test_the_grid_marks_who_may_open_each_shared_card(repo):
    result = CliRunner().invoke(users, ["list", "--repo", str(repo)])
    assert result.exit_code == 0, result.output
    grid = result.output[result.output.index("WHO CAN OPEN WHAT") :].splitlines()
    assert grid[0].split() == [
        "WHO",
        "CAN",
        "OPEN",
        "WHAT",
        "render",
        "own",
        "desk",
        "lab",
        "ops",
        "search",
    ]
    assert grid[1].split() == ["alice", "editor", "●", "●", "·", "●", "●"]
    assert grid[2].split() == ["bob", "viewer", "●", "●", "●", "·", "●"]
    assert "unused tiers: spare" in result.output


def test_json_is_one_document_on_stdout(repo):
    result = CliRunner().invoke(users, ["list", "--json", "--repo", str(repo)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["deployment"] == "demo"


def test_naming_a_login_answers_for_that_login(repo):
    output = CliRunner().invoke(users, ["list", "alice", "--repo", str(repo)]).output
    assert "opens: desk, alice, ops, search" in output


def test_naming_a_card_answers_who_may_open_it(repo):
    output = CliRunner().invoke(users, ["list", "lab", "--repo", str(repo)]).output
    assert "logins: bob" in output


def test_an_unknown_name_is_refused(repo):
    assert CliRunner().invoke(users, ["list", "nobody", "--repo", str(repo)]).exit_code != 0


def test_a_sign_in_counts_for_the_card_and_for_who_opened_it():
    newest = last_sign_ins(
        [
            {
                "ts": "2026-02-01T00:00:00+00:00",
                "body": {"decision": "allowed", "subject": "search", "detail": "opener=alice"},
            },
            {
                "ts": "2026-01-01T00:00:00+00:00",
                "body": {"decision": "allowed", "subject": "alice"},
            },
        ]
    )
    assert newest == {"search": "2026-02-01T00:00:00+00:00", "alice": "2026-02-01T00:00:00+00:00"}


def test_a_refused_login_is_not_a_sign_in():
    entry = {"ts": "2026-01-01T00:00:00+00:00", "body": {"decision": "refused", "subject": "bob"}}
    assert last_sign_ins([entry]) == {}


def test_live_degrades_to_nothing_without_a_runtime_or_a_record(repo, monkeypatch):
    def no_runtime(_config):
        raise RuntimeError("no runtime")

    monkeypatch.setattr(roster_report, "get_runtime_command", no_runtime)
    result = CliRunner().invoke(users, ["list", "--live", "--repo", str(repo)])
    assert result.exit_code == 0, result.output
    assert "last sign-in: unavailable" in result.output


def test_live_inspects_with_the_runtime_and_not_its_compose_command(repo, monkeypatch):
    seen = []

    class Done:
        returncode = 0
        stdout = "running\n"

    def run(command, **_kwargs):
        seen.append(command)
        return Done()

    monkeypatch.setattr(roster_report, "get_runtime_command", lambda _c: ["podman", "compose"])
    monkeypatch.setattr(roster_report.subprocess, "run", run)
    report = _report(repo)
    roster_report.attach_live(report, {}, repo)
    assert seen[0][:2] == ["podman", "inspect"]
    assert seen[0][-1] == "demo-web-desk"
    assert _card(report, "desk")["state"] == "running"


def test_a_tier_principal_admits_that_tiers_people_only(repo):
    report = _report(repo)
    assert "ops" in _person(report, "alice")["opens"]
    assert "ops" not in _person(report, "bob")["opens"]
