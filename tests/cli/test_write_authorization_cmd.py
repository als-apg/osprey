"""``osprey write-authorization``: approve, list and revoke without a web terminal.

Two places it runs: a laptop with no deployment repo, where the owner is this
account and its grants sit in its own state directory; and a deployment repo,
where it acts on the control-context tree as the owner the request names. The
approval records ``approver.identity`` as this process's acting identity and
``surface: "cli"``, and refuses what the web terminal's card refuses.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.write_authorization_cmd import write_authorization
from osprey_connectors import control_context, posture_store
from osprey_connectors import write_authorization as wa
from osprey_connectors.identity import acting_identity
from osprey_connectors_tests._harness import isolate_verdict_environment, write_control_context

FAR_FUTURE_NS = time.time_ns() + 10 * 365 * 24 * 3600 * 10**9

LIMITS = {
    "SR:A": {"writable": True, "min_value": 0.0, "max_value": 10.0},
    "SR:ENUM": {"writable": True},
}


def _config(directory: Path, *, writes: str = "allowed") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "channel_limits.json").write_text(json.dumps(LIMITS), encoding="utf-8")
    path = directory / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "control_system": {
                    "type": "epics",
                    "writes": writes,
                    "connector": {
                        "virtual_accelerator": {
                            "gateways": {
                                "read_only": {"address": "localhost", "port": 5074},
                            },
                        },
                    },
                    "limits_checking": {
                        "enabled": True,
                        "mode": "exclusive",
                        "database_path": "channel_limits.json",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def laptop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host with no deployment repo: this account's own state directory and a config."""
    root = tmp_path / "agent-data"
    isolate_verdict_environment(monkeypatch, root)
    for name in (
        wa.AUTHORIZATION_ENV_VAR,
        wa.LEDGER_DIR_ENV_VAR,
        wa.REQUESTS_DIR_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    elsewhere = tmp_path / "not-a-repo"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("OSPREY_CONFIG", str(_config(tmp_path / "project")))
    return root


def run(*args: str) -> Any:
    return CliRunner().invoke(write_authorization, list(args), catch_exceptions=False)


def ask(**fields: Any) -> wa.WriteRequest:
    values: dict[str, Any] = {
        "steps": 5,
        "expires_at_ns": FAR_FUTURE_NS,
        "lease_s": 30.0,
        "label": "run-1",
        "completion": "restore",
    }
    values.update(fields)
    channels = values.pop("channels", {"SR:A": (1.0, 9.0)})
    return wa.write_request("va", channels, **values)


@pytest.mark.usefixtures("laptop")
class TestLaptop:
    """No terminal and no tree: the account approves its own program's request."""

    def test_approve_by_id_records_the_cli_and_this_account(self):
        request = ask()
        result = run("approve", request.id, "--yes")
        assert result.exit_code == 0, result.output
        grant = wa.approved_grant(request.id)
        assert grant is not None
        assert (grant.approver.identity, grant.approver.surface) == (acting_identity(), "cli")
        assert grant.owner == grant.holder == acting_identity()
        assert "Approved" in result.output

    def test_no_record_yet_still_approves(self):
        assert control_context.record_path() is not None
        assert not control_context.record_path().exists()  # type: ignore[union-attr]
        request = ask()
        assert run("approve", request.id, "--yes").exit_code == 0
        assert wa.approved_grant(request.id) is not None

    def test_list_shows_pending_then_in_force_and_revoke_ends_it(self, laptop):
        write_control_context(laptop, switch={})
        request = ask(label=":warning:")
        listed = json.loads(run("list", "--json").output)
        assert [r["id"] for r in listed["requests"]] == [request.id]
        assert ":warning:" in run("list").output  # a label is text, not an emoji code
        assert run("approve", request.id, "--yes").exit_code == 0
        listed = json.loads(run("list", "--json").output)
        assert listed["requests"] == []
        assert [(a["id"], a["in_force"]) for a in listed["authorizations"]] == [(request.id, True)]
        result = run("revoke", request.id)
        assert result.exit_code == 0, result.output
        with pytest.raises(wa.AuthorizationRefused) as refused:
            wa.approved_grant(request.id)
        assert refused.value.reason == "auth_revoked"
        listed = json.loads(run("list", "--json").output)
        assert listed["authorizations"][0]["in_force"] is False

    @pytest.mark.parametrize(
        ("channels", "reason"),
        [({"SR:A": (0.0, 11.0)}, "outside_limits"), ({"SR:ENUM": (0.0, 1.0)}, "not_numeric")],
    )
    def test_a_range_the_limits_do_not_hold_is_refused(self, channels, reason):
        request = ask(channels=channels)
        result = run("approve", request.id, "--yes")
        assert result.exit_code != 0
        assert reason in result.output
        assert wa.approved_grant(request.id) is None

    def test_a_never_tier_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_CONFIG", str(_config(tmp_path / "never", writes="never")))
        request = ask()
        result = run("approve", request.id, "--yes")
        assert result.exit_code != 0 and "never" in result.output
        assert wa.approved_grant(request.id) is None

    def test_a_missing_request_and_another_owners_are_refused(self, tmp_path):
        assert run("approve", uuid.uuid4().hex, "--yes").exit_code != 0
        foreign = wa.parse_request(
            {**ask().to_payload(), "id": uuid.uuid4().hex, "owner": "someone-else"}
        )
        path = tmp_path / f"{foreign.id}.json"
        path.write_text(json.dumps(foreign.to_payload()), encoding="utf-8")
        result = run("approve", str(path), "--yes")
        assert result.exit_code != 0 and "this account" in result.output

    def test_an_id_with_no_pending_request_says_whether_it_was_approved(self):
        unknown = uuid.uuid4().hex
        result = run("approve", unknown, "--yes")
        assert result.exit_code != 0
        assert f"No pending request {unknown}" in result.output
        assert "osprey write-authorization list" in result.output
        request = ask()
        assert run("approve", request.id, "--yes").exit_code == 0
        assert wa.withdraw_request(request.id)  # the holder withdraws it once granted
        result = run("approve", request.id, "--yes")
        assert result.exit_code != 0
        assert f"Request {request.id} was already approved" in result.output

    def test_without_yes_a_declined_prompt_approves_nothing(self):
        request = ask()
        result = CliRunner().invoke(write_authorization, ["approve", request.id], input="n\n")
        assert result.exit_code != 0
        assert wa.approved_grant(request.id) is None


class TestDeployment:
    """On the deployment host: the owner's directory in the tree."""

    @pytest.fixture
    def repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        root = tmp_path / "deployment"
        root.mkdir()
        (root / "profile.yml").write_text("name: demo\n", encoding="utf-8")
        _config(root / "build")
        isolate_verdict_environment(monkeypatch, tmp_path / "laptop-root")
        for name in (wa.LEDGER_DIR_ENV_VAR, wa.REQUESTS_DIR_ENV_VAR):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(root)
        return root

    def _put(
        self,
        repo: Path,
        owner: str,
        holder: str = "optimizer",
        channels=None,
        target="va",
        *,
        on_host: bool = False,
    ) -> wa.WriteRequest:
        """File a request as a program in a container does, or *on_host* with no ledger bound."""
        request = wa.parse_request(
            {
                "schema": 1,
                "id": uuid.uuid4().hex,
                "owner": owner,
                "holder": holder,
                "target": target,
                "channels": channels or {"SR:A": {"min": 1.0, "max": 2.0}},
                "steps": 3,
                "requested_at_ns": time.time_ns(),
                "expires_at_ns": FAR_FUTURE_NS,
                "lease_s": 30.0,
                "label": "run-9",
                "completion": "none",
            }
        )
        if on_host:
            directory = self._own(repo, owner) / wa.REQUESTS_DIR_NAME
        else:
            directory = repo / "var" / "write_ledger" / "requests"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{request.id}.json").write_text(json.dumps(request.to_payload()))
        return request

    @staticmethod
    def _own(repo: Path, owner: str) -> Path:
        return repo / "var" / "agent_data" / "control_target" / owner / "authorizations"

    @staticmethod
    def _stage_map(repo: Path, login_tiers: dict[str, str]) -> None:
        """The login map a build stages for the shared containers.

        The as-built ``build/config.yml`` names the service, as every build's
        does: the map is read from the services that drive the target there.
        """
        path = repo / "build" / "services" / "bluesky" / "config.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"control_system": {"writes": "allowed", "login_tiers": login_tiers}}
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        as_built = repo / "build" / "config.yml"
        built = yaml.safe_load(as_built.read_text(encoding="utf-8")) if as_built.is_file() else {}
        built = built if isinstance(built, dict) else {}
        built.setdefault("services", {}).setdefault("bluesky", {})
        as_built.write_text(yaml.safe_dump(built), encoding="utf-8")

    def test_a_request_filed_on_the_host_is_listed_approved_by_id_and_counted(self, repo):
        request = self._put(repo, "alice", on_host=True)
        listed = json.loads(run("list", "--json").output)
        assert [r["id"] for r in listed["requests"]] == [request.id]
        result = run("approve", request.id, "--yes")
        assert result.exit_code == 0, result.output
        ledger = wa.Ledger(
            authorization=request.id,
            holder_pid=None,
            holder_start=None,
            pid_ns=None,
            lease_until_ns=FAR_FUTURE_NS,
            steps=(wa.LedgerStep(n=1, at_ns=1, channels=("SR:A",)),),
        )
        (self._own(repo, "alice") / f"{request.id}.ledger.json").write_text(
            json.dumps(ledger.to_payload())
        )
        listed = json.loads(run("list", "--json").output)
        assert listed["requests"] == []
        assert [(a["id"], a["used"], a["in_force"]) for a in listed["authorizations"]] == [
            (request.id, 1, True)
        ]

    def test_the_tier_is_judged_where_the_program_runs(self, repo):
        """A container program meets the staged login map; a program on the host, the render."""
        self._stage_map(repo, {"alice": "asked"})
        in_container = self._put(repo, "gmartino")
        result = run("approve", in_container.id, "--yes")
        assert result.exit_code != 0 and "unknown" in result.output
        on_host = self._put(repo, "gmartino", on_host=True)
        result = run("approve", on_host.id, "--yes")
        assert result.exit_code == 0, result.output
        listed = self._put(repo, "alice")
        result = run("approve", listed.id, "--yes")
        assert result.exit_code == 0, result.output

    def test_approve_writes_into_the_owners_tree_directory(self, repo):
        request = self._put(repo, "alice")
        result = run("approve", request.id, "--yes")
        assert result.exit_code == 0, result.output
        grants_dir = repo / "var" / "agent_data" / "control_target" / "alice" / "authorizations"
        (grant,) = wa.list_grants(grants_dir)[0]
        assert grant.id == request.id
        assert (grant.owner, grant.approver.identity, grant.approver.surface) == (
            "alice",
            acting_identity(),
            "cli",
        )

    def test_a_target_the_deployment_does_not_configure_is_refused_before_any_write(self, repo):
        request = self._put(repo, "alice", target="standin")
        result = run("approve", request.id, "--yes")
        assert result.exit_code != 0
        assert "not a control target" in result.output
        # Nothing was written: no grant, and no record created on the bogus target.
        alice = repo / "var" / "agent_data" / "control_target" / "alice"
        assert not (alice / "control_context.json").exists()
        assert wa.list_grants(alice / "authorizations") == ((), ())

    def test_another_owners_program_on_the_channel_is_not_an_overlap(self, repo):
        first = self._put(repo, "alice", holder="optimizer")
        assert run("approve", first.id, "--yes").exit_code == 0
        second = self._put(repo, "bob", holder="tuning-scripts")
        result = run("approve", second.id, "--yes")
        assert result.exit_code == 0, result.output
        bob = repo / "var" / "agent_data" / "control_target" / "bob" / "authorizations"
        assert [grant.id for grant in wa.list_grants(bob)[0]] == [second.id]

    def test_a_grant_whose_steps_are_spent_is_listed_ended_with_its_count(self, repo):
        request = self._put(repo, "alice")
        assert run("approve", request.id, "--yes").exit_code == 0
        ledger = wa.Ledger(
            authorization=request.id,
            holder_pid=None,
            holder_start=None,
            pid_ns=None,
            lease_until_ns=FAR_FUTURE_NS,
            steps=tuple(wa.LedgerStep(n=n, at_ns=n, channels=("SR:A",)) for n in (1, 2, 3)),
        )
        (repo / "var" / "write_ledger" / f"{request.id}.ledger.json").write_text(
            json.dumps(ledger.to_payload())
        )
        listed = json.loads(run("list", "--json").output)
        assert [(a["used"], a["steps"], a["in_force"]) for a in listed["authorizations"]] == [
            (3, 3, False)
        ]

    def test_list_and_revoke_find_the_owner(self, repo):
        request = self._put(repo, "alice")
        assert run("approve", request.id, "--yes").exit_code == 0
        listed = json.loads(run("list", "--json").output)
        assert [(a["owner"], a["in_force"]) for a in listed["authorizations"]] == [("alice", True)]
        assert run("revoke", request.id).exit_code == 0
        listed = json.loads(run("list", "--json").output)
        assert listed["authorizations"][0]["revoked"] is not None

    def test_a_tree_bound_holder_is_refused_after_the_stop(self, repo, monkeypatch):
        request = self._put(repo, "alice")
        assert run("approve", request.id, "--yes").exit_code == 0
        tree = repo / "var" / "agent_data" / "control_target"
        control_context.set_switch(["va"], "off", path=tree / "alice" / "control_context.json")
        monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
        monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
        posture_store.invalidate_cache()
        control_context.invalidate_cache()
        status, grant = wa._find_grant("alice", request.id)
        assert status == "ok" and grant is not None and grant.revoked is not None

    def test_a_programs_self_issued_grant_is_listed_and_revoked_in_the_ledger(self, repo):
        record = repo / "var" / "agent_data" / "control_target" / "alice" / "control_context.json"
        with control_context.held_epoch(record, "va") as epoch:
            pass
        control_context.set_switch(["va"], "allowed", epoch=epoch, path=record)
        now = time.time_ns()
        grant = wa.Grant(
            id=uuid.uuid4().hex,
            owner="alice",
            holder="tuning-scripts",
            target="va",
            switch_epoch=epoch,
            channels={"SR:A": wa.ChannelRange(0.0, 1.0)},
            steps=50,
            issued_at_ns=now,
            expires_at_ns=FAR_FUTURE_NS,
            lease_s=30.0,
            approver=wa.Approver(identity="tuning-scripts", surface="write_switch", at_ns=now),
            label="scan",
        )
        ledger = repo / "var" / "write_ledger"
        wa.issue_grant(grant, directory=ledger, peers=())
        listed = json.loads(run("list", "--json").output)
        assert [(a["id"], a["self_issued"], a["in_force"]) for a in listed["authorizations"]] == [
            (grant.id, True, True)
        ]
        assert "self-issued" in run("list").output
        result = run("revoke", grant.id)
        assert result.exit_code == 0, result.output
        (stored,) = wa.list_grants(ledger)[0]
        assert stored.revoked is not None and stored.revoked.by == acting_identity()


def test_the_card_prints_a_channel_name_as_written(capsys) -> None:
    """`SR:MAG:...` is a channel, not markup: `:MAG:` must not become an emoji."""
    from osprey.cli.write_authorization_cmd import _show_card

    _show_card(
        {
            "holder": "optimizer",
            "target": "va",
            "owner": "alice",
            "label": "",
            "channels": [
                {
                    "address": "SR:MAG:QF:01:CURRENT:SP",
                    "min": 10.0,
                    "max": 60.0,
                    "limit_min": 0.0,
                    "limit_max": 300.0,
                    "problem": None,
                }
            ],
            "steps": 6,
            "completion": "apply_best",
            "expires_at": "2026-11-04T16:00:00+00:00",
            "lease_s": 60.0,
        }
    )
    out = capsys.readouterr().out
    assert "SR:MAG:QF:01:CURRENT:SP" in out
    assert "\N{LEFT-POINTING MAGNIFYING GLASS}" not in out
