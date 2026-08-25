"""``osprey reset`` — what it is allowed to destroy, and what it must not.

Reset is the only OSPREY verb that removes data, so these tests are written
around the two ways it can be wrong rather than around its features.

**It can destroy the wrong deployment.** A compose project name is the repo's
directory NAME, so two clones or worktrees of one deployment on a host share a
project name and a volume namespace. Everything under "the scoping rule" below
exists to pin the two-condition gate — project name AND this checkout's
``com.osprey.repo-id`` — and, above all, the refusal: a same-named resource
carrying a different identity stops the whole reset and removes nothing.

The refusal's *wording* is tested as carefully as its behaviour, because it has
two branches that know different amounts and neither may borrow the other's
confidence. A container carries compose's ``working_dir`` label, so the other
repo's path can be named; a volume carries no path label at all, and since the
identity is a one-way hash of that path there is nothing to name. Branch B must
say so rather than guess, and must not send an operator to a repo it cannot
identify. Both are tested apart, each asserting what its branch says AND what it
must not.

**It can promise something other than what it does.** For a destructive verb a
plan that overstates is a lie an operator acts on, and a plan that understates
destroys something they did not agree to. So the plan and the execution are the
same object, and the tests assert the equality in BOTH directions: every removal
argv names something the plan printed, and everything the plan printed got a
removal argv.

No test here reaches a container runtime. Every one drives
:class:`~osprey.deployment.reset.RuntimeProbe` through an injected runner that
records argv, which is also what lets the argv themselves be asserted on — that
none is a ``prune``, an ``-a``, a wildcard, or a ``down -v``.
"""

from __future__ import annotations

import ast
import os
import stat
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.reset_cmd import reset as reset_command
from osprey.deployment import reset as reset_mod
from osprey.deployment.compose_generator import REPO_ID_LABEL, repo_identity
from osprey.deployment.reset import (
    MINTED_ENV_BANNERS,
    ForeignCheckoutError,
    ResetOutcome,
    Resource,
    RuntimeProbe,
    confirmation_token,
    partition_by_identity,
    plan_reset,
    reset_deployment,
    strip_minted_env_blocks,
)

PROJECT = "als-exemplar"
OTHER_IDENTITY = "0123456789ab"
OTHER_PATH = "/Users/someone/clones/als-exemplar"

# A rendered config with the two keys reset reads off it: the project name the
# running resources were labelled with, and enough shape for the config loader.
_RENDERED_CONFIG = f"""\
project_name: {PROJECT}
build_dir: ./build
deployed_services:
  - event_dispatcher
services:
  event_dispatcher:
    path: services/event_dispatcher
claude_code:
  provider: anthropic
"""


# ---------------------------------------------------------------------------
# A container runtime that only exists in this file
# ---------------------------------------------------------------------------


class FakeRuntime:
    """A recording stand-in for ``docker``/``podman``.

    It answers the real argv :class:`RuntimeProbe` builds, so the probe's
    command construction and its JSON label parsing are both under test rather
    than stubbed past. Every call is appended to :attr:`calls`, which is what the
    plan-versus-execution assertions read.
    """

    def __init__(
        self,
        *,
        containers: dict[str, dict[str, str]] | None = None,
        volumes: dict[str, dict[str, str]] | None = None,
        images: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.containers = dict(containers or {})
        self.volumes = dict(volumes or {})
        self.images = dict(images or {})
        self.calls: list[list[str]] = []
        #: Removal argv only — the destructive surface, isolated for assertions.
        self.removals: list[list[str]] = []

    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        _, *args = argv
        stdout = ""

        if args[:2] == ["ps", "-a"]:
            stdout = "".join(f"{name}\n" for name in self.containers)
        elif args[:2] == ["volume", "ls"]:
            stdout = "".join(f"{name}\n" for name in self.volumes)
        elif args[:2] == ["container", "inspect"]:
            return self._inspect(self.containers.get(args[2]))
        elif args[:2] == ["volume", "inspect"]:
            return self._inspect(self.volumes.get(args[2]))
        elif args[:2] == ["image", "inspect"]:
            return self._inspect(self.images.get(args[2]))
        elif args[:1] == ["rm"]:
            self.removals.append(list(argv))
            return self._remove(self.containers, args[-1], "container")
        elif args[:2] == ["volume", "rm"]:
            self.removals.append(list(argv))
            return self._remove(self.volumes, args[2], "volume")
        elif args[:2] == ["image", "rm"]:
            self.removals.append(list(argv))
            return self._remove(self.images, args[2], "image")
        else:  # pragma: no cover - an unexpected argv is a test bug, loudly
            raise AssertionError(f"FakeRuntime got an argv it does not know: {argv}")

        return subprocess.CompletedProcess(argv, 0, stdout, "")

    def _inspect(self, labels: dict[str, str] | None) -> subprocess.CompletedProcess:
        if labels is None:
            return subprocess.CompletedProcess([], 1, "", "Error: no such object")
        import json

        return subprocess.CompletedProcess([], 0, json.dumps(labels), "")

    def _remove(self, store, name, kind) -> subprocess.CompletedProcess:
        if name not in store:
            return subprocess.CompletedProcess([], 1, "", f"Error: no such {kind}: {name}")
        del store[name]
        return subprocess.CompletedProcess([], 0, "", "")

    def removed_names(self) -> list[str]:
        """The exact resource each removal argv named, in order."""
        return [argv[-1] for argv in self.removals]


def ours(repo_root: Path, **extra: str) -> dict[str, str]:
    """Labels a resource created by a deploy of *repo_root* would carry."""
    return {
        reset_mod.COMPOSE_PROJECT_LABEL: PROJECT,
        REPO_ID_LABEL: repo_identity(repo_root),
        **extra,
    }


def theirs(**extra: str) -> dict[str, str]:
    """Labels a resource from a DIFFERENT checkout of the same name carries."""
    return {
        reset_mod.COMPOSE_PROJECT_LABEL: PROJECT,
        REPO_ID_LABEL: OTHER_IDENTITY,
        "com.docker.compose.project.working_dir": OTHER_PATH,
        **extra,
    }


def unlabelled(**extra: str) -> dict[str, str]:
    """Labels a pre-identity resource carries: the project, and nothing else."""
    return {reset_mod.COMPOSE_PROJECT_LABEL: PROJECT, **extra}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A deployment repo in the three-zone shape, with a build and some state."""
    root = tmp_path / PROJECT
    (root / "build").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "personas").mkdir()
    (root / "var" / "agent_data" / "memory").mkdir(parents=True)
    (root / "var" / "audit").mkdir(parents=True)
    (root / "profile.yml").write_text("preset: control-assistant\n", encoding="utf-8")
    (root / "build" / "config.yml").write_text(_RENDERED_CONFIG, encoding="utf-8")
    (root / "data" / "channels.json").write_text("{}", encoding="utf-8")
    (root / "var" / "agent_data" / "memory" / "notes.md").write_text("remember", encoding="utf-8")
    (root / "var" / "audit" / "2026-08-11.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
    env = root / ".env"
    env.write_text(
        "# ── Seeded by `osprey init` from your shell ──\n"
        "ANTHROPIC_API_KEY=sk-provider-secret\n"
        "\n"
        f"# {MINTED_ENV_BANNERS[0]}\n"
        "OSPREY_DISPATCH_TOKEN=minted-dispatch-secret\n"
        "OSPREY_ARTIFACTS_TOKEN=minted-artifacts-secret\n",
        encoding="utf-8",
    )
    os.chmod(env, 0o600)
    return root


@pytest.fixture
def no_down(monkeypatch):
    """Record the ``down`` instead of running one, and prove it ran first.

    Returns the recorder list, whose length at any later point says whether the
    stack was taken down before that point.
    """
    calls: list[Path] = []
    monkeypatch.setattr(reset_mod, "down_deployment", lambda root: calls.append(Path(root)))
    return calls


#: The image axes are read from ``os.environ`` at resolve time, so an exported
#: ``OSPREY_IMAGE_TAG`` in the developer's shell would rename every tag these
#: tests expect. Unset for the whole module; the axis tests set them back.
_IMAGE_AXIS_VARS = ("OSPREY_IMAGE_REGISTRY", "OSPREY_IMAGE_TAG")


@pytest.fixture(autouse=True)
def _axes_unset(monkeypatch):
    for name in _IMAGE_AXIS_VARS:
        monkeypatch.delenv(name, raising=False)


def make_probe(fake: FakeRuntime) -> RuntimeProbe:
    return RuntimeProbe("docker", run=fake)


def run_reset(repo: Path, fake: FakeRuntime, **kwargs) -> list[str]:
    """Drive a full reset, returning the emitted lines."""
    emitted: list[str] = []
    kwargs.setdefault("assume_yes", True)
    reset_deployment(repo, probe=make_probe(fake), emit=emitted.append, **kwargs)
    return emitted


# ---------------------------------------------------------------------------
# The scoping rule: project name AND this checkout's identity
# ---------------------------------------------------------------------------


def test_a_resource_labelled_for_this_checkout_is_removed(repo, no_down):
    fake = FakeRuntime(
        containers={"als-exemplar-dispatch": ours(repo)},
        volumes={"als-exemplar_dispatch_workspace": ours(repo)},
    )
    run_reset(repo, fake)

    assert fake.removed_names() == [
        "als-exemplar-dispatch",
        "als-exemplar_dispatch_workspace",
    ]


def _flowed(message: str) -> str:
    """``message`` with every run of whitespace collapsed to one space.

    The refusal wraps its prose itself, because the renderer prints cause lines
    exactly as given. So a multi-word phrase in it is split at whatever column
    the wrap fell on, and only a flowed view can assert on the words rather than
    on the line breaks.
    """
    return " ".join(message.split())


def test_a_same_named_resource_from_another_checkout_refuses(repo):
    fake = FakeRuntime(volumes={"als-exemplar_dispatch_workspace": theirs()})

    with pytest.raises(ForeignCheckoutError) as excinfo:
        plan_reset(repo, probe=make_probe(fake))

    assert OTHER_PATH in str(excinfo.value)
    assert OTHER_IDENTITY in str(excinfo.value)


def test_the_foreign_refusal_removes_nothing(repo, no_down):
    """The whole point of refusing: it is a true no-op, not a partial reset."""
    fake = FakeRuntime(
        containers={"als-exemplar-dispatch": ours(repo)},
        volumes={"als-exemplar_dispatch_workspace": theirs()},
    )

    with pytest.raises(ForeignCheckoutError):
        reset_deployment(repo, probe=make_probe(fake), assume_yes=True, emit=lambda _: None)

    assert fake.removals == []
    assert no_down == []
    assert (repo / "build" / "config.yml").is_file()
    assert (repo / "var" / "agent_data" / "memory" / "notes.md").is_file()
    assert "minted-dispatch-secret" in (repo / ".env").read_text(encoding="utf-8")


# -- the two refusal branches, named and tested apart --------------------------
#
# The refusal has exactly two things it can know about where a foreign resource
# came from, and its wording has to claim the right one. Branch A: a container
# carries compose's own working_dir label, so the path can be named. Branch B: a
# volume carries no path label at all — the identity is a one-way hash, so there
# is nothing to name and the message has to say that instead of guessing. They
# are tested separately, and each asserts both what its branch DOES say and what
# it must NOT.


def test_refusal_branch_a_a_container_lets_the_other_repo_path_be_named(repo):
    """Compose records the ``--project-directory`` it was pinned to; that is the path."""
    fake = FakeRuntime(containers={"als-exemplar-dispatch": theirs()})

    with pytest.raises(ForeignCheckoutError) as excinfo:
        plan_reset(repo, probe=make_probe(fake))

    message = str(excinfo.value)
    assert f"recorded at {OTHER_PATH}" in message
    assert OTHER_IDENTITY in message
    # Branch A must not carry branch B's admission.
    assert "cannot be derived back out of the identity hash" not in message


def test_refusal_branch_a_reads_ospreys_own_label_when_compose_left_none(repo):
    labels = theirs()
    del labels["com.docker.compose.project.working_dir"]
    labels["osprey.project.root"] = "/opt/deployments/als-exemplar"
    fake = FakeRuntime(containers={"als-exemplar-dispatch": labels})

    with pytest.raises(ForeignCheckoutError) as excinfo:
        plan_reset(repo, probe=make_probe(fake))

    assert "recorded at /opt/deployments/als-exemplar" in str(excinfo.value)


def test_refusal_branch_b_a_volume_only_refusal_names_the_hash_and_admits_no_path(repo):
    """The real shape of a foreign volume: project + identity labels, no path label.

    Spelled out as a literal label map rather than derived from ``theirs()``,
    because this IS the branch — a volume is labelled by the compose template
    (``com.osprey.repo-id``) plus compose's own project label, and compose does
    not put ``working_dir`` on volumes the way it does on containers.
    """
    fake = FakeRuntime(
        volumes={
            "als-exemplar_dispatch_workspace": {
                reset_mod.COMPOSE_PROJECT_LABEL: PROJECT,
                REPO_ID_LABEL: OTHER_IDENTITY,
            }
        }
    )

    with pytest.raises(ForeignCheckoutError) as excinfo:
        plan_reset(repo, probe=make_probe(fake))

    message = str(excinfo.value)
    # It names the identity, which is the only thing it actually knows.
    assert OTHER_IDENTITY in message
    # And says plainly that the path is not recoverable.
    assert "no path is recorded on this resource" in message
    assert "cannot be derived back out of the identity hash" in message
    # It must not claim a path in any form, least of all this repo's own — which
    # is the path an invented one would most plausibly be.
    assert "recorded at" not in message
    assert str(repo) not in message.split("\n", 1)[1]


def test_refusal_branch_b_does_not_send_the_operator_to_a_repo_it_cannot_name(repo):
    """With no path recorded anywhere, "go run it over there" is advice to nowhere."""
    fake = FakeRuntime(
        volumes={"vol": {reset_mod.COMPOSE_PROJECT_LABEL: PROJECT, REPO_ID_LABEL: OTHER_IDENTITY}}
    )

    with pytest.raises(ForeignCheckoutError) as excinfo:
        plan_reset(repo, probe=make_probe(fake))

    message = str(excinfo.value)
    assert "cannot point you at the repo they came from" in message
    assert "remove them by hand" in message


def test_the_refusal_says_a_recorded_path_is_gone_when_it_is_gone(repo):
    """ "Recorded at X" is a label fact; whether X is there is a filesystem fact."""
    fake = FakeRuntime(containers={"c": theirs()})  # OTHER_PATH does not exist

    with pytest.raises(ForeignCheckoutError) as excinfo:
        plan_reset(repo, probe=make_probe(fake))

    message = str(excinfo.value)
    assert "no such directory on this host now" in message
    assert "leftovers from a deployment whose repo" in message
    # The remedy must not tell them to go reset a directory that is not there.
    assert f"Run `osprey reset` from {OTHER_PATH}" not in message
    # Nor claim an impossibility: recreating that path WOULD restore the identity,
    # so the message explains why reset here cannot rather than that nothing can.
    assert "no OSPREY command can" not in message


def test_the_refusal_points_at_a_recorded_path_that_really_is_there(repo, tmp_path):
    other = tmp_path / "other-checkout"
    other.mkdir()
    labels = theirs()
    labels["com.docker.compose.project.working_dir"] = str(other)
    fake = FakeRuntime(containers={"c": labels})

    with pytest.raises(ForeignCheckoutError) as excinfo:
        plan_reset(repo, probe=make_probe(fake))

    message = str(excinfo.value)
    assert f"osprey reset --repo {other}" in message
    assert "no such directory on this host now" not in message


def test_the_refusal_does_not_claim_a_different_checkout_only_a_different_path(repo):
    """A renamed directory is the SAME checkout with a new identity.

    The identity is a hash of the repo path, so moving or renaming a directory
    changes it. An operator who did that meets this refusal, and being told the
    resources "belong to a different checkout" would be flatly wrong — they are
    their own. The message claims only what the hash proves (a different repo
    PATH) and names the rename case explicitly.
    """
    fake = FakeRuntime(containers={"c": theirs()})

    with pytest.raises(ForeignCheckoutError) as excinfo:
        plan_reset(repo, probe=make_probe(fake))

    # Flowed, because the refusal wraps its own prose to a terminal width: a
    # phrase assertion against the raw text is an assertion about where the
    # wrap happened to fall.
    message = _flowed(str(excinfo.value))
    assert "created from a different copy of this repo" in message
    assert "renamed or moved" in message
    assert "these are your own resources under its old path" in message
    # The claim it must never make, however the sentence is reworded.
    assert "belong to a different checkout" not in message


def test_the_refusal_is_precise_about_what_it_scanned(repo):
    """It scanned containers and volumes. Images are not discovered before the refusal."""
    fake = FakeRuntime(containers={"c": theirs()})

    with pytest.raises(ForeignCheckoutError) as excinfo:
        plan_reset(repo, probe=make_probe(fake))

    assert "container(s) and volume(s) are named" in str(excinfo.value)
    assert not [argv for argv in fake.calls if argv[1:3] == ["image", "inspect"]]


def test_containers_and_volumes_are_both_reported_in_one_refusal(repo):
    fake = FakeRuntime(
        containers={"als-exemplar-dispatch": theirs()},
        volumes={"als-exemplar_dispatch_workspace": theirs()},
    )

    with pytest.raises(ForeignCheckoutError) as excinfo:
        plan_reset(repo, probe=make_probe(fake))

    assert len(excinfo.value.resources) == 2
    assert "container als-exemplar-dispatch" in str(excinfo.value)
    assert "volume als-exemplar_dispatch_workspace" in str(excinfo.value)


def test_an_unlabelled_resource_is_never_removed(repo, no_down):
    """The gate is AND. A project-name match alone proves nothing and removes nothing."""
    fake = FakeRuntime(
        containers={"als-exemplar-dispatch": unlabelled()},
        volumes={"als-exemplar_dispatch_workspace": unlabelled()},
    )
    run_reset(repo, fake)

    assert fake.removed_names() == []
    assert "als-exemplar_dispatch_workspace" in fake.volumes


def test_the_plan_lists_unlabelled_resources_rather_than_passing_over_them(repo):
    fake = FakeRuntime(volumes={"als-exemplar_dispatch_workspace": unlabelled()})
    plan = plan_reset(repo, probe=make_probe(fake))
    rendered = "\n".join(plan.render())

    assert "NOT REMOVED" in rendered
    assert "als-exemplar_dispatch_workspace" in rendered
    assert REPO_ID_LABEL in rendered


def test_the_unlabelled_volume_remedy_does_not_promise_a_relabel(repo):
    """A named volume is never relabelled by a later deploy. The text must not say it is.

    The container remedy ("`osprey up` recreates it with the label") is true and
    is offered; extending it to volumes would send an operator to run a deploy
    that cannot possibly make their volume provable.
    """
    fake = FakeRuntime(volumes={"als-exemplar_dispatch_workspace": unlabelled()})
    plan = plan_reset(repo, probe=make_probe(fake))
    volume_section = _unidentified_section(plan)

    assert "never relabelled by a later" in volume_section
    assert "remove it by hand" in volume_section
    assert "osprey up" not in volume_section


def test_the_container_remedy_is_offered_only_for_containers(repo):
    fake = FakeRuntime(containers={"als-exemplar-dispatch": unlabelled()})
    plan = plan_reset(repo, probe=make_probe(fake))
    section = _unidentified_section(plan)

    assert "recreates it with the label" in section
    assert "never relabelled" not in section


@pytest.mark.parametrize(
    ("labels", "bucket"),
    [
        ({REPO_ID_LABEL: "aaaaaaaaaaaa"}, "ours"),
        ({REPO_ID_LABEL: "bbbbbbbbbbbb"}, "foreign"),
        ({}, "unidentified"),
        ({REPO_ID_LABEL: ""}, "unidentified"),
    ],
)
def test_partition_puts_every_resource_in_exactly_one_bucket(labels, bucket):
    """Total and exclusive: an empty label is unprovable, not a match on ``""``."""
    resource = Resource("volume", "v", labels)
    partition = partition_by_identity([resource], "aaaaaaaaaaaa")

    buckets = {
        "ours": partition.ours,
        "foreign": partition.foreign,
        "unidentified": partition.unidentified,
    }
    assert buckets.pop(bucket) == [resource]
    assert all(other == [] for other in buckets.values())


def test_two_spellings_of_one_repo_are_one_checkout(repo, tmp_path, no_down):
    """A symlinked path must not read as a foreign checkout of the same name."""
    link = tmp_path / "via-symlink"
    link.symlink_to(repo, target_is_directory=True)
    fake = FakeRuntime(volumes={"als-exemplar_dispatch_workspace": ours(repo)})

    plan = plan_reset(link, probe=make_probe(fake))

    assert [r.name for r in plan.volumes] == ["als-exemplar_dispatch_workspace"]


# ---------------------------------------------------------------------------
# The plan IS the program
# ---------------------------------------------------------------------------


def test_everything_the_plan_lists_gets_a_removal_argv(repo, no_down):
    inventory = {
        "containers": {"c1": ours(repo), "c2": ours(repo)},
        "volumes": {"v1": ours(repo)},
        "images": {f"{PROJECT}:local": {reset_mod.OSPREY_PROJECT_LABEL: PROJECT}},
    }
    plan = plan_reset(repo, probe=make_probe(FakeRuntime(**inventory)))
    planned = [r.name for r in (*plan.containers, *plan.volumes, *plan.images)]

    fake = FakeRuntime(**inventory)
    run_reset(repo, fake)

    assert fake.removed_names() == planned
    assert planned == ["c1", "c2", "v1", f"{PROJECT}:local"]


def test_no_removal_argv_names_anything_the_plan_did_not_list(repo, no_down):
    """Including the resources reset deliberately left alone."""
    fake = FakeRuntime(
        containers={"mine": ours(repo), "unprovable": unlabelled()},
        volumes={"mine-vol": ours(repo), "unprovable-vol": unlabelled()},
    )
    plan = plan_reset(repo, probe=make_probe(fake))
    listed = {r.name for r in (*plan.containers, *plan.volumes, *plan.images)}

    run_reset(repo, fake)

    assert set(fake.removed_names()) <= listed
    assert "unprovable" not in fake.removed_names()
    assert "unprovable-vol" not in fake.removed_names()


def test_nothing_is_re_discovered_after_the_confirmation(repo, no_down):
    """A second scan could widen the set past what the operator agreed to."""
    fake = FakeRuntime(containers={"mine": ours(repo)}, volumes={"mine-vol": ours(repo)})
    run_reset(repo, fake)

    first_removal = next(i for i, argv in enumerate(fake.calls) if argv in fake.removals)
    after = fake.calls[first_removal:]

    assert not [argv for argv in after if argv[1:3] == ["ps", "-a"]]
    assert not [argv for argv in after if argv[1:3] == ["volume", "ls"]]


def test_no_destructive_argv_is_a_glob_a_prune_or_an_all(repo, no_down):
    """The removal surface, read straight off the recorded argv."""
    fake = FakeRuntime(
        containers={"mine": ours(repo)},
        volumes={"mine-vol": ours(repo)},
        images={f"{PROJECT}:local": {reset_mod.OSPREY_PROJECT_LABEL: PROJECT}},
    )
    run_reset(repo, fake)

    assert fake.removals, "the test needs at least one removal to be meaningful"
    for argv in fake.removals:
        assert "prune" not in argv
        assert "-a" not in argv and "--all" not in argv
        assert "-v" not in argv and "--volumes" not in argv
        assert not any("*" in token for token in argv)
        assert not any(token.startswith("--filter") for token in argv)


def test_the_down_runs_before_the_first_removal(repo, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(reset_mod, "down_deployment", lambda root: order.append("down"))
    fake = FakeRuntime(containers={"mine": ours(repo)})

    original = fake.__call__

    def recording(argv, **kwargs):
        if argv[1:2] == ["rm"]:
            order.append("rm")
        return original(argv, **kwargs)

    reset_deployment(
        repo, probe=RuntimeProbe("docker", run=recording), assume_yes=True, emit=lambda _: None
    )

    assert order == ["down", "rm"]


def test_a_failing_down_aborts_before_anything_is_removed(repo, monkeypatch):
    def boom(_root):
        raise RuntimeError("`compose down` failed (exit 1)")

    monkeypatch.setattr(reset_mod, "down_deployment", boom)
    fake = FakeRuntime(containers={"mine": ours(repo)}, volumes={"mine-vol": ours(repo)})

    with pytest.raises(RuntimeError, match="compose down"):
        reset_deployment(repo, probe=make_probe(fake), assume_yes=True, emit=lambda _: None)

    assert fake.removals == []
    assert (repo / "var" / "agent_data" / "memory" / "notes.md").is_file()
    assert (repo / "build" / "config.yml").is_file()


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_declining_the_confirmation_is_a_true_no_op(repo, no_down, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "not-the-name")
    fake = FakeRuntime(containers={"mine": ours(repo)}, volumes={"mine-vol": ours(repo)})
    emitted: list[str] = []

    outcome = reset_deployment(repo, probe=make_probe(fake), emit=emitted.append)

    assert outcome is ResetOutcome.DECLINED
    assert fake.removals == []
    assert no_down == []
    assert (repo / "build" / "config.yml").is_file()
    assert "Nothing was touched" in "\n".join(emitted)


def test_the_confirmation_token_is_this_repos_name(repo, no_down, monkeypatch):
    typed: list[str] = []

    def fake_input(prompt: str) -> str:
        typed.append(prompt)
        return confirmation_token(repo)

    monkeypatch.setattr("builtins.input", fake_input)
    fake = FakeRuntime(containers={"mine": ours(repo)})

    assert (
        reset_deployment(repo, probe=make_probe(fake), emit=lambda _: None)
        is ResetOutcome.COMPLETED
    )
    assert f"Type '{PROJECT}' to confirm" in typed[0]


def test_an_interrupt_at_the_prompt_says_nothing_was_touched(repo, no_down, monkeypatch):
    def interrupt(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    fake = FakeRuntime(containers={"mine": ours(repo)})
    emitted: list[str] = []

    assert (
        reset_deployment(repo, probe=make_probe(fake), emit=emitted.append) is ResetOutcome.DECLINED
    )
    assert "Nothing was touched" in "\n".join(emitted)
    assert fake.removals == []
    assert no_down == []


def test_the_prompt_counts_what_the_plan_lists(repo, no_down, monkeypatch):
    """The last line before the gate must not name things that are not going."""
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda p: prompts.append(p) or "no")
    fake = FakeRuntime(volumes={"mine-vol": ours(repo)})

    reset_deployment(repo, probe=make_probe(fake), emit=lambda _: None)

    assert "1 volume" in prompts[0]
    assert "container" not in prompts[0]


def test_a_removal_that_fails_is_not_reported_as_a_complete_reset(repo, no_down):
    """The plan said this volume would go. It did not. The report must say so.

    An operator who reads "Reset complete." believes the data in that volume is
    gone and stops looking for it. The rest of the reset still runs — one stuck
    volume is not a reason to leave them with a half-reset deployment — but the
    closing line is the difference between a report and a claim.
    """
    fake = FakeRuntime(volumes={"stuck-vol": ours(repo), "fine-vol": ours(repo)})
    original = fake._remove

    def refuse(store, name, kind):
        if name == "stuck-vol":
            return subprocess.CompletedProcess([], 1, "", "volume is in use")
        return original(store, name, kind)

    fake._remove = refuse
    emitted = run_reset(repo, fake)
    report = "\n".join(emitted)

    assert "Reset complete" not in report
    assert "could not" in report
    assert "stuck-vol" in report
    assert "volume is in use" in report
    # The rest of the reset still ran.
    assert "fine-vol" not in fake.volumes
    assert not (repo / "build").exists()


def test_an_already_absent_resource_is_not_reported_as_a_failure(repo, no_down):
    """The ``down`` usually removed the containers already; that is not an error."""
    fake = FakeRuntime(containers={"mine": ours(repo)})
    plan_probe = make_probe(fake)
    plan = plan_reset(repo, probe=plan_probe)
    del fake.containers["mine"]  # as a compose `down` would have left it

    emitted: list[str] = []
    probe = make_probe(fake)
    reset_deployment(repo, probe=probe, assume_yes=True, emit=emitted.append)

    assert plan.containers  # the plan did list it
    assert probe.failures == []
    assert "Reset complete" in "\n".join(emitted)


def test_dry_run_prints_the_plan_and_touches_nothing(repo, no_down):
    fake = FakeRuntime(containers={"mine": ours(repo)}, volumes={"mine-vol": ours(repo)})
    emitted = run_reset(repo, fake, dry_run=True, assume_yes=False)

    assert "--dry-run: nothing was stopped, removed, or written." in emitted
    assert "mine-vol" in "\n".join(emitted)
    assert fake.removals == []
    assert no_down == []
    assert (repo / "build" / "config.yml").is_file()


def test_dry_run_still_refuses_on_a_foreign_checkout(repo):
    """A dry run answers "what would happen", and what would happen is a refusal."""
    fake = FakeRuntime(volumes={"theirs-vol": theirs()})

    with pytest.raises(ForeignCheckoutError):
        reset_deployment(repo, probe=make_probe(fake), dry_run=True, emit=lambda _: None)


# ---------------------------------------------------------------------------
# var/audit
# ---------------------------------------------------------------------------


def test_the_audit_log_survives_a_default_reset(repo, no_down):
    run_reset(repo, FakeRuntime())

    assert (repo / "var" / "audit" / "2026-08-11.jsonl").is_file()


def test_the_plan_names_the_audit_log_as_kept(repo):
    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))
    kept = "\n".join(plan.render()).split("WILL BE KEPT", 1)[1]

    assert "var/audit/" in kept
    assert "--purge-audit" in kept


def test_purge_audit_destroys_it_and_says_so_in_the_removal_list(repo, no_down):
    plan = plan_reset(repo, probe=make_probe(FakeRuntime()), purge_audit=True)
    removed = "\n".join(plan.render()).split("WILL BE REMOVED", 1)[1].split("WILL BE KEPT", 1)[0]
    assert "var/audit/" in removed

    run_reset(repo, FakeRuntime(), purge_audit=True)

    assert not (repo / "var" / "audit" / "2026-08-11.jsonl").exists()


def test_purge_audit_is_gated_by_the_same_typed_confirmation(repo, no_down, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _p: "")
    fake = FakeRuntime()

    reset_deployment(repo, probe=make_probe(fake), purge_audit=True, emit=lambda _: None)

    assert (repo / "var" / "audit" / "2026-08-11.jsonl").is_file()


def test_purge_audit_does_not_claim_the_audit_log_is_kept(repo):
    """The kept section must not still advertise something now on the removal list."""
    plan = plan_reset(repo, probe=make_probe(FakeRuntime()), purge_audit=True)
    kept = "\n".join(plan.render()).split("WILL BE KEPT", 1)[1]

    assert "the safety audit log — what the agent was asked" not in kept


# ---------------------------------------------------------------------------
# .env.users / .env.auth — the web tier's write-once credential files
#
# Neither is touched by a reset, and neither is inside build/ or var/, so
# nothing else in the plan implies their fate. Undisclosed, an operator would
# learn that every web-terminal password still works only by trying one —
# after wiping the deployment those passwords belong to.
# ---------------------------------------------------------------------------


def _kept_section(repo: Path) -> str:
    return "\n".join(plan_reset(repo, probe=make_probe(FakeRuntime())).render()).split(
        "WILL BE KEPT", 1
    )[1]


@pytest.mark.parametrize(
    ("filename", "phrase"),
    [
        (".env.users", "runtime secrets every web-terminal container reads"),
        (".env.auth", "password hashes and cookie-signing secrets"),
    ],
)
def test_the_plan_discloses_a_web_credential_file_it_will_keep(repo, filename, phrase):
    (repo / filename).write_text("OSPREY_AUTH_PW_HASH_ALICE=x\n", encoding="utf-8")

    kept = _kept_section(repo)

    assert filename in kept
    assert phrase in kept
    # Written once and never rewritten, which is the reason the line exists:
    # a file nothing refreshes on its own is one a reset must not silently keep.
    assert "written once" in kept


@pytest.mark.parametrize("filename", [".env.users", ".env.auth"])
def test_a_web_credential_file_that_is_not_there_is_not_disclosed(repo, filename):
    """The kept section names what this deployment has, not what it could have.

    Every other entry in it is a real path in this repo; an entry for a file
    that does not exist would send an operator looking for one.
    """
    assert not (repo / filename).exists()

    assert filename not in _kept_section(repo)


def test_the_two_web_credential_files_disclose_different_refreshes(repo):
    """One removal costs a registry deploy its secrets, the other everyone's password.

    Mutually exclusive wording, pinned: a shared clause would be wrong for one
    of them, and this is the section an operator reads before typing a
    confirmation.
    """
    (repo / ".env.users").write_text("OSPREY_API_KEY=x\n", encoding="utf-8")
    (repo / ".env.auth").write_text("OSPREY_AUTH_PW_HASH_ALICE=x\n", encoding="utf-8")

    kept = _kept_section(repo)

    assert "registry-mode deploy expects it to be there already" in kept
    assert "mints a NEW password for every user" in kept
    assert "osprey users decommission" in kept


def test_reset_keeps_both_web_credential_files_on_disk(repo, no_down):
    """The disclosure and the behaviour, asserted against each other."""
    (repo / ".env.users").write_text("OSPREY_API_KEY=x\n", encoding="utf-8")
    (repo / ".env.auth").write_text("OSPREY_AUTH_PW_HASH_ALICE=x\n", encoding="utf-8")

    run_reset(repo, FakeRuntime())

    assert (repo / ".env.users").read_text(encoding="utf-8") == "OSPREY_API_KEY=x\n"
    assert (repo / ".env.auth").read_text(encoding="utf-8") == "OSPREY_AUTH_PW_HASH_ALICE=x\n"


# ---------------------------------------------------------------------------
# .env — minted tokens go, provider keys stay
# ---------------------------------------------------------------------------


def test_minted_tokens_go_and_provider_keys_stay(repo, no_down):
    run_reset(repo, FakeRuntime())
    text = (repo / ".env").read_text(encoding="utf-8")

    assert "ANTHROPIC_API_KEY=sk-provider-secret" in text
    assert "OSPREY_DISPATCH_TOKEN" not in text
    assert "minted-artifacts-secret" not in text
    assert "Auto-generated service auth tokens" not in text


def test_the_operators_own_lines_come_out_byte_identical(repo, no_down):
    """``.env`` is hand-edited. A reset strips a block; it does not reformat a file."""
    (repo / ".env").write_text(
        "# my own note\n"
        "\n"
        "ANTHROPIC_API_KEY='sk with spaces'\n"
        f"# {MINTED_ENV_BANNERS[0]}\n"
        "OSPREY_DISPATCH_TOKEN=x\n"
        "\n"
        "# trailing note\n"
        "OLOG_PASSWORD=hunter2\n",
        encoding="utf-8",
    )
    run_reset(repo, FakeRuntime())

    assert (repo / ".env").read_text(encoding="utf-8") == (
        "# my own note\n"
        "\n"
        "ANTHROPIC_API_KEY='sk with spaces'\n"
        "\n"
        "# trailing note\n"
        "OLOG_PASSWORD=hunter2\n"
    )


def test_every_minted_block_shape_is_stripped():
    """All three blocks a deploy writes, including the bluesky ones."""
    text = "".join(f"# {banner}\nKEY_{i}=v\n" for i, banner in enumerate(MINTED_ENV_BANNERS))
    planned = {f"KEY_{i}" for i in range(len(MINTED_ENV_BANNERS))}
    stripped, removed = strip_minted_env_blocks("KEEP=1\n" + text, planned)

    assert stripped == "KEEP=1\n"
    assert removed == [f"KEY_{i}" for i in range(len(MINTED_ENV_BANNERS))]


def test_a_token_minted_after_the_plan_survives_the_reset(repo, no_down, monkeypatch):
    """ "Nothing is re-discovered after the confirmation" has to be true of ``.env`` too.

    A concurrent ``osprey up`` — or an operator in another terminal — can append
    a minted block between the plan being printed and the confirmation being
    typed. That value was never shown, so it was never agreed to. The rewrite
    necessarily re-reads the file, but the re-read may only decide WHERE the
    planned keys are, never WHICH keys go.
    """
    env = repo / ".env"

    def append_a_new_token(_prompt: str) -> str:
        env.write_text(
            env.read_text(encoding="utf-8") + "OSPREY_LATE_TOKEN=minted-after-the-plan\n",
            encoding="utf-8",
        )
        return confirmation_token(repo)

    monkeypatch.setattr("builtins.input", append_a_new_token)
    emitted: list[str] = []

    reset_deployment(repo, probe=make_probe(FakeRuntime()), emit=emitted.append)

    text = env.read_text(encoding="utf-8")
    assert "OSPREY_LATE_TOKEN=minted-after-the-plan" in text
    assert "OSPREY_LATE_TOKEN" not in "\n".join(emitted)
    # The keys that WERE planned still went.
    assert "OSPREY_DISPATCH_TOKEN" not in text
    assert "ANTHROPIC_API_KEY=sk-provider-secret" in text


def test_a_surviving_key_keeps_the_banner_that_explains_it(repo, no_down):
    """A block is only removed whole when every key under it was planned."""
    (repo / ".env").write_text(
        f"# {MINTED_ENV_BANNERS[0]}\nPLANNED_TOKEN=a\nUNPLANNED_TOKEN=b\n",
        encoding="utf-8",
    )
    stripped, removed = strip_minted_env_blocks(
        (repo / ".env").read_text(encoding="utf-8"), {"PLANNED_TOKEN"}
    )

    assert removed == ["PLANNED_TOKEN"]
    assert stripped == (f"# {MINTED_ENV_BANNERS[0]}\nUNPLANNED_TOKEN=b\n")


def test_the_kept_count_does_not_undercount_a_name_shared_with_a_minted_key(repo):
    """An operator's own line is kept even when a minted key shares its name.

    Counting by subtracting minted names from a whole-file parse dropped it from
    the tally, so the plan under-reported what it was keeping while the strip
    correctly left it alone. Both halves now come from the same function.
    """
    (repo / ".env").write_text(
        "OSPREY_DISPATCH_TOKEN=my-own-value-outside-any-block\n"
        "ANTHROPIC_API_KEY=sk-x\n"
        f"# {MINTED_ENV_BANNERS[0]}\n"
        "OSPREY_DISPATCH_TOKEN=minted\n",
        encoding="utf-8",
    )
    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))

    assert set(plan.env_kept_keys) == {"OSPREY_DISPATCH_TOKEN", "ANTHROPIC_API_KEY"}


def test_the_env_file_keeps_its_mode_across_the_rewrite(repo, no_down):
    run_reset(repo, FakeRuntime())

    assert stat.S_IMODE((repo / ".env").stat().st_mode) == 0o600


def test_an_env_with_no_minted_block_is_not_rewritten(repo, no_down):
    (repo / ".env").write_text("ANTHROPIC_API_KEY=sk-x\n", encoding="utf-8")
    before = (repo / ".env").stat().st_mtime_ns

    run_reset(repo, FakeRuntime())

    assert (repo / ".env").stat().st_mtime_ns == before


def test_no_secret_value_reaches_the_printed_plan(repo):
    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))
    rendered = "\n".join(plan.render())

    assert "minted-dispatch-secret" not in rendered
    assert "sk-provider-secret" not in rendered
    # Names, on the other hand, are how an operator recognizes what is going.
    assert "OSPREY_DISPATCH_TOKEN" in rendered


def test_the_plan_counts_the_provider_keys_it_is_keeping(repo):
    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))

    assert plan.env_kept_keys == ("ANTHROPIC_API_KEY",)


def test_every_banner_the_deploy_writes_is_one_reset_knows_about():
    """The drift gate on the ``.env`` block format.

    ``MINTED_ENV_BANNERS`` is matched against text already written into operators'
    ``.env`` files, so it is a data format rather than a message. A fourth block
    added to the deploy path — or an existing banner reworded — would silently
    orphan those values: reset would report the tokens stripped while leaving
    them in the file. This reads the writer's own call sites instead of trusting
    that anyone remembered.
    """
    source = Path(reset_mod.__file__).with_name("container_lifecycle.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))

    written = {
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_append_env_block"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }

    assert written, "found no _append_env_block call sites — the gate has stopped working"
    assert written <= set(MINTED_ENV_BANNERS), (
        f"container_lifecycle writes .env block(s) reset does not recognize: "
        f"{sorted(written - set(MINTED_ENV_BANNERS))}. Add them to MINTED_ENV_BANNERS, "
        f"keeping the old spellings for .env files already on disk."
    )


# ---------------------------------------------------------------------------
# The filesystem
# ---------------------------------------------------------------------------


def test_agent_data_is_destroyed_and_left_as_an_empty_directory(repo, no_down):
    run_reset(repo, FakeRuntime())
    agent_data = repo / "var" / "agent_data"

    assert agent_data.is_dir()
    assert list(agent_data.iterdir()) == []


def _relocate_agent_data(repo: Path, base_dir: str) -> None:
    """Point this build's ``agent_data.base_dir`` somewhere other than the default."""
    import yaml

    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    config["agent_data"] = {"base_dir": base_dir}
    (repo / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_a_relocated_agent_data_root_is_planned_printed_and_wiped(repo, no_down):
    """``agent_data.base_dir`` is a real key, and reading the default instead is silent.

    A deployment that moved its agent data got neither the wipe nor a word about
    it: the plan said the agent's memory was destroyed while the directory sat
    untouched. For the one verb whose contract is "what is printed is what
    happens", reading the configured key is the contract.
    """
    _relocate_agent_data(repo, "var/memory")
    memory = repo / "var" / "memory"
    memory.mkdir(parents=True)
    (memory / "notes.md").write_text("the agent's memory", encoding="utf-8")

    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))
    rendered = "\n".join(plan.render())
    assert memory in plan.paths
    assert "var/memory/" in rendered
    assert "the agent's memory" in rendered

    run_reset(repo, FakeRuntime())

    assert not (memory / "notes.md").exists()
    assert memory.is_dir()  # recreated empty, as the default location would be


def test_the_default_location_is_not_touched_when_the_config_moved_it(repo, no_down):
    """The mirror image: reset must not wipe a directory this deployment does not use."""
    _relocate_agent_data(repo, "var/memory")
    (repo / "var" / "memory").mkdir(parents=True)
    stale = repo / "var" / "agent_data" / "memory" / "notes.md"

    run_reset(repo, FakeRuntime())

    assert stale.is_file()


@pytest.mark.parametrize("spelling", ["{outside}", "~/{name}"])
def test_an_agent_data_root_outside_the_repo_is_disclosed_and_never_deleted(
    repo, no_down, tmp_path, monkeypatch, spelling
):
    """The containment boundary: reset deletes directories, so it stops at the repo.

    ``base_dir`` may be absolute or ``~``-relative, so a config can point it at a
    shared filesystem or a home directory. Reset's authority to delete comes from
    a directory being inside the repo the operator named — outside it, that
    authority does not exist whatever the config says. Both spellings that can
    escape the repo are covered.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    outside = tmp_path / "shared-agent-data"
    outside.mkdir()
    target = Path(spelling.format(outside=outside, name="shared-agent-data"))
    resolved = home / "shared-agent-data" if spelling.startswith("~") else outside
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "notes.md").write_text("not reset's to delete", encoding="utf-8")
    _relocate_agent_data(repo, str(target))

    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))
    rendered = "\n".join(plan.render())

    assert resolved not in plan.paths
    assert plan.external_paths == [resolved]
    assert "NOT REMOVED" in rendered
    assert str(resolved) in rendered
    assert "OUTSIDE this repo" in rendered

    run_reset(repo, FakeRuntime())

    assert (resolved / "notes.md").is_file()


def test_an_agent_data_root_that_is_the_repo_itself_is_never_deleted(repo, no_down):
    """``base_dir: "."`` is the most dangerous value this key can take.

    It resolves to the repo root, and removing that would take the tracked
    source zone and ``.git`` with it — the one thing reset promises never to
    touch. The containment check excludes the root itself, not just paths above
    it, so this lands in NOT REMOVED like any other out-of-bounds root.
    """
    _relocate_agent_data(repo, ".")

    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))

    assert repo not in plan.paths
    assert plan.external_paths == [repo]

    # The REASON is pinned, not just the refusal. This path is not "outside the
    # repo" — it IS the repo — so printing the boundary rationale here would
    # state something false about the very path it is printing.
    section = _unidentified_section(plan)
    assert "resolves to the repo root" in section
    assert "would take the source zone and .git with it" in section
    assert "OUTSIDE this repo" not in section

    run_reset(repo, FakeRuntime())

    assert (repo / "profile.yml").is_file()
    assert (repo / "data" / "channels.json").is_file()
    assert repo.is_dir()


def test_an_outside_root_and_a_repo_root_get_their_own_reasons(repo, no_down, tmp_path):
    """Both refusals can appear at once, and neither may borrow the other's reason."""
    outside = tmp_path / "shared-agent-data"
    outside.mkdir()
    _relocate_agent_data(repo, str(outside))
    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))
    plan.external_paths = [repo, outside]

    section = "\n".join(plan.render()).split("NOT REMOVED", 1)[1].split("Networks are", 1)[0]

    assert "resolves to the repo root" in section
    assert "OUTSIDE this repo" in section
    # The repo-root line must come with its own rationale, not the boundary one.
    root_line, outside_line = section.index(str(repo)), section.index(str(outside))
    assert root_line < section.index("would take the source zone") < outside_line


def test_an_unreadable_build_config_does_not_block_a_reset(repo, no_down):
    """A reset is what an operator reaches for when a deployment is already broken."""
    (repo / "build" / "config.yml").write_text("{{{ not yaml", encoding="utf-8")

    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))

    assert "derived from this directory's name" in plan.project_name_source
    assert repo / "var" / "agent_data" in plan.paths


def test_the_build_zone_is_deleted_outright(repo, no_down):
    """Its ABSENCE is what ``osprey up`` reads as "no build found"."""
    run_reset(repo, FakeRuntime())

    assert not (repo / "build").exists()


def test_the_source_zone_is_untouched(repo, no_down):
    run_reset(repo, FakeRuntime())

    assert (repo / "profile.yml").read_text(encoding="utf-8") == "preset: control-assistant\n"
    assert (repo / "data" / "channels.json").is_file()
    assert (repo / "personas").is_dir()


def test_a_repo_with_nothing_to_reset_says_so_and_skips_the_gate(tmp_path, no_down, monkeypatch):
    def never(_prompt):  # pragma: no cover - reaching it is the failure
        raise AssertionError("a no-op reset must not prompt")

    monkeypatch.setattr("builtins.input", never)
    root = tmp_path / PROJECT
    root.mkdir()
    (root / "profile.yml").write_text("preset: hello-world\n", encoding="utf-8")
    emitted: list[str] = []

    outcome = reset_deployment(root, probe=make_probe(FakeRuntime()), emit=emitted.append)

    assert outcome is ResetOutcome.NOTHING_TO_DO
    assert "Nothing to reset" in "\n".join(emitted)


# ---------------------------------------------------------------------------
# Which project name, and where it came from
# ---------------------------------------------------------------------------


def test_the_project_name_comes_from_the_build_when_there_is_one(repo):
    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))

    assert plan.project == PROJECT
    assert "build/config.yml" in plan.project_name_source


def test_without_a_build_the_name_is_derived_and_the_plan_says_it_is_a_derivation(repo):
    """An operator is the only one who can tell whether a guessed name is right."""
    (repo / "build" / "config.yml").unlink()
    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))

    assert plan.project == PROJECT
    assert "derived from this directory's name" in plan.project_name_source
    assert "derived from this directory's name" in "\n".join(plan.render())


# ---------------------------------------------------------------------------
# Images: verified by project label, because there is no per-checkout image
# ---------------------------------------------------------------------------


def test_the_project_image_is_removed_when_its_label_says_it_is_ours(repo, no_down):
    fake = FakeRuntime(images={f"{PROJECT}:local": {reset_mod.OSPREY_PROJECT_LABEL: PROJECT}})
    run_reset(repo, fake)

    assert fake.removed_names() == [f"{PROJECT}:local"]


def test_a_same_named_image_belonging_to_something_else_survives(repo, no_down):
    fake = FakeRuntime(images={f"{PROJECT}:local": {reset_mod.OSPREY_PROJECT_LABEL: "other"}})
    run_reset(repo, fake)

    assert fake.removals == []
    assert f"{PROJECT}:local" in fake.images


def test_an_unlabelled_image_is_not_assumed_to_be_ours(repo, no_down):
    fake = FakeRuntime(images={f"{PROJECT}:local": {}})
    run_reset(repo, fake)

    assert fake.removals == []


def test_a_missing_image_tag_is_not_an_error_and_not_in_the_plan(repo, no_down):
    plan = plan_reset(repo, probe=make_probe(FakeRuntime()))

    assert plan.images == []


def _with_web_terminals(repo: Path) -> None:
    """Give the build a web-terminal roster with a local-mode persona catalog."""
    import yaml

    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    config["facility"] = {"name": "ALS", "prefix": "als", "timezone": "UTC"}
    config["registry"] = {"url": "registry.example.org"}
    config["modules"] = {
        "web_terminals": {
            "enabled": True,
            "nginx_port": 8080,
            "web_base_port": 9000,
            "artifact_base_port": 9100,
            "ariel_base_port": 9200,
            "lattice_base_port": 9300,
            "image_source": "local",
            "users": [{"name": "alice", "index": 0, "persona": "control-room"}],
            "personas": {"control-room": {"project": "acc-control", "project_path": "/x"}},
        }
    }
    (repo / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_a_web_terminal_personas_local_image_is_a_candidate(repo, no_down):
    """Reset absorbs ``deploy nuke``, whose image half is the persona tags."""
    _with_web_terminals(repo)
    persona_tag = "acc-control:local"
    fake = FakeRuntime(
        images={
            f"{PROJECT}:local": {reset_mod.OSPREY_PROJECT_LABEL: PROJECT},
            persona_tag: {reset_mod.OSPREY_PROJECT_LABEL: PROJECT},
        }
    )
    run_reset(repo, fake)

    assert sorted(fake.removed_names()) == sorted([f"{PROJECT}:local", persona_tag])


def test_a_persona_image_belonging_to_another_deployment_survives(repo, no_down):
    """Image tags are host-global, so the project label is checked tag by tag."""
    _with_web_terminals(repo)
    persona_tag = "acc-control:local"
    fake = FakeRuntime(images={persona_tag: {reset_mod.OSPREY_PROJECT_LABEL: "someone-else"}})
    run_reset(repo, fake)

    assert fake.removals == []
    assert persona_tag in fake.images


def test_an_unreadable_roster_leaves_the_project_image_removable(repo, no_down, caplog):
    """A broken persona catalog degrades reset's image half; it never blocks the reset."""
    import yaml

    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    config["modules"] = {"web_terminals": {"enabled": True, "users": "not-a-roster"}}
    (repo / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    fake = FakeRuntime(images={f"{PROJECT}:local": {reset_mod.OSPREY_PROJECT_LABEL: PROJECT}})

    run_reset(repo, fake)

    assert fake.removed_names() == [f"{PROJECT}:local"]
    assert not (repo / "build").exists()


def test_no_image_is_removed_when_any_foreign_resource_exists(repo, no_down):
    """The load-bearing half of the image argument, asserted rather than reasoned.

    Images are the one class with no per-checkout identity to gate on, so the
    case for removing them at all rests on an ORDERING claim: reset reaches image
    removal only after the container/volume scan came back with nothing foreign.
    If that ordering ever inverted — images discovered or removed before the
    refusal — a shared ``<project>:local`` tag could be pulled out from under a
    live sibling deployment while reset was in the act of refusing to touch its
    volumes.

    So this sets up an image that WOULD qualify on its own (its
    ``com.osprey.project`` matches) beside a single foreign volume, and asserts
    the image is neither inspected nor removed.
    """
    fake = FakeRuntime(
        volumes={"als-exemplar_dispatch_workspace": theirs()},
        images={f"{PROJECT}:local": {reset_mod.OSPREY_PROJECT_LABEL: PROJECT}},
    )

    with pytest.raises(ForeignCheckoutError):
        reset_deployment(repo, probe=make_probe(fake), assume_yes=True, emit=lambda _: None)

    assert fake.removals == []
    assert f"{PROJECT}:local" in fake.images
    assert not [argv for argv in fake.calls if argv[1:3] == ["image", "inspect"]]
    assert not [argv for argv in fake.calls if argv[1:3] == ["image", "rm"]]


def test_the_plan_states_how_images_are_verified(repo):
    """Images are scoped differently from containers, so the plan must not imply otherwise."""
    fake = FakeRuntime(images={f"{PROJECT}:local": {reset_mod.OSPREY_PROJECT_LABEL: PROJECT}})
    rendered = "\n".join(plan_reset(repo, probe=make_probe(fake)).render())

    assert f"{reset_mod.OSPREY_PROJECT_LABEL}={PROJECT} verified" in rendered


# ---------------------------------------------------------------------------
# Images: the tag is the one the BUILD produced, not the ``:local`` literal
# ---------------------------------------------------------------------------
#
# ``OSPREY_IMAGE_REGISTRY`` / ``OSPREY_IMAGE_TAG`` (over ``images.registry`` /
# ``images.tag``) rename every image OSPREY builds. Reset names the image it
# wants to remove BEFORE it looks at the host, so a restated ``:local`` literal
# does not fail loudly — it silently names an image that was never built, finds
# nothing, and leaves the real one on the disk forever. Both halves of the
# naming are therefore pinned here: the tag reset asks for, and the roster
# filter that decides which other images are ours.

_AXIS_REGISTRY = "registry.example.org/beam"
_AXIS_TAG = "2026.08.19"
_AXIS_IMAGE = f"{_AXIS_REGISTRY}/{PROJECT}:{_AXIS_TAG}"


def _with_image_axes(repo: Path, registry: str, tag: str) -> None:
    """Pin the axes in the rendered config, the way a profile would."""
    import yaml

    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    config["images"] = {"registry": registry, "tag": tag}
    (repo / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_the_project_image_tag_is_local_while_no_axis_is_set(repo):
    """The unchanged case, asserted so the axis work cannot move it by accident."""
    assert reset_mod._candidate_image_tags(repo, PROJECT) == [f"{PROJECT}:local"]


def test_the_environment_axes_rename_the_image_reset_removes(repo, no_down, monkeypatch):
    """The deploy built ``<registry>/<project>:<tag>``, so that is what must go.

    The ``:local`` image is present on this host too, and is asserted to survive:
    reset removes the image THIS deployment's configuration names, and naming a
    second tag on a suffix guess is how a shared host loses a sibling's image.
    """
    monkeypatch.setenv("OSPREY_IMAGE_REGISTRY", _AXIS_REGISTRY)
    monkeypatch.setenv("OSPREY_IMAGE_TAG", _AXIS_TAG)
    fake = FakeRuntime(
        images={
            _AXIS_IMAGE: {reset_mod.OSPREY_PROJECT_LABEL: PROJECT},
            f"{PROJECT}:local": {reset_mod.OSPREY_PROJECT_LABEL: PROJECT},
        }
    )

    run_reset(repo, fake)

    assert fake.removed_names() == [_AXIS_IMAGE]
    assert f"{PROJECT}:local" in fake.images


def test_the_configured_axes_rename_the_image_reset_removes(repo, no_down):
    """The config layer, which a deploy carries in ``build/config.yml`` with no env at all."""
    _with_image_axes(repo, _AXIS_REGISTRY, _AXIS_TAG)
    fake = FakeRuntime(images={_AXIS_IMAGE: {reset_mod.OSPREY_PROJECT_LABEL: PROJECT}})

    run_reset(repo, fake)

    assert fake.removed_names() == [_AXIS_IMAGE]


def test_the_environment_axis_wins_over_the_configured_one(repo):
    """Same precedence the render resolves, because both must name one image."""
    _with_image_axes(repo, "stale.example.org", "stale")

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("OSPREY_IMAGE_REGISTRY", _AXIS_REGISTRY)
        patch.setenv("OSPREY_IMAGE_TAG", _AXIS_TAG)

        assert reset_mod._candidate_image_tags(repo, PROJECT) == [_AXIS_IMAGE]


def test_the_axes_still_apply_when_there_is_no_rendered_config(repo, monkeypatch):
    """A build directory that was already removed must not silently restore ``:local``."""
    (repo / "build" / "config.yml").unlink()
    monkeypatch.setenv("OSPREY_IMAGE_REGISTRY", _AXIS_REGISTRY)
    monkeypatch.setenv("OSPREY_IMAGE_TAG", _AXIS_TAG)

    assert reset_mod._candidate_image_tags(repo, PROJECT) == [_AXIS_IMAGE]


def test_an_unreadable_config_degrades_to_the_project_name_and_the_axes(repo, monkeypatch):
    """An unparseable build is a warning and a fallback, never an exception."""
    (repo / "build" / "config.yml").write_text("{not: valid: yaml:", encoding="utf-8")
    monkeypatch.setenv("OSPREY_IMAGE_TAG", _AXIS_TAG)

    assert reset_mod._candidate_image_tags(repo, PROJECT) == [f"{PROJECT}:{_AXIS_TAG}"]


def _roster_of(monkeypatch, *images: str) -> None:
    """Make the roster resolve to exactly *images*, whatever the persona rules say.

    The persona naming rules produce ``:local`` images and only ``:local``
    images, so the filter's other half — an image the AXES named — cannot be
    reached through them. Substituting the roster is what puts that half under
    test without asserting a persona shape the code does not produce.
    """
    from osprey.deployment.web_terminals import personas as personas_mod

    roster = [{"name": f"u{index}", "image": image} for index, image in enumerate(images)]
    monkeypatch.setattr(personas_mod, "resolve_personas", lambda *args, **kwargs: roster)


def test_a_roster_image_the_axes_named_is_a_candidate(repo, monkeypatch):
    """The filter's second half: ours by name, not by a ``:local`` suffix it no longer carries."""
    _with_web_terminals(repo)
    _with_image_axes(repo, _AXIS_REGISTRY, _AXIS_TAG)
    qmd_image = f"{_AXIS_REGISTRY}/{PROJECT}-qmd:{_AXIS_TAG}"
    _roster_of(monkeypatch, qmd_image)

    assert reset_mod._candidate_image_tags(repo, PROJECT) == [_AXIS_IMAGE, qmd_image]


def test_a_pulled_upstream_image_sharing_the_axis_tag_is_not_a_candidate(repo, monkeypatch):
    """The reason the filter matches whole names instead of the tag.

    A registry-mode roster image is PULLED, and its tag can coincide with the
    axis tag. Widening the filter to ``endswith(":<tag>")`` would put an
    upstream image on the removal list, where only its project label stands
    between it and a ``docker image rm``.
    """
    _with_web_terminals(repo)
    _with_image_axes(repo, _AXIS_REGISTRY, _AXIS_TAG)
    pulled = f"{_AXIS_REGISTRY}/web-terminal:{_AXIS_TAG}"
    _roster_of(monkeypatch, pulled)

    assert reset_mod._candidate_image_tags(repo, PROJECT) == [_AXIS_IMAGE]


def test_a_local_roster_image_stays_a_candidate_under_a_set_axis(repo, monkeypatch):
    """``:local`` is the persona convention and the axes do not move it."""
    _with_web_terminals(repo)
    _with_image_axes(repo, _AXIS_REGISTRY, _AXIS_TAG)
    persona_tag = "acc-control:local"
    _roster_of(monkeypatch, persona_tag)

    assert reset_mod._candidate_image_tags(repo, PROJECT) == [_AXIS_IMAGE, persona_tag]


# ---------------------------------------------------------------------------
# Networks: the plan says what reset does NOT do
# ---------------------------------------------------------------------------


def test_the_plan_is_honest_that_reset_removes_no_network(repo):
    """Nothing labels a network with a checkout identity, so nothing here can scope one."""
    rendered = "\n".join(plan_reset(repo, probe=make_probe(FakeRuntime())).render())

    assert "Networks are removed by the `down`" in rendered
    assert "reset never removes one on its own" in rendered


def test_reset_issues_no_network_argv_at_all(repo, no_down):
    fake = FakeRuntime(containers={"mine": ours(repo)})
    run_reset(repo, fake)

    assert not [argv for argv in fake.calls if "network" in argv]


# ---------------------------------------------------------------------------
# A runtime that is not running
# ---------------------------------------------------------------------------


def test_an_unreachable_runtime_refuses_instead_of_reporting_an_empty_plan(repo, monkeypatch):
    """An empty filtered listing from a downed daemon looks exactly like "already clean"."""
    monkeypatch.setattr(
        "osprey.deployment.runtime_helper.verify_runtime_is_running",
        lambda config=None: (False, "Docker daemon is not running"),
    )

    with pytest.raises(RuntimeError, match="not running"):
        reset_deployment(repo, assume_yes=True, emit=lambda _: None)


def test_a_listing_that_fails_is_raised_rather_than_read_as_empty(repo):
    def failing(argv, **kwargs):
        if argv[1:3] == ["ps", "-a"]:
            return subprocess.CompletedProcess(argv, 1, "", "permission denied")
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(RuntimeError, match="Could not list this project's containers"):
        plan_reset(repo, probe=RuntimeProbe("docker", run=failing))


# ---------------------------------------------------------------------------
# The CLI surface
# ---------------------------------------------------------------------------


def test_cli_dry_run_prints_the_plan_and_exits_zero(repo, monkeypatch):
    fake = FakeRuntime(volumes={"mine-vol": ours(repo)})
    monkeypatch.setattr(reset_mod, "_default_probe", lambda root: make_probe(fake))

    result = CliRunner().invoke(reset_command, ["--repo", str(repo), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "WILL BE REMOVED" in result.output
    assert "WILL BE KEPT" in result.output
    assert fake.removals == []


def test_cli_refuses_and_names_the_other_checkout(repo, monkeypatch):
    fake = FakeRuntime(volumes={"mine-vol": theirs()})
    monkeypatch.setattr(reset_mod, "_default_probe", lambda root: make_probe(fake))

    result = CliRunner().invoke(reset_command, ["--repo", str(repo)])

    assert result.exit_code != 0
    assert OTHER_PATH in result.output
    assert fake.removals == []


@pytest.mark.parametrize(
    ("argv", "typed", "expected_code"),
    [
        (["--dry-run"], None, 0),
        ([], "wrong-name", 1),
        ([], PROJECT, 0),
    ],
    ids=["dry-run-is-success", "decline-is-failure", "completed-is-success"],
)
def test_cli_exit_codes_distinguish_a_decline_from_a_dry_run(
    repo, no_down, monkeypatch, argv, typed, expected_code
):
    """A declined reset must stop a shell chain; a dry run and a no-op must not.

    ``osprey reset && osprey build && osprey up -d`` is the documented start-over
    recipe. Exiting 0 on a decline rebuilds and restarts the deployment the
    operator just refused to wipe, with the refusal scrolled off the top.
    """
    fake = FakeRuntime(volumes={"mine-vol": ours(repo)})
    monkeypatch.setattr(reset_mod, "_default_probe", lambda root: make_probe(fake))
    if typed is not None:
        monkeypatch.setattr("builtins.input", lambda _p: typed)

    result = CliRunner().invoke(reset_command, ["--repo", str(repo), *argv])

    assert result.exit_code == expected_code, result.output


def test_a_partial_reset_exits_three_rather_than_claiming_success(repo, no_down, monkeypatch):
    """A survivor on the plan must stop a chain, distinctly from a decline.

    Exit 0 here would be the false-completion defect moved out of the prose and
    into ``$?``: `osprey reset && <handoff>` would proceed on the assertion that
    the deployment is clean while a volume — and the data in it — is still
    there. 3 rather than 1 so a caller that genuinely wants to continue on a
    partial reset can say so explicitly, and 3 rather than 2 because click
    already spends 2 on usage errors (see the reserved-code test below).
    """
    fake = FakeRuntime(volumes={"stuck-vol": ours(repo), "fine-vol": ours(repo)})
    original = fake._remove

    def refuse(store, name, kind):
        if name == "stuck-vol":
            return subprocess.CompletedProcess([], 1, "", "volume is in use")
        return original(store, name, kind)

    fake._remove = refuse
    monkeypatch.setattr(reset_mod, "_default_probe", lambda root: make_probe(fake))

    result = CliRunner().invoke(reset_command, ["--repo", str(repo), "-y"])

    assert result.exit_code == 3, result.output
    assert "could not" in result.output
    assert "stuck-vol" in result.output
    assert "Reset complete" not in result.output
    # Still distinct from a decline, which is 1 and touches nothing.
    assert "fine-vol" not in fake.volumes


def test_every_outcome_has_an_exit_code(repo):
    """A new outcome must not silently fall through to success.

    The mapping is a table rather than a chain of branches so that adding a
    member without deciding its code is a KeyError at the call site. This
    asserts the table is total, which is what makes that guarantee real.
    """
    from osprey.deployment.reset import EXIT_CODES

    assert set(EXIT_CODES) == set(ResetOutcome)
    assert EXIT_CODES[ResetOutcome.COMPLETED] == 0
    assert EXIT_CODES[ResetOutcome.DRY_RUN] == 0
    assert EXIT_CODES[ResetOutcome.NOTHING_TO_DO] == 0
    assert EXIT_CODES[ResetOutcome.DECLINED] == 1
    assert EXIT_CODES[ResetOutcome.COMPLETED_WITH_FAILURES] == 3
    # 2 is click's, and this verb must never take it.
    assert 2 not in EXIT_CODES.values()


def test_two_belongs_to_click_and_this_verb_never_takes_it(repo):
    """Documentation-by-test of the reserved code, so nobody "closes the gap".

    ``click.UsageError.exit_code`` is 2, so a mistyped flag already exits 2 —
    verified here through the real command rather than asserted from the class
    attribute. If reset also used 2 for a partly-reset deployment, a calling
    script could not tell "your command line was wrong" from "your data is
    half-gone", which is the ambiguity the whole mapping exists to remove.
    """
    from osprey.deployment.reset import EXIT_CODES

    result = CliRunner().invoke(reset_command, ["--repo", str(repo), "--bogus-flag"])

    assert result.exit_code == 2
    assert 2 not in EXIT_CODES.values()


def test_an_interrupt_mid_teardown_reports_the_partial_state_not_a_clean_refusal(repo, monkeypatch):
    """The other way to reach a partly-reset deployment, and it must signal the same.

    An interrupt AT the prompt is a decline (1, nothing touched). An interrupt
    AFTER it leaves exactly the state a stuck removal leaves, so it exits with
    the partial code. Reporting 1 here would tell a script "nothing was touched"
    about a half-wiped deployment — the same false-completion defect, reached by
    a different route.
    """
    from osprey.deployment.reset import PARTIAL_RESET_EXIT_CODE

    def interrupt_during_teardown(_root):
        raise KeyboardInterrupt

    monkeypatch.setattr(reset_mod, "down_deployment", interrupt_during_teardown)
    fake = FakeRuntime(volumes={"mine-vol": ours(repo)})
    monkeypatch.setattr(reset_mod, "_default_probe", lambda root: make_probe(fake))

    result = CliRunner().invoke(reset_command, ["--repo", str(repo), "-y"])

    assert result.exit_code == PARTIAL_RESET_EXIT_CODE == 3
    assert "partly-reset state" in result.output
    assert "Nothing was touched" not in result.output


def test_a_declined_reset_exits_nonzero_without_a_second_error_line(repo, no_down, monkeypatch):
    """The reason was printed with the plan; Click must not restate it as "Error:"."""
    fake = FakeRuntime(volumes={"mine-vol": ours(repo)})
    monkeypatch.setattr(reset_mod, "_default_probe", lambda root: make_probe(fake))
    monkeypatch.setattr("builtins.input", lambda _p: "nope")

    result = CliRunner().invoke(reset_command, ["--repo", str(repo)])

    assert result.exit_code == 1
    assert "confirmation did not match" in result.output
    assert "Error:" not in result.output


def test_cli_exit_code_is_zero_when_there_was_nothing_to_reset(tmp_path, no_down, monkeypatch):
    root = tmp_path / PROJECT
    root.mkdir()
    (root / "profile.yml").write_text("preset: hello-world\n", encoding="utf-8")
    monkeypatch.setattr(reset_mod, "_default_probe", lambda r: make_probe(FakeRuntime()))

    result = CliRunner().invoke(reset_command, ["--repo", str(root)])

    assert result.exit_code == 0, result.output
    assert "Nothing to reset" in result.output


def test_cli_help_names_what_survives(repo):
    result = CliRunner().invoke(reset_command, ["--help"])

    assert "var/audit/" in result.output
    assert "--purge-audit" in result.output


def test_cli_help_documents_the_exit_status_contract(repo):
    """A script author has to be able to discover what `reset && ...` promises."""
    result = CliRunner().invoke(reset_command, ["--help"])

    assert "Exit status" in result.output
    for code in ("0 ", "1 ", "3 "):
        assert code in result.output
    assert "nothing was touched" in result.output
    assert "partly reset" in result.output
    # The reservation is documented too, so an operator reading --help learns
    # why there is no 2 rather than assuming an omission.
    assert "2 is never this command's" in result.output


# ---------------------------------------------------------------------------
# Registration: `osprey reset` reaches THIS command
# ---------------------------------------------------------------------------


def test_reset_resolves_to_this_modules_command_object():
    """Object identity, not equality — the failure mode here is invisible by repr.

    ``reset`` takes ``LazyGroup``'s default ``getattr(mod, cmd_name)`` path
    because, unlike ``up``/``down``/``restart``, there is no legacy ``deploy
    reset`` subcommand for it to collide with. That is a fact about the tree, not
    a permanent property: the day something else in the import path answers to
    the name ``reset``, this lookup starts returning a DIFFERENT command that
    still prints ``<Command reset>``, and an equality assertion would pass while
    ``osprey reset`` quietly meant something else. Identity is what catches it.
    """
    from osprey.cli.main import cli

    resolved = cli.get_command(None, "reset")

    assert resolved is reset_command, (
        "osprey reset resolved to a different command object than "
        "reset_cmd.reset. Compare their .params — the reprs are identical."
    )
    assert "reset" in cli.list_commands(None)


def test_the_registered_reset_is_repo_scoped_not_project_scoped():
    """The clearest signal that a wrong object was registered, per verb family."""
    from osprey.cli.main import cli

    params = {param.name for param in cli.get_command(None, "reset").params}

    assert "repo" in params
    assert {"dry_run", "assume_yes", "purge_audit"} <= params
    assert "project" not in params
    assert "config" not in params


def test_no_legacy_reset_namesake_exists_to_collide_with():
    """The precondition that makes the elif-free registration correct.

    Recorded as an assertion rather than left as a claim in a comment: if Phase 3
    (or anything else) ever adds a ``reset`` attribute to ``deploy_cmd``, the
    reasoning behind omitting an elif branch stops holding, and this says so at
    the moment it changes rather than after a mis-resolution ships.
    """
    from osprey.cli import deploy_cmd

    assert not hasattr(deploy_cmd, "reset")


def _unidentified_section(plan) -> str:
    """Just the "NOT REMOVED" block, without the network note that follows it."""
    rendered = "\n".join(plan.render())
    return rendered.split("NOT REMOVED", 1)[1].split("Networks are removed", 1)[0]
