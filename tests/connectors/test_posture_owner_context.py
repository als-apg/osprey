"""The owner's tree record and its verdict, out of one no-follow read.

A service that admits work on somebody's behalf asks two questions of the
read-only control-context tree before it writes: which target does that owner's
record name, and does the record narrow the service's target. Asked as two reads
they can disagree — a record rewritten between them answers the target from one
file and the narrowing from another — so
:func:`~osprey_connectors.posture_store.owner_context` answers both from one.

These tests pin what that reader adds over the verdict reader: the record comes
back normalised against the caller's baseline (an unswitched record follows the
deployment), a record on another target comes back as it is so the caller can
refuse the mismatch, an absent record is an answer rather than a failure, a
planted link refuses, and a process with no tree bound refuses rather than
falling back to a host record it does not own.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from osprey_connectors import control_context, posture_store
from osprey_connectors.posture_store import StoreVerdict

OWNER = "alice"


def _provision_tree(root: Path) -> Path:
    """A marked, readable tree at ``<root>/control_target`` — what the build leaves."""
    tree = root / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    return tree


def _write_tree_record(
    tree: Path, owner: str, *, target: str, generation: int, posture=None
) -> Path:
    """One owner's record inside the tree, written the way its owner writes it."""
    path = tree / owner / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(
            target=target, generation=generation, posture=dict(posture or {})
        ),
        path=path,
    )
    posture_store.invalidate_cache()
    return path


@pytest.fixture
def bound_tree(tmp_path, monkeypatch):
    """A provisioned read-only tree bound in, with no owner named anywhere."""
    tree = _provision_tree(tmp_path)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    posture_store.invalidate_cache()
    yield tree
    posture_store.invalidate_cache()


def test_owner_context_is_public_api():
    assert "owner_context" in posture_store.__all__
    assert "OwnerContext" in posture_store.__all__
    params = inspect.signature(posture_store.owner_context).parameters
    assert list(params) == ["owner", "target", "baseline"]
    assert params["baseline"].kind is inspect.Parameter.KEYWORD_ONLY


def test_a_switched_record_on_the_target_comes_back_with_its_verdict(bound_tree):
    _write_tree_record(bound_tree, OWNER, target="va", generation=4)

    answer = posture_store.owner_context(OWNER, "va", baseline="live")

    assert answer.record is not None
    assert answer.record.target == "va"
    assert answer.record.generation == 4
    assert answer.detail == posture_store.StoreVerdictDetail(StoreVerdict.PERMITTED, None)


def test_a_generation_zero_record_follows_the_baseline(bound_tree):
    """Generation 0 is a claim, never a switch: the record follows the deployment."""
    _write_tree_record(bound_tree, OWNER, target="va", generation=0)

    answer = posture_store.owner_context(OWNER, "live", baseline="live")

    assert answer.record is not None
    assert answer.record.target == "live"
    assert answer.record.generation == 0
    assert answer.detail.verdict is StoreVerdict.PERMITTED


def test_a_switched_record_on_another_target_is_returned_as_it_is(bound_tree):
    """The caller compares targets and refuses the mismatch; the reader never moves it."""
    _write_tree_record(bound_tree, OWNER, target="va", generation=2)

    answer = posture_store.owner_context(OWNER, "live", baseline="live")

    assert answer.record is not None
    assert answer.record.target == "va"
    assert answer.record.generation == 2
    # Nothing narrows "live" in this record, so the verdict itself grants: the
    # mismatch is the caller's refusal, not the record clause's.
    assert answer.detail.verdict is StoreVerdict.PERMITTED


def test_a_narrowed_record_answers_narrowing_beside_the_record(bound_tree):
    _write_tree_record(bound_tree, OWNER, target="live", generation=1, posture={"live": "sandbox"})

    answer = posture_store.owner_context(OWNER, "live", baseline="live")

    assert answer.record is not None
    assert answer.record.posture == {"live": "sandbox"}
    assert answer.detail == posture_store.StoreVerdictDetail(StoreVerdict.NARROWING, None)


def test_the_verdict_matches_the_record_verdict_reader(bound_tree):
    """One clause, two spellings: both readers decide the same record the same way."""
    _write_tree_record(bound_tree, OWNER, target="live", generation=1, posture={"va": "sandbox"})

    for target in ("live", "va", None):
        answer = posture_store.owner_context(OWNER, target, baseline="live")
        assert answer.detail == posture_store.record_verdict_detail(target, OWNER)


@pytest.mark.usefixtures("bound_tree")
def test_a_missing_record_is_an_answer_not_a_failure():
    answer = posture_store.owner_context(OWNER, "live", baseline="live")

    assert answer.record is None
    assert answer.detail == posture_store.StoreVerdictDetail(StoreVerdict.PERMITTED, None)


def test_a_missing_record_under_an_existing_owner_directory_is_an_answer(bound_tree):
    (bound_tree / OWNER).mkdir()

    answer = posture_store.owner_context(OWNER, "live", baseline="live")

    assert answer.record is None
    assert answer.detail.verdict is StoreVerdict.PERMITTED


def test_a_planted_link_at_the_record_refuses(bound_tree, tmp_path):
    elsewhere = _write_tree_record(
        _provision_tree(tmp_path / "elsewhere"), "mallory", target="live", generation=5
    )
    (bound_tree / OWNER).mkdir()
    os.symlink(elsewhere, bound_tree / OWNER / control_context.RECORD_FILENAME)

    answer = posture_store.owner_context(OWNER, "live", baseline="live")

    assert answer.record is None
    assert answer.detail.verdict is StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert "symbolic link" in answer.detail.reason


def test_a_planted_link_at_the_owner_directory_refuses(bound_tree, tmp_path):
    other = _provision_tree(tmp_path / "elsewhere")
    _write_tree_record(other, "mallory", target="live", generation=5)
    os.symlink(other / "mallory", bound_tree / OWNER)

    answer = posture_store.owner_context(OWNER, "live", baseline="live")

    assert answer.record is None
    assert answer.detail.verdict is StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert "symbolic link" in answer.detail.reason


def test_an_unreadable_record_refuses_without_a_record(bound_tree):
    path = bound_tree / OWNER / control_context.RECORD_FILENAME
    path.parent.mkdir()
    path.write_text("{not json", encoding="utf-8")

    answer = posture_store.owner_context(OWNER, "live", baseline="live")

    assert answer.record is None
    assert answer.detail.verdict is StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert answer.detail.reason


def test_no_tree_bound_refuses_and_never_reads_the_host_record(tmp_path, monkeypatch):
    """The host fallback exists for the verdict reader, never for this one."""
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
    posture_store.invalidate_cache()

    def _explode(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("the host record must not be read")

    monkeypatch.setattr(posture_store, "recorded_posture", _explode)

    answer = posture_store.owner_context(OWNER, "live", baseline="live")

    assert answer.record is None
    assert answer.detail.verdict is StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert posture_store.CONTROL_CONTEXT_TREE_ENV_VAR in answer.detail.reason


def test_an_unusable_bind_refuses(monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "control_target")

    answer = posture_store.owner_context(OWNER, "live", baseline="live")

    assert answer.record is None
    assert answer.detail.verdict is StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert posture_store.CONTROL_CONTEXT_TREE_ENV_VAR in answer.detail.reason


@pytest.mark.parametrize("owner", [posture_store.NO_OWNER, 7, "", "..", "a/b"])
@pytest.mark.usefixtures("bound_tree")
def test_an_owner_that_is_not_a_name_refuses(owner):
    answer = posture_store.owner_context(owner, "live", baseline="live")

    assert answer.record is None
    assert answer.detail.verdict is StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert answer.detail.reason


def test_an_unprovisioned_tree_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tmp_path / "missing"))

    answer = posture_store.owner_context(OWNER, "live", baseline="live")

    assert answer.record is None
    assert answer.detail.verdict is StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE


# --- a synthetic consumer ---------------------------------------------------


def _admit(owner, service_target: str, *, baseline: str) -> str | None:
    """A backend's admission check: the refusal reason, or ``None`` to admit.

    The shape a service receiving a proxy-minted owner takes: it refuses when
    the owner's record is unreadable or narrows its target, and when the
    record names another target than the one the service drives.
    """
    answer = posture_store.owner_context(owner, service_target, baseline=baseline)
    if answer.detail.verdict is not StoreVerdict.PERMITTED:
        return answer.detail.reason or answer.detail.verdict.value
    record_target = answer.record.target if answer.record is not None else baseline
    if record_target != service_target:
        return f"the owner's control context is on {record_target!r}, not {service_target!r}"
    return None


def test_a_consumer_admits_an_owner_on_its_own_target(bound_tree):
    _write_tree_record(bound_tree, OWNER, target="va", generation=3)

    assert _admit(OWNER, "va", baseline="live") is None


def test_a_consumer_refuses_an_owner_switched_elsewhere(bound_tree):
    _write_tree_record(bound_tree, OWNER, target="va", generation=3)

    assert "'va'" in _admit(OWNER, "live", baseline="live")


@pytest.mark.usefixtures("bound_tree")
def test_a_consumer_refuses_an_owner_with_no_record_off_the_baseline():
    assert _admit(OWNER, "va", baseline="live") is not None
    assert _admit(OWNER, "live", baseline="live") is None


def test_a_consumer_refuses_a_narrowed_owner(bound_tree):
    _write_tree_record(bound_tree, OWNER, target="live", generation=1, posture={"live": "sandbox"})

    assert _admit(OWNER, "live", baseline="live") is not None


def test_a_consumer_with_no_tree_bound_refuses(monkeypatch):
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)

    reason = _admit(OWNER, "live", baseline="live")

    assert reason is not None
    assert posture_store.CONTROL_CONTEXT_TREE_ENV_VAR in reason
