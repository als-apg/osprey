"""The plan store: one current plan per owner, and the contract's status transitions."""

import threading

import pytest
from orbit_bump.plans import (
    PlanConflict,
    PlanNotCurrent,
    PlanNotFound,
    PlanStore,
)


def _plan(model="ring", status="ready", **extra):
    return {"model": model, "plane": "x", "orm_version": 1, "status": status, **extra}


@pytest.fixture
def store():
    return PlanStore()


def test_set_current_assigns_a_unique_opaque_id(store):
    a = store.set_current("alice", _plan())
    b = store.set_current("bob", _plan())
    assert isinstance(a["id"], str) and a["id"]
    assert a["id"] != b["id"]
    assert a["owner"] == "alice"
    assert store.current("alice")["id"] == a["id"]


def test_ids_are_unique_across_stores():
    assert (
        PlanStore().set_current("o", _plan())["id"] != PlanStore().set_current("o", _plan())["id"]
    )


def test_returned_plans_are_copies(store):
    plan = store.set_current("alice", _plan())
    plan["status"] = "applied"
    store.current("alice")["status"] = "applied"
    assert store.current("alice")["status"] == "ready"


def test_set_current_refuses_a_new_plan_in_a_non_initial_status(store):
    with pytest.raises(ValueError, match="applied"):
        store.set_current("alice", _plan(status="applied"))


def test_set_current_supersedes_the_previous_ready_plan(store):
    old = store.set_current("alice", _plan())
    new = store.set_current("alice", _plan())
    assert store.current("alice")["id"] == new["id"]
    assert store.get(old["id"])["status"] == "superseded"


def test_set_current_supersedes_a_not_applicable_plan(store):
    old = store.set_current("alice", _plan(status="not_applicable"))
    store.set_current("alice", _plan())
    assert store.get(old["id"])["status"] == "superseded"


def test_set_current_refused_while_the_previous_plan_is_applying(store):
    old = store.set_current("alice", _plan())
    store.claim(old["id"], "alice")
    with pytest.raises(PlanConflict, match="applying"):
        store.set_current("alice", _plan())
    assert store.current("alice")["id"] == old["id"]
    assert store.current("alice")["status"] == "applying"


def test_a_new_plan_after_an_applied_one(store):
    old = store.set_current("alice", _plan())
    store.claim(old["id"], "alice")
    store.transition(old["id"], "applied", applied_at="t")
    new = store.set_current("alice", _plan())
    assert store.current("alice")["id"] == new["id"]
    superseded = store.get(old["id"])
    assert superseded["status"] == "superseded"
    assert superseded["applied_at"] == "t"


def test_set_current_leaves_other_owners_alone(store):
    a = store.set_current("alice", _plan())
    store.set_current("bob", _plan())
    assert store.current("alice")["id"] == a["id"]
    assert store.current("alice")["status"] == "ready"


def test_current_is_none_without_a_plan(store):
    assert store.current("nobody") is None


def test_discard_then_current_is_none(store):
    plan = store.set_current("alice", _plan())
    discarded = store.discard("alice")
    assert discarded["id"] == plan["id"]
    assert store.current("alice") is None
    assert store.get(plan["id"])["status"] == "discarded"


def test_discard_without_a_plan_is_none(store):
    assert store.discard("alice") is None


def test_discard_refused_while_applying(store):
    plan = store.set_current("alice", _plan())
    store.claim(plan["id"], "alice")
    with pytest.raises(PlanConflict, match="applying"):
        store.discard("alice")
    assert store.current("alice")["status"] == "applying"


def test_discard_after_partially_applied_then_current_is_none(store):
    plan = store.set_current("alice", _plan())
    store.claim(plan["id"], "alice")
    store.transition(plan["id"], "partially_applied")
    store.discard("alice")
    assert store.current("alice") is None
    assert store.get(plan["id"])["status"] == "discarded"


def test_discard_for_orm_change_discards_ready_and_not_applicable_on_that_model(store):
    a = store.set_current("alice", _plan(model="ring"))
    b = store.set_current("bob", _plan(model="ring", status="not_applicable"))
    c = store.set_current("carol", _plan(model="other"))
    discarded = store.discard_for_orm_change("ring")
    assert sorted(discarded) == sorted([a["id"], b["id"]])
    assert store.current("alice") is None
    assert store.current("bob") is None
    assert store.current("carol")["id"] == c["id"]
    assert store.get(a["id"])["status"] == "discarded"


def test_discard_for_orm_change_leaves_terminal_and_applying_plans(store):
    done = store.set_current("alice", _plan())
    store.claim(done["id"], "alice")
    store.transition(done["id"], "applied")
    busy = store.set_current("bob", _plan())
    store.claim(busy["id"], "bob")
    assert store.discard_for_orm_change("ring") == []
    assert store.current("alice")["status"] == "applied"
    assert store.current("bob")["status"] == "applying"


def test_claim_moves_ready_to_applying_writing(store):
    plan = store.set_current("alice", _plan())
    claimed = store.claim(plan["id"], "alice")
    assert claimed["status"] == "applying"
    assert claimed["phase"] == "writing"


def test_claim_refuses_a_non_current_id_naming_the_current(store):
    old = store.set_current("alice", _plan())
    new = store.set_current("alice", _plan())
    with pytest.raises(PlanNotCurrent) as err:
        store.claim(old["id"], "alice")
    assert err.value.current_id == new["id"]
    assert new["id"] in str(err.value)


def test_claim_refuses_another_owners_plan(store):
    plan = store.set_current("alice", _plan())
    with pytest.raises(PlanNotCurrent) as err:
        store.claim(plan["id"], "bob")
    assert err.value.current_id is None


def test_claim_refuses_an_unknown_id(store):
    store.set_current("alice", _plan())
    with pytest.raises(PlanNotCurrent):
        store.claim("no-such-id", "alice")


@pytest.mark.parametrize("status", ["not_applicable"])
def test_claim_refuses_a_plan_that_is_not_ready(store, status):
    plan = store.set_current("alice", _plan(status=status))
    with pytest.raises(PlanConflict, match=status):
        store.claim(plan["id"], "alice")


def test_claim_refuses_an_applying_or_applied_plan(store):
    plan = store.set_current("alice", _plan())
    store.claim(plan["id"], "alice")
    with pytest.raises(PlanConflict, match="applying"):
        store.claim(plan["id"], "alice")
    store.transition(plan["id"], "applied")
    with pytest.raises(PlanConflict, match="applied"):
        store.claim(plan["id"], "alice")


def test_release_moves_applying_back_to_ready(store):
    plan = store.set_current("alice", _plan())
    store.claim(plan["id"], "alice")
    released = store.release(plan["id"])
    assert released["status"] == "ready"
    assert "phase" not in released
    assert store.claim(plan["id"], "alice")["status"] == "applying"


def test_release_refuses_a_plan_that_is_not_applying(store):
    plan = store.set_current("alice", _plan())
    with pytest.raises(PlanConflict, match="ready"):
        store.release(plan["id"])


def test_release_refuses_an_unknown_id(store):
    with pytest.raises(PlanNotFound):
        store.release("nope")


def test_two_concurrent_claims_exactly_one_wins(store):
    plan = store.set_current("alice", _plan())
    barrier = threading.Barrier(8)
    wins, losses = [], []

    def go():
        barrier.wait()
        try:
            store.claim(plan["id"], "alice")
            wins.append(1)
        except PlanConflict:
            losses.append(1)

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1
    assert len(losses) == 7


# --- transition(): the contract's table ---------------------------------------


@pytest.mark.parametrize("to", ["superseded", "discarded"])
def test_ready_to_bookkeeping(store, to):
    plan = store.set_current("alice", _plan())
    assert store.transition(plan["id"], to)["status"] == to
    assert store.current("alice") is None


@pytest.mark.parametrize("to", ["ready", "applied", "partially_applied", "discarded"])
def test_applying_to(store, to):
    plan = store.set_current("alice", _plan())
    store.claim(plan["id"], "alice")
    after = store.transition(plan["id"], to, correctors=[{"outcome": "confirmed"}])
    assert after["status"] == to
    assert after["correctors"] == [{"outcome": "confirmed"}]
    assert "phase" not in after


def test_applying_phase_update_keeps_status(store):
    plan = store.set_current("alice", _plan())
    store.claim(plan["id"], "alice")
    after = store.transition(plan["id"], "applying", phase="settling")
    assert after["status"] == "applying"
    assert after["phase"] == "settling"


@pytest.mark.parametrize("start", ["not_applicable", "applied", "partially_applied"])
@pytest.mark.parametrize("to", ["superseded", "discarded"])
def test_bookkeeping_from_terminal_or_not_applicable(store, start, to):
    plan = store.set_current("alice", _plan(status="ready" if start != "not_applicable" else start))
    if start != "not_applicable":
        store.claim(plan["id"], "alice")
        store.transition(plan["id"], start)
    assert store.transition(plan["id"], to)["status"] == to


@pytest.mark.parametrize("start", ["not_applicable", "applied", "partially_applied"])
@pytest.mark.parametrize("to", ["applying", "ready"])
def test_never_back_to_applying_or_ready(store, start, to):
    plan = store.set_current("alice", _plan(status="ready" if start != "not_applicable" else start))
    if start != "not_applicable":
        store.claim(plan["id"], "alice")
        store.transition(plan["id"], start)
    with pytest.raises(PlanConflict):
        store.transition(plan["id"], to)


@pytest.mark.parametrize("start", ["superseded", "discarded"])
@pytest.mark.parametrize(
    "to", ["ready", "applying", "applied", "partially_applied", "superseded", "discarded"]
)
def test_superseded_and_discarded_are_terminal(store, start, to):
    plan = store.set_current("alice", _plan())
    store.transition(plan["id"], start)
    with pytest.raises(PlanConflict):
        store.transition(plan["id"], to)


def test_ready_cannot_jump_to_applied(store):
    plan = store.set_current("alice", _plan())
    with pytest.raises(PlanConflict):
        store.transition(plan["id"], "applied")


def test_transition_refuses_an_unknown_status(store):
    plan = store.set_current("alice", _plan())
    with pytest.raises(ValueError, match="bogus"):
        store.transition(plan["id"], "bogus")


def test_transition_refuses_identity_fields(store):
    plan = store.set_current("alice", _plan())
    store.claim(plan["id"], "alice")
    with pytest.raises(ValueError, match="owner"):
        store.transition(plan["id"], "applied", owner="mallory")
    assert store.get(plan["id"])["status"] == "applying"


def test_transition_unknown_id(store):
    with pytest.raises(PlanNotFound):
        store.transition("nope", "discarded")


def test_get_unknown_id(store):
    with pytest.raises(PlanNotFound):
        store.get("nope")
