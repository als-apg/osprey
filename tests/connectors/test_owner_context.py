"""The owner ladder and the binding a queued plan's owner rides in on.

A narrowing is one person's, so every lookup needs a name as well as a target.
These tests pin the name side of that: the sentinel for work that belongs to
nobody, the four rungs :func:`~osprey_connectors.posture_store.current_owner`
walks, and the block :func:`~osprey_connectors.posture_store.bind_owner` opens
around a plan wrapper's body.

Two properties are worth more than the rest and are pinned hardest. First, the
sentinel has no string spelling that anything parses back — an account named
``"<no owner>"`` must never resolve to "no owner", because the name is what
keys the directory a verdict reads. Second, the binding is gone the moment the
block ends: a leaked owner gates the next plan's writes against the previous
person's chip, and that is the failure with no visible symptom.

Two conventions run through the file. Tests about what a *lookup* answers use
:func:`current_owner`; tests about what the *variable* holds read
``_owner_var`` directly, because an unbound variable is not the same
observation as a ladder that walked past it to the process account. And the
tests that want "no owner" to be the ladder's final answer stamp the
control-context tree, which is the deployment shape where that is true: a
lane's queueserver holds the whole tree and no chip of its own.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading

import pytest

from osprey_connectors import posture_store
from osprey_connectors.posture_store import (
    NO_OWNER,
    RESERVED_OWNER_KWARG,
    bind_owner,
    current_owner,
)


def bound_owner():
    """What the context variable itself holds, with no ladder in the way."""
    return posture_store._owner_var.get()


@pytest.fixture(autouse=True)
def clean_owner_environment(monkeypatch):
    """No stamped owner, no tree bind, and no owner left bound by a neighbour.

    Both env rungs are cleared because the suite runs in whatever environment
    the lane exports, and a deployment that stamped either one would otherwise
    decide half of these tests. The variable is reset on the way out as well as
    the way in: a test that binds an owner and then fails mid-block would
    otherwise hand its owner to the next test.
    """
    monkeypatch.delenv(posture_store.CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    posture_store._owner_var.set(NO_OWNER)
    yield
    posture_store._owner_var.set(NO_OWNER)


@pytest.fixture
def tree_bound(monkeypatch):
    """Stamp the tree bind: the shape where the ladder's last rung is nobody."""
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "/var/osprey/control")


# --- the sentinel ----------------------------------------------------------


def test_the_sentinel_prints_but_does_not_parse():
    """It is loggable in both spellings, and no string is equal to it."""
    assert str(NO_OWNER) == "<no owner>"
    assert repr(NO_OWNER) == "<no owner>"
    assert f"plan ran under {NO_OWNER}" == "plan ran under <no owner>"

    # The printable form is for a warning line, not a value anyone round-trips.
    assert NO_OWNER != "<no owner>"
    assert "<no owner>" != NO_OWNER
    assert not hasattr(posture_store, "parse_owner")


def test_the_sentinel_survives_a_logging_call(caplog):
    """The one line saying a plan ran at the ceiling must not raise."""
    logger = logging.getLogger("owner_context_test")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        logger.warning("no narrowing read for %s", current_owner(NO_OWNER))
    assert "no narrowing read for <no owner>" in caplog.text


def test_the_sentinel_is_compared_by_identity():
    """One instance; a second construction is a different object.

    Callers ask ``owner is NO_OWNER``. Nothing may make a freshly built
    instance compare equal to the module's, or a value from anywhere could
    claim to be the absence of an owner.
    """
    assert current_owner(NO_OWNER) is NO_OWNER
    other = type(NO_OWNER)()
    assert other is not NO_OWNER
    assert other != NO_OWNER


def test_the_reserved_kwarg_is_the_spelling_both_ends_agree_on():
    assert RESERVED_OWNER_KWARG == "_osprey_owner"


@pytest.mark.parametrize(
    "name",
    ["NO_OWNER", "RESERVED_OWNER_KWARG", "CONTROL_OWNER_ENV_VAR", "bind_owner", "current_owner"],
)
def test_the_owner_names_are_exported(name):
    """The bridge and the write monitor import these by name from here."""
    assert name in posture_store.__all__
    assert hasattr(posture_store, name)


# --- the ladder, rung by rung ----------------------------------------------


def test_rung_one_an_explicit_owner_wins(monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "stamped")
    with bind_owner({RESERVED_OWNER_KWARG: "bound"}):
        assert current_owner("asked") == "asked"


def test_rung_one_an_explicit_no_owner_is_naming_one(monkeypatch):
    """``NO_OWNER`` passed in is an answer; ``None`` is the request to look."""
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "stamped")
    with bind_owner({RESERVED_OWNER_KWARG: "bound"}):
        assert current_owner(NO_OWNER) is NO_OWNER
        assert current_owner(None) == "bound"


def test_rung_two_the_bound_owner_beats_the_stamp(monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "stamped")
    with bind_owner({RESERVED_OWNER_KWARG: "bound"}):
        assert current_owner() == "bound"


def test_rung_three_the_stamped_owner_is_read(monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "stamped")
    assert current_owner() == "stamped"


def test_rung_three_is_read_past_an_unbound_variable(monkeypatch, tree_bound):
    """An empty binding is not an answer — the stamp below it still is."""
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "stamped")
    with bind_owner({}):
        assert bound_owner() is NO_OWNER
        assert current_owner() == "stamped"


@pytest.mark.parametrize("stamp", ["", "   ", "\t\n"])
def test_rung_three_a_blank_stamp_is_the_unset_case(monkeypatch, tree_bound, stamp):
    """A rendered-but-empty ``environment:`` entry names nobody."""
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, stamp)
    assert current_owner() is NO_OWNER


@pytest.mark.parametrize("stamp", ["ada", "  ada  ", "svc-lane"])
def test_rung_three_only_ever_yields_a_string(monkeypatch, stamp):
    """Whatever the environment holds, this rung answers with a ``str``.

    The value becomes a path component under the control-state tree, so a rung
    that could answer with something else would push the type check down into
    every reader.
    """
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, stamp)
    resolved = current_owner()
    assert isinstance(resolved, str)
    assert resolved == stamp.strip()


def test_rung_four_the_tree_bind_means_owned_or_nothing(tree_bound, monkeypatch):
    """A container holding the whole tree holds no chip of its own."""
    monkeypatch.setattr(posture_store, "acting_identity", lambda: "queueserver")
    assert current_owner() is NO_OWNER


def test_rung_four_a_blank_tree_bind_is_not_a_bind(monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "   ")
    monkeypatch.setattr(posture_store, "acting_identity", lambda: "ada")
    assert current_owner() == "ada"


def test_rung_four_falls_through_to_the_acting_identity(monkeypatch):
    """Off a tree-holding container, the process account is the person."""
    monkeypatch.setattr(posture_store, "acting_identity", lambda: "ada")
    assert current_owner() == "ada"


def test_the_last_rung_answers_with_a_real_name_unmocked():
    """The unmocked ladder still resolves something usable as a directory."""
    resolved = current_owner()
    assert isinstance(resolved, str)
    assert resolved
    assert resolved.strip() == resolved


def test_a_higher_rung_never_consults_the_identity_ladder(monkeypatch, tree_bound):
    """The last rung is reached only when nothing above it answered.

    Not an optimisation: a tree-holding container must not read the account it
    runs as *at all*, because that account names nobody whose chip anyone set,
    and an answer from there would be a name a verdict then looks up.
    """
    calls: list[int] = []

    def _counted():
        calls.append(1)
        return "process-account"

    monkeypatch.setattr(posture_store, "acting_identity", _counted)

    assert current_owner("asked") == "asked"
    with bind_owner({RESERVED_OWNER_KWARG: "bound"}):
        assert current_owner() == "bound"
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "stamped")
    assert current_owner() == "stamped"
    monkeypatch.delenv(posture_store.CONTROL_OWNER_ENV_VAR)
    assert current_owner() is NO_OWNER

    assert calls == []


# --- the binding -----------------------------------------------------------


def test_the_owner_is_visible_in_the_block_and_gone_after(tree_bound):
    with bind_owner({RESERVED_OWNER_KWARG: "ada", "detectors": ["d1"]}):
        assert current_owner() == "ada"
    assert bound_owner() is NO_OWNER
    assert current_owner() is NO_OWNER


def test_the_block_yields_kwargs_without_the_reserved_key():
    kwargs = {RESERVED_OWNER_KWARG: "ada", "detectors": ["d1"], "num": 3}
    with bind_owner(kwargs) as clean:
        assert clean == {"detectors": ["d1"], "num": 3}
        assert RESERVED_OWNER_KWARG not in clean


def test_the_callers_mapping_is_not_mutated():
    """Two wrappers reading the same mapping must not race over the pop."""
    kwargs = {RESERVED_OWNER_KWARG: "ada", "num": 3}
    with bind_owner(kwargs) as clean:
        clean["num"] = 4
    assert kwargs == {RESERVED_OWNER_KWARG: "ada", "num": 3}


@pytest.mark.parametrize(
    "claimed",
    [{}, {RESERVED_OWNER_KWARG: None}, {RESERVED_OWNER_KWARG: ""}, {RESERVED_OWNER_KWARG: "  "}],
)
def test_a_nameless_plan_binds_the_sentinel(tree_bound, claimed):
    """No name on the item means the ceiling is all that governs the plan."""
    with bind_owner(dict(claimed)):
        assert bound_owner() is NO_OWNER
        assert current_owner() is NO_OWNER


@pytest.mark.parametrize("claimed", [17, ["ada"], {"who": "ada"}, object()])
def test_a_non_string_claim_is_not_an_owner(claimed):
    with bind_owner({RESERVED_OWNER_KWARG: claimed}):
        assert bound_owner() is NO_OWNER


def test_a_claimed_owner_is_stripped():
    with bind_owner({RESERVED_OWNER_KWARG: "  ada  "}):
        assert current_owner() == "ada"


def test_the_binding_is_released_when_the_body_raises():
    """A plan that fails must not leave its owner gating the next one."""
    with pytest.raises(RuntimeError, match="plan failed"):
        with bind_owner({RESERVED_OWNER_KWARG: "ada"}):
            assert current_owner() == "ada"
            raise RuntimeError("plan failed")
    assert bound_owner() is NO_OWNER


def test_one_runs_owner_does_not_reach_the_next_run(tree_bound):
    with bind_owner({RESERVED_OWNER_KWARG: "ada"}):
        assert current_owner() == "ada"
    with bind_owner({}):
        assert current_owner() is NO_OWNER
    assert bound_owner() is NO_OWNER


# --- the documented invariant ----------------------------------------------


async def test_the_owner_reaches_a_task_created_in_the_block():
    """A task copies the context at creation, so the async path sees it."""

    async def read_it():
        return current_owner()

    with bind_owner({RESERVED_OWNER_KWARG: "ada"}):
        assert await asyncio.create_task(read_it()) == "ada"


async def test_to_thread_copies_the_context_but_run_in_executor_does_not():
    """Exactly the split the module docstring claims — no more, no less."""
    with bind_owner({RESERVED_OWNER_KWARG: "ada"}):
        assert await asyncio.to_thread(bound_owner) == "ada"

        loop = asyncio.get_running_loop()
        assert await loop.run_in_executor(None, bound_owner) is NO_OWNER


def test_a_bare_thread_reads_the_default():
    seen: list[object] = []
    with bind_owner({RESERVED_OWNER_KWARG: "ada"}):
        worker = threading.Thread(target=lambda: seen.append(bound_owner()))
        worker.start()
        worker.join()
    assert seen == [NO_OWNER]


def test_a_copied_context_carries_the_owner_across_a_hand_copy():
    """``copy_context().run`` is the hand copy the docstring points at."""
    with bind_owner({RESERVED_OWNER_KWARG: "ada"}):
        context = contextvars.copy_context()
    assert context.run(bound_owner) == "ada"
