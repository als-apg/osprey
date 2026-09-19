"""The four copies of the acting-identity ladder, pinned to one another.

The ladder is written out four times on purpose. ``osprey_connectors.identity``
holds it; ``osprey.utils.identity`` re-exports it under the historical path; and
the two stdlib hooks — ``osprey_target_state`` and ``osprey_hook_log`` — restate
it, because a hook runs outside the osprey venv where neither package exists.
Restating is the only option available to them, so the cost of restating has to
be paid somewhere, and it is paid here.

What drift would cost is worth being concrete about. The identity chooses a
directory: ``var/audit/<identity>/`` for the ledger, and the per-identity
control-state directory for the narrowing a write is checked against. A hook
that resolved one rung differently from the writer would read a narrowing
somebody else published, or read none where one applies, and would do it
silently — nothing crashes when two copies disagree, the answers simply stop
being about the same person. That is only ever visible at a write, long after
the edit that caused it.

So this module asserts agreement across the whole environment matrix rather
than sampling it: 16 values for each env rung (unset, blank, whitespace-padded,
ordinary, and each way a value can fail to be one path component) against 10
outcomes for the local-account rung (ordinary names, blank ones, a traversal,
and the three ways :func:`getpass.getuser` fails) — 2560 rows, every one of
which all four copies must answer alike.

Agreement alone would be satisfied by four copies that are identically wrong,
so two other things anchor it. The package ladder is the reference every other
copy is compared against, and its semantics are pinned independently in
``tests/utils/test_identity.py``. And the rows that name a deployment shape —
a terminal user, a service container, a laptop, and a sandbox child — carry
their own expected answers below, stated rather than derived.

The sandbox row is the one that motivated the rule the ladder's docstring
states: ``OSPREY_AUDIT_IDENTITY`` may never join a scrub list. An executor
sandbox is severed from the ``OSPREY_TERMINAL_`` family by design, and inside a
container the process account names ``osprey`` or ``root``, so rung 2 is the
only rung left that names the person. This module builds a real child
environment with the real scrubber and checks that all four copies still say
``alice``.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest

from osprey.mcp_server.sandbox_env import scrub_sandbox_child_env
from osprey.utils import identity as re_export
from osprey_connectors import identity as ladder

#: The shipped hooks, as files. They are loaded from here by path rather than
#: imported as ``osprey.templates...`` so that the ``sys.path`` entry
#: ``osprey_target_state`` inserts for its own siblings is this module's to undo
#: — an entry left behind lets any later ``import osprey_hook_log`` anywhere in
#: the worker resolve to the template copy.
HOOKS_DIR = Path(__file__).resolve().parents[2] / "src/osprey/templates/claude_code/claude/hooks"

#: The hooks that carry their own restatement of the ladder.
HOOK_MODULES: tuple[str, ...] = ("osprey_target_state", "osprey_hook_log")

#: One axis of the matrix, used for both env rungs. Four kinds of value, because
#: each is a different way a deployment renders one: absent; rendered empty (an
#: uninterpolated compose variable arrives as ``""``); a name a real account or
#: service key would carry; and a value that is not one path component, which a
#: rung must decline rather than turn into a directory somewhere else.
ENV_VALUES: tuple[str | None, ...] = (
    None,
    "",
    "   ",
    "alice",
    "  alice\n",
    "channel_finder",
    "sidecar",
    "alice.smith",
    "svc-web-terminal",
    "user@example.org",
    ".",
    "..",
    "../elsewhere",
    "a/b",
    "/absolute",
    "back\\slash",
)

#: The other axis: what the local-account rung does when it is reached. The
#: three exceptions are the real failure modes — ``KeyError`` on Python 3.12 and
#: earlier, ``OSError`` on 3.13+, both ordinary for a uid with no passwd entry in
#: a slim image, and anything else at all.
ACCOUNT_OUTCOMES: tuple[tuple[str, object], ...] = (
    ("named", "pinned-local-account"),
    ("root", "root"),
    ("osprey", "osprey"),
    ("padded", "  spaced-account  "),
    ("empty", ""),
    ("whitespace", "   "),
    ("traversal", ".."),
    ("keyerror", KeyError("uid")),
    ("oserror", OSError("no passwd entry")),
    ("other-failure", RuntimeError("something else entirely")),
)

#: The row count the matrix below is expected to walk. Spelled out so that
#: thinning an axis has to be a deliberate edit to this number rather than a
#: quietly smaller sweep.
MATRIX_ROWS = 2560


def _load_hook(name: str):
    """Execute the shipped hook file and hand back the module object.

    By path, with a private module name: the hooks are shipped template files
    rather than library modules, and loading one under its own name would make
    this module's copy the one a later sibling import in the same worker binds.
    """
    spec = importlib.util.spec_from_file_location(
        f"_identity_drift_{name}", HOOKS_DIR / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hooks() -> dict:
    """The two shipped hooks as module objects, keyed by hook name.

    A hook is deliberately absent from ``sys.modules`` once loaded — see
    :func:`_load_hook` — so anything wanting one asks for it here rather than
    looking it up by name.

    The two loads are undone afterwards: ``osprey_target_state`` puts its own
    directory on ``sys.path`` and imports ``osprey_hook_log`` as a sibling, and
    both of those outlive the load unless they are taken back out.
    """
    path_before = list(sys.path)
    sibling_before = sys.modules.get("osprey_hook_log")
    try:
        yield {name: _load_hook(name) for name in HOOK_MODULES}
    finally:
        sys.path[:] = path_before
        if sibling_before is None:
            sys.modules.pop("osprey_hook_log", None)
        else:
            sys.modules["osprey_hook_log"] = sibling_before


@pytest.fixture(scope="module")
def resolvers(hooks: dict) -> dict:
    """The four ``acting_identity`` callables, keyed by where each one lives."""
    return {
        "osprey_connectors.identity": ladder.acting_identity,
        "osprey.utils.identity": re_export.acting_identity,
        **{name: module.acting_identity for name, module in hooks.items()},
    }


def pin_account(monkeypatch: pytest.MonkeyPatch, outcome: object) -> None:
    """Make the local-account rung answer *outcome*, or raise it.

    Pinned in every test, including the ones that never reach this rung:
    unpinned, :func:`getpass.getuser` reads the real environment, so an
    assertion about the env rungs would otherwise depend on who ran pytest.
    """
    if isinstance(outcome, BaseException):

        def _raise() -> str:
            raise outcome

        monkeypatch.setattr("getpass.getuser", _raise)
    else:
        monkeypatch.setattr("getpass.getuser", lambda: outcome)


def set_rung(monkeypatch: pytest.MonkeyPatch, name: str, value: str | None) -> None:
    """Set env var *name* to *value*, or unset it when *value* is ``None``."""
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


def answers(resolvers: dict) -> dict[str, str]:
    """What every copy resolves for the environment as it stands right now."""
    return {where: resolve() for where, resolve in resolvers.items()}


class TestTheMatrixItself:
    """The sweep's own shape, so that a thinner sweep cannot pass unnoticed."""

    def test_axes_carry_no_duplicate_values(self) -> None:
        """A repeated value buys no coverage while inflating the row count."""
        assert len(set(ENV_VALUES)) == len(ENV_VALUES)
        assert len({name for name, _ in ACCOUNT_OUTCOMES}) == len(ACCOUNT_OUTCOMES)

    def test_the_sweep_is_the_full_product_of_the_axes(self) -> None:
        """Both rungs take every value, against every account outcome."""
        assert len(ENV_VALUES) ** 2 * len(ACCOUNT_OUTCOMES) == MATRIX_ROWS


class TestOneImplementation:
    """Two of the four copies are meant to be the same object, not a likeness."""

    def test_the_re_export_is_the_ladder_itself(self) -> None:
        """``osprey.utils.identity`` is an import path, not a second ladder."""
        assert re_export.acting_identity is ladder.acting_identity

    @pytest.mark.parametrize(
        "constant",
        ["TERMINAL_USER_ENV", "AUDIT_IDENTITY_ENV", "IDENTITY_ENV_LADDER", "UNKNOWN_IDENTITY"],
    )
    def test_every_copy_declares_the_same_constant(self, constant: str, hooks: dict) -> None:
        """The names are a deployment contract; a copy that spells one
        differently resolves a different rung from the same environment.

        A copy that omits the constant altogether reports as ``None`` here
        rather than as an ``AttributeError``, so the hook that dropped it is
        named by the assertion instead of by a traceback.
        """
        expected = getattr(ladder, constant)
        for name, module in hooks.items():
            assert getattr(module, constant, None) == expected, name
        assert getattr(re_export, constant) == expected


class TestEveryCopyAgrees:
    """The sweep: 2560 environments, one answer each."""

    @pytest.mark.parametrize(
        "outcome", [pytest.param(value, id=name) for name, value in ACCOUNT_OUTCOMES]
    )
    def test_copies_agree_on_every_env_row(
        self, outcome: object, resolvers: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every disagreement in the sweep is reported, not just the first.

        A drift that shows up on one rung usually shows up on a family of rows,
        and the family is what says which rung moved.
        """
        pin_account(monkeypatch, outcome)
        reference = "osprey_connectors.identity"

        disagreements = []
        for terminal, audit in itertools.product(ENV_VALUES, repeat=2):
            set_rung(monkeypatch, ladder.TERMINAL_USER_ENV, terminal)
            set_rung(monkeypatch, ladder.AUDIT_IDENTITY_ENV, audit)

            resolved = answers(resolvers)
            if len(set(resolved.values())) > 1:
                disagreements.append((terminal, audit, resolved[reference], resolved))

        assert not disagreements, disagreements


class TestTheRowsThatNameADeployment:
    """Four shapes, with the answer stated rather than derived from the ladder."""

    @pytest.fixture(autouse=True)
    def _container_account(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The process account inside a container names nobody — so a copy that
        fell through to it would answer ``osprey``, visibly wrong here."""
        pin_account(monkeypatch, "osprey")

    def test_a_terminal_user_names_the_person(
        self, resolvers: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``web-<user>`` container: a real person is behind it."""
        set_rung(monkeypatch, ladder.TERMINAL_USER_ENV, "alice")
        set_rung(monkeypatch, ladder.AUDIT_IDENTITY_ENV, "bluesky-web")

        assert answers(resolvers) == dict.fromkeys(resolvers, "alice")

    def test_a_service_container_names_its_own_key(
        self, resolvers: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A container hosting no single user files under its service key."""
        set_rung(monkeypatch, ladder.TERMINAL_USER_ENV, None)
        set_rung(monkeypatch, ladder.AUDIT_IDENTITY_ENV, "sidecar")

        assert answers(resolvers) == dict.fromkeys(resolvers, "sidecar")

    def test_a_laptop_names_the_local_account(
        self, resolvers: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The single-user case, where the process account is the person."""
        pin_account(monkeypatch, "carol")
        set_rung(monkeypatch, ladder.TERMINAL_USER_ENV, None)
        set_rung(monkeypatch, ladder.AUDIT_IDENTITY_ENV, None)

        assert answers(resolvers) == dict.fromkeys(resolvers, "carol")

    def test_a_scrubbed_sandbox_child_still_names_the_person(
        self, resolvers: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row the never-scrub rule exists for.

        The child environment is built by the real scrubber from a real
        ``web-alice`` parent, so the assertion is about what the sandbox
        actually hands its child rather than about a hand-written copy of it.
        """
        parent = {
            "PATH": "/usr/local/bin:/usr/bin",
            ladder.TERMINAL_USER_ENV: "alice",
            ladder.AUDIT_IDENTITY_ENV: "alice",
            "OSPREY_WEB_PORT": "8080",
            "OSPREY_TERMINAL_BIND_HOST": "127.0.0.1",
        }

        child = scrub_sandbox_child_env(parent)

        # The severing is real, or the row proves nothing: the whole
        # ``OSPREY_TERMINAL_`` prefix goes, and so does the web port.
        assert ladder.TERMINAL_USER_ENV not in child
        assert "OSPREY_TERMINAL_BIND_HOST" not in child
        assert "OSPREY_WEB_PORT" not in child
        # Negative control: an environment emptied wholesale would satisfy every
        # absence above while saying nothing about what survives.
        assert child["PATH"] == parent["PATH"]
        assert child[ladder.AUDIT_IDENTITY_ENV] == "alice"

        for name in ladder.IDENTITY_ENV_LADDER:
            set_rung(monkeypatch, name, child.get(name))

        assert answers(resolvers) == dict.fromkeys(resolvers, "alice")
