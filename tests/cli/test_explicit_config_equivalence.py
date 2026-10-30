"""Requirement 1: the live render still equals the frozen baseline render.

The explicit-profile-config work moves the declarative defaults out of
``src/osprey/templates/apps/*/config.yml.j2`` and into the presets, and rewrites
the framework template to carry only derived keys. Nothing about the *rendered*
``build/config.yml`` is supposed to change. ``tests/fixtures/explicit_config``
holds a freeze of the render taken by ``freeze.py`` against the live tree (not
against a pre-conversion baseline); this module re-runs the same ``osprey init``
+ ``osprey build`` against the **live** tree and asserts the two agree, document
by document, leaf by leaf. What it guarantees is therefore "the render has not
moved since the last freeze": a change that moves it declares the moved leaves
below or re-freezes the fixtures.

What "the same" means
---------------------
A **cell** is one ``(preset, channel_finder_mode)`` pair, listed in
``cells.json``. Per cell the CLI is invoked exactly as the freeze invoked it —
same verbs, same ``--set channel_finder_mode=``, same fixed project name, same
``*_API_KEY``-stripped environment — but with no baseline export and no
``PYTHONPATH`` shadowing, so it renders through whatever is checked out. The
captured documents are masked by :func:`freeze.mask_document`, which is imported
rather than re-implemented: the fixture and the assertion cannot drift into
masking different things.

Allowed differences: the delta table
------------------------------------
:data:`CELL_DELTAS` is the *complete* list of leaves a cell is allowed to differ
from its fixture by, save the host-resolved ones below. It is exact in both
directions. A difference that is not
declared fails, and a declared difference that is not observed fails too, so a
delta cannot outlive the change that justified it. A change that moves the
render either declares its leaves here or re-freezes the fixtures with
``freeze.py``; the three-way write switch declares, so the frozen baseline
render stays the proof of render-identity. A delta is spelt
like this::

    "hello-world/unset": (
        Delta(document="root", path="hooks.debug", fixture=ABSENT, live=False),
    ),

``execution.environment.*`` can never surface as a delta: the freeze strips that
whole mapping (it resolves to an interpreter path).

The staged login map is not a delta either. ``control_system.login_tiers`` is
staged into the shared containers' ``build/services/*`` copies only, never into
a root, tier or app render, so the completeness check compares those copies
with it removed and the map is frozen on its own at
``<cell>/staged/login_tiers.yml``. :func:`test_staged_login_map_equals_frozen_map`
compares it exactly — same presence, same logins, same tier per login — save
the logins :data:`STAGED_LOGIN_DELTAS` declares, exact in both directions. Both
sides build as the pinned account (``freeze.FROZEN_BUILD_ACCOUNT``), which is
how the build account's own entry can be compared at all.

One leaf is exempt from the table rather than declared in it. ``osprey build``
answers the presets' ``container_runtime: auto`` with the runtime that served
the build, so the rendered value states a fact about the building host: it is
``docker`` where these fixtures were frozen and ``auto`` on a host with no
working runtime. :data:`HOST_RESOLVED_LEAVES` compares that leaf for presence
and for being one of the answers the build can give, never for which one. A
table entry could not do the job, because a delta that covered a runtime-less
host would go stale on every host that resolves one.

A delta may also carry ``pending=<task>``, which makes it a prediction rather
than an allowance: a difference an upcoming task is expected to introduce. It is
not asserted and does not go stale while it is unobserved, but the moment the
render does produce it the module fails and says to drop the marker — so the
prediction is checked against reality exactly once, deliberately, instead of
quietly becoming an allowance nobody re-read.

Running it
----------
The ``slow`` marker is registered in ``pyproject.toml`` and needs no opt-in flag:
these tests run in an ordinary suite run and are deselected with ``-m 'not
slow'``. On their own::

    uv run --extra dev pytest tests/cli/test_explicit_config_equivalence.py -q

A full run is minutes, not seconds: each cell is a real ``osprey build``, and a
control-assistant cell derives a large virtual-accelerator channel set. Each cell
is therefore built once per module and shared between the tests that read it.
Under the repo's ``--dist loadgroup`` scheduler an unmarked file goes to one
worker whole (``tests/README.md``), so that cache holds in the parallel lane too
and no ``xdist_group`` is needed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import yaml

from tests.fixtures.explicit_config.freeze import (
    FIXTURE_ROOT,
    LOGIN_TIERS_FILE,
    PROJECT_NAME,
    STAGED_DIR,
    UNSET_MODE_DIR,
    _cli_env,
    _failure_reason,
    _run_cli,
    collect_login_tiers,
    collect_rendered_configs,
    mask_document,
    verify_capture_is_complete,
)

pytestmark = pytest.mark.slow


# ─────────────────────────────────────────────────────────────────────────────
# Absence, as a value
# ─────────────────────────────────────────────────────────────────────────────


class _Absent:
    """A leaf that one side of a comparison does not carry at all."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<absent>"


#: Sentinel for "this document has no such leaf". Distinguishes a key that
#: gained a ``null`` from a key that appeared or vanished.
ABSENT = _Absent()


# ─────────────────────────────────────────────────────────────────────────────
# The delta table
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Delta:
    """One leaf a cell is allowed to differ from its fixture by.

    Spelled either as an exact pair of values (*fixture* and *live*, each of
    which may be :data:`ABSENT`) or, for a list leaf that only grows, as
    *added*: the members the live render appends, the assertion being that the
    live list is the fixture's members plus those, sorted.

    *pending* names a task that has not landed yet. A pending delta is a
    prediction rather than an allowance: it is not asserted, and it is not stale
    while it goes unobserved. When it *is* observed the module fails and says to
    drop the marker, so the difference becomes exact from that moment on and
    cannot arrive unnoticed in the meantime.

    Attributes:
        document: Render the leaf belongs to — ``"root"`` or a render name.
        path: Dotted path of the leaf inside that document.
        fixture: Value the frozen document carries.
        live: Value the live render carries.
        added: List members the live render gains, instead of *fixture*/*live*.
        pending: Task this difference waits on, or ``None`` when it is expected
            to be observed already.
    """

    document: str
    path: str
    fixture: Any = ABSENT
    live: Any = ABSENT
    added: tuple[Any, ...] | None = None
    pending: str | None = None

    def __post_init__(self) -> None:
        exact = self.fixture is not ABSENT or self.live is not ABSENT
        if self.added is not None and exact:
            raise ValueError(
                f"{self.document}/{self.path}: give added= or fixture=/live=, not both"
            )
        if self.added is None and not exact:
            raise ValueError(f"{self.document}/{self.path}: a delta must say what differs")

    @property
    def key(self) -> tuple[str, str]:
        """The ``(document, path)`` this delta claims."""
        return (self.document, self.path)

    def explain_mismatch(self, fixture_value: Any, live_value: Any) -> str | None:
        """Why an observed difference is not the one this delta declares.

        Args:
            fixture_value: What the frozen document carries at this leaf.
            live_value: What the live render carries at this leaf.

        Returns:
            A one-line reason, or ``None`` when the observation is exactly what
            the delta declares.
        """
        if self.added is not None:
            if not isinstance(fixture_value, list) or not isinstance(live_value, list):
                return (
                    f"{self.document}/{self.path}: declared as list gains, but the leaf is "
                    f"{_render_value(fixture_value)} -> {_render_value(live_value)}"
                )
            expected = sorted([*fixture_value, *self.added])
            if live_value != expected:
                return (
                    f"{self.document}/{self.path}: declared to gain {list(self.added)}, "
                    f"but the live list is {_render_value(live_value)}"
                )
            return None
        if fixture_value != self.fixture or live_value != self.live:
            return (
                f"{self.document}/{self.path}: declared {_render_value(self.fixture)} -> "
                f"{_render_value(self.live)}, observed {_render_value(fixture_value)} -> "
                f"{_render_value(live_value)}"
            )
        return None


#: The leaves the three-way write switch moves in every render, root, tier or
#: app: the governed tools' per-tool approval policies retire, so the presets no
#: longer spell ``approval.tools.channel_write: always`` or
#: ``approval.tools.execute: selective`` and the position decides instead.
def _retired_governed_policies(document: str) -> tuple[Delta, ...]:
    return (
        Delta(document=document, path="approval.tools.channel_write", fixture="always"),
        Delta(document=document, path="approval.tools.execute", fixture="selective"),
    )


#: The control-assistant renders the feature moves: its root and its
#: ``readwrite`` tier open at ``asked`` instead of ``allowed``. ``admin``,
#: ``readonly`` and the apps keep their writes posture.
_ASKED_RENDERS = ("root", "readwrite")

#: Every render a control-assistant cell freezes.
_CONTROL_ASSISTANT_RENDERS = (
    "root",
    "admin",
    "channel-finder",
    "knowledge",
    "logbook",
    "readonly",
    "readwrite",
)


def _control_assistant_deltas() -> tuple[Delta, ...]:
    deltas: list[Delta] = []
    for document in _CONTROL_ASSISTANT_RENDERS:
        deltas.extend(_retired_governed_policies(document))
        if document in _ASKED_RENDERS:
            deltas.append(
                Delta(
                    document=document,
                    path="control_system.writes",
                    fixture="allowed",
                    live="asked",
                )
            )
    return tuple(deltas)


#: Every leaf each cell is allowed to differ from its frozen fixture by, keyed by
#: the cell directory as ``cells.json`` spells it. A cell absent from this
#: mapping must equal its fixture exactly. See the module docstring for how to
#: spell a delta. The fixtures stay frozen at the baseline: the render-identity
#: proof is the baseline render plus exactly these leaves.
CELL_DELTAS: dict[str, tuple[Delta, ...]] = {
    "hello-world/unset": _retired_governed_policies("root"),
    **{
        f"control-assistant/{mode}": _control_assistant_deltas()
        for mode in ("in_context", "hierarchical", "middle_layer", "graph")
    },
}

#: Logins whose staged tier posture each cell is allowed to differ from its
#: frozen ``staged/login_tiers.yml`` by, as ``login -> (fixture, live)`` — the
#: login-map twin of :data:`CELL_DELTAS`, exact in both directions. A login that
#: lands on the deployment's default or on ``readwrite`` follows that render's
#: ``control_system.writes`` from ``allowed`` to ``asked``; one granted another
#: tier (``admin``'s ``allowed``, ``readonly``'s ``never``) does not move.
STAGED_LOGIN_DELTAS: dict[str, dict[str, tuple[Any, Any]]] = {
    f"control-assistant/{mode}": {
        "alice": ("allowed", "asked"),
        "osprey-freeze-builder": ("allowed", "asked"),
    }
    for mode in ("in_context", "hierarchical", "middle_layer", "graph")
}

#: Build-relative ``config.yml`` documents a cell's build gains over the count
#: its freeze recorded, keyed by cell directory — the completeness twin of
#: :data:`CELL_DELTAS`. Empty while the fixtures match the live render.
BUILD_CONFIG_GAINS: dict[str, tuple[str, ...]] = {}


#: Substrings a refused cell's live refusal must still contain, keyed by cell
#: directory. The full baseline text is in ``cells.json``, but it names the app
#: template, which this feature removes; what must survive is *why* the mode was
#: refused. A cell absent from this mapping is only asserted to be refused at the
#: same stage.
REFUSAL_TOKENS: dict[str, tuple[str, ...]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# The cells
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cell:
    """One ``(preset, channel_finder_mode)`` pair, as ``cells.json`` records it.

    Attributes:
        preset: Preset name as ``osprey init --preset`` spells it.
        mode: ``channel_finder_mode`` to set, or ``None`` to set none.
        directory: Fixture directory, ``<preset>/<mode or unset>``.
        status: ``frozen`` or ``refused`` at the baseline.
        reason: The baseline refusal text, for a refused cell.
        stage: Which verb refused, for a refused cell.
    """

    preset: str
    mode: str | None
    directory: str
    status: str
    reason: str | None = None
    stage: str | None = None


def _load_cells() -> list[Cell]:
    """Every cell the freeze recorded.

    Returns:
        The cells in ``cells.json`` order.
    """
    records = json.loads((FIXTURE_ROOT / "cells.json").read_text(encoding="utf-8"))
    return [
        Cell(
            preset=record["preset"],
            mode=record["mode"],
            directory=record["directory"],
            status=record["status"],
            reason=record.get("reason"),
            stage=record.get("stage"),
        )
        for record in records
    ]


CELLS = _load_cells()
FROZEN_CELLS = [cell for cell in CELLS if cell.status == "frozen"]
REFUSED_CELLS = [cell for cell in CELLS if cell.status == "refused"]
META = json.loads((FIXTURE_ROOT / "meta.json").read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Rendering a cell against the live tree
# ─────────────────────────────────────────────────────────────────────────────


class Rendered(NamedTuple):
    """What one live build produced.

    Attributes:
        documents: Masked documents, keyed by render name (``root`` + renders).
        build_configs_checked: Build-relative paths of every ``config.yml`` in
            the build tree, each proved equal to one of *documents*.
        login_tiers: The login map staged into the shared containers, or
            ``None`` when the build stages none.
    """

    documents: dict[str, dict[str, Any]]
    build_configs_checked: list[str]
    login_tiers: dict[str, Any] | None


class Refused(NamedTuple):
    """A cell the live CLI would not produce.

    Attributes:
        stage: The verb that failed, ``init`` or ``build``.
        reason: The tail of what the CLI said.
    """

    stage: str
    reason: str


class CellRenderer:
    """Builds each cell at most once per module and hands out the result.

    A cell is a real ``osprey init`` + ``osprey build``; a control-assistant cell
    costs minutes. Several tests read the same cell, so the outcome — rendered or
    refused — is cached under the cell directory.
    """

    def __init__(self, root: Path) -> None:
        """Args:
        root: An existing directory to build the cells under.
        """
        self._root = root
        self._outcomes: dict[str, Rendered | Refused] = {}

    def attempt(self, cell: Cell) -> Rendered | Refused:
        """Build *cell*, or return why the CLI refused it.

        Args:
            cell: The cell to build.

        Returns:
            The cached :class:`Rendered` or :class:`Refused` for this cell.
        """
        if cell.directory not in self._outcomes:
            self._outcomes[cell.directory] = self._build(cell)
        return self._outcomes[cell.directory]

    def render(self, cell: Cell) -> Rendered:
        """Build *cell*, failing the test if the CLI refuses it.

        Args:
            cell: The cell to build.

        Returns:
            The cell's rendered documents.
        """
        outcome = self.attempt(cell)
        if isinstance(outcome, Refused):
            pytest.fail(
                f"{cell.directory}: osprey {outcome.stage} refused a cell the baseline "
                f"renders.\n{outcome.reason}"
            )
        return outcome

    def _build(self, cell: Cell) -> Rendered | Refused:
        """Run the CLI for one cell against the live tree.

        Args:
            cell: The cell to build.

        Returns:
            Its outcome.
        """
        scratch = self._root / cell.directory.replace("/", "-")
        scratch.mkdir(parents=True, exist_ok=True)

        # No source directories: the CLI imports OSPREY as installed, which is
        # the whole point of this test. _cli_env still drops every *_API_KEY, so
        # the environment matches the one the fixtures were frozen under.
        env = _cli_env([])

        init_args = ["init", PROJECT_NAME, "--preset", cell.preset, "--no-git"]
        if cell.mode is not None:
            init_args += ["--set", f"channel_finder_mode={cell.mode}"]
        result = _run_cli(init_args, cwd=scratch, env=env)
        if result.returncode != 0:
            return Refused("init", _failure_reason(result))

        project = scratch / PROJECT_NAME
        # --skip-deps: the comparison reads only rendered documents, never a
        # venv. The one key a venv reaches is the interpreter under
        # execution.environment, which mask_document deletes, so the masked
        # documents are the ones the freeze's full build captured.
        result = _run_cli(["build", "--skip-deps"], cwd=project, env=env)
        if result.returncode != 0:
            return Refused("build", _failure_reason(result))

        captured = collect_rendered_configs(project)
        masked = {name: mask_document(document or {}) for name, document in captured.items()}
        return Rendered(
            masked, verify_capture_is_complete(project, masked), collect_login_tiers(project)
        )


@pytest.fixture(scope="module")
def rendered_cells(tmp_path_factory: pytest.TempPathFactory) -> CellRenderer:
    """One renderer for the whole module, so no cell is built twice."""
    return CellRenderer(tmp_path_factory.mktemp("explicit-config-equivalence"))


def _fixture_login_tiers(cell: Cell) -> dict[str, Any] | None:
    """The frozen staged login map of *cell*, or ``None`` when it froze none.

    Args:
        cell: A cell the freeze recorded as frozen.

    Returns:
        The parsed ``<cell>/staged/login_tiers.yml``.
    """
    path = FIXTURE_ROOT / cell.directory / STAGED_DIR / LOGIN_TIERS_FILE
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _fixture_documents(cell: Cell) -> dict[str, dict[str, Any]]:
    """The frozen documents of *cell*, keyed by render name.

    Args:
        cell: A cell the freeze recorded as frozen.

    Returns:
        ``{"root": …}`` plus one entry per render.
    """
    directory = FIXTURE_ROOT / cell.directory
    return {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for path in sorted(directory.glob("*.yml"))
    }


# ─────────────────────────────────────────────────────────────────────────────
# Leaf-level comparison
# ─────────────────────────────────────────────────────────────────────────────


def dotted_leaves(document: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a document to dotted paths, one entry per leaf.

    Mappings are descended into; everything else — scalars and sequences — is a
    leaf, so a list difference reads as one line naming the list rather than as
    a wall of per-index entries.

    Args:
        document: A masked config document.

    Returns:
        ``{"a.b.c": value}`` for every leaf reached.
    """
    leaves: dict[str, Any] = {}

    def walk(node: Any, prefix: str) -> None:
        # An empty mapping is a leaf value of its own — a key that lost its
        # members is not the same as a key that vanished — except at the root,
        # where an empty document simply has no leaves.
        if isinstance(node, Mapping) and (node or not prefix):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        else:
            leaves[prefix] = node

    walk(document, "")
    return leaves


def leaf_differences(
    fixture_document: Mapping[str, Any], live_document: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Every dotted leaf on which two documents disagree.

    Args:
        fixture_document: The frozen document.
        live_document: The document the live render produced.

    Returns:
        ``{path: (fixture_value, live_value)}``, either value possibly
        :data:`ABSENT`.
    """
    left = dotted_leaves(fixture_document)
    right = dotted_leaves(live_document)
    differences: dict[str, tuple[Any, Any]] = {}
    for path in sorted(set(left) | set(right)):
        fixture_value = left.get(path, ABSENT)
        live_value = right.get(path, ABSENT)
        if fixture_value is ABSENT and live_value is ABSENT:
            continue
        if fixture_value != live_value:
            differences[path] = (fixture_value, live_value)
    return differences


def _render_value(value: Any) -> str:
    """A value, short enough to read in a failure message.

    Args:
        value: Any leaf value, or :data:`ABSENT`.

    Returns:
        Its ``repr``, truncated.
    """
    text = repr(value)
    return text if len(text) <= 160 else f"{text[:157]}..."


def _describe(document: str, path: str, fixture_value: Any, live_value: Any) -> str:
    """One readable line for one differing leaf.

    Args:
        document: Render the leaf belongs to.
        path: Dotted path of the leaf.
        fixture_value: What the frozen document carries.
        live_value: What the live render carries.

    Returns:
        A single line naming the document, the leaf, and what changed. For two
        lists it names the members gained and lost instead of printing both.
    """
    where = f"{document}: {path}"
    if isinstance(fixture_value, list) and isinstance(live_value, list):
        gained = [item for item in live_value if item not in fixture_value]
        lost = [item for item in fixture_value if item not in live_value]
        if gained or lost:
            return f"{where}: list gains {_render_value(gained)}, loses {_render_value(lost)}"
        return f"{where}: same list members, different order"
    return f"{where}: {_render_value(fixture_value)} -> {_render_value(live_value)}"


# ─────────────────────────────────────────────────────────────────────────────
# Leaves the build resolves from the host
# ─────────────────────────────────────────────────────────────────────────────

#: Leaves whose rendered value states a fact about the machine that built them,
#: mapped to the answers the build can give. ``osprey build`` answers the
#: presets' ``container_runtime: auto`` with the runtime that served the build:
#: ``docker`` on the host that froze the fixtures, and ``auto`` left standing on
#: a host with no working runtime. Comparing the value would therefore pass only
#: on hosts configured like the freezing one.
#:
#: This is not a delta. :data:`CELL_DELTAS` is exact in both directions, so an
#: entry covering a runtime-less host would go stale on every host that has one.
#: Nor can the leaf be masked away: ``test_explicit_config_partition`` reads the
#: same fixtures as the ledger of what a preset states, and pins
#: ``container_runtime`` as a key the preset states as ``auto`` and the render
#: resolves — so it has to survive in the baseline.
#:
#: The leaf is still compared, just not for which answer: both sides must carry
#: one of these, so a leaf that vanishes, or that resolves to something which is
#: not a runtime, is still a difference.
HOST_RESOLVED_LEAVES: Mapping[str, frozenset[str]] = {
    "container_runtime": frozenset({"auto", "docker", "podman"}),
}


def is_host_resolution(path: str, fixture_value: Any, live_value: Any) -> bool:
    """Whether a differing leaf is only the build answering for its host.

    Args:
        path: Dotted path of the leaf.
        fixture_value: What the frozen document carries.
        live_value: What the live render carries.

    Returns:
        True when *path* is host-resolved and both sides carry one of the
        answers the build can give for it.
    """
    answers = HOST_RESOLVED_LEAVES.get(path)
    if answers is None:
        return False
    return fixture_value in answers and live_value in answers


# ─────────────────────────────────────────────────────────────────────────────
# The comparison machinery itself
#
# These build no deployment and pin the two properties the cell assertions
# rely on: a difference that is
# not declared is caught, and a declared difference that stops happening is
# caught too.
# ─────────────────────────────────────────────────────────────────────────────


def test_leaf_differences_reports_appearance_and_disappearance() -> None:
    """A key gained or lost reads as a difference against :data:`ABSENT`."""
    differences = leaf_differences({"hooks": {"debug": True}}, {"hooks": {"debug": True, "x": 1}})
    assert differences == {"hooks.x": (ABSENT, 1)}

    differences = leaf_differences({"hooks": {"debug": True}}, {})
    assert differences == {"hooks.debug": (True, ABSENT)}

    assert leaf_differences({"a": {"b": [1, 2]}}, {"a": {"b": [1, 2]}}) == {}


def test_dotted_leaves_treats_a_list_as_one_leaf() -> None:
    """Sequences are leaves, so a list difference is one line, not one per index."""
    assert dotted_leaves({"api": {"providers": ["a", "b"]}}) == {"api.providers": ["a", "b"]}
    assert dotted_leaves({"web": {"panels": {}}}) == {"web.panels": {}}


def test_a_list_gains_delta_accepts_only_the_declared_members() -> None:
    """``added=`` means the fixture's members plus exactly those, sorted."""
    delta = Delta(document="root", path="api.providers", added=("argo", "ds4"))
    assert (
        delta.explain_mismatch(["anthropic", "openai"], ["anthropic", "argo", "ds4", "openai"])
        is None
    )

    mismatch = delta.explain_mismatch(["anthropic"], ["anthropic", "argo", "ds4", "surprise"])
    assert mismatch is not None
    assert "surprise" in mismatch


def test_an_exact_delta_accepts_only_the_declared_pair() -> None:
    """``fixture=``/``live=`` pin both sides, :data:`ABSENT` included."""
    delta = Delta(document="root", path="hooks.debug", fixture=ABSENT, live=False)
    assert delta.explain_mismatch(ABSENT, False) is None

    mismatch = delta.explain_mismatch(ABSENT, True)
    assert mismatch is not None
    assert "hooks.debug" in mismatch

    mismatch = delta.explain_mismatch(True, False)
    assert mismatch is not None


def test_a_delta_must_say_what_differs() -> None:
    """A delta spelled with neither form, or with both, is a coding error."""
    with pytest.raises(ValueError, match="must say what differs"):
        Delta(document="root", path="api.providers")
    with pytest.raises(ValueError, match="not both"):
        Delta(document="root", path="api.providers", live=1, added=("a",))


def test_a_host_resolved_leaf_differs_only_between_the_builds_answers() -> None:
    """The build answering per host is not a difference; anything else still is."""
    assert is_host_resolution("container_runtime", "docker", "auto")
    assert is_host_resolution("container_runtime", "auto", "podman")

    # A leaf that vanished, or that is not a runtime at all, is a real difference.
    assert not is_host_resolution("container_runtime", "docker", ABSENT)
    assert not is_host_resolution("container_runtime", "docker", "containerd")

    # The exemption is per leaf, not a blanket one for those values.
    assert not is_host_resolution("execution_method", "docker", "auto")


def test_every_pending_delta_names_a_task() -> None:
    """A pending entry says what it is waiting for, so it can be retired."""
    for cell, deltas in CELL_DELTAS.items():
        for delta in deltas:
            if delta.pending is not None:
                assert delta.pending.strip(), f"{cell} {delta.key}: empty pending marker"


def _write_build_config(build: Path, relative: str, document: Mapping[str, Any]) -> None:
    path = build / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(document)), encoding="utf-8")


def test_a_service_copy_is_compared_without_its_login_map(tmp_path: Path) -> None:
    """Only ``build/services/*`` copies may carry the map; it is frozen apart from them."""
    root = {"project_name": PROJECT_NAME, "control_system": {"writes": "allowed"}}
    staged = {**root, "control_system": {"writes": "allowed", "login_tiers": {"a": "never"}}}
    build = tmp_path / "build"
    _write_build_config(build, "config.yml", root)
    _write_build_config(build, "services/bluesky/config.yml", staged)
    _write_build_config(build, "services/dispatch_worker/config.yml", staged)

    assert verify_capture_is_complete(tmp_path, {"root": root}) == [
        "config.yml",
        "services/bluesky/config.yml",
        "services/dispatch_worker/config.yml",
    ]
    assert collect_login_tiers(tmp_path) == {"a": "never"}

    # The map anywhere but a service copy is a document of its own.
    _write_build_config(build, ".image/x/build/config.yml", staged)
    with pytest.raises(RuntimeError, match="match no captured render"):
        verify_capture_is_complete(tmp_path, {"root": root})


def test_staged_copies_that_disagree_on_the_login_map_refuse(tmp_path: Path) -> None:
    """One cell stages one map; two different ones cannot be frozen as one."""
    build = tmp_path / "build"
    _write_build_config(
        build, "services/bluesky/config.yml", {"control_system": {"login_tiers": {"a": "never"}}}
    )
    _write_build_config(
        build,
        "services/dispatch_worker/config.yml",
        {"control_system": {"login_tiers": {"a": "allowed"}}},
    )
    with pytest.raises(RuntimeError, match="different login maps"):
        collect_login_tiers(tmp_path)

    assert collect_login_tiers(tmp_path / "nothing-built") is None


def test_the_build_account_is_pinned_for_both_sides(monkeypatch: pytest.MonkeyPatch) -> None:
    """The freeze and this module build as the same account, whoever runs them."""
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "someone")
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "someone-else")
    env = _cli_env([])
    assert "OSPREY_TERMINAL_USER" not in env
    assert env["OSPREY_AUDIT_IDENTITY"] == META["build_account"]


def test_a_list_difference_names_what_it_gained_and_lost() -> None:
    """The failure line summarizes a list instead of printing both of them."""
    line = _describe("root", "api.providers", ["a", "b"], ["a", "c"])
    assert "gains ['c']" in line
    assert "loses ['b']" in line


# ─────────────────────────────────────────────────────────────────────────────
# The assertions
# ─────────────────────────────────────────────────────────────────────────────


FROZEN_IDS = [cell.directory for cell in FROZEN_CELLS]
REFUSED_IDS = [cell.directory for cell in REFUSED_CELLS]


def test_every_recorded_cell_is_covered() -> None:
    """The cell list, the fixture directories and ``meta.json`` agree."""
    assert CELLS, "cells.json is empty; regenerate the fixtures"
    assert [cell.directory for cell in CELLS] == META["cells"]
    assert sorted(META["completeness"]) == sorted(cell.directory for cell in FROZEN_CELLS)
    stale = sorted(set(CELL_DELTAS) - {cell.directory for cell in FROZEN_CELLS})
    assert not stale, f"CELL_DELTAS names cells that are not frozen: {stale}"
    stale = sorted(set(STAGED_LOGIN_DELTAS) - {cell.directory for cell in FROZEN_CELLS})
    assert not stale, f"STAGED_LOGIN_DELTAS names cells that are not frozen: {stale}"


@pytest.mark.parametrize("cell", FROZEN_CELLS, ids=FROZEN_IDS)
def test_live_render_equals_frozen_baseline(cell: Cell, rendered_cells: CellRenderer) -> None:
    """The live render differs from the frozen one by exactly its declared deltas."""
    rendered = rendered_cells.render(cell)
    fixture_documents = _fixture_documents(cell)

    assert set(rendered.documents) == set(fixture_documents), (
        f"{cell.directory}: the live build emitted a different set of renders. "
        f"frozen={sorted(fixture_documents)} live={sorted(rendered.documents)}"
    )

    observed: dict[tuple[str, str], tuple[Any, Any]] = {}
    for name in sorted(fixture_documents):
        for path, values in leaf_differences(
            fixture_documents[name], rendered.documents[name]
        ).items():
            observed[(name, path)] = values

    declared = {delta.key: delta for delta in CELL_DELTAS.get(cell.directory, ())}

    problems: list[str] = []
    for (document, path), (fixture_value, live_value) in sorted(observed.items()):
        if is_host_resolution(path, fixture_value, live_value):
            continue
        delta = declared.get((document, path))
        if delta is None:
            problems.append(f"  undeclared  {_describe(document, path, fixture_value, live_value)}")
        elif delta.pending is not None:
            problems.append(
                f"  pending now real  {_describe(document, path, fixture_value, live_value)} "
                f"— predicted for {delta.pending}, which has evidently landed. Check the "
                f"observation against the prediction, then drop pending= so it is asserted."
            )
        else:
            mismatch = delta.explain_mismatch(fixture_value, live_value)
            if mismatch is not None:
                problems.append(f"  wrong delta {mismatch}")
    for document, path in sorted(set(declared) - set(observed)):
        if declared[(document, path)].pending is not None:
            continue
        problems.append(
            f"  stale delta {document}: {path} is declared in CELL_DELTAS but the live "
            f"render matches the fixture; drop the entry"
        )

    assert not problems, (
        f"{cell.directory}: the live render does not match "
        f"tests/fixtures/explicit_config/{cell.directory} within its declared deltas.\n"
        + "\n".join(problems)
    )


@pytest.mark.parametrize("cell", FROZEN_CELLS, ids=FROZEN_IDS)
def test_live_render_is_as_complete_as_the_baseline(
    cell: Cell, rendered_cells: CellRenderer
) -> None:
    """The live build emits the same renders and the same number of config documents."""
    rendered = rendered_cells.render(cell)
    expected = META["completeness"][cell.directory]

    tiers = sorted(name for name in rendered.documents if name != "root")
    assert tiers == expected["renders"], (
        f"{cell.directory}: renders changed. frozen={expected['renders']} live={tiers}"
    )
    documents = sorted(f"{name}.yml" for name in rendered.documents)
    assert documents == expected["documents"], (
        f"{cell.directory}: captured documents changed. "
        f"frozen={expected['documents']} live={documents}"
    )
    gains = BUILD_CONFIG_GAINS.get(cell.directory, ())
    missing_gains = sorted(set(gains) - set(rendered.build_configs_checked))
    assert not missing_gains, (
        f"{cell.directory}: declared build-config gains are absent: {missing_gains}"
    )
    expected_count = expected["build_configs_checked"] + len(gains)
    assert len(rendered.build_configs_checked) == expected_count, (
        f"{cell.directory}: the build tree holds "
        f"{len(rendered.build_configs_checked)} config.yml documents, the baseline "
        f"{expected['build_configs_checked']} plus {len(gains)} declared. Live paths: "
        f"{sorted(rendered.build_configs_checked)}"
    )


@pytest.mark.parametrize("cell", FROZEN_CELLS, ids=FROZEN_IDS)
def test_staged_login_map_equals_frozen_map(cell: Cell, rendered_cells: CellRenderer) -> None:
    """The login map the build stages equals the frozen one exactly, presence included."""
    rendered = rendered_cells.render(cell)
    frozen = _fixture_login_tiers(cell)
    expected_presence = META["completeness"][cell.directory]["login_tiers"]
    assert (frozen is not None) == expected_presence, (
        f"{cell.directory}: meta.json says login_tiers={expected_presence}, the fixture "
        f"{'has' if frozen is not None else 'has no'} {STAGED_DIR}/{LOGIN_TIERS_FILE}"
    )
    live = rendered.login_tiers
    if frozen is None or live is None:
        assert live == frozen, f"{cell.directory}: frozen={frozen} live={live}"
        return
    declared = STAGED_LOGIN_DELTAS.get(cell.directory, {})
    problems: list[str] = []
    for login in sorted(set(frozen) | set(live)):
        observed = (frozen.get(login, ABSENT), live.get(login, ABSENT))
        expected = declared.get(login)
        if expected is None:
            if observed[0] != observed[1]:
                problems.append(
                    f"  undeclared  {login}: {_render_value(observed[0])} -> "
                    f"{_render_value(observed[1])}"
                )
        elif observed[0] == observed[1]:
            problems.append(f"  stale delta {login}: declared {expected}, but it did not move")
        elif observed != expected:
            problems.append(
                f"  wrong delta {login}: declared {expected[0]!r} -> {expected[1]!r}, observed "
                f"{_render_value(observed[0])} -> {_render_value(observed[1])}"
            )
    assert not problems, (
        f"{cell.directory}: the staged login map moved outside STAGED_LOGIN_DELTAS. "
        f"frozen={frozen} live={live}\n" + "\n".join(problems)
    )


@pytest.mark.parametrize("cell", REFUSED_CELLS, ids=REFUSED_IDS)
def test_refused_cell_is_still_refused(cell: Cell, rendered_cells: CellRenderer) -> None:
    """A cell the baseline refused is refused by the live tree, for the same reason."""
    outcome = rendered_cells.attempt(cell)
    assert isinstance(outcome, Refused), (
        f"{cell.directory}: the baseline refused this cell at {cell.stage}, the live tree "
        f"renders it. If that is intended it is a behaviour change, not a fixture repair.\n"
        f"baseline reason: {cell.reason}"
    )
    assert outcome.stage == cell.stage, (
        f"{cell.directory}: baseline refused at {cell.stage}, live refuses at "
        f"{outcome.stage}.\nlive reason: {outcome.reason}"
    )
    missing = [
        token for token in REFUSAL_TOKENS.get(cell.directory, ()) if token not in outcome.reason
    ]
    assert not missing, (
        f"{cell.directory}: the live refusal no longer names {missing}.\n"
        f"live reason: {outcome.reason}"
    )


def test_unset_mode_cells_pass_no_mode() -> None:
    """A cell with no mode is the unset one, and names its directory accordingly."""
    for cell in CELLS:
        expected = cell.mode or UNSET_MODE_DIR
        assert cell.directory == f"{cell.preset}/{expected}"
