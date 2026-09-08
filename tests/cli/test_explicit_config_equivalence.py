"""Requirement 1: the live render still equals the frozen baseline render.

The explicit-profile-config work moves the declarative defaults out of
``src/osprey/templates/apps/*/config.yml.j2`` and into the presets, and rewrites
the framework template to carry only derived keys. Nothing about the *rendered*
``build/config.yml`` is supposed to change. ``tests/fixtures/explicit_config``
holds what the render produced at the baseline commit; this module re-runs the
same ``osprey init`` + ``osprey build`` against the **live** tree and asserts the
two agree, document by document, leaf by leaf.

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
delta cannot outlive the change that justified it. Requirement 1 sanctions two,
which land with the tasks that cause them:

- the two standalone presets gain ``{amsc-i2, argo, ds4, stanford}`` in the
  ``api.providers`` name list, because the packaged provider catalog is what
  feeds the render::

      "ariel-standalone/unset": (
          Delta(document="root", path="api.providers",
                added=("amsc-i2", "argo", "ds4", "stanford")),
      ),

- ``hello-world`` gains ``hooks.debug: false``, because the posture floor makes
  that key unconditional::

      "hello-world/unset": (
          Delta(document="root", path="hooks.debug", fixture=ABSENT, live=False),
      ),

``execution.environment.*`` is named in Requirement 1 too, but the freeze strips
that whole mapping (it resolves to an interpreter path), so it can never surface
as a delta and needs no entry here.

Three more deltas are declared that Requirement 1 did not foresee, two from
later gating work rather than from the conversion. ``approval.tools.entry_publish``
reaches every render made from a preset that spells an ARIEL approval policy,
and the three panel-rail verbs — ``approval.tools.add_panel_to_rail``,
``approval.tools.remove_panel_from_rail`` and ``approval.tools.register_panel``
— reach every render made from a preset that names its approval policy tool by
tool. The fixtures were frozen while all four were gated nowhere, so those
leaves are genuinely new. ``web.feedback.email`` goes the other way: the three
root presets stopped shipping a recipient, so every document they render
carries ``""`` where the frozen one carries the address the baseline shipped.

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
quietly becoming an allowance nobody re-read. The control-assistant persona
entries below were carried this way until the app templates were deleted and
the observation matched the prediction on every cell.

Running it
----------
The ``slow`` marker is registered in ``pyproject.toml`` and needs no opt-in flag:
these tests run in an ordinary suite run and are deselected with ``-m 'not
slow'``. On their own::

    uv run --extra dev pytest tests/cli/test_explicit_config_equivalence.py -q

A full run is minutes, not seconds: each cell is a real ``osprey build``, and a
control-assistant cell derives a large virtual-accelerator channel set. Each cell
is therefore built once per session and shared between the tests that read it.
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
    PROJECT_NAME,
    UNSET_MODE_DIR,
    _cli_env,
    _failure_reason,
    _run_cli,
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
        document: Render the leaf belongs to — ``"root"`` or a persona name.
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


#: Every leaf each cell is allowed to differ from its frozen fixture by, keyed by
#: the cell directory as ``cells.json`` spells it. A cell absent from this
#: mapping must equal its fixture exactly. See the module docstring for the
#: Requirement 1 deltas and how to spell one.
def _control_assistant_persona_deltas() -> tuple[Delta, ...]:
    """The per-persona differences the framework template introduces.

    Both are functionally inert, and both are the same in all four
    channel-finder modes, so every control-assistant cell carries them. They
    were predicted before the app templates were deleted and observed exactly
    as predicted — fixture value present, live key absent — once the personas
    rendered through the framework template.

    Returns:
        The knowledge-persona and logbook-persona deltas.
    """
    return (
        # knowledge persona runs with `claude_code.servers.ariel.enabled: false`,
        # and the framework template gates these three on that flag. Their only readers
        # are the ARIEL surfaces, which are off for this persona, so dropping them
        # changes nothing it does.
        Delta(
            document="knowledge",
            path="logbook.composition.provider",
            fixture="anthropic",
            live=ABSENT,
        ),
        Delta(
            document="knowledge",
            path="ariel.enhancement_modules.semantic_processor.provider",
            fixture="anthropic",
            live=ABSENT,
        ),
        Delta(
            document="knowledge",
            path="ariel.enhancement_modules.semantic_processor.model.model_id",
            fixture="haiku",
            live=ABSENT,
        ),
        # logbook persona: the app template wrote `pipeline_mode: {{ default_pipeline }}`
        # with the variable undefined, so both keys froze as null. The framework
        # template omits the whole block instead of writing an empty one.
        Delta(
            document="logbook",
            path="channel_finder.pipeline_mode",
            fixture=None,
            live=ABSENT,
        ),
        Delta(
            document="logbook",
            path="channel_finder.pipelines",
            fixture=None,
            live=ABSENT,
        ),
    )


def _standalone_catalog_delta() -> tuple[Delta, ...]:
    """The provider entries a standalone preset gains from the packaged catalog.

    The two standalone app templates carried a shorter ``api.providers`` list
    than the control-assistant one; the packaged ``providers.yml`` is one
    catalog for every preset, so their renders gain exactly the four entries
    they lacked (Requirement 1).

    Returns:
        The ``api.providers`` delta, the same for every standalone cell.
    """
    return (
        Delta(document="root", path="api.providers", added=("amsc-i2", "argo", "ds4", "stanford")),
    )


def _entry_publish_deltas(*documents: str) -> tuple[Delta, ...]:
    """The approval policy every render with an ARIEL approval table gained.

    ``entry_publish`` is the half of a logbook write that reaches the facility,
    and it was gated nowhere: it sat in neither ``permissions_ask`` nor the
    approval hook, so a headless read-only query could publish. Gating it put
    ``approval.tools.entry_publish: always`` beside the existing
    ``entry_create`` line in every preset that spells an ARIEL approval policy,
    so every document those presets render gains the leaf. The fixtures were
    frozen before that, which is why it reads as a difference here rather than
    as a render the conversion changed.

    channel-finder-standalone runs no ARIEL server and names no approval table
    for it, so its cells gain nothing and are absent below.

    Args:
        documents: The rendered documents the cell emits, ``root`` plus one per
            persona.

    Returns:
        One delta per document.
    """
    return tuple(
        Delta(
            document=document,
            path="approval.tools.entry_publish",
            fixture=ABSENT,
            live="always",
        )
        for document in documents
    )


#: The panel-rail verbs that moved behind the approval hook.
_RAIL_TOOLS = ("add_panel_to_rail", "remove_panel_from_rail", "register_panel")


def _rail_tool_deltas(*documents: str) -> tuple[Delta, ...]:
    """The panel-rail policy every render with a named approval table gained.

    The rail axis decides what an operator can launch at all, which the
    on-screen verbs beside it do not: ``remove_panel_from_rail`` costs them the
    ability to launch the panel back, ``add_panel_to_rail`` puts an entry in
    front of them, and ``register_panel`` adds a proxied upstream. Gating the
    three put ``approval.tools.<tool>: always`` in every preset that names its
    approval policy tool by tool, so every document those presets render gains
    the three leaves. The fixtures were frozen before that, which is why it
    reads as a difference here rather than as a render the conversion changed.

    channel-finder-standalone names no ``approval.tools`` entry at all — its
    rail verbs fall to ``approval.default_policy``, which the freeze already
    recorded — so its cells gain nothing and are absent below.

    Args:
        documents: The rendered documents the cell emits, ``root`` plus one per
            persona.

    Returns:
        One delta per rail verb per document.
    """
    return tuple(
        Delta(
            document=document,
            path=f"approval.tools.{tool}",
            fixture=ABSENT,
            live="always",
        )
        for document in documents
        for tool in _RAIL_TOOLS
    )


def _retired_upstream_link_deltas(*documents: str) -> tuple[Delta, ...]:
    """The three leaves naming the upstream project the presets stopped rendering.

    ``web.feedback.email``, ``web.feedback.github_repo`` and ``web.docs_url``
    named the OSPREY maintainers, their tracker and their documentation site.
    Rendered live they landed in every deployment's own ``profile.yml`` as
    though the facility had chosen them; each preset now documents the key as a
    commented example instead. The code defaults in
    ``interfaces/web_terminal/feedback_destination.py`` still apply when
    nothing spells the key, so the running deployment is unchanged — only the
    rendered document is three leaves shorter.

    Args:
        documents: The rendered documents the cell emits, ``root`` plus one per
            persona.

    Returns:
        Three deltas per document.
    """
    return tuple(
        Delta(document=document, path=path, fixture=fixture, live=ABSENT)
        for document in documents
        for path, fixture in (
            ("web.docs_url", "https://als-apg.github.io/osprey"),
            ("web.feedback.email", "thellert@lbl.gov"),
            ("web.feedback.github_repo", "als-apg/osprey"),
        )
    )


def _query_max_rows_deltas(*documents: str) -> tuple[Delta, ...]:
    """The middle-layer SQL row cap the presets now state.

    ``run_sql`` capped its answer at a number fixed in the tool, so a facility
    could not decide how much of its channel table was worth a turn of the
    agent's context. The cap is now ``channel_finder.query_max_rows``, stated at
    its previous value in the two presets that carry a ``channel_finder`` block,
    so every document they render gains the leaf. The fixtures were frozen
    before the key existed, which is why it reads as a difference here rather
    than as a render that changed.

    ``hello-world`` and ``ariel-standalone`` name no ``channel_finder`` block
    and gain nothing, so their cells are absent below.

    Args:
        documents: The rendered documents the cell emits, ``root`` plus one per
            persona.

    Returns:
        One delta per document.
    """
    return tuple(
        Delta(
            document=document,
            path="channel_finder.query_max_rows",
            fixture=ABSENT,
            live=500,
        )
        for document in documents
    )


def _dispatch_max_turns_deltas() -> tuple[Delta, ...]:
    """The dispatch worker's turn ceiling, now written into its service block.

    How many agentic turns one dispatched run may take was a literal in the
    worker's request model, so a facility could set the two clock budgets and
    not this one. ``dispatch.max_turns`` is the third budget, and the build
    writes it into ``services.dispatch_worker`` beside ``timeout_sec`` and
    ``inactivity_sec`` on every deploy. The fixtures were frozen before the key
    existed, and only the root document carries a service block, so this is one
    delta rather than one per persona.

    Only ``control-assistant`` deploys a dispatch worker; the other presets gain
    nothing and are absent below.

    Returns:
        The single root-document delta.
    """
    return (
        Delta(
            document="root",
            path="services.dispatch_worker.max_turns",
            fixture=ABSENT,
            live=25,
        ),
    )


#: The documents a control-assistant cell renders: the root config plus one per
#: persona in the preset's roster.
_CONTROL_ASSISTANT_DOCUMENTS = (
    "admin",
    "knowledge",
    "logbook",
    "readonly",
    "readwrite",
    "root",
)


CELL_DELTAS: dict[str, tuple[Delta, ...]] = {
    # The posture floor makes `hooks.debug` unconditional, and hello-world is the
    # one preset whose app template never carried it (Requirement 1). The other
    # presets already render `hooks.debug: true`, so they gain nothing.
    "hello-world/unset": (
        Delta(document="root", path="hooks.debug", fixture=ABSENT, live=False),
        *_entry_publish_deltas("root"),
        *_rail_tool_deltas("root"),
    ),
    "ariel-standalone/unset": _standalone_catalog_delta()
    + _entry_publish_deltas("root")
    + _rail_tool_deltas("root")
    + _retired_upstream_link_deltas("root"),
    "channel-finder-standalone/in_context": _standalone_catalog_delta()
    + _retired_upstream_link_deltas("root")
    + _query_max_rows_deltas("root"),
    "channel-finder-standalone/hierarchical": _standalone_catalog_delta()
    + _retired_upstream_link_deltas("root")
    + _query_max_rows_deltas("root"),
    "channel-finder-standalone/middle_layer": _standalone_catalog_delta()
    + _retired_upstream_link_deltas("root")
    + _query_max_rows_deltas("root"),
    "control-assistant/in_context": _control_assistant_persona_deltas()
    + _entry_publish_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _rail_tool_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _retired_upstream_link_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _dispatch_max_turns_deltas()
    + _query_max_rows_deltas(*_CONTROL_ASSISTANT_DOCUMENTS),
    "control-assistant/hierarchical": _control_assistant_persona_deltas()
    + _entry_publish_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _rail_tool_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _retired_upstream_link_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _dispatch_max_turns_deltas()
    + _query_max_rows_deltas(*_CONTROL_ASSISTANT_DOCUMENTS),
    "control-assistant/middle_layer": _control_assistant_persona_deltas()
    + _entry_publish_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _rail_tool_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _retired_upstream_link_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _dispatch_max_turns_deltas()
    + _query_max_rows_deltas(*_CONTROL_ASSISTANT_DOCUMENTS),
    "control-assistant/graph": _control_assistant_persona_deltas()
    + _entry_publish_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _rail_tool_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _retired_upstream_link_deltas(*_CONTROL_ASSISTANT_DOCUMENTS)
    + _dispatch_max_turns_deltas()
    + _query_max_rows_deltas(*_CONTROL_ASSISTANT_DOCUMENTS),
}


#: Substrings a refused cell's live refusal must still contain, keyed by cell
#: directory. The full baseline text is in ``cells.json``, but it names the app
#: template, which this feature removes; what must survive is *why* the mode was
#: refused. A cell absent from this mapping is only asserted to be refused at the
#: same stage.
REFUSAL_TOKENS: dict[str, tuple[str, ...]] = {
    "channel-finder-standalone/graph": ("channel_finder_mode", "services.graphdb"),
}


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
        documents: Masked documents, keyed by render name (``root`` + personas).
        build_configs_checked: Build-relative paths of every ``config.yml`` in
            the build tree, each proved equal to one of *documents*.
    """

    documents: dict[str, dict[str, Any]]
    build_configs_checked: list[str]


class Refused(NamedTuple):
    """A cell the live CLI would not produce.

    Attributes:
        stage: The verb that failed, ``init`` or ``build``.
        reason: The tail of what the CLI said.
    """

    stage: str
    reason: str


class CellRenderer:
    """Builds each cell at most once per session and hands out the result.

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
        result = _run_cli(["build"], cwd=project, env=env)
        if result.returncode != 0:
            return Refused("build", _failure_reason(result))

        captured = collect_rendered_configs(project)
        masked = {name: mask_document(document or {}) for name, document in captured.items()}
        return Rendered(masked, verify_capture_is_complete(project, masked))


@pytest.fixture(scope="session")
def rendered_cells(tmp_path_factory: pytest.TempPathFactory) -> CellRenderer:
    """One renderer for the whole session, so no cell is built twice."""
    return CellRenderer(tmp_path_factory.mktemp("explicit-config-equivalence"))


def _fixture_documents(cell: Cell) -> dict[str, dict[str, Any]]:
    """The frozen documents of *cell*, keyed by render name.

    Args:
        cell: A cell the freeze recorded as frozen.

    Returns:
        ``{"root": …}`` plus one entry per persona.
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
    """The live build emits the same personas and the same number of config documents."""
    rendered = rendered_cells.render(cell)
    expected = META["completeness"][cell.directory]

    personas = sorted(name for name in rendered.documents if name != "root")
    assert personas == expected["personas"], (
        f"{cell.directory}: personas changed. frozen={expected['personas']} live={personas}"
    )
    documents = sorted(f"{name}.yml" for name in rendered.documents)
    assert documents == expected["documents"], (
        f"{cell.directory}: captured documents changed. "
        f"frozen={expected['documents']} live={documents}"
    )
    assert len(rendered.build_configs_checked) == expected["build_configs_checked"], (
        f"{cell.directory}: the build tree holds "
        f"{len(rendered.build_configs_checked)} config.yml documents, the baseline "
        f"{expected['build_configs_checked']}. Live paths: "
        f"{sorted(rendered.build_configs_checked)}"
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
