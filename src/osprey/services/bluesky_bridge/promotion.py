"""Prepares a validated session-tier plan for promotion to a permanent catalog.

This module is thin glue, not a promotion engine: it never opens a pull/merge
request, never commits, and never touches ``plan_loader.py``'s directory
layers or trust order. All it does is (1) refuse to hand off a session plan
whose *current* on-disk content lacks a passing validation record — the same
check the load gate (task 2.4) and promote gate (task 2.5) perform — and (2)
copy that plan's exact bytes into a checkout of the repo that owns the target
``preset``/``facility`` plan directory, so the file is ready to ``git add``.

**The promotion workflow this glue supports:**

1. Author + validate a session plan (``write_plan`` /
   ``validate_plan`` — task 2.3), then run it a few times via
   ``launch_run`` until it looks worth keeping.
2. Call :func:`prepare_promotion` with the plan's ``name`` and a filesystem
   path to a checkout of the repo that owns the target catalog directory —
   one of the directories already listed in ``bluesky.plan_dirs`` (config.yml,
   ``preset`` tier) or ``BLUESKY_PLAN_DIRS`` (env, ``facility`` tier); see
   ``plan_loader.py``'s module docstring for the tier mapping. This raises
   unless the plan has a passing validation record for its exact current
   content.
3. Call :func:`stage_promotion` on the result to write the file into that
   checkout.
4. Use the existing **``osprey-contribute``** skill (branch → commit → push
   → PR, GitHub Flow) to open the PR/MR proposing the addition — exactly as
   for any other change to that repo. This module supplies the file and a
   suggested branch name / PR title / PR body; it does not re-implement any
   part of that workflow.

**The trust boundary — read this before wiring anything to auto-promote:**
Nothing in this module (or step 1-3 above) raises the plan's provenance.
A staged file sitting in a target repo's working tree, or even a pushed
branch with an open PR, is still exactly as trusted as it was in the
session: unreviewed. Provenance rises from ``session`` to ``preset``/
``facility`` **only** at the moment a human reviews and merges that PR/MR
into the branch the deployment actually serves plans from. The merge is the
trust boundary — not this module, not validation, not the PR being open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .plan_validation import hash_plan_body
from .session_dir import resolve_session_plan_dir
from .validation_record import validation_records


class UnknownSessionPlanError(FileNotFoundError):
    """No session-tier plan file named ``name`` exists."""


class UnvalidatedPlanError(ValueError):
    """The session plan's current on-disk content has no passing validation record."""


@dataclass(frozen=True)
class PromotionRequest:
    """Everything a contributor needs to propose a session plan for promotion.

    ``target_path`` is where the file should land inside a checkout of the
    repo that owns the target ``preset``/``facility`` plan directory —
    :func:`prepare_promotion` never writes there itself; :func:`stage_promotion`
    does that as an explicit, separate step.
    """

    name: str
    body: str
    content_hash: str
    source_path: Path
    target_path: Path
    suggested_branch: str
    suggested_pr_title: str
    suggested_pr_body: str


def prepare_promotion(name: str, catalog_dir: str | Path) -> PromotionRequest:
    """Build a `PromotionRequest` for a validated session plan.

    Reads ``name``'s current on-disk content from the bridge's session
    directory (``session_dir.resolve_session_plan_dir()``) and refuses to
    proceed unless that exact content already has a passing validation
    record (``validation_record.validation_records``) — an edit made after
    the last successful ``validate_plan`` call changes the content
    hash and drops the record, exactly as it does for the load gate (2.4)
    and promote gate (2.5). Never writes anywhere; purely a read + gate check.

    Args:
        name: Session plan name (the file stem written by ``write_plan``).
        catalog_dir: Filesystem path of the target `preset`/`facility` plan
            directory, inside a local checkout of the repo that owns it.

    Returns:
        A `PromotionRequest` describing what to write where, plus a
        suggested branch name and PR title/body for the ``osprey-contribute``
        skill to use verbatim or adapt.

    Raises:
        UnknownSessionPlanError: No such session plan file exists.
        UnvalidatedPlanError: The file's current content has no passing
            validation record — call ``validate_plan`` first.
    """
    source_path = resolve_session_plan_dir() / f"{name}.py"
    if not source_path.is_file():
        raise UnknownSessionPlanError(
            f"No session plan named {name!r} in {source_path.parent} — call write_plan first."
        )

    body = source_path.read_text(encoding="utf-8")
    content_hash = hash_plan_body(body)
    if not validation_records.has_passing_record(content_hash):
        raise UnvalidatedPlanError(
            f"Session plan {name!r} (content hash {content_hash[:12]}...) has no "
            "passing validation record for its CURRENT content — call "
            "validate_plan and confirm it passes before promoting. "
            "Editing the file after validation invalidates this check, by design."
        )

    target_path = Path(catalog_dir) / f"{name}.py"
    return PromotionRequest(
        name=name,
        body=body,
        content_hash=content_hash,
        source_path=source_path,
        target_path=target_path,
        suggested_branch=f"feature/promote-{name}-plan",
        suggested_pr_title=f"Add {name} scan plan to the catalog",
        suggested_pr_body=(
            f"Promotes the session-authored `{name}` scan plan (validated "
            f"content hash `{content_hash}`) to a permanent catalog plan.\n\n"
            "This file was authored and dry-run validated (mock devices only) "
            "in an OSPREY session; it has never run against real hardware. "
            "Merging this PR is what raises its provenance from `session` to "
            "reviewed — review the plan body as you would any other code "
            "change before approving."
        ),
    )


def stage_promotion(request: PromotionRequest) -> Path:
    """Write ``request.body`` to ``request.target_path``, creating parent dirs.

    The one filesystem side effect this module performs: copying the
    validated session plan's exact bytes into a checkout of the target
    catalog repo so the ``osprey-contribute`` skill has something to
    ``git add``. Byte-for-byte identical to the session file — no
    re-generation, no re-formatting — so the content hash a reviewer or a
    future re-validation computes is unchanged by the move.

    Returns:
        The path written (``request.target_path``).
    """
    request.target_path.parent.mkdir(parents=True, exist_ok=True)
    request.target_path.write_text(request.body, encoding="utf-8")
    return request.target_path
