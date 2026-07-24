"""Request bodies for the Bluesky bridge's HTTP routes (see ``app.py``).

Pure Pydantic models — no runner, registry, or connector state — so they are
import-clean of the bluesky stack and safe to import from anywhere the bridge
needs the wire shapes. ``app.py`` re-exports these for backwards-compatible
``bluesky_bridge.app import RunRequest`` call sites.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """A scan launch intent (`POST /runs`).

    Intentionally generic: `plan_name` names a plan the registry (`plans.py`,
    plus any facility-injected plans from `plan_loader.py`) resolves, and
    `plan_args` is forwarded to the runner unmodified via `do_launch` ->
    `PlanRunner.reinitialize(run.request)`.
    """

    plan_name: str
    plan_args: dict[str, Any] = Field(default_factory=dict)


class DraftRunRequest(BaseModel):
    """Body for `POST /draft/run`: launch the shared draft at a pinned revision.

    ``draft_revision`` is the caller's last-seen draft revision (from `GET
    /draft`, a PATCH response, or an SSE frame). The launched
    ``plan_name``/``plan_args`` come exclusively from the server-side draft
    snapshot taken at that exact revision — never from this body.
    """

    draft_revision: int


class PlanSessionWriteRequest(BaseModel):
    """Request body for `POST /plans/session`: author a session-tier plan file.

    ``body`` is the author's own source (``PARAMS`` + ``build_plan``, per the
    layered directory catalog's file contract) — it is never exec'd by this
    route. The remaining fields become the generated `PLAN_METADATA` block
    prepended to it; together they must satisfy `plan_metadata.PlanMetadata`'s
    contract once the session-tier load gate (task 2.4) parses the file.
    """

    name: str
    description: str = ""
    category: str
    required_devices: list[str] = Field(default_factory=list)
    writes: bool
    body: str


class PlanValidateRequest(BaseModel):
    """Request body for `POST /plans/validate`: validate a session plan by name.

    ``sample_args`` supplies the stage-3 dry run's `PARAMS` field values
    directly (the simpler of the two options `plan_validation.py`'s docstring
    calls out — deriving minimal samples from the `PARAMS` schema would need
    per-type generation logic this bridge does not otherwise have); omit it
    for a `PARAMS` with no required fields.
    """

    name: str
    sample_args: dict[str, Any] | None = None
    dry_run_timeout: float = 30.0
