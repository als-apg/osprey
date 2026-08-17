"""Request bodies for the Bluesky bridge's HTTP routes (see ``app.py``).

Pure Pydantic models — no execution or connector state — so they are
import-clean of the bluesky stack and safe to import from anywhere the bridge
needs the wire shapes.

The retired direct-execute routes (``POST /runs``, ``POST /draft/run``) parse
nothing — they answer a fixed refusal — so no model here serves them. The queue
surface defines its own request bodies in ``queue.py``, next to the routes that
read them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class PlanSessionWriteRequest(BaseModel):
    """Request body for `POST /plans/session`: author a session-tier plan file.

    ``body`` is the author's own source (``PARAMS`` + ``build_plan``, per the
    layered directory catalog's file contract) — it is never exec'd by this
    route. The remaining three fields become the generated `PLAN_METADATA`
    block prepended to it, and are exactly `plan_metadata.PlanMetadata`'s
    fields: which channels the plan touches is not declared here at all, it is
    read off the role-typed fields of the ``body``'s own `PARAMS` model, so
    there is nothing here for an author to keep in sync with the code.

    Unknown keys are rejected too (``extra="forbid"``), uniform with
    `plan_metadata.PlanMetadata` — a stale client still POSTing a retired key
    like ``category`` or ``required_devices`` fails loudly naming that key
    rather than getting a silent 200 with the surplus dropped.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    writes: bool
    body: str


class PlanValidateRequest(BaseModel):
    """Request body for `POST /plans/validate`: validate a session plan by name.

    ``sample_args`` supplies the stage-3 dry run's `PARAMS` field values
    directly (the simpler of the two options `plan_validation.py`'s docstring
    calls out — deriving minimal samples from the `PARAMS` schema would need
    per-type generation logic this bridge does not otherwise have); omit it
    for a `PARAMS` with no required fields.

    Unknown keys are rejected too (``extra="forbid"``), uniform with
    `PlanSessionWriteRequest` and `plan_metadata.PlanMetadata`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    sample_args: dict[str, Any] | None = None
    dry_run_timeout: float = 30.0
