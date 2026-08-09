"""Request bodies for the Bluesky bridge's HTTP routes (see ``app.py``).

Pure Pydantic models — no execution or connector state — so they are
import-clean of the bluesky stack and safe to import from anywhere the bridge
needs the wire shapes.

The retired direct-execute routes (``POST /runs``, ``POST /draft/run``) had
bodies here too; they answer a fixed refusal now and parse nothing, so those
models are gone. The queue surface defines its own request bodies in
``queue.py``, next to the routes that read them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PlanSessionWriteRequest(BaseModel):
    """Request body for `POST /plans/session`: author a session-tier plan file.

    ``body`` is the author's own source (``PARAMS`` + ``build_plan``, per the
    layered directory catalog's file contract) — it is never exec'd by this
    route. The remaining fields become the generated `PLAN_METADATA` block
    prepended to it; together they must satisfy `plan_metadata.PlanMetadata`'s
    contract once the session-tier load gate parses the file.
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
