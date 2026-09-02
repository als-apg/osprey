"""The example artifact the gallery seeds into an empty workspace.

A first visit to WORKSPACE otherwise shows nothing. The gallery writes one
shipped, interactive Plotly page into the store the first time it starts
against a workspace, when ``artifact_server.example_artifact`` is on. The
bytes ship inside the package (``examples/interactive-plot.html``, made by
``examples/make_interactive_plot.py``), so no deployment needs plotly to
show it.

Two rules keep it honest:

* **It is the person's, not the agent's.** The entry carries
  :data:`~osprey.stores.artifact_store.EXAMPLE_ORIGIN`; every listing the
  agent reads passes ``exclude_examples=True``, so the agent never lists,
  cites or reports it.
* **Deleting it sticks.** Removing the entry through any store path writes
  :data:`~osprey.stores.artifact_store.EXAMPLE_REMOVED_SENTINEL` beside the
  index, and this module never seeds while that file exists.
"""

from __future__ import annotations

import logging
from functools import cache
from importlib import resources

from osprey.stores.artifact_store import (
    EXAMPLE_ORIGIN,
    ArtifactEntry,
    ArtifactStore,
    artifact_mutation_actor,
)

logger = logging.getLogger(__name__)

EXAMPLE_TITLE = "Example: an interactive plot"
EXAMPLE_DESCRIPTION = (
    "Shipped with the control-assistant preset so the workspace is not empty on first visit. "
    "Synthetic data, not a session result. Delete it when you no longer need it. "
    "Hover for values, drag to zoom, double-click to reset. "
    'For a real one, ask the agent: "Plot the beam current for the last two hours."'
)
EXAMPLE_FILENAME = "example-interactive-plot.html"

_RESOURCE = "interactive-plot.html"


@cache
def example_artifact_html() -> str:
    """The shipped page, read once from the installed package."""
    return (
        resources.files("osprey.interfaces.artifacts.examples")
        .joinpath(_RESOURCE)
        .read_text(encoding="utf-8")
    )


def seed_example_artifact(store: ArtifactStore) -> ArtifactEntry | None:
    """Write the example into *store* if it is not there and was never removed.

    Returns the new entry, or ``None`` when nothing was written. Safe to call
    on every launch: a store that already holds an example, or whose example
    was deleted, is left alone.
    """
    if store.example_removed:
        return None
    if any(e.origin == EXAMPLE_ORIGIN for e in store.list_entries()):
        return None
    # "system": the activity feed reports only agent mutations, and this is
    # neither the agent's work nor a person's click.
    with artifact_mutation_actor("system"):
        entry = store.save_file(
            file_content=example_artifact_html().encode("utf-8"),
            filename=EXAMPLE_FILENAME,
            artifact_type="plot_html",
            title=EXAMPLE_TITLE,
            description=EXAMPLE_DESCRIPTION,
            mime_type="text/html",
            tool_source="osprey",
            origin=EXAMPLE_ORIGIN,
            run_id="",
            session_id="",
        )
    logger.info("Seeded the example artifact into %s", store.artifact_dir)
    return entry
