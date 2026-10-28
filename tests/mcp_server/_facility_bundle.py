"""OKF bundle builders shared by the facility-knowledge MCP tool tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def write_doc(path: Path, body: str) -> None:
    """Write a dedented document, creating its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(body).lstrip(), encoding="utf-8")


def write_minimal_bundle(root: Path) -> Path:
    """Write a minimal OKF bundle at ``root``: an index and two concept types."""
    write_doc(
        root / "index.md",
        """\
        # Facility Bundle
        * [Accelerator Overview](/accelerator_overview.md) - High-level accelerator overview.
        """,
    )
    write_doc(
        root / "accelerator_overview.md",
        """\
        ---
        type: facility_overview
        title: Accelerator Overview
        description: Overview of the accelerator complex.
        ---

        The facility uses a synchrotron ring to produce bright X-ray light.
        """,
    )
    write_doc(
        root / "tables" / "beam_params.md",
        """\
        ---
        type: data_table
        title: Beam Parameters
        description: Live beam parameter PVs from the EPICS control system.
        tags: [epics, beam]
        ---

        Contains PV names for all beam parameters.
        """,
    )
    return root


def install_bundle(monkeypatch, root: Path) -> None:
    """Inject an ``OKFBundle`` over ``root`` into the server module's ``_bundle`` global."""
    import osprey.mcp_server.facility_knowledge.server as srv
    from osprey.services.facility_knowledge.okf.bundle import OKFBundle

    monkeypatch.setattr(srv, "_bundle", OKFBundle(root))
