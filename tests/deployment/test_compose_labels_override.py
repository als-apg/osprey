"""OSPREY's attribution labels, generated once for every rendered service.

The render writes ``build/osprey-labels.override.yml``: one entry per service
it rendered, each carrying the project name, the checkout identity, the
project path and the config digest. Every compose invocation passes it last,
so a service template does not have to spell those labels for ``osprey
status``, ``down``, ``reset`` and ``set`` to treat its containers as the
deployment's.
"""

from __future__ import annotations

from osprey.deployment import compose_generator, reset, status_display
from osprey.deployment.compose_generator import (
    _inject_project_metadata,
    project_label_values,
)


def test_status_reads_the_project_label_the_render_writes() -> None:
    """Every reader of a label key reads the key the render writes."""
    assert status_display.PROJECT_LABEL is compose_generator.PROJECT_LABEL
    assert compose_generator.PROJECT_ROOT_LABEL in reset.PATH_EVIDENCE_LABELS


def test_the_template_context_labels_are_project_label_values(tmp_path) -> None:
    """The template context and the override read one source of label values."""
    cfg = {"project_name": "site-fixture", "project_root": str(tmp_path)}
    assert _inject_project_metadata(cfg)["osprey_labels"] == project_label_values(cfg)
