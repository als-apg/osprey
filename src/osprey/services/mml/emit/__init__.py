"""Emitters that turn ``ao.json`` + ``mapping.yaml`` into deployment artifacts.

Only the provenance context is re-exported here. Each emitter lives in its own
module and is imported from it, because the lanes do not cost the same: the
virtual-accelerator lane in :mod:`~osprey.services.mml.emit.va` reaches the
served model's own bindings schema and calibration arithmetic, and re-exporting
it would pull that weight into every import of this package, the CLI's included.
"""

from osprey.services.mml.emit.context import EmitContext, build_context, require_knowledge_extra

__all__ = ["EmitContext", "build_context", "require_knowledge_extra"]
