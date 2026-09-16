"""Emitters that turn ``ao.json`` + ``mapping.yaml`` into deployment artifacts."""

from osprey.services.mml.emit.context import EmitContext, build_context, require_knowledge_extra

__all__ = ["EmitContext", "build_context", "require_knowledge_extra"]
