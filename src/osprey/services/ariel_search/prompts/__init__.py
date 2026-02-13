"""ARIEL prompt templates.

This module provides prompt templates for RAG answer generation.
Agent system prompts are now in osprey.services.ariel_search.agent.executor.

See 02_SEARCH_MODULES.md Section 5 for specification.
"""

from osprey.services.ariel_search.prompts.rag_answer import RAG_PROMPT_TEMPLATE

__all__ = [
    "RAG_PROMPT_TEMPLATE",
]
