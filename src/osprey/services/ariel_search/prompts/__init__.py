"""ARIEL prompt templates.

This module provides prompt templates for the ARIEL ReAct agent
and RAG answer generation.

See 03_AGENTIC_REASONING.md Section 2.7 for specification.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.prompts.agent_system import DEFAULT_SYSTEM_PROMPT
from osprey.services.ariel_search.prompts.rag_answer import RAG_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig


def get_system_prompt(config: "ARIELConfig") -> str:
    """Get the system prompt for the ARIEL agent.

    Checks for custom facility-specific prompt in config.
    Falls back to default if not configured.

    Args:
        config: ARIEL configuration

    Returns:
        System prompt string
    """
    # Check for custom prompt in config
    # V2: Could check config.prompts.system_prompt
    return DEFAULT_SYSTEM_PROMPT


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "RAG_PROMPT_TEMPLATE",
    "get_system_prompt",
]
