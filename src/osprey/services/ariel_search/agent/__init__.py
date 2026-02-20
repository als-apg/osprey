"""ARIEL Agent module for agentic orchestration.

This module provides the AgentExecutor for ReAct-style agent execution
with search tools. Use AGENT mode for agentic orchestration where the
agent decides what to search and synthesizes answers.

The Agent is a peer to the RAGPipeline - they are not nested.
Use direct search calls for KEYWORD/SEMANTIC, RAGPipeline for RAG,
and Agent for agentic orchestration (AGENT mode).

Example:
    from osprey.services.ariel_search.agent import AgentExecutor, AgentResult

    executor = AgentExecutor(repository, config, embedder_loader)
    result = await executor.execute(
        query="What caused the beam loss yesterday?",
        max_results=10,
    )
    print(result.answer)
    print(result.sources)
"""

from osprey.services.ariel_search.agent.executor import (
    AgentExecutor,
    AgentResult,
)

__all__ = [
    "AgentExecutor",
    "AgentResult",
]
