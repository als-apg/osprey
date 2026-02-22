"""ARIEL Agent module — self-contained ReAct agent for agentic search.

Uses Osprey's own completion API (LiteLLM) with tool calling instead of
LangGraph. This is a transitional implementation that preserves the
AgentExecutor / AgentResult interface.
"""

from osprey.services.ariel_search.agent.executor import AgentExecutor, AgentResult

__all__ = ["AgentExecutor", "AgentResult"]
