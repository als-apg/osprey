"""Decision Subsystem for XOpt Optimizer.

This subsystem routes the workflow based on machine state assessment,
selecting the appropriate optimization strategy (exploration, optimization,
or abort).

Supports two modes (configured via xopt_optimizer.decision.mode):
- "llm": LLM-based decision making with structured output
- "mock": Fast placeholder that always selects exploration (default for tests)
"""

from .node import DECISION_SYSTEM_PROMPT, StrategyDecision, create_decision_node

__all__ = ["create_decision_node", "StrategyDecision", "DECISION_SYSTEM_PROMPT"]
