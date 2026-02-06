"""Single LLM processor implementation for ARIEL RAP pipeline.

One LLM call for RAG answer generation.
Extracted from search/rag.py LLM call logic.
"""

from __future__ import annotations

import logging
import re

from osprey.services.ariel_search.pipeline.types import (
    AssembledContext,
    ProcessedResult,
    ProcessorConfig,
)
from osprey.services.ariel_search.prompts import RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class SingleLLMProcessor:
    """Processor that makes a single LLM call for RAG.

    Generates an answer from the assembled context using
    a single LLM completion call.
    """

    @property
    def processor_type(self) -> str:
        """Type of processor."""
        return "single_llm"

    async def process(
        self,
        query: str,
        context: AssembledContext,
        config: ProcessorConfig,
    ) -> ProcessedResult:
        """Generate an answer using LLM.

        Args:
            query: Original query string
            context: Assembled context with formatted text
            config: Processing configuration with LLM settings

        Returns:
            ProcessedResult with generated answer
        """
        if not context.items:
            return ProcessedResult(
                answer=(
                    "I don't have enough information to answer this question based on "
                    "the available logbook entries."
                ),
                items=[],
                reasoning="No context provided",
                citations=[],
            )

        # Build the RAG prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context.text,
            question=query,
        )

        # Generate answer using LLM
        try:
            from osprey.models.completion import get_chat_completion

            llm_kwargs: dict = {
                "provider": config.provider,
                "model_id": config.model_id,
                "temperature": config.temperature,
            }

            if config.base_url:
                llm_kwargs["base_url"] = config.base_url

            response = get_chat_completion(
                message=prompt,
                **llm_kwargs,
            )

            # Handle different response types
            if isinstance(response, str):
                answer = response
            else:
                # For structured outputs or thinking blocks, convert to string
                answer = str(response)

            if not answer:
                answer = "I was unable to generate an answer."

        except ImportError:
            logger.warning("osprey.models.completion not available for RAG")
            answer = (
                "LLM not available for answer generation. "
                f"Found {len(context.items)} relevant entries."
            )
        except Exception as e:
            logger.error(f"LLM call failed for RAG: {e}")
            answer = f"Error generating answer: {e}"

        # Extract citations from the answer
        citations = self._extract_citations(answer)

        # If no citations extracted, use all context items
        if not citations:
            citations = [item.entry["entry_id"] for item in context.items]

        return ProcessedResult(
            answer=answer,
            items=context.items,
            reasoning=None,
            citations=citations,
        )

    def _extract_citations(self, text: str) -> list[str]:
        """Extract citation IDs from text.

        Looks for [#id] patterns in the answer.

        Args:
            text: Text to search for citations

        Returns:
            List of unique entry IDs cited
        """
        if not text:
            return []

        # Find [#XXX] patterns
        pattern = r"\[#(\w+)\]"
        matches = re.findall(pattern, text)

        # Dedupe while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                unique.append(match)

        return unique


__all__ = ["SingleLLMProcessor"]
