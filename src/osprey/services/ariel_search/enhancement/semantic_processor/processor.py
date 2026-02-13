"""ARIEL semantic processor module.

This module extracts keywords and generates summaries from logbook entries
to enable keyword search and improve retrieval quality.

See 01_DATA_LAYER.md Section 6.5-6.6 for specification.
"""

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule
from osprey.services.ariel_search.enhancement.semantic_processor.migration import (
    SemanticProcessorMigration,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from osprey.services.ariel_search.database.migration import BaseMigration
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = get_logger("ariel")


# Pydantic model for structured output
class SemanticProcessorResult(BaseModel):
    """Structured output from semantic processing.

    Follows the pattern from MemoryContentExtraction in osprey.capabilities.memory.
    """

    keywords: list[str] = Field(
        description="Key terms, equipment names, and concepts from the entry"
    )
    summary: str = Field(description="Concise summary capturing the main point of the entry")


# Default prompt template for semantic processing
DEFAULT_PROMPT_TEMPLATE = """Extract keywords and generate a summary from this logbook entry.

Entry text:
{text}

Instructions:
1. Extract 5-15 keywords that capture:
   - Equipment names (e.g., "vacuum pump VP-103", "RF cavity")
   - Technical terms (e.g., "beam current", "temperature")
   - Actions taken (e.g., "replaced", "calibrated", "adjusted")
   - Problem types (e.g., "fault", "alarm", "leak")
   - System areas (e.g., "storage ring", "injector")

2. Generate a 1-2 sentence summary that captures:
   - What happened
   - What action was taken (if any)
   - The outcome or current status

Return ONLY valid JSON matching this schema:
{{
  "keywords": ["keyword1", "keyword2", ...],
  "summary": "Brief summary of the entry"
}}"""


class SemanticProcessorModule(BaseEnhancementModule):
    """Extract keywords and generate summaries from logbook entries.

    Uses a small LLM to extract keywords and generate summaries.
    Follows Osprey's zero-argument constructor pattern with lazy loading.
    """

    def __init__(self) -> None:
        """Initialize the module.

        Zero-argument constructor (Osprey pattern).
        LLM is lazy-loaded on first enhance() call.
        """
        self._model_config: dict[str, Any] = {}
        self._prompt_template: str = DEFAULT_PROMPT_TEMPLATE

    @property
    def name(self) -> str:
        """Return module identifier."""
        return "semantic_processor"

    @property
    def migration(self) -> "type[BaseMigration]":
        """Return migration class for this module."""
        return SemanticProcessorMigration

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the module with settings from config.yml.

        Args:
            config: The enhancement_modules.semantic_processor config dict
        """
        self._model_config = config.get("model", {})
        if config.get("prompt_template"):
            self._prompt_template = config["prompt_template"]

    async def enhance(
        self,
        entry: "EnhancedLogbookEntry",
        conn: "AsyncConnection",
    ) -> None:
        """Extract keywords and summary from entry and store in database.

        Args:
            entry: The entry to enhance
            conn: Database connection from pool
        """
        raw_text = entry.get("raw_text", "")

        if not raw_text.strip():
            logger.debug(f"Skipping empty entry {entry.get('entry_id')}")
            return

        try:
            # Extract keywords and summary using LLM
            result = await self._process_text(raw_text)

            if result:
                await self._store_results(
                    entry_id=entry["entry_id"],
                    keywords=result.keywords,
                    summary=result.summary,
                    conn=conn,
                )

        except Exception as e:
            logger.warning(f"Failed to process entry {entry.get('entry_id')}: {e}")

    async def _process_text(self, text: str) -> SemanticProcessorResult | None:
        """Process text using LLM to extract keywords and summary.

        Args:
            text: Entry text to process

        Returns:
            SemanticProcessorResult or None if processing failed
        """
        try:
            from osprey.models.completion import get_chat_completion

            # Build prompt
            prompt = self._prompt_template.format(text=text[:8000])  # Truncate for safety

            # Get completion
            response = get_chat_completion(
                message=prompt,
                model_config=self._model_config if self._model_config else None,
            )

            # Handle different response types
            if isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            return self._parse_response(response_text)

        except ImportError:
            logger.warning("osprey.models.completion not available, skipping semantic processing")
            return None
        except Exception as e:
            logger.warning(f"LLM processing failed: {e}")
            return None

    def _parse_response(self, response_text: str) -> SemanticProcessorResult | None:
        """Parse LLM response into structured result.

        Args:
            response_text: Raw LLM response

        Returns:
            SemanticProcessorResult or None if parsing failed
        """
        try:
            # Try to extract JSON from response
            # Handle cases where LLM wraps JSON in markdown code blocks
            text = response_text.strip()

            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            text = text.strip()

            # Parse JSON
            data = json.loads(text)

            # Validate with Pydantic
            return SemanticProcessorResult(**data)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None

    async def _store_results(
        self,
        entry_id: str,
        keywords: list[str],
        summary: str,
        conn: "AsyncConnection",
    ) -> None:
        """Store keywords and summary in enhanced_entries table.

        Args:
            entry_id: Entry ID
            keywords: Extracted keywords
            summary: Generated summary
            conn: Database connection
        """
        await conn.execute(
            """
            UPDATE enhanced_entries
            SET keywords = %s,
                summary = %s
            WHERE entry_id = %s
            """,
            [keywords, summary, entry_id],
        )

    async def health_check(self) -> tuple[bool, str]:
        """Check if module is ready.

        Verifies that the LLM model is accessible.

        Returns:
            Tuple of (healthy, message)
        """
        try:
            from osprey.models.completion import get_chat_completion

            # Quick test with minimal prompt
            response = get_chat_completion(
                message="Say OK",
                model_config=self._model_config if self._model_config else None,
            )

            if response:
                return (True, "OK")
            return (False, "Empty response from LLM")

        except ImportError:
            return (False, "osprey.models.completion not available")
        except Exception as e:
            return (False, str(e))
