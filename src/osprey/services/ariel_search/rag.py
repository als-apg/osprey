"""ARIEL RAG pipeline.

Deterministic RAG pipeline: retrieve → fuse → assemble → generate.
A peer to AgentExecutor — both are top-level execution strategies.

The user switches between agent (exploratory, non-deterministic) and
RAG (direct question-answering, deterministic, auditable).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.models import (
    DiagnosticLevel,
    PipelineDetails,
    RAGStageStats,
    SearchDiagnostic,
)
from osprey.services.ariel_search.prompts import RAG_PROMPT_TEMPLATE
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = get_logger("ariel")

# Empty-context answer when no entries are found
_NO_CONTEXT_ANSWER = (
    "I don't have enough information to answer this question based on "
    "the available logbook entries."
)


@dataclass(frozen=True)
class RAGResult:
    """Result from RAG pipeline execution.

    Attributes:
        answer: LLM-generated answer text
        entries: EnhancedLogbookEntry dicts used as context
        citations: Entry IDs cited in the answer
        retrieval_count: Total entries retrieved before fusion/truncation
        context_truncated: Whether context was truncated to fit limits
    """

    answer: str
    entries: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    citations: tuple[str, ...] = field(default_factory=tuple)
    retrieval_count: int = 0
    context_truncated: bool = False
    diagnostics: tuple[SearchDiagnostic, ...] = field(default_factory=tuple)
    pipeline_details: PipelineDetails | None = None


class RAGPipeline:
    """Deterministic RAG pipeline: retrieve → fuse → assemble → generate.

    Composes keyword and semantic search with RRF fusion, token-aware
    context assembly, and LLM answer generation.
    """

    def __init__(
        self,
        repository: ARIELRepository,
        config: ARIELConfig,
        embedder_loader: Callable[[], BaseEmbeddingProvider],
        *,
        fusion_k: int = 60,
        max_context_chars: int = 12000,
        max_chars_per_entry: int = 2000,
        prompt_template: str | None = None,
    ) -> None:
        self._repository = repository
        self._config = config
        self._embedder_loader = embedder_loader
        self._fusion_k = fusion_k
        self._max_context_chars = max_context_chars
        self._max_chars_per_entry = max_chars_per_entry
        self._prompt_template = prompt_template or RAG_PROMPT_TEMPLATE

    # === Public API ===

    async def execute(
        self,
        query: str,
        *,
        max_results: int = 10,
        similarity_threshold: float | None = None,
        start_date: Any | None = None,
        end_date: Any | None = None,
        author: str | None = None,
        source_system: str | None = None,
        temperature: float | None = None,
    ) -> RAGResult:
        """Execute the RAG pipeline.

        Args:
            query: Natural language query
            max_results: Maximum entries to retrieve per search
            similarity_threshold: Minimum similarity for semantic search
            start_date: Filter entries after this time
            end_date: Filter entries before this time
            author: Filter by author name (ILIKE match)
            source_system: Filter by source system (exact match)
            temperature: Override LLM temperature (None uses config default)

        Returns:
            RAGResult with answer, entries, and citations
        """
        if not query.strip():
            pd = PipelineDetails(
                pipeline_type="rag",
                rag_stats=RAGStageStats(),
                step_summary="Empty query — no retrieval performed",
            )
            return RAGResult(answer=_NO_CONTEXT_ANSWER, pipeline_details=pd)

        diags: list[SearchDiagnostic] = []

        # 1. Retrieve — run enabled search modules in parallel
        keyword_results, semantic_results, retrieve_diags = await self._retrieve(
            query,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            start_date=start_date,
            end_date=end_date,
            author=author,
            source_system=source_system,
        )
        diags.extend(retrieve_diags)

        kw_count = len(keyword_results)
        sem_count = len(semantic_results)

        # 2. Fuse — RRF if both returned results, otherwise use whichever returned
        entries = self._fuse(keyword_results, semantic_results, max_results)

        retrieval_count = len(entries)

        if not entries:
            pd = PipelineDetails(
                pipeline_type="rag",
                rag_stats=RAGStageStats(
                    keyword_retrieved=kw_count,
                    semantic_retrieved=sem_count,
                    fused_count=0,
                    context_included=0,
                ),
                step_summary=f"Retrieved {kw_count} keyword + {sem_count} semantic, 0 after fusion",
            )
            return RAGResult(
                answer=_NO_CONTEXT_ANSWER,
                retrieval_count=0,
                diagnostics=tuple(diags),
                pipeline_details=pd,
            )

        # 3. Assemble — build context window
        context_text, included_entries, truncated = self._assemble_context(entries)
        if truncated:
            diags.append(
                SearchDiagnostic(
                    level=DiagnosticLevel.INFO,
                    source="rag.assemble",
                    message="Context was truncated to fit token limits",
                )
            )

        # 4. Generate — LLM call
        answer, gen_diag = await self._generate(query, context_text, temperature=temperature)
        if gen_diag is not None:
            diags.append(gen_diag)

        # 5. Detect which context entries are cited in the answer
        context_ids = [e["entry_id"] for e in included_entries]
        citations = self._find_cited_ids(answer, context_ids)
        if not citations:
            citations = context_ids  # fallback: all context entries

        # Build pipeline details
        pd = PipelineDetails(
            pipeline_type="rag",
            rag_stats=RAGStageStats(
                keyword_retrieved=kw_count,
                semantic_retrieved=sem_count,
                fused_count=retrieval_count,
                context_included=len(included_entries),
                context_truncated=truncated,
            ),
            step_summary=(
                f"Retrieved {kw_count} keyword + {sem_count} semantic, "
                f"{retrieval_count} after fusion, {len(included_entries)} in context"
            ),
        )

        return RAGResult(
            answer=answer,
            entries=tuple(included_entries),
            citations=tuple(citations),
            retrieval_count=retrieval_count,
            context_truncated=truncated,
            diagnostics=tuple(diags),
            pipeline_details=pd,
        )

    # === Retrieval ===

    async def _retrieve(
        self,
        query: str,
        *,
        max_results: int,
        similarity_threshold: float | None,
        start_date: Any | None,
        end_date: Any | None,
        author: str | None = None,
        source_system: str | None = None,
    ) -> tuple[
        list[tuple[EnhancedLogbookEntry, float, list[str]]],
        list[tuple[EnhancedLogbookEntry, float]],
        list[SearchDiagnostic],
    ]:
        """Run keyword and/or semantic search in parallel.

        Uses the pipeline's configured retrieval_modules list if available,
        otherwise falls back to checking which search modules are enabled.

        Returns:
            Tuple of (keyword_results, semantic_results, diagnostics)
        """
        tasks: dict[str, Any] = {}
        diags: list[SearchDiagnostic] = []

        # Determine which retrieval modules to use
        retrieval_modules = self._config.get_pipeline_retrieval_modules("rag")

        if "keyword" in retrieval_modules and self._config.is_search_module_enabled("keyword"):
            from osprey.services.ariel_search.search.keyword import keyword_search

            tasks["keyword"] = keyword_search(
                query,
                self._repository,
                self._config,
                max_results=max_results,
                start_date=start_date,
                end_date=end_date,
                author=author,
                source_system=source_system,
            )

        if "semantic" in retrieval_modules and self._config.is_search_module_enabled("semantic"):
            from osprey.services.ariel_search.search.semantic import semantic_search

            try:
                embedder = self._embedder_loader()
            except Exception as e:
                logger.warning(f"Failed to load embedder, skipping semantic search: {e}")
                diags.append(
                    SearchDiagnostic(
                        level=DiagnosticLevel.WARNING,
                        source="rag.retrieve.semantic",
                        message=f"Failed to load embedder, skipping semantic search: {e}",
                        category="embedding",
                    )
                )
                embedder = None

            if embedder is not None:
                tasks["semantic"] = semantic_search(
                    query,
                    self._repository,
                    self._config,
                    embedder,
                    max_results=max_results,
                    similarity_threshold=similarity_threshold,
                    start_date=start_date,
                    end_date=end_date,
                    author=author,
                    source_system=source_system,
                )

        keyword_results: list[tuple[EnhancedLogbookEntry, float, list[str]]] = []
        semantic_results: list[tuple[EnhancedLogbookEntry, float]] = []

        if not tasks:
            return keyword_results, semantic_results, diags

        # Run in parallel
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        keys = list(tasks.keys())

        failed_keys: list[str] = []
        for key, result in zip(keys, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(f"RAG {key} retrieval failed: {result}")
                diags.append(
                    SearchDiagnostic(
                        level=DiagnosticLevel.ERROR,
                        source=f"rag.retrieve.{key}",
                        message=f"{key.title()} retrieval failed: {result}",
                        category="search",
                    )
                )
                failed_keys.append(key)
            elif key == "keyword":
                keyword_results = result
            elif key == "semantic":
                semantic_results = result

        if len(failed_keys) == len(tasks) and len(tasks) > 0:
            diags.append(
                SearchDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    source="rag.retrieve",
                    message="All retrieval modules failed",
                    category="search",
                )
            )

        return keyword_results, semantic_results, diags

    # === Fusion ===

    def _fuse(
        self,
        keyword_results: list[tuple[EnhancedLogbookEntry, float, list[str]]],
        semantic_results: list[tuple[EnhancedLogbookEntry, float]],
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Fuse keyword and semantic results using Reciprocal Rank Fusion.

        When only one source returned results, passes them through directly.
        When both returned results, combines using RRF scoring.

        Returns:
            List of entry dicts sorted by fused score, limited to max_results.
        """
        if not keyword_results and not semantic_results:
            return []

        # Convert to uniform format: {entry_id: (entry_dict, score)}
        scored: dict[str, tuple[dict[str, Any], float]] = {}
        k = self._fusion_k

        # Score keyword results
        for rank, (entry, _score, _highlights) in enumerate(keyword_results):
            entry_id = entry["entry_id"]
            rrf_score = 1.0 / (k + rank + 1)
            if entry_id in scored:
                existing_entry, existing_score = scored[entry_id]
                scored[entry_id] = (existing_entry, existing_score + rrf_score)
            else:
                scored[entry_id] = (dict(entry), rrf_score)

        # Score semantic results
        for rank, (entry, _similarity) in enumerate(semantic_results):
            entry_id = entry["entry_id"]
            rrf_score = 1.0 / (k + rank + 1)
            if entry_id in scored:
                existing_entry, existing_score = scored[entry_id]
                scored[entry_id] = (existing_entry, existing_score + rrf_score)
            else:
                scored[entry_id] = (dict(entry), rrf_score)

        # Sort by fused score descending
        sorted_items = sorted(scored.values(), key=lambda x: x[1], reverse=True)

        return [{**entry, "_score": score} for entry, score in sorted_items[:max_results]]

    # === Context Assembly ===

    def _assemble_context(
        self,
        entries: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], bool]:
        """Assemble entries into a context string for the LLM.

        Uses ENTRY #id | timestamp | Author: name format per spec Section 5.5.2.

        Returns:
            Tuple of (context_text, included_entries, truncated)
        """
        parts: list[str] = []
        included: list[dict[str, Any]] = []
        total_chars = 0
        truncated = False

        for entry in entries:
            formatted = self._format_entry(entry)

            if total_chars + len(formatted) > self._max_context_chars:
                remaining = self._max_context_chars - total_chars
                if remaining > 100:
                    formatted = formatted[:remaining] + "..."
                    parts.append(formatted)
                    total_chars += len(formatted)
                    included.append(entry)
                truncated = True
                break

            parts.append(formatted)
            total_chars += len(formatted)
            included.append(entry)

        return "\n---\n".join(parts), included, truncated

    def _format_entry(self, entry: dict[str, Any]) -> str:
        """Format a single entry for RAG context.

        Uses ENTRY #id | timestamp | Author: name format.
        """
        metadata = entry.get("metadata", {})
        title = metadata.get("title", "")
        author = entry.get("author", "Unknown")
        timestamp = entry.get("timestamp")
        timestamp_str = timestamp.isoformat() if timestamp else "Unknown"

        header = f"ENTRY #{entry['entry_id']} | {timestamp_str} | Author: {author}"
        if title:
            header += f" | {title}"

        content = entry.get("raw_text", "")
        if len(content) > self._max_chars_per_entry:
            content = content[: self._max_chars_per_entry] + "..."

        return f"{header}\n{content}\n"

    # === LLM Generation ===

    async def _generate(
        self,
        query: str,
        context: str,
        *,
        temperature: float | None = None,
    ) -> tuple[str, SearchDiagnostic | None]:
        """Generate an answer using the LLM.

        Args:
            query: Original query
            context: Assembled context string
            temperature: Override temperature (None uses config default)

        Returns:
            Tuple of (answer_text, optional_diagnostic)
        """
        prompt = self._prompt_template.format(context=context, question=query)

        try:
            from osprey.models.completion import get_chat_completion

            llm_kwargs: dict[str, Any] = {
                "provider": self._config.reasoning.provider,
                "model_id": self._config.reasoning.model_id,
                "temperature": temperature
                if temperature is not None
                else self._config.reasoning.temperature,
            }

            response = get_chat_completion(message=prompt, **llm_kwargs)

            if isinstance(response, str):
                answer = response
            else:
                answer = str(response)

            return (answer if answer else "I was unable to generate an answer.", None)

        except ImportError:
            logger.warning("osprey.models.completion not available for RAG")
            return (
                "LLM not available for answer generation.",
                SearchDiagnostic(
                    level=DiagnosticLevel.WARNING,
                    source="rag.generate",
                    message="LLM module not available for answer generation",
                ),
            )
        except Exception as e:
            logger.error(f"LLM call failed for RAG: {e}")
            return (
                f"Error generating answer: {e}",
                SearchDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    source="rag.generate",
                    message=f"LLM call failed: {e}",
                ),
            )

    # === Citation Detection ===

    @staticmethod
    def _find_cited_ids(text: str, candidate_ids: list[str]) -> list[str]:
        """Find which candidate entry IDs appear in the answer text.

        Checks each candidate ID for presence in the text (case-sensitive).
        Returns IDs in candidate order (not text order).

        Args:
            text: LLM-generated answer text
            candidate_ids: Entry IDs from the retrieval context

        Returns:
            List of candidate IDs that appear as substrings in the text.
        """
        if not text or not candidate_ids:
            return []
        return [eid for eid in candidate_ids if eid in text]


__all__ = ["RAGPipeline", "RAGResult"]
