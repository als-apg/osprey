"""ARIEL RAG answer prompt template.

This module provides the prompt template for RAG (Retrieval-Augmented Generation)
answer generation.

See 02_SEARCH_MODULES.md Section 5 for specification.
"""

# RAG prompt template - uses [#entry_id] citation format per spec Section 5.6
RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about facility operations based on logbook entries.

Use the following logbook entries as context to answer the question. Each entry has an ID, timestamp, author, and content.

**Important:**
- Only answer based on the information provided in the entries
- If the entries don't contain relevant information, say "I don't have enough information to answer this question based on the available logbook entries"
- When referencing information, cite the entry ID in brackets with a hash, e.g., [#12345]
- Be concise but thorough

**Context (Logbook Entries):**
{context}

**Question:** {question}

**Answer:**"""

__all__ = ["RAG_PROMPT_TEMPLATE"]
