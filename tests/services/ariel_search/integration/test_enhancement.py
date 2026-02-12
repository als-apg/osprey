"""Integration tests for ARIEL enhancement modules with real Ollama.

Tests the enhancement pipeline with real embedding generation (INT-007).

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def is_ollama_available() -> bool:
    """Check if Ollama is available for tests."""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.requires_ollama
class TestEnhancementWithOllama:
    """Test enhancement modules with real Ollama service."""

    async def test_text_embedding_generation(self, repository, migrated_pool, seed_entry_factory):
        """TextEmbeddingModule generates embeddings with correct dimensions.

        Steps:
        1. Create TextEmbeddingModule with nomic-embed-text config
        2. Generate embedding for sample entry
        3. Verify embedding has 768 dimensions
        4. Verify embedding stored in correct table
        """
        if not is_ollama_available():
            pytest.skip("Ollama not available - run 'ollama pull nomic-embed-text'")

        from osprey.services.ariel_search.enhancement.text_embedding import (
            TextEmbeddingModule,
        )

        # Create test entry
        entry = seed_entry_factory(
            entry_id="enhance-embed-001",
            raw_text="The storage ring current dropped to 480mA after a vacuum event.",
        )
        await repository.upsert_entry(entry)

        # Create and configure embedding module
        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [
                    {"name": "nomic-embed-text", "dimension": 768, "max_input_tokens": 8192}
                ],
                "provider": {"base_url": "http://localhost:11434"},
            }
        )

        # Generate embedding using connection
        async with migrated_pool.connection() as conn:
            await module.enhance(entry, conn)

        # Verify embedding was stored
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT embedding FROM text_embeddings_nomic_embed_text
                WHERE entry_id = 'enhance-embed-001'
            """)
            row = await result.fetchone()

        assert row is not None, "Embedding was not stored"
        embedding = row[0]

        # Check dimension (pgvector stores as string or list)
        if isinstance(embedding, str):
            dim = embedding.count(",") + 1
        else:
            dim = len(embedding)

        assert dim == 768, f"Expected 768 dimensions, got {dim}"

    async def test_text_embedding_health_check(self):
        """TextEmbeddingModule health check verifies Ollama connectivity."""
        if not is_ollama_available():
            pytest.skip("Ollama not available")

        from osprey.services.ariel_search.enhancement.text_embedding import (
            TextEmbeddingModule,
        )

        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [{"name": "nomic-embed-text", "dimension": 768}],
                "provider": {"base_url": "http://localhost:11434"},
            }
        )

        healthy, message = await module.health_check()
        assert healthy is True
        assert "connected" in message.lower() or "ok" in message.lower()

    async def test_text_embedding_handles_empty_text(
        self, repository, migrated_pool, seed_entry_factory
    ):
        """TextEmbeddingModule skips entries with empty text."""
        if not is_ollama_available():
            pytest.skip("Ollama not available")

        from osprey.services.ariel_search.enhancement.text_embedding import (
            TextEmbeddingModule,
        )

        # Create entry with empty text
        entry = seed_entry_factory(
            entry_id="enhance-empty-001",
            raw_text="   ",  # Whitespace only
        )
        await repository.upsert_entry(entry)

        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [{"name": "nomic-embed-text", "dimension": 768}],
                "provider": {"base_url": "http://localhost:11434"},
            }
        )

        # Should not raise, just skip
        async with migrated_pool.connection() as conn:
            await module.enhance(entry, conn)

        # Should not have stored embedding
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT COUNT(*) FROM text_embeddings_nomic_embed_text
                WHERE entry_id = 'enhance-empty-001'
            """)
            row = await result.fetchone()
            assert row[0] == 0

    async def test_text_embedding_truncates_long_text(
        self, repository, migrated_pool, seed_entry_factory
    ):
        """TextEmbeddingModule truncates text exceeding max tokens."""
        if not is_ollama_available():
            pytest.skip("Ollama not available")

        from osprey.services.ariel_search.enhancement.text_embedding import (
            TextEmbeddingModule,
        )

        # Create entry with very long text
        long_text = "Beam status update. " * 10000  # ~200k chars
        entry = seed_entry_factory(
            entry_id="enhance-long-001",
            raw_text=long_text,
        )
        await repository.upsert_entry(entry)

        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [
                    {
                        "name": "nomic-embed-text",
                        "dimension": 768,
                        "max_input_tokens": 1000,  # Low limit for test
                    }
                ],
                "provider": {"base_url": "http://localhost:11434"},
            }
        )

        # Should not raise despite long text (truncates internally)
        async with migrated_pool.connection() as conn:
            await module.enhance(entry, conn)

        # Should have stored embedding
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT embedding FROM text_embeddings_nomic_embed_text
                WHERE entry_id = 'enhance-long-001'
            """)
            row = await result.fetchone()
            assert row is not None

    async def test_cleanup(self, migrated_pool):
        """Clean up enhancement test data."""
        async with migrated_pool.connection() as conn:
            # Clean embeddings first
            try:
                await conn.execute("""
                    DELETE FROM text_embeddings_nomic_embed_text
                    WHERE entry_id LIKE 'enhance-%'
                """)
            except Exception:
                pass

            # Clean entries
            await conn.execute("""
                DELETE FROM enhanced_entries
                WHERE entry_id LIKE 'enhance-%'
            """)


@pytest.mark.requires_ollama
class TestMultipleEmbeddingModels:
    """Test enhancement with multiple embedding models."""

    async def test_multiple_models_generate_embeddings(
        self, repository, migrated_pool, seed_entry_factory, integration_ariel_config
    ):
        """Enhancement module can use multiple embedding models."""
        if not is_ollama_available():
            pytest.skip("Ollama not available")

        # Check if all-minilm is available
        try:
            import requests

            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                if "all-minilm" not in model_names:
                    pytest.skip("all-minilm not available - run 'ollama pull all-minilm'")
        except Exception:
            pytest.skip("Cannot check Ollama models")

        from osprey.services.ariel_search.enhancement.text_embedding import (
            TextEmbeddingModule,
        )
        from osprey.services.ariel_search.enhancement.text_embedding.migration import (
            TextEmbeddingMigration,
        )

        # Ensure the all-minilm table exists (nomic-embed-text created by default migration)
        migration = TextEmbeddingMigration(models=[("all-minilm", 384)])
        async with migrated_pool.connection() as conn:
            await migration.up(conn)

        entry = seed_entry_factory(
            entry_id="enhance-multi-001",
            raw_text="RF cavity frequency adjusted by 10 kHz for optimal beam lifetime.",
        )
        await repository.upsert_entry(entry)

        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [
                    {"name": "nomic-embed-text", "dimension": 768},
                    {"name": "all-minilm", "dimension": 384},
                ],
                "provider": {"base_url": "http://localhost:11434"},
            }
        )

        async with migrated_pool.connection() as conn:
            await module.enhance(entry, conn)

        # Check both tables have embeddings
        async with migrated_pool.connection() as conn:
            # nomic-embed-text (768 dims)
            result1 = await conn.execute("""
                SELECT embedding FROM text_embeddings_nomic_embed_text
                WHERE entry_id = 'enhance-multi-001'
            """)
            row1 = await result1.fetchone()

            # all-minilm (384 dims)
            result2 = await conn.execute("""
                SELECT embedding FROM text_embeddings_all_minilm
                WHERE entry_id = 'enhance-multi-001'
            """)
            row2 = await result2.fetchone()

        assert row1 is not None, "nomic-embed-text embedding not stored"
        assert row2 is not None, "all-minilm embedding not stored"

    async def test_cleanup(self, migrated_pool):
        """Clean up multi-model test data."""
        async with migrated_pool.connection() as conn:
            for table in [
                "text_embeddings_nomic_embed_text",
                "text_embeddings_all_minilm",
            ]:
                try:
                    await conn.execute(f"""
                        DELETE FROM {table}
                        WHERE entry_id LIKE 'enhance-multi-%'
                    """)  # noqa: S608
                except Exception:
                    pass

            await conn.execute("""
                DELETE FROM enhanced_entries
                WHERE entry_id LIKE 'enhance-multi-%'
            """)


class TestEnhancementWithoutOllama:
    """Tests that work without Ollama (skip gracefully)."""

    async def test_enhancement_module_health_check_when_unavailable(self):
        """Health check returns false when Ollama unavailable."""
        from osprey.services.ariel_search.enhancement.text_embedding import (
            TextEmbeddingModule,
        )

        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [{"name": "nomic-embed-text", "dimension": 768}],
                "provider": {"base_url": "http://localhost:99999"},  # Invalid port
            }
        )

        healthy, message = await module.health_check()
        assert healthy is False

    async def test_enhancement_module_configured_correctly(self):
        """Enhancement module configures models from dict."""
        from osprey.services.ariel_search.enhancement.text_embedding import (
            TextEmbeddingModule,
        )

        module = TextEmbeddingModule()
        module.configure(
            {
                "models": [
                    {"name": "model-a", "dimension": 512},
                    {"name": "model-b", "dimension": 768},
                ],
            }
        )

        assert len(module._models) == 2
        assert module._models[0]["name"] == "model-a"
        assert module._models[1]["dimension"] == 768
