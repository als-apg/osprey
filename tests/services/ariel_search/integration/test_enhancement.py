"""Integration tests for ARIEL enhancement modules with real Ollama.

Tests the enhancement pipeline with real embedding generation (INT-007).

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

import pytest

# xdist_group("docker"): pins every container-starting test file onto one worker, so
# a run has a single testcontainers session and a single ryuk reaper -- concurrent
# reaper starts race the Docker daemon's port mapper. It also serializes the shared
# database: the session ``database_url`` fixture prefers a running dev Postgres with
# ONE shared ``ariel_test`` database over a per-worker container, so parallel workers
# would otherwise collide on migrations/seed/truncate.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]


#: Rows the embedding module enhances, the multi-model ones among them.
ENHANCE_PREFIX = "enhance-"


def is_ollama_available() -> bool:
    """Check if Ollama is available for tests."""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.requires_ollama
@pytest.mark.usefixtures("litellm_callback_pool")
class TestEnhancementWithOllama:
    """Test enhancement modules with real Ollama service."""

    async def test_text_embedding_generation(
        self, repository, migrated_pool, seed_entry_factory, seeded_prefixes
    ):
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
            entry_id=f"{ENHANCE_PREFIX}embed-001",
            raw_text="The storage ring current dropped to 480mA after a vacuum event.",
        )
        seeded_prefixes.add(ENHANCE_PREFIX)
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
        self, repository, migrated_pool, seed_entry_factory, seeded_prefixes
    ):
        """TextEmbeddingModule skips entries with empty text."""
        if not is_ollama_available():
            pytest.skip("Ollama not available")

        from osprey.services.ariel_search.enhancement.text_embedding import (
            TextEmbeddingModule,
        )

        # Create entry with empty text
        entry = seed_entry_factory(
            entry_id=f"{ENHANCE_PREFIX}empty-001",
            raw_text="   ",  # Whitespace only
        )
        seeded_prefixes.add(ENHANCE_PREFIX)
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
        self, repository, migrated_pool, seed_entry_factory, seeded_prefixes
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
            entry_id=f"{ENHANCE_PREFIX}long-001",
            raw_text=long_text,
        )
        seeded_prefixes.add(ENHANCE_PREFIX)
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


@pytest.mark.requires_ollama
@pytest.mark.usefixtures("litellm_callback_pool")
class TestMultipleEmbeddingModels:
    """Test enhancement with multiple embedding models."""

    @pytest.mark.usefixtures("integration_ariel_config")
    async def test_multiple_models_generate_embeddings(
        self, repository, migrated_pool, seed_entry_factory, seeded_prefixes
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
            entry_id=f"{ENHANCE_PREFIX}multi-001",
            raw_text="RF cavity frequency adjusted by 10 kHz for optimal beam lifetime.",
        )
        seeded_prefixes.add(ENHANCE_PREFIX)
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


#: Ollama endpoint the served-limit checks ask, and the model they ask about.
OLLAMA_URL = "http://localhost:11434"
SERVED_MODEL = "nomic-embed-text"


def _served_context_length() -> int:
    """Return the input window Ollama serves ``SERVED_MODEL`` with, or skip.

    Only ``/api/show`` is asked; no model is pulled.
    """
    try:
        import requests

        response = requests.post(f"{OLLAMA_URL}/api/show", json={"model": SERVED_MODEL}, timeout=5)
    except Exception as exc:
        pytest.skip(f"Ollama not reachable at {OLLAMA_URL}: {exc}")
    if response.status_code != 200:
        pytest.skip(f"Ollama does not serve {SERVED_MODEL} (/api/show {response.status_code})")
    model_info = response.json()["model_info"]
    return int(model_info[f"{model_info['general.architecture']}.context_length"])


@pytest.mark.requires_ollama
class TestServedInputLimit:
    """The presets' input limit is the served window, and the cut fits it."""

    @pytest.mark.parametrize("preset", ["control-assistant", "ariel-standalone"])
    async def test_preset_limit_is_the_served_window(self, preset):
        """A preset states the window the server reports, never a remembered number."""
        import importlib.resources

        import yaml

        served = _served_context_length()
        path = importlib.resources.files("osprey.profiles") / "presets" / f"{preset}.yml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["config"]
        models = config["ariel.enhancement_modules.text_embedding.models"]
        (entry,) = [m for m in models if m["name"] == SERVED_MODEL]

        assert entry["max_input_tokens"] == served

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("!?.,;:" * 4000, id="punctuation"),
            pytest.param(
                " ".join(f"SR{s:02d}C:BPM{b}:SA:X" for s in range(1, 13) for b in range(1, 9)) * 20,
                id="device-names",
            ),
            pytest.param("加速器光束電流安定" * 3000, id="cjk"),
            pytest.param("&lt;p&gt;Beam &amp; RF&lt;/p&gt; " * 1000, id="html-escaped"),
        ],
    )
    async def test_a_cut_entry_fits_without_server_truncation(self, text):
        """A cut text embeds with server truncation off, within the served window."""
        import requests

        from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
            fit_to_input_limit,
        )

        served = _served_context_length()
        assert len(text) >= 20_000

        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": SERVED_MODEL,
                "input": fit_to_input_limit(text, served),
                "truncate": False,
            },
            timeout=60,
        )

        assert response.status_code == 200, response.text
        assert response.json()["prompt_eval_count"] <= served


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
