The pgvector IVFFlat `lists` count for ARIEL's embedding index is now a config
key, `ariel.enhancement_modules.text_embedding.index_lists` (default 224, the
previous fixed value). The rule of thumb is roughly one list per 1000 entries.
The value is fixed when the index is created, so changing it later means
dropping the index and recreating it.
