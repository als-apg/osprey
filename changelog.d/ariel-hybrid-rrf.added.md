ARIEL: hybrid search takes an opt-in `ariel.search_modules.hybrid.settings.strategy:
rrf`, which fuses qmd's BM25 ranking with ARIEL's configured semantic provider by
reciprocal-rank fusion and never calls qmd's reranker. `qmd` stays the default.
