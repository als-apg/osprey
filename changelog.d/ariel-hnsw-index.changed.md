ARIEL's embedding index is now an HNSW index over the pgvector column. An
existing deployment is moved across by `osprey ariel migrate`, which drops the
old index and builds the new one; on a large corpus that build is the cost of
the move.
