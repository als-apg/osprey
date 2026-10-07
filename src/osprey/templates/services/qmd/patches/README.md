# qmd patches

The qmd sidecar image installs `@tobilu/qmd` at the version the Dockerfile pins
(`QMD_VERSION`) and then applies the patches in this directory to the installed
package's `dist/`, the compiled JavaScript that npm ships and that the daemon
runs. The Dockerfile names the set it applies in `OSPREY_QMD_PATCHES`, in order,
and fails the build if a listed patch is missing, if a file here is not listed,
or if any hunk does not apply exactly (`patch -p1 --fuzz=0`). The image reports
what it carries: `OSPREY_QMD_VERSION` in its identity file and the
`com.osprey.qmd.version` label read `<release>+osprey.<N>`, and the
`com.osprey.qmd.patches` label lists the patch files.

None of the patches changes the index format or the embedder, so an index built
by the plain release is served unchanged and no reindex follows from adding or
dropping one.

| Patch | What it changes |
|---|---|
| `0001-searchvec-hydrate-by-key.patch` | `searchVec` resolves its vector matches with `content_vectors.hash IN (...)`, which uses the table's `(hash, seq)` primary key, and keeps the matched chunks in JavaScript; the released code matched on the concatenated `hash \|\| '_' \|\| seq`, which scans every chunk row. Document bodies are read only for the results returned. Same rows, same order. |
| `0002-in-memory-vector-scan.patch` | `qmd mcp` loads the stored vectors once into shared memory and answers vector searches with an exact cosine scan over a pool of worker threads, instead of a sqlite-vec KNN query, which reads every vector out of SQLite pages per search. A collection-scoped search masks other collections out before it takes the top k, so a small collection is never crowded out of its own results by a large one sharing the index. sqlite-vec stays the storage of record: when another process (`qmd embed`, `qmd update`) commits a change, the next search goes to sqlite-vec while the copy reloads, so results are never stale. Costs 3 KiB of RAM per 768-dimension chunk. `QMD_VEC_SCAN=vec0` turns it off; `QMD_VEC_SCAN_THREADS` sets the pool size (default `min(16, CPUs)`). Applies after 0001. |
| `0003-natural-language-lex.patch` | A lex query written as a question (it contains a stopword or ends with `?`, and uses no quotes or `-negation`) is searched as an OR of its content words, without prefix expansion, ranked by bm25. The released parse ANDs every word as a prefix, so a question matches almost nothing and the prefixes of function words make it slow. Keyword queries and the explicit syntax keep their parse. Unlike 0001 and 0002 this changes results, not only their cost: the keyword leg of a hybrid query returns ranked candidates for a question where it used to return none. |

Each patch file opens with a short description; the diff below it is against
the published package, with paths relative to the package root.

## Upstream

Each patch mirrors a change proposed to qmd itself (https://github.com/tobi/qmd):

- 0001: the same lookup by key, with bodies read after the limit, is part of the
  partitioned vector index on qmd's `main` branch
  (https://github.com/tobi/qmd/pull/1024), not yet in a release.
- 0002: https://github.com/tobi/qmd/pull/1041
- 0003: https://github.com/tobi/qmd/pull/1040

## Dropping the carriage

When `QMD_VERSION` is bumped to a release that contains a patch's change, delete
that patch file, remove it from `OSPREY_QMD_PATCHES`, and set
`OSPREY_QMD_BUILD` to the new release (with a `+osprey.<N>` suffix only while
patches remain). A patch that no longer applies to the new release fails the
build, so the carriage cannot silently outlive the version it was written for.
