`services.qmd.corpora` declares further corpora, each served by its own sidecar
at `services.qmd.port` + 2 onwards. A corpus is `managed` (the sidecar indexes
its `source`) or `prebuilt` (the sidecar serves an index built elsewhere, such
as on a GPU host, from `index_dir`). A new `qmd` health category shows each
sidecar's documents and how many still have no vectors.
