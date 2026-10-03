.. _ariel-picture-search:

===================================
Picture Captions and Picture Search
===================================

ARIEL can read the pictures attached to logbook entries in two ways, each an
:ref:`enhancement module <Enhancement Pipeline>` of its own:

- ``image_caption`` asks a vision model to describe each copied picture and list
  its visible text. The caption is folded into the entry's searchable text, so
  keyword, semantic and hybrid search find an entry by what its plots show.
- ``image_embedding`` turns each copied picture into a vector with a multimodal
  embedding model served by a site-run ``llama-server``. ``hybrid`` search then
  also ranks pictures against the query text, so an entry known only by its
  pictures can still come back.

Both read the picture bytes ARIEL copied at ingest
(``ariel.attachments.copy_on_ingest``), so they see only copied pictures.
The agent's ``attachment_view`` tool, which returns one stored picture to the
agent, needs neither module. See :doc:`/reference/contracts/ariel` for the
tool and result contract.


On by default, skipped when unavailable
========================================

The ``control-assistant`` and ``ariel-standalone`` presets turn captions,
picture search and the view tool on. Like semantic search without Ollama, each
picture module degrades gracefully: it runs when its server and model answer,
and otherwise it is skipped while the rest of ARIEL keeps working.

A deployment **without the caption model** (Ollama not running, or
``qwen3-vl:4b`` not pulled) sees:

- ``osprey ariel status`` printing a line that names the skipped module and
  why, for example
  ``image_caption: skipped, model qwen3-vl:4b not available on ollama (pull it, or set ariel.enhancement_modules.image_caption.enabled: false)``;
- entries searchable by their text alone, as before captions existed.

A deployment **without llama-server** sees:

- ``osprey ariel status`` printing, for example,
  ``image_embedding: skipped, llama-cpp not reachable at http://localhost:8080 (start llama-server, see the picture-search guide, or set ariel.enhancement_modules.image_embedding.enabled: false)``;
- ``hybrid_search`` answering on text only, with the diagnostic
  "Picture search unavailable --- results are matched on text only, so entries
  known only by their pictures are missing.";
- ``capabilities.attachments.picture_search_unavailable`` set to the reason
  (``unreachable``, ``model``, ``auth`` or ``config``) once the picture lane has
  failed.

To turn a module off, set its ``enabled`` key to ``false`` in ``profile.yml``:

.. code-block:: yaml

   config:
     ariel.enhancement_modules.image_caption.enabled: false    # no captions
     ariel.enhancement_modules.image_embedding.enabled: false  # no picture search
     ariel.attachments.view.enabled: false                     # no attachment_view tool


Picture formats
===============

The format table below is the one ARIEL uses everywhere: what it copies,
renders, captions, embeds and shows. *Accepted* formats are rendered and
viewable; *reserved* formats are recognised by MIME type and recorded as
skipped.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Format
     - Status
     - MIME types
   * - ``png``
     - accepted
     - ``image/png``
   * - ``jpeg``
     - accepted
     - ``image/jpeg``
   * - ``gif``
     - accepted
     - ``image/gif``
   * - ``webp``
     - accepted
     - ``image/webp``
   * - ``bmp``
     - accepted
     - ``image/bmp``
   * - ``tiff``
     - accepted
     - ``image/tiff``
   * - ``svg``
     - reserved
     - ``image/svg+xml``
   * - ``pdf``
     - reserved
     - ``application/pdf``
   * - ``text``
     - reserved
     - ``text/plain``
   * - ``heif``
     - reserved
     - ``image/heic``, ``image/avif``
   * - ``video``
     - reserved
     - ``video/mp4``, ``video/quicktime``, ``video/webm``, ``video/x-msvideo``
   * - ``audio``
     - reserved
     - ``audio/mp4``, ``audio/mpeg``, ``audio/flac``, ``audio/ogg``, ``audio/wav``
   * - ``archive``
     - reserved
     - ``application/zip``, ``application/gzip``, ``application/x-bzip2``,
       ``application/x-xz``, ``application/x-7z-compressed``,
       ``application/vnd.rar``

``capabilities.attachments.formats`` reports the same two lists at run time.


Captions
========

Captions come from any chat provider with a vision-capable model. The presets
use Ollama with ``qwen3-vl:4b``:

.. code-block:: bash

   ollama pull qwen3-vl:4b

The provider and model are the module's own
(``ariel.enhancement_modules.image_caption.provider`` and
``.model.model_id``), never the deployment's main model. On a CPU, a local
vision model can spend minutes on one picture; the measured values are in
:ref:`ariel-picture-search-measurements`, and
``image_caption.timeout_seconds`` (default 1320) is sized from them.


The llama-server for picture search
===================================

OSPREY ships no image or service for the embedding server: the site builds and
runs ``llama-server`` itself, and the ``llama-cpp`` provider talks to it (see
the provider table in :doc:`/how-to/llm-providers/configure-providers`). It
serves both picture vectors at enhancement time and query vectors at search
time.

Build
-----

Use llama.cpp tag ``b11277`` (commit
``eae11d2217fe9225d1aaba48773b6cca45ae4de9``). It contains PR #29556, the
multimodal embedding support picture search depends on; earlier releases,
including Homebrew's ``llama.cpp 0.5.0``, do not.

.. code-block:: bash

   git clone --depth 1 --branch b11277 https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   cmake -B build -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release   # add -DGGML_METAL=ON on a Mac
   cmake --build build --target llama-server

Model files
-----------

Both files come from the Hugging Face repository
``DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF`` at revision
``6a1b927414664e0e17dd379913e3416a1ae1b48d``. The vectors depend on both, so
check them:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - File
     - Bytes
     - sha256
   * - ``Qwen.Qwen3-VL-Embedding-2B.Q4_K_M.gguf``
     - 1107410528
     - ``42a4ebc629ecc6514649e12b1529b857f54900273bb854f853c970fb90edd09d``
   * - ``mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf``
     - 819395136
     - ``3f89a7768ffa6606935319f71bf56bb71871249ba549bf1080a0caea7a088613``

.. code-block:: bash

   sha256sum Qwen.Qwen3-VL-Embedding-2B.Q4_K_M.gguf mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf

Command
-------

.. code-block:: bash

   llama-server --embedding --pooling last -m Qwen.Qwen3-VL-Embedding-2B.Q4_K_M.gguf --mmproj mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf --alias qwen3-vl-embedding-2b --host 127.0.0.1 --port 8080 --no-webui --no-slots -c 8192 -b 2048 -ub 2048 -np 2 --image-max-tokens 256

Add ``--n-gpu-layers 99`` on a GPU build. The other flags matter as follows:

- ``--alias qwen3-vl-embedding-2b`` is the model id the server advertises on
  ``/v1/models``. It must equal ``ariel.enhancement_modules.image_embedding.model``
  in the profile, or ``osprey ariel status`` reports the module skipped with
  reason ``model``.
- ``--image-max-tokens 256`` caps each picture at 256 vision tokens, about
  512x512 px of area for Qwen3-VL. On a CPU-only server, a query that arrives
  while a picture is being embedded waits about one picture time; without the
  cap the measured query p95 under bulk embedding failed the 2 s target, with
  it the p95 passes (see :ref:`ariel-picture-search-measurements`).
- ``--host 127.0.0.1`` binds the server to the loopback interface. See the
  next section for why.
- The command passes no ``--media-path``, so the server reads no local files.

Why the server listens on localhost only
----------------------------------------

Build ``b11277`` has no switch to disable remote image-URL fetch. Its
``handle_media`` function (``tools/server/server-common.cpp:1088-1102``)
downloads any ``http`` or ``https`` URL a request names as an ``image_url``,
up to 10 MiB with a 10 s timeout; only ``file://`` URLs are gated, by
``--media-path``. A request that names an ``http`` URL makes the server fetch
it. Probed live against ``b11277``, the server sent this request to the URL a
client named:

.. code-block:: text

   GET /orbit_kick.png HTTP/1.1
   Host: 127.0.0.1:18999
   User-Agent: llama-cpp/b1-eae11d2
   Connection: close
   Accept: */*

Anyone who can reach the server can therefore make it issue requests from the
host's network position. Binding to ``127.0.0.1`` limits that to processes on
the same host. OSPREY itself sends only ``data:`` URLs.

How containers reach it
-----------------------

The ``llama-cpp`` provider's default address is ``http://localhost:8080``;
``LLAMA_CPP_HOST`` overrides it.

- **Docker Desktop (macOS, Windows).** A container reaches a server on the
  host's loopback through the same fallback table OSPREY uses for Ollama: a
  server configured as ``localhost`` is also tried as ``host.docker.internal``
  from a Docker container (``host.containers.internal`` from a Podman one). Nothing needs to be configured.
- **Linux host.** A localhost-bound server is reachable only from containers
  on host networking. Run ``ariel-sync`` with host networking by setting
  ``services.ariel_sync.network: host``, and do the same for any
  bridge-networked container that serves ``hybrid_search``; web terminals
  already use host networking. In ``profile.yml`` that is ``network: host``
  under the service's ``config:`` key:

  .. code-block:: yaml

     services:
       ariel_sync:
         config:
           network: host

  On the default bridge network, ``osprey ariel status`` inside ``ariel-sync``
  reports ``image_embedding`` unreachable and
  ``picture_search_unavailable: "unreachable"``. See
  :doc:`/how-to/deploy-project/networking` for what host networking changes.


Fusion of picture and text results
==================================

``hybrid`` search merges the picture lane into the text results with two
thresholds: a picture match below a cosine similarity of ``0.45``
(``min_similarity``) is never trusted, and an entry found only through its
pictures is admitted only within ``0.08`` (``relative_margin``) of the best
picture match. **These defaults are uncalibrated**: no labelled set of
logbook pictures and queries existed to calibrate them against, so no set size
or precision can be stated. The one check they pass is that ``0.45`` is below
the 1024-dimension cosine (``0.540``) between the query "orbit kick near BPM 7"
and the matching probe picture in the recorded fixtures.
``scripts/benchmark/fusion_calibration.py --set <set.json> --dsn <store>
--threshold <precision>`` reruns the calibration once a labelled set exists.


.. _ariel-picture-search-measurements:

Measured values
===============

The values below were measured with the command above on one native x86_64
Linux host, CPU only (llama-server built without CUDA and started without
GPU access, Ollama with no GPU visible), while other users kept the host
lightly loaded (load average 1.3--2.9 on 64 threads). They describe that host;
a different CPU gives different times.

**Host:** ``uname -m`` ``x86_64``; CPU ``AMD EPYC 7313 16-Core Processor``
(2 sockets x 16 cores x 2 threads, 64 logical CPUs); 503 GiB RAM; docker host
architecture ``x86_64``; Rocky Linux 8.10, docker 28.0.4. llama-server ran
with 32 threads and 2 slots.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Value
     - Result
   * - CPU time per picture (s/picture, 1024x768 rendition)
     - 3.77 s mean (771 tokens) without the cap; 1.14 s (237 tokens) with
       ``--image-max-tokens 256``
   * - Pictures per hour (derived, 3600 / s/picture)
     - about 3,150 with the cap; about 950 without
   * - llama-server peak RSS
     - 4.74 GiB with the cap; 7.57 GiB without; 3.44 GiB loaded and idle
   * - Idle query embed
     - p50 0.059 s, p95 0.072 s
   * - Query p95 under bulk ``image_embedding`` (2 s target)
     - 3.83 s without the cap (fails); 0.94 s with ``--image-max-tokens 256``
       (passes). The latency gate failed, so the documented command carries
       the cap
   * - Query timeout chosen
     - 5 s (``max(5, 2 x 0.94)``); ``image_embedding.timeout_seconds`` 120
   * - Render worker ``VmSize``
     - 30.6 MiB after start-up (target: at most 512 MiB), under a 1 GiB address-space
       limit; a 48 Mpx JPEG renders in 0.43 s with a peak of 119 MiB
   * - ``qwen3-vl:4b`` with ``think:false`` (Ollama, CPU)
     - 75.7 s, 199.4 s and 119.4 s per caption, mean 131.5 s; resident 4.25
       GiB. ``think:false`` is not honoured: every reply still carries
       thinking output
   * - Captions per hour at the defaults (derived, 3600 / 131.5 s)
     - about 27
   * - Upgrade fold, 135,000 rows (half captioned)
     - 57.6 s (PostgreSQL 16.15)
   * - v2 full-text index build, 135,000 rows
     - 27.2 s (99 MB index)
   * - Trigram index build, 135,000 rows
     - 2.6 s (24 MB index)
   * - Fusion calibration
     - Uncalibrated: no labelled set; defaults ``relative_margin`` 0.08 and
       ``min_similarity`` 0.45 kept


.. _ariel-picture-search-upgrade:

Upgrade notes
=============

**Profile keys.** An existing deployment's ``profile.yml`` is explicit and does
not gain the new keys; ``osprey validate`` lists them as drift. Add the lines
shown below, or run ``osprey profile expand`` (it writes every lacking leaf, with
the same side-effect caveat as ``--providers`` below); until then
``image_caption`` and ``image_embedding`` keep their code defaults (off) and the
view tool its default (on).

.. code-block:: yaml

   config:
     ariel.attachments.copy_on_ingest: images
     ariel.attachments.view.enabled: true
     ariel.enhancement_modules.image_caption.enabled: true
     ariel.enhancement_modules.image_caption.provider: ollama
     ariel.enhancement_modules.image_caption.model.model_id: qwen3-vl:4b
     ariel.enhancement_modules.image_embedding.enabled: true
     ariel.enhancement_modules.image_embedding.provider: llama-cpp
     ariel.enhancement_modules.image_embedding.model: qwen3-vl-embedding-2b
     ariel.enhancement_modules.image_embedding.dimensions: 1024

**The** ``llama-cpp`` **provider entry.** A deployment that keeps its own
``providers.yml`` does not need a ``llama-cpp`` entry: the adapter's default
address and ``LLAMA_CPP_HOST`` already work. To declare it anyway, paste the
packaged entry:

.. code-block:: yaml

   providers:
     llama-cpp:
       api_key: llama-cpp              # ignored by a server started without --api-key
       base_url: ${LLAMA_CPP_HOST:-http://localhost:8080}
       default_model: qwen3-vl-embedding-2b
       models:                         # the name llama-server advertises (--alias)
         - qwen3-vl-embedding-2b

``osprey profile expand --providers`` also writes it, with side effects: it fills
in every key the profile leaves to its preset and stamps provenance, which turns
on the strict preset-drift check (see
:ref:`Preset drift, and osprey profile expand <profile-preset-drift>`).

**Search indexes.** The upgrade builds a v2 full-text index and a trigram index
over attachment text, and keeps the v1 indexes (the full-text index and, with
the semantic processor on, its search index); a later release drops them. The
build times on 135,000 rows are in :ref:`ariel-picture-search-measurements`.

**Captions already present upstream.** The upgrade folds captions that came with
the upstream entries into each entry's searchable text and marks those entries'
text embeddings as owed again. Count them with the fold's own predicate, so null
captions are not counted:

.. code-block:: sql

   SELECT count(*) FROM enhanced_entries WHERE attachments @? '$[*] ? (@.caption != null && @.caption != "")'

With a paid embedding provider, drain that backlog in a maintenance window,
passing the count as the limit (the default limit is 100):

.. code-block:: bash

   osprey ariel enhance --module text_embedding --limit <N>

**Emptied attachment lists.** An upstream entry whose attachment list becomes
empty keeps its stored attachment rows. This is a known limitation.

**Re-running the picture modules.** ``--force`` does not apply to them:

   --force does not re-run image_caption/image_embedding: their results are kept per picture and model. Change model.model_id (captions) or model/dimensions (embeddings) to re-run, or use --retry-failed for per-picture failures.

**Changing the server.** Changing ``--image-max-tokens`` or the model files
changes every picture vector while the table name stays the same, so re-embed
with ``osprey ariel purge --embeddings-only`` followed by a catch-up (which
re-embeds text too).
