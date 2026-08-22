============
Search Modes
============

ARIEL's search system is built around **search modules** --- leaf-level functions that each implement a single retrieval strategy over the logbook. The framework ships three: keyword full-text search, embedding-based semantic similarity, and ``hybrid``, a merge of the two answered by a separate search sidecar container (qmd). All three produce a common ``ARIELSearchResult``. Higher-level reasoning over results --- multi-step retrieval, answer synthesis, custom prompting --- lives in the Osprey agent layer, which calls these search modules through ARIEL's MCP tools.

**Dispatch is registry-driven.** A search request names a mode as a plain string --- ``"keyword"``, ``"semantic"``, ``"hybrid"``. The ``ARIELSearchService`` looks that name up in Osprey's central registry and calls the module's own ``execute``; it carries no per-mode branch of its own. The registry is the only source of routable modes, so the service, the web interface's capabilities API and the agent's MCP tools cannot disagree about which modes exist, and adding a module needs no change to the service.

.. note::

   ``sql_query`` is a **tool, not a search mode.** It runs read-only SQL against the same database, which is precision filtering --- exact matches, exhaustive date and author ranges, counts --- not ranked retrieval. It has no relevance score to merge with the others, so it is exposed only as its own MCP tool and never appears as a ``--mode`` value.

Search Architecture
-------------------

.. code-block:: text

   User Query
       ↓
   ARIELSearchService.search(mode="keyword" | "semantic" | "hybrid" | ...)
       ↓
   registry lookup  →  that module's execute()
       ↓
   ARIELSearchResult (entries, search_modes_used)

The service refuses a mode that is not registered, or is registered but disabled in configuration, rather than quietly falling back to another one.

**CLI usage:**

.. code-block:: bash

   osprey ariel search "RF cavity fault"                  # ariel.default_search_mode
   osprey ariel search "RF cavity fault" --mode keyword
   osprey ariel search "RF cavity fault" --mode semantic
   osprey ariel search "RF cavity fault" --mode hybrid

The ``--mode`` choices are read from the registry when the command runs, so a facility that registers its own search module gets it as a choice --- and in ``--help`` --- without any code change.


Search Modules
==============

Search modules are leaf-level functions that execute a single search strategy against the database. Each module exports a ``get_tool_descriptor()`` function that describes its capabilities, input schema, and execution function. The web interface discovers modules through this descriptor via ARIEL's capabilities API; each built-in module is exposed to the Osprey agent through its own ARIEL MCP tool (``keyword_search``, ``semantic_search``, ``hybrid_search``). The framework ships with the following built-in search modules:

.. tab-set::

   .. tab-item:: Keyword Search

      **Module:** ``search/keyword.py``

      PostgreSQL full-text search with optional fuzzy matching fallback. Best for specific terms, equipment names, PV names, and exact phrases.

      **Query syntax:**

      .. code-block:: text

         # Simple terms (implicit AND)
         RF cavity fault

         # Boolean operators
         RF AND cavity
         vacuum OR pressure
         beam NOT injection

         # Quoted phrases
         "RF cavity trip"

         # Field prefixes
         author:smith
         date:2024-06

         # Combined
         author:jones "beam loss" date:2024-01

         # Pattern tokens
         SR01C___BPM*
         /SR0[1-4]C___BPM\d+/

      Pattern tokens and query expansion are covered in :ref:`vocabulary-expansion`.

      **How it works:**

      1. Validates and preprocesses the query --- empty queries return immediately, queries longer than 1,000 characters are truncated, and unbalanced quotes are auto-balanced by removing the last unmatched quote
      2. Parses the query to extract field filters (``author:``, ``date:``), quoted phrases, and remaining search terms
      3. Builds a PostgreSQL ``tsquery`` using the function appropriate for the query shape:

         - ``plainto_tsquery`` --- for simple terms (implicit AND)
         - ``websearch_to_tsquery`` --- for queries with Boolean operators (AND, OR, NOT)
         - ``phraseto_tsquery`` --- for quoted phrases

         When multiple components are present (e.g. terms *and* phrases), they are combined with ``&&`` (tsquery AND).

      4. Executes full-text search against the ``raw_text`` column with ``ts_rank`` scoring, applying any field filters (``author ILIKE``, date range) and time range constraints
      5. If no results and fuzzy fallback is enabled, falls back to ``pg_trgm`` trigram similarity (default threshold: 0.3)
      6. Returns results as ``(entry, score, highlights)`` tuples --- highlights are generated via ``ts_headline``

      **Configuration:**

      .. code-block:: yaml

         ariel:
           search_modules:
             keyword:
               enabled: true
               settings:
                 patterns_enabled: true          # default: true
                 pattern_timeout_seconds: 10.0   # default: 10.0

   .. tab-item:: Semantic Search

      **Module:** ``search/semantic.py``

      Embedding-based similarity search using pgvector. Best for conceptual queries where exact keywords may not appear in the text.

      **How it works:**

      1. Resolves the similarity threshold using a 3-tier priority:

         a. Per-query ``similarity_threshold`` parameter (highest)
         b. Config value (``search_modules.semantic.settings.similarity_threshold``)
         c. Hardcoded default: 0.5 (lowest)

      2. Determines the embedding model from config (``search_modules.semantic.model``) and resolves provider credentials via Osprey's centralized ``api.providers`` configuration
      3. Generates a query embedding using the configured provider, with a dimension-mismatch warning if the returned embedding size does not match the configured ``embedding_dimension``
      4. Searches the per-model embedding table using cosine distance (``<=>`` operator)
      5. Filters results by similarity threshold and optional time range
      6. Returns results as ``(entry, similarity_score)`` tuples

      **Configuration:**

      .. code-block:: yaml

         ariel:
           search_modules:
             semantic:
               enabled: true
               provider: ollama
               model: nomic-embed-text
               settings:
                 similarity_threshold: 0.5
                 embedding_dimension: 768

      **Requirements:** Ollama (or another embedding provider) running with the configured model, embedding table populated via the ``text_embedding`` :ref:`enhancement module <Enhancement Pipeline>`, and the pgvector extension installed in PostgreSQL.

   .. tab-item:: hybrid (qmd sidecar)

      **Module:** ``search/qmd.py``

      Hybrid keyword-plus-semantic search, answered by the **qmd search sidecar** --- a separate container that indexes a markdown mirror of the logbook and returns one merged ranking. Best when a question mixes specific terms with a described situation, or when keyword search returned too little.

      Unlike the other two modes, ``hybrid`` does not search PostgreSQL. It needs two things running together:

      1. the ``services.qmd`` sidecar (see :ref:`qmd-search-sidecar`), and
      2. the ``qmd_export`` :ref:`enhancement module <Enhancement Pipeline>`, which writes the markdown mirror the sidecar indexes.

      Either one alone is useless: an export with no sidecar indexes nothing, and a sidecar with no export searches an empty corpus. The shipped ``control-assistant`` and ``ariel-standalone`` templates enable both, together with the sidecar itself.

      ``hybrid`` also does not degrade the way semantic search does. A query against a sidecar that is not there is reported as *search is down*, deliberately, so that the agent cannot read an outage as "nothing matched".

      **Configuration:**

      .. code-block:: yaml

         ariel:
           search_modules:
             hybrid:
               enabled: true
               settings:
                 rerank: true          # default
                 candidate_limit: 40   # default

      .. warning::

         The knobs **must** stay under ``settings:``. ARIEL's search-config loader keeps only ``enabled``, ``provider``, ``model`` and ``settings``, and drops every other key without a word. A knob written as a sibling of ``enabled`` is inert forever --- no error, no warning, just the defaults.

      **The rerank decision.** ``rerank`` turns on qmd's LLM reranker, which reorders candidates for quality. It is the single most important knob here, because it is the dominant latency term. Measured against a 134,996-entry logbook:

      .. list-table::
         :header-rows: 1
         :widths: 40 30 30

         * - Corpus
           - p95, ``rerank: false``
           - p95, ``rerank: true``
         * - 135,000 entries
           - 811 ms
           - 3927 ms
         * - 2,000 entries
           - --
           - 1587 ms

      Reranking costs roughly **4x** the query budget, and its cost barely grows with corpus size --- so no logbook is small enough to outrun it. ``hybrid_search`` is an agent tool with no interactive budget to protect, so it ships with the quality path on. Set ``rerank: false`` for the fast path. (The OKF bundle, which backs an interactive panel, defaults the other way; see :doc:`../okf-bundle`.)

      ``candidate_limit`` is how many candidates the reranker considers. Lowering it trades recall for latency.

      **Filtering is best-effort.** ``hybrid_search`` ranks the corpus first and applies the date, author and source filters *afterwards*, to the top of that ranking --- not inside the database. A selective filter can therefore return fewer entries than you asked for even when more matching entries exist. Read a short result set as "the ranked window ran out", not as "there is nothing else". When a filter has to be exhaustive, use ``keyword_search`` or ``sql_query``, which filter in the database.

      .. admonition:: Known limitation --- the first reranked query times out
         :class: warning

         The first query with ``rerank: true`` loads a 610 MB model on CPU, and that load exceeds the client's default timeout. A deployment that enables reranking gets a hard failure on its **first** query; subsequent queries are fine. Run one throwaway query after starting the sidecar, or start with ``rerank: false``.

      .. admonition:: Known limitation --- entry IDs that are not numeric
         :class: note

         qmd normalises the document paths it reports: ``_`` and ``%`` both become ``-``, runs collapse, and a leading one is dropped. ARIEL rehydrates each hit from the document's title rather than the reported path, so the entries you get back are correct. But two entry IDs that differ *only* in characters qmd collapses --- ``beam_current_setpoint`` and ``beam-current-setpoint``, say --- index as one document, and one of them becomes unreachable through this mode.

         Over the real 134,996-entry ALS logbook the measured collision rate is **0.0000%**: every ALS entry ID is a 4-6 digit decimal string, so no real pair can collide. This matters only for a facility whose entry IDs are not numeric.

**Registering a custom search module:**

To add your own search module, create a Python module that exports ``get_tool_descriptor()`` (and optionally ``get_parameter_descriptors()``), then register it through your application's registry configuration:

.. code-block:: python

   from osprey.registry.helpers import extend_framework_registry
   from osprey.registry.base import ArielSearchModuleRegistration

   app_config = extend_framework_registry(
       ariel_search_modules=[
           ArielSearchModuleRegistration(
               name="my_search",
               module_path="my_app.search.my_module",
               description="Custom search module for my facility",
           ),
       ],
   )

Once registered and enabled in ``config.yml`` (``search_modules.my_search.enabled: true``), the module is routable by name --- through ``osprey ariel search --mode my_search`` and through the web interface's capabilities API --- with no change to ``ARIELSearchService``. Making it callable by the Osprey agent additionally requires a matching ARIEL MCP tool (contributions welcome). The ``get_tool_descriptor()`` function must return a ``SearchToolDescriptor``, whose ``search_mode`` field is simply the registered module name:

:class:`~osprey.services.ariel_search.search.base.SearchToolDescriptor` — a frozen dataclass whose key fields are ``execute`` (the async search function), ``format_result`` (formats results for agent consumption), and ``args_schema`` (a Pydantic model for input validation). See the class definition in the source for the full field list.

Modules may also export ``get_parameter_descriptors()`` to declare tunable parameters for the frontend capabilities API. Each :class:`~osprey.services.ariel_search.search.base.ParameterDescriptor` describes a single knob --- its name, type, default, range, and UI grouping --- so the web interface can render controls dynamically.

**What a module returns.** ``execute`` may return a plain list of entries, as it always could, or a ``ModuleOutput(entries, diagnostics, expansion)`` when it has something to report alongside them: ``diagnostics`` for the notices the caller should see (a truncated query, a skipped step), and ``expansion`` for the :ref:`vocabulary groups <vocabulary-expansion>` that actually reached the query it ran. A bare list stays valid and needs no change.

**Opting into the shared query work.** Two descriptor fields, both off by default, let a module receive work the service has already done. Declare neither and the module is called with exactly the arguments it is called with today.

* ``accepts_expansion`` --- set it to ``True`` and the service passes ``query_expansion=`` when, and only when, expansion was resolved for that request. The module never sees the ``expand_query`` request argument itself: whether to expand is settled once, in the service.
* ``query_parser`` --- an optional callable that takes the raw query string and returns a ``ParsedKeywordQuery`` (search text, field filters, phrases, pattern spans, diagnostics). The service calls it once and hands the result to the module as ``parsed=``. Keyword search declares its own ``parse_keyword_query`` here; a module that wants a different query language supplies its own parser, and one that wants none is never given ARIEL's.

.. admonition:: Collaboration Welcome
   :class: outreach

   If you implement a search module that could benefit other facilities --- for example, a structured-metadata search, a time-series correlation search, or a cross-entry linking search --- we encourage you to open a pull request so it becomes natively available in Osprey.


.. _vocabulary-expansion:

Vocabulary Expansion and Pattern Search
=======================================

Control rooms search in shorthand and write logbook entries in words. A search
for ``t/s the bpm offset`` will not find the entry that says "troubleshoot the
beam position monitor offset" --- stemming clips word endings, it does not know
that ``bpm`` and "beam position monitor" are the same thing.

A **vocabulary** is where a facility says that they are. It is a data file
listing each concept once: the words your prose uses, and the spellings your
operators type. Before a search runs, ARIEL rewrites the query using that file,
so the shorthand spelling reaches the prose spelling and the other way round.
It is plain dictionary matching --- no model, no network call, microseconds ---
and every rewrite is reported back with the results, so a hit that looks
surprising always arrives with the reason it appeared.

Nothing is built in. With no vocabulary configured, no query is ever rewritten.
The ``control-assistant`` template ships a twenty-concept example at
``data/ariel/vocabulary.yml`` as a starting point and a format tutorial; it is
meant to be edited down to what your control room actually says.

.. note::

   Thanks to Chris Tennant (Jefferson Lab), whose idea this feature started
   from.

The vocabulary file
~~~~~~~~~~~~~~~~~~~

One top-level key, ``concepts``, holding a non-empty list. Each entry has three
required keys and no others:

.. code-block:: yaml

   concepts:
     - canonical: beam position monitor   # the words logbook prose uses
       kind: acronym                      # acronym | shorthand
       forms: [bpm, bpms]                 # what operators type instead

     - canonical: troubleshoot
       kind: shorthand
       forms: [ts, t/s]

``canonical``
   Write it exactly as it appears in entry text. It is what a matched form is
   rewritten into. Canonicals must be unique.

``kind``
   ``acronym`` for a genuine initialism, ``shorthand`` for a clipped or slang
   spelling of an ordinary word. It selects a direction gate (below).

``forms``
   One or more spellings operators type. A form may not repeat its own
   canonical.

Anything else is refused rather than ignored --- an unknown key, a missing
field, an unknown ``kind``, a duplicate canonical, an empty list, a file that is
not a YAML mapping. The check reports **every** problem in one pass, so a file
with three mistakes takes one run to fix, not three.

**How text is matched.** The file and the query go through the same rule before
anything is compared: lowercase; ``-`` becomes a space (so ``Beam-Position
Monitor`` and ``beam position monitor`` are one phrase); ``/`` is **kept** (so
``t/s`` and ``i/o`` stay a single token); runs of whitespace collapse to one.
Nothing else is stripped, and no stemming happens here --- PostgreSQL's stemmer
runs afterwards, on the text this produces. Forms match on whole tokens;
multi-word forms match as phrases, longest first, left to right, and a stretch
of text that has already matched is never matched again.

**The two direction gates.** Matching a form and adding its canonical is
*always* on: type ``bpm`` and the search also finds "beam position monitor".
The reverse direction --- spelling the canonical out and *also* searching the
short form --- is what ``kind`` gates:

.. list-table::
   :header-rows: 1
   :widths: 45 30 25

   * - Setting
     - Applies to
     - Default
   * - ``ariel.vocabulary.canonical_to_acronym``
     - ``kind: acronym`` concepts
     - ``true``
   * - ``ariel.vocabulary.canonical_to_shorthand``
     - ``kind: shorthand`` concepts
     - ``false``

The defaults are deliberately asymmetric. An initialism means one thing, so a
search for "beam position monitor" should also reach the entries that wrote
"BPM". An ordinary word is not so lucky: expanding "calibration" into ``cal``
pulls in every entry that abbreviated something else that way.

**Forms that mean two things.** Binding one form to two concepts is legal, and a
query containing it expands to *both* canonicals rather than picking a winner.
``vocab-check`` warns about it and lists every canonical the form reaches, so
the widening is a decision you made rather than something you discover from odd
results. Two other things warn without failing: a form that can never fire
because a longer form always wins on the same text, and a word PostgreSQL's
English text search discards outright, which would make expanding to it match
nothing.

What gets expanded, and where
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expansion runs once, in the search service, before the request reaches a search
module. What it looks at depends on the mode:

``keyword``
   The search terms and quoted phrases only. The values of ``author:`` and
   ``date:`` filters are never expanded, and neither is the body of a pattern
   token --- ``author:ts bpm`` expands ``bpm`` and leaves the author alone. A
   query carrying explicit boolean operators (``AND``, ``OR``, ``NOT``) runs
   unexpanded, with an INFO diagnostic saying so: that query syntax cannot group
   the alternatives. Keyword queries are also truncated at 1,000 characters, so
   a form past that point is not seen.

``semantic`` and ``hybrid``
   The whole query, because the whole query is what gets embedded or sent to the
   sidecar.

The same query can therefore report different expansions in different modes.
``expand_modes`` limits which modes expand at all; unset --- the default ---
means every enabled search module.

.. admonition:: Hybrid feeds the expanded query to a reranker
   :class: note

   ``hybrid`` hands its query to the qmd sidecar's LLM reranker, so expansion
   changes what the reranker sees and can change the order of the results, not
   only which ones come back. If your reranked ordering gets worse with
   expansion on, drop that one mode rather than the feature:

   .. code-block:: yaml

      ariel:
        vocabulary:
          expand_modes: [keyword, semantic]

Turning expansion off, for a deployment or for one search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ariel.vocabulary.enabled: false`` switches the whole thing off: nothing is
read, and a stale ``path`` left behind is inert rather than a startup failure.

For a single search, every surface takes an ``expand_query`` argument --- the
**Expand vocabulary** toggle in the web interface's advanced options, and an
argument on the ``keyword_search``, ``semantic_search`` and ``hybrid_search``
MCP tools. Leave it unset and it follows ``ariel.vocabulary.expand_by_default``;
pass ``false`` and that one search runs on exactly the words you typed, with no
expansions reported. It is the switch to reach for when you want to see what a
query finds on its own, and the Osprey agent can use it the same way when a
rewrite looks like it widened a search too far. With the vocabulary disabled the
toggle is not advertised at all and the argument does nothing.

What you see when a query expands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Results carry an ``expanded_terms`` list --- one entry per rewritten span, each
pairing the matched span, as normalized, with the terms that were added:

.. code-block:: json

   [{"original": "ts", "alternatives": ["troubleshoot", "timing system"]}]

It reports only what the search actually used: a mode that skipped expansion, or
a request that turned it off, reports an empty list. In the web interface the
pairs render as a strip above the results in expert mode. The MCP tools carry
the same list in their result envelopes, and ``keyword_search`` and
``semantic_search`` also carry ``diagnostics`` --- the notices explaining a
truncated query, a skipped expansion, or a pattern searched as text.
``osprey ariel search`` does not print the expansion --- use the panel or an MCP
tool to see it.

Pattern tokens in keyword search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keyword search accepts two ways of writing a control-system name you only partly
know. A glob is matched at word boundaries; an explicit ``/…/`` regular
expression matches anywhere in the entry text unless you anchor it yourself.
Both are case-insensitive, and both work alongside expansion in the same query.

**Globs.** A token is a glob only if it contains ``*``. Inside such a token
``?`` matches a single character; on its own it is ordinary text, so
``did the BPM trip?`` and ``SR01C___BPM?`` are just words.

.. code-block:: text

   SR01C___BPM*     matches SR01C___BPM3 and bare SR01C___BPM, not XSR01C___BPM3
   SR0?C___BPM*     matches SR01C___BPM3 and SR02C___BPM, not SR011C___BPM3

**Explicit regular expressions.** A ``/…/`` token is a PostgreSQL regular
expression. The slashes must start and end a whole token, which is what keeps a
slash *inside* a word from opening one --- ``t/s`` and ``i/o`` stay shorthand.
Spaces and escaped slashes are allowed inside the body, so ``/beam loss/`` is
one pattern.

.. code-block:: text

   /SR0[1-4]C___BPM\d+/       one pattern
   t/s /SR0[1-4]C___BPM\d+/   one pattern, plus the t/s expansion

One trailing ``?`` is stripped from the query before patterns are read, so
asking ``/SR0[1-4]C___BPM\d+/?`` as a question still works; a ``?`` *inside* the
body is left alone (``/BPMs?/`` keeps it).

**A pattern that is too generic is searched as text.** A pattern needs at least
three consecutive literal characters to use the trigram index; below that it
would scan the whole logbook. Such a token is neither rejected nor dropped --- it
goes back into the query as ordinary search text (a rejected ``/…/`` contributes
the words of its body, without the slashes), and the search says so in its
diagnostics. That is why ``test UNION SELECT * FROM users --``, ``5 * 3`` and
``flow rate 5 / 6 / 7`` produce no pattern at all. The one exception is an empty
pattern (``//``), which is ignored outright with an INFO notice.

**When a pattern is too slow, you are told.** A pattern query runs under
``pattern_timeout_seconds``; an ordinary keyword search never opens that
envelope. On expiry the search returns a timeout diagnostic --- through MCP, a
``search_timeout`` error envelope --- rather than an empty result, so the agent
can tell "too slow" from "nothing matched". A regular expression PostgreSQL
cannot compile comes back the same way, as an ``invalid_pattern`` error naming
the pattern. Both carry the expansion groups the query was going to use.

Setting ``patterns_enabled: false`` makes ``*`` and ``/…/`` ordinary words again
--- the escape hatch for a logbook whose prose is full of literal asterisks.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

   ariel:
     vocabulary:
       enabled: true                       # default: false
       path: data/ariel/vocabulary.yml     # required when enabled
       expand_by_default: true             # default: true
       canonical_to_acronym: true          # default: true
       canonical_to_shorthand: false       # default: false
       # expand_modes: [keyword, semantic] # default: every enabled module

     search_modules:
       keyword:
         enabled: true
         settings:
           patterns_enabled: true          # default: true
           pattern_timeout_seconds: 10.0   # default: 10.0

A relative ``path`` resolves against the directory holding the ``config.yml``
that was loaded --- the same rule as ``qmd_export.mirror_path`` and
``facility_knowledge.bundle_path`` --- so the panel, the CLI and the MCP server
all read the same file no matter where the process was started.

``enabled: true`` with no ``path``, an ``expand_modes`` entry naming a module
that is unknown or not enabled, and a malformed pattern knob are all
configuration errors that name the offending key. Nothing is quietly defaulted.

.. warning::

   The two keyword knobs **must** stay under ``settings:``. ARIEL's
   search-config loader keeps only ``enabled``, ``provider``, ``model`` and
   ``settings``, and drops every other key without a word. A knob written as a
   sibling of ``enabled`` is inert forever --- no error, no warning, just the
   defaults.

Checking the file, and reading the status
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   osprey ariel vocab-check                            # the configured file
   osprey ariel vocab-check data/ariel/vocabulary.yml  # any file
   osprey ariel vocab-check --json                     # one JSON document

Neither this command nor the vocabulary line of ``osprey ariel status`` needs a
database. ``vocab-check`` exits ``0`` when the file loads and ``1`` when it does
not, listing every error it found. Warnings are printed either way and never
fail the check.

``osprey ariel status`` reports one line --- ``Vocabulary: OK (20 concepts)``,
``Vocabulary: INVALID (1 errors). Run: osprey ariel vocab-check``, or
``Vocabulary: disabled`` --- and its ``--json`` output carries the same verdict
under ``vocabulary``. Run ``vocab-check`` on every edit, before you deploy: the
file is read once, when the configuration is parsed, so a mistake in it is a
startup problem rather than a search that quietly stops expanding.

When the vocabulary is broken
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A vocabulary that cannot be loaded takes search down, loudly, and leaves
everything else running:

* The web panel starts in **CONFIGURATION INVALID** mode. A banner names the
  errors and the fix, and the search form is disabled --- it does not look like
  a search that found nothing.
* A search request returns ``503`` naming ``ariel.vocabulary.path``, not a
  generic failure.
* Browsing entries, status and publishing keep working, and so does the settings
  editor --- which is the point: you can set ``ariel.vocabulary.enabled: false``
  or repoint ``ariel.vocabulary.path`` from the panel itself, then restart.
* The MCP server refuses to start, naming the same key, so the Osprey agent
  never gets a half-working search.

If the broken file is baked into a deployed image, the panel's editor cannot
reach it: ``vocabulary.yml`` ships inside the image, like the channel databases.
Fix the file in the project source, then re-run the build and deploy again.

.. admonition:: What expansion does not do
   :class: note

   * **Forms are literal text.** A form cannot be a regular expression or a
     wildcard; write out the spellings you want matched.
   * **No fuzzy matching.** A misspelled form is simply not a form. (Keyword
     search's own trigram fallback still applies to the query as a whole.)
   * **No disambiguation by context.** A form bound to two concepts always
     expands to both. Split the forms (``t/s`` versus ``ts``) or drop a binding
     if that is too wide.
   * **Boolean keyword queries are never expanded** --- the syntax cannot group
     the alternatives.
   * **Pattern matches are not highlighted.** Result highlights come from
     full-text search; a pattern match contributes none.
   * **The vocabulary is not editable from the panel.** It is project data, not
     a runtime setting: edit it in the source and redeploy.


.. _choosing-semantic-or-hybrid:

Choosing Between Semantic and Hybrid
====================================

Both modes retrieve by meaning rather than by matching words, and a deployment
can run either, both, or neither. They differ in what they depend on, what they
cost per query, and how much of their ranking you can inspect.

``hybrid`` is the stronger default for most deployments. It combines BM25 with
vector search and an LLM reranker, and its models ship inside the qmd sidecar's
image, so it needs no embedding provider on the host and no pgvector extension
in the logbook database. It is also the mode the shipped templates set as
``default_search_mode``.

``semantic`` is worth keeping — or choosing — when any of the following applies:

* **You want a stronger embedding model than the sidecar bakes in.** ``semantic``
  takes its embeddings from a configured provider, so a facility with its own
  inference endpoint can point it at a far larger model than the 300M embedder
  qmd ships.
* **You need the ranking to be inspectable.** ``semantic`` is cosine distance
  over a pgvector column and nothing else --- no reranker. It does apply
  deterministic :ref:`vocabulary expansion <vocabulary-expansion>` when one is
  configured, and reports every term it added; drop ``semantic`` from
  ``ariel.vocabulary.expand_modes`` for the pure path. Any result can then be
  explained with a single SQL query, which matters where retrieval has to be
  auditable.
* **You want it composable with structured filters.** The embeddings live in
  ``text_embeddings_*`` tables in the logbook database, so they join against
  ordinary columns and are reachable from the ``sql_query`` tool. ``hybrid``
  ranks a markdown mirror and post-filters instead.
* **Query latency matters more than ranking quality.** Reranking dominates a
  hybrid query's cost; a pure vector lookup is the fast path. (``hybrid`` can
  also be run with ``rerank: false``.)

Running both is a reasonable configuration: they are independent modules, and
``default_search_mode`` decides only which one answers when the caller names no
mode.


Need behavior beyond these search modules --- multi-step reasoning, answer
synthesis, custom prompting? That lives in the Osprey agent layer; see
:doc:`osprey-integration` under "Extending the integration."


See Also
========

:doc:`data-ingestion`
    How data gets into the system --- facility adapters, enhancement modules, and database schema

:doc:`osprey-integration`
    MCP tools, service factory, and search result structure

:doc:`web-interface`
    Web interface architecture and capabilities API
