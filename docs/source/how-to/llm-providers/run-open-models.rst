.. _how-to-run-open-models:

Run Open & Local Models
=======================

OSPREY runs Claude Code as its agent harness, and the model behind it is a
configuration choice. This guide covers running the OSPREY agent on
**open-weight and locally hostable models**.

Two independent axes
--------------------

- **The agent harness** (the program that drives the model through tool calls) is
  Claude Code, the one harness OSPREY ships. Its code sits in one adapter package
  (:ref:`extending-agent-harness`).
- **The model** is a configuration choice. Any endpoint OSPREY can reach — remote
  or self-hosted — is set in configuration, not in code.

How open models are routed
--------------------------

Open models are most often served behind an **OpenAI-compatible** API — that is
how `CBORG <https://cborg.lbl.gov/>`__, `Ollama <https://ollama.com/>`__, and
`vLLM <https://vllm.ai/>`__ all expose them — so OSPREY reaches them through the
local ``Anthropic ↔ OpenAI`` translation proxy it starts for any OpenAI-protocol
provider. How that translation works, and the routing every provider follows
(:ref:`diagram <provider-routing-diagram>`), are in :doc:`configure-providers`.

One nuance for CBORG: the built-in ``cborg`` provider uses CBORG's
Anthropic-compatible endpoint, which serves the **Claude** models — those go
direct, with no proxy. CBORG's *open* models live on its OpenAI-compatible side,
so to run one you add your own provider entry pointing there. It then goes
through the translation proxy like any other OpenAI endpoint:

In ``providers.yml``, beside the deployment's ``profile.yml``:

.. code-block:: yaml

   providers:
     cborg-open:
       api_key: ${CBORG_OPEN_API_KEY}
       base_url: https://api.cborg.lbl.gov/v1   # keep the /v1
       default_model: gpt-oss-120b
       models:
         - gpt-oss-120b

and in ``profile.yml``:

.. code-block:: yaml

   provider: cborg-open

The entry names ``gpt-oss-120b`` as its default, so the profile needs no
``model:``. The provider serves no Claude models, so the build prints one line
saying Claude Code's haiku, sonnet and opus aliases point at ``gpt-oss-120b``.

A provider you add yourself takes its key from ``<NAME>_API_KEY`` — uppercased,
dashes to underscores — so ``cborg-open`` reads ``CBORG_OPEN_API_KEY``. Put
``CBORG_OPEN_API_KEY=<your CBORG key>`` in the repo's ``.env``; the same CBORG
key serves both entries.

Any model ID from CBORG's catalogue works in the ``models`` list — the
*Benchmark snapshot* box on this page shows which ones hold up in practice. The
provider list and the keys behind it are in :doc:`configure-providers`.

For a model you serve yourself with vLLM, the packaged ``vllm`` provider is
keyless and points at ``http://localhost:8000/v1``; set ``VLLM_BASE_URL`` for a
server elsewhere and list its ``--served-model-name`` under ``models`` in your
copy of ``providers.yml``.

Which models are known to work
------------------------------

For an agentic system "runs" means sustaining a multi-step tool-calling loop across
the **full OSPREY end-to-end suite**, not just emitting plausible text. The
following open families were verified to complete that suite end-to-end:

- ``gpt-oss`` (20B and 120B)
- ``gemma``
- ``cborg-coder``
- ``qwen-3`` / ``qwen-3-coder``

This set was chosen from the open models **available on the CBORG provider** — it
reflects CBORG's catalogue at benchmark time, not an exhaustive survey of open
models. Capability varies widely across the set; the snapshot below is the real
signal, not the bare "it runs" list.

Benchmark it yourself
---------------------

.. note::

   This benchmark is at its very beginning, not an exhaustive test of OSPREY with open
   or frontier models. It will grow: more Tier 2 tasks, and facilities beyond the
   storage ring, such as a free-electron laser or other scientific instruments.
   Contributions are very welcome; see :doc:`/contributing/index`.

The numbers are reproducible. OSPREY ships the benchmark toolchain under
``scripts/benchmark/`` (see its `README
<https://github.com/als-apg/osprey/blob/main/scripts/benchmark/README.md>`_): it runs the agentic tests of
``tests/e2e/`` — those where the model under test drives the agent — across a
matrix of models and renders a per-test dashboard. The
whole run is declared in one file, ``scripts/benchmark/matrix.yaml`` — each row
names a ``provider`` and a model ``id``; the launcher resolves credentials,
derives the route (proxy for OpenAI-protocol models, direct for Anthropic),
wires the judge, and runs one isolated worker per (model, seed) cell. Adding a
model — or a self-hosted ``vllm`` server — is a config edit, not a script edit.
Every answer the tests judge is graded by ``claude-haiku-5-5`` on als-apg. Each
test carries its tier and a plain-words description on an ``agentic`` marker,
and the board scores its first attempt:

.. list-table::
   :header-rows: 1
   :widths: 18 46 36

   * - Tier
     - What it covers
     - What the board shows
   * - 1 · basic
     - The stack works, the model calls the right tools, a simple workflow goes through.
     - Median time and tokens per task; all green expected.
   * - 2 · benchmark
     - Harder tasks a capable model may still miss.
     - Each test's first-attempt result per seed.
   * - 3 · facility scale
     - Not built yet: facility-scale tasks, such as searching a whole logbook.
     -

.. code-block:: bash

   # see the resolved plan without running anything
   scripts/benchmark/matrix.py --dry-run

   # one model (substring filter), serially
   scripts/benchmark/matrix.py --only gpt-oss-20b --parallel 1

   # a smoke run: only the tests a pytest -k expression selects
   scripts/benchmark/matrix.py --only gpt-oss-20b --select vacuum_burst --results-dir results-smoke

   # the whole grid, then render the dashboard
   scripts/benchmark/build_board_images.sh   # once per revision
   scripts/benchmark/matrix.py --parallel 4
   scripts/benchmark/matrix_dashboard.py --results-dir results --out dashboard.html

   # re-render the copy this page links to
   scripts/benchmark/matrix_dashboard.py --results-dir results --artifact \
       --out docs/source/_static/benchmark-dashboard.html

.. admonition:: Benchmark snapshot — OSPREY [FULL-RUN: version]
   :class: important

   First-attempt results on the agentic tests of the e2e suite.
   The judge changed to ``claude-haiku-5-5``, so these numbers are not
   comparable with the June and October 2026 runs.

   - **Measured against:** OSPREY ``[FULL-RUN: version]``
   - **Run:** [FULL-RUN: date] · open models via CBORG · ``kolibri-1`` self-hosted on vLLM · Anthropic references via als-apg · 3 seeds each
   - **Scope:** 34 tests per seed: 28 in Tier 1, 6 in Tier 2
   - **Scoring:** first attempt only; a timeout counts as a failure (the model did not finish within the 1800s cap), and a test that dies in setup or that the provider refused (auth, network, unserved model) scores nothing. Tier 1: passes over all seeds and the median time per task. Tier 2: passes over all seeds.

   .. list-table::
      :header-rows: 1
      :widths: 30 20 16 16

      * - Model
        - Provider
        - Tier 1 (passed · median task)
        - Tier 2 (passed)
      * - ``cborg-coder``
        - CBORG
        - [FULL-RUN]
        - [FULL-RUN]
      * - ``gemma-4``
        - CBORG
        - [FULL-RUN]
        - [FULL-RUN]
      * - ``qwen-3-coder``
        - CBORG
        - [FULL-RUN]
        - [FULL-RUN]
      * - ``gpt-oss-120b``
        - CBORG
        - [FULL-RUN]
        - [FULL-RUN]
      * - ``qwen-3``
        - CBORG
        - [FULL-RUN]
        - [FULL-RUN]
      * - ``gpt-oss-20b``
        - CBORG
        - [FULL-RUN]
        - [FULL-RUN]
      * - ``kolibri-1``
        - vLLM · self-hosted
        - [FULL-RUN]
        - [FULL-RUN]
      * - ``claude-haiku-5-5`` *(ref)*
        - als-apg
        - [FULL-RUN]
        - [FULL-RUN]
      * - ``claude-sonnet-5-5`` *(ref)*
        - als-apg
        - [FULL-RUN]
        - [FULL-RUN]
      * - ``claude-opus-5-5`` *(ref)*
        - als-apg
        - [FULL-RUN]
        - [FULL-RUN]

   Anthropic models are a control/ceiling reference. These figures are a
   point-in-time snapshot and *will* drift — regenerate with ``scripts/benchmark/``.

   .. FULL-RUN: re-render _static/benchmark-dashboard.html with the --artifact command above.

   :download:`Download the full interactive dashboard (HTML) <../../_static/benchmark-dashboard.html>`

.. seealso::

   - :doc:`configure-providers` — providers, the translation proxy, model selection.
   - `scripts/benchmark/README.md
     <https://github.com/als-apg/osprey/blob/main/scripts/benchmark/README.md>`_
     — the full benchmark contract.
