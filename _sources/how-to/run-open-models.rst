Run Open & Local Models
=======================

OSPREY is built so the volatile pieces — the model and the agent harness — can be
swapped without touching the framework. This guide covers what already works
today: running the Osprey agent on **open-weight and locally hostable models**.

Two independent axes
--------------------

- **The agent harness** (the program that drives the model through tool calls) is
  swappable in *intent*. Today there is one, Claude Code; support for additional
  coding-agent harnesses is planned.
- **The model** is already swappable. The Osprey agent speaks the Anthropic
  Messages API; any OpenAI-protocol endpoint — remote or self-hosted — is reached
  through a local translation proxy that OSPREY starts automatically. Open models
  are a configuration choice, not a code change.

How open models are routed
--------------------------

Open models are most often served behind an **OpenAI-compatible** API — that is how
CBORG, Ollama, and vLLM all expose them — while the Osprey agent emits **Anthropic
Messages** calls. When the endpoint speaks the OpenAI protocol, OSPREY bridges the
two with a local ``Anthropic ↔ OpenAI`` translation proxy that starts automatically
once you select an OpenAI-protocol provider — you never invoke it yourself. This is
identical whether the model is self-hosted (``ollama``, ``vllm``, no API key) or
served by an OpenAI-only remote aggregator.

One nuance for CBORG: the built-in ``cborg`` provider uses CBORG's
Anthropic-compatible endpoint, which serves the **Claude** models — those go
direct, with no proxy. CBORG's *open* models live on its OpenAI-compatible side,
so to run one you add your own provider entry pointing there. It then goes
through the translation proxy like any other OpenAI endpoint:

.. code-block:: yaml

   api:
     providers:
       cborg-open:
         api_key: ${CBORG_API_KEY}
         base_url: https://api.cborg.lbl.gov/v1   # keep the /v1
         models:
           haiku: gpt-oss-120b
           sonnet: gpt-oss-120b
           opus: gpt-oss-120b

   claude_code:
     provider: cborg-open
     default_model: sonnet

Any model ID from CBORG's catalogue works in the ``models`` block — the
benchmark table below shows which ones hold up in practice. The provider list
and ``config.yml`` keys are in :doc:`configure-providers`.

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

The numbers are reproducible. OSPREY ships the benchmark toolchain under
``scripts/benchmark/`` (see its ``README.md``): it runs the model-driving part of
``tests/e2e/`` — the tests that actually exercise the model under test — across a
matrix of models and renders a per-test pass-rate dashboard. The
whole run is declared in one file, ``scripts/benchmark/matrix.yaml`` — each row
names a ``provider`` and a model ``id``; the launcher resolves credentials,
derives the route (proxy for OpenAI-protocol models, direct for Anthropic),
wires the judge, and runs one isolated worker per (model, seed) cell. Adding a
model — or a provider like the local DeepSeek (``ds4``) server — is a config
edit, not a script edit.

.. code-block:: bash

   # see the resolved plan without running anything
   scripts/benchmark/matrix.py --dry-run

   # one model (substring filter), serially
   scripts/benchmark/matrix.py --only gpt-oss-20b --parallel 1

   # the whole grid, then render the dashboard
   scripts/benchmark/matrix.py --parallel 4
   scripts/benchmark/matrix_dashboard.py --results-dir results --out dashboard.html

The dashboard derives every count from the run data at render time — it never
hard-codes a number, so it stays accurate as the suite grows.

.. admonition:: Benchmark snapshot — OSPREY v2026.6.2
   :class: important

   Pass rates for the open and self-hosted models below, measured on the
   model-driving subset of the e2e suite; single-seed Anthropic columns are a
   control/ceiling reference.

   - **Measured against:** OSPREY ``v2026.6.2``
   - **Run:** 2026-06-25 · open subjects via CBORG · Anthropic reference columns via als-apg · ``deepseek-v4`` self-hosted on a Mac Studio (keyless ``ds4`` server, single seed)
   - **Scope:** the model-driving subset of ``tests/e2e/`` — 36 tests per seed
   - **Scoring:** pass rate = passed / (passed + failed + timeout + errors); a timeout counts as a failure (the model did not finish within the 1800s cap). Mean is over completed seeds.

   .. list-table::
      :header-rows: 1
      :widths: 26 12 10 10 10 12

      * - Model
        - Provider
        - Seed 1
        - Seed 2
        - Seed 3
        - Mean
      * - ``cborg-coder``
        - CBORG
        - 94%
        - 97%
        - 92%
        - **94%**
      * - ``gemma-4``
        - CBORG
        - 89%
        - 94%
        - 92%
        - **92%**
      * - ``qwen-3-coder``
        - CBORG
        - 94%
        - 83%
        - 94%
        - **91%**
      * - ``gpt-oss-120b``
        - CBORG
        - 92%
        - 81%
        - 81%
        - **84%**
      * - ``qwen-3``
        - CBORG
        - 89%
        - 78%
        - 83%
        - **83%**
      * - ``gpt-oss-20b``
        - CBORG
        - 67%
        - 67%
        - 56%
        - **63%**
      * - ``deepseek-v4-flash``
        - ds4 · macstudio
        - 94%
        - N/A
        - N/A
        - **94%**
      * - ``deepseek-v4-pro``
        - ds4 · macstudio
        - 97%
        - N/A
        - N/A
        - **97%**
      * - ``claude-haiku-4-5-20251001`` *(ref)*
        - als-apg
        - 100%
        - —
        - —
        - **100%**
      * - ``claude-sonnet-4-6`` *(ref)*
        - als-apg
        - 100%
        - —
        - —
        - **100%**
      * - ``claude-opus-4-6`` *(ref)*
        - als-apg
        - 100%
        - —
        - —
        - **100%**

   Anthropic columns are single-seed control/ceiling references. These figures are
   a point-in-time snapshot and *will* drift — regenerate with ``scripts/benchmark/``.

   :download:`Download the full interactive dashboard (HTML) <../_static/benchmark-dashboard.html>`

.. seealso::

   - :doc:`configure-providers` — providers, the translation proxy, model selection.
   - ``scripts/benchmark/README.md`` — the full benchmark contract.
