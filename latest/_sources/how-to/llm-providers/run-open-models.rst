.. _how-to-run-open-models:

Run Open & Local Models
=======================

Use this guide to run the OSPREY agent on **open-weight and locally hostable
models**. It shows how to route such a model, which open models are known to
work, and how to benchmark one yourself. OSPREY runs Claude Code as its agent
harness, and the model behind it is a configuration choice.

Two independent axes
--------------------

- **The agent harness** (the program that drives the model through tool calls) is
  Claude Code, the one harness OSPREY ships. Its code sits in one adapter package
  (:ref:`extending-agent-harness`).
- **The model** is a configuration choice. Any endpoint OSPREY can reach — remote
  or self-hosted — is set in configuration, not in code.

How open models are routed
--------------------------

OSPREY reaches open models through the local ``Anthropic ↔ OpenAI``
translation proxy. OSPREY starts this proxy for any OpenAI-protocol provider.
Open models are most often served behind an **OpenAI-compatible** API; that is
how CBORG, Ollama, and vLLM all expose them. For how that translation works,
and the routing every provider follows
(:ref:`diagram <provider-routing-diagram>`), see :doc:`configure-providers`.

CBORG needs one extra step. The built-in ``cborg`` provider uses CBORG's
Anthropic-compatible endpoint, which serves the **Claude** models. Those go
direct, with no proxy. CBORG's *open* models live on its OpenAI-compatible
side. To run one, add your own provider entry that points there; it then goes
through the translation proxy like any other OpenAI endpoint.

Add the entry in ``providers.yml``, beside the deployment's ``profile.yml``:

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

A provider you add yourself takes its key from ``<NAME>_API_KEY``, with the
name uppercased and dashes turned to underscores. So ``cborg-open`` reads
``CBORG_OPEN_API_KEY``. Put ``CBORG_OPEN_API_KEY=<your CBORG key>`` in the
repo's ``.env``. The same CBORG key serves both entries.

Any model ID from CBORG's catalogue works in the ``models`` list. The
*Historical benchmark snapshot* box on this page shows which ones hold up in
practice. For the provider list and the keys behind it, see
:doc:`configure-providers`.

Which models are known to work
------------------------------

These open families were verified to complete the **full OSPREY end-to-end
suite**. For an agentic system, "runs" means sustaining a multi-step
tool-calling loop across that suite, not just emitting plausible text.

- ``gpt-oss`` (20B and 120B)
- ``gemma`` (``gemma-4``)
- ``cborg-coder``
- ``qwen-3`` / ``qwen-3-coder``

This set was chosen from the open models **available on the CBORG provider**.
It reflects CBORG's catalogue at benchmark time, not an exhaustive survey of
open models. Capability varies widely across the set. The snapshot below is the
real signal, not the bare "it runs" list.

Benchmark it yourself
---------------------

To reproduce the numbers, use the benchmark toolchain OSPREY ships under
``scripts/benchmark/`` (see its `README
<https://github.com/als-apg/osprey/blob/main/scripts/benchmark/README.md>`_). It runs the
model-driving part of ``tests/e2e/``, the tests that actually exercise the
model under test. It runs them across a matrix of models and renders a per-test
pass-rate dashboard.

One file declares the whole run: ``scripts/benchmark/matrix.yaml``. Each row
names a ``provider`` and a model ``id``. The launcher then:

- resolves credentials;
- derives the route (proxy for OpenAI-protocol models, direct for Anthropic);
- wires the judge;
- runs one isolated worker per (model, seed) cell.

Adding a model, or a provider like the local DeepSeek (``ds4``) server, is a
config edit, not a script edit.

.. code-block:: bash

   # see the resolved plan without running anything
   scripts/benchmark/matrix.py --dry-run

   # one model (substring filter), serially
   scripts/benchmark/matrix.py --only gpt-oss-20b --parallel 1

   # the whole grid, then render the dashboard
   scripts/benchmark/matrix.py --parallel 4
   scripts/benchmark/matrix_dashboard.py --results-dir results --out dashboard.html

The dashboard derives every count from the run data at render time. It never
hard-codes a number, so it stays accurate as the suite grows.

.. admonition:: Historical benchmark snapshot — OSPREY v2026.6.2
   :class: important

   Pass rates for the open and self-hosted models below, measured on the
   model-driving subset of the e2e suite in June 2026. This snapshot is
   historical: it is not regenerated with each release. Single-seed Anthropic
   columns are a control/ceiling reference.

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

   :download:`Download the full interactive dashboard (HTML) <../../_static/benchmark-dashboard.html>`

.. seealso::

   - :doc:`configure-providers` — providers, the translation proxy, model selection.
   - `scripts/benchmark/README.md
     <https://github.com/als-apg/osprey/blob/main/scripts/benchmark/README.md>`_
     — the full benchmark contract.
