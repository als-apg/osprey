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
served by a remote aggregator. The provider list and ``config.yml`` keys are in
:doc:`configure-providers`.

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
``scripts/benchmark/`` (see its ``README.md``): it runs the entire ``tests/e2e/``
suite against a matrix of models and renders a per-test pass-rate dashboard. The
portable layer is the e2e harness's environment-variable contract
(``OSPREY_E2E_FORCE_PROVIDER``, ``OSPREY_E2E_FORCE_MODEL``, …) — no per-test edits,
no hard-wired provider; the shell/ssh scripts are copy-and-adapt operator examples.

.. code-block:: bash

   # one model, locally
   scripts/benchmark/run_e2e_for_model.sh <model-id> 1

   # a grid, then render the dashboard
   MATRIX_MODELS="gpt-oss-20b gemma-4" MATRIX_SEEDS="1 2" MATRIX_PARALLEL=4 \
     scripts/benchmark/matrix_driver.sh
   scripts/benchmark/matrix_dashboard.py --results-dir results --out dashboard.html

The dashboard derives every count from the run data at render time — it never
hard-codes a number, so it stays honest as the suite grows.

.. admonition:: Benchmark snapshot — OSPREY 2026.6.2 (pending release re-run)
   :class: important

   *Results table held until the* ``v2026.6.2`` *release re-run.* Once the tag is
   cut, the full matrix is re-run against it and pinned here with provenance:

   - **Measured against:** OSPREY ``v2026.6.2`` (commit ``<sha>``)
   - **Run:** ``<date>`` · open subjects via CBORG, Anthropic reference columns via
     als-apg
   - **Scope:** the model-driving subset of ``tests/e2e/``

   These figures are a point-in-time snapshot and *will* drift — they describe the
   pinned version only. Regenerate with ``scripts/benchmark/`` for any later release.

   *(Inline pass-rate table + downloadable dashboard land here at release.)*

.. seealso::

   - :doc:`configure-providers` — providers, the translation proxy, model selection.
   - ``scripts/benchmark/README.md`` — the full benchmark contract.
