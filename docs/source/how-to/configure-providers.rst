Configure LLM Providers
=======================

Osprey uses `LiteLLM <https://docs.litellm.ai/>`_ as a unified adapter layer
for AI model providers. This guide explains how to select a provider, configure
API keys, and add custom providers.

Available Providers
-------------------

.. list-table::
   :header-rows: 1
   :widths: 15 35 10 10 20

   * - Name
     - Description
     - API Key
     - Base URL
     - Default Model
   * - ``anthropic``
     - Anthropic (Claude models)
     - Required
     - No
     - ``claude-haiku-4-5-20251001``
   * - ``openai``
     - OpenAI (GPT models)
     - Required
     - No
     - ``gpt-5``
   * - ``google``
     - Google (Gemini models)
     - Required
     - No
     - ``gemini-2.5-flash``
   * - ``cborg``
     - LBNL CBorg proxy
     - Required
     - Required
     - ``anthropic/claude-haiku``
   * - ``amsc``
     - American Science Cloud proxy
     - Required
     - Required
     - ``claude-haiku``
   * - ``als-apg``
     - ALS Accelerator Physics Group AWS proxy
     - Required
     - Required
     - ``claude-haiku-4-5-20251001``
   * - ``argo``
     - ANL Argo proxy
     - Required
     - Required
     - ``claudesonnet45``
   * - ``stanford``
     - Stanford AI Playground
     - Required
     - Required
     - ``gpt-4o``
   * - ``asksage``
     - AskSage proxy
     - Required
     - Required
     - ``google-claude-45-haiku``
   * - ``ollama``
     - Ollama (local models)
     - No
     - Required
     - ``mistral:7b``
   * - ``vllm``
     - vLLM inference server
     - No
     - Required
     - *(depends on served model)*

Providers marked **Required** under *Base URL* are OpenAI-compatible proxies
that need a custom endpoint. Providers without a required base URL use the
vendor's default API endpoint.

Setting Up API Keys
-------------------

Set the API key as an environment variable before running Osprey. The
conventional variable names are:

.. code-block:: bash

   # Direct vendors
   export ANTHROPIC_API_KEY="sk-ant-..."
   export OPENAI_API_KEY="sk-..."
   export GOOGLE_API_KEY="AIza..."

   # Institutional proxies
   export CBORG_API_KEY="..."
   export AMSC_API_KEY="..."
   export ALS_APG_API_KEY="..."
   export ARGO_API_KEY="..."
   export STANFORD_API_KEY="..."
   export ASKSAGE_API_KEY="..."

Ollama and vLLM run locally and do not require an API key.

Configuring a Provider in config.yml
-------------------------------------

The ``provider`` section in your project's ``config.yml`` selects the active
provider and its settings.

**Anthropic (direct API)**

.. code-block:: yaml

   provider:
     name: anthropic
     model_id: claude-haiku-4-5-20251001

**OpenAI-compatible proxy (CBORG example)**

.. code-block:: yaml

   provider:
     name: cborg
     model_id: anthropic/claude-haiku
     base_url: https://api.cborg.lbl.gov

**Local Ollama**

.. code-block:: yaml

   provider:
     name: ollama
     model_id: mistral:7b
     base_url: http://localhost:11434

**Self-hosted vLLM**

.. code-block:: yaml

   provider:
     name: vllm
     model_id: meta-llama/Llama-3-8b
     base_url: http://localhost:8000/v1

Verifying Connectivity
----------------------

After configuring a provider, check that the API key and endpoint are working:

.. code-block:: bash

   osprey health

The health command makes a minimal API call using the cheapest available model
for the selected provider.

Adding a Custom Provider
------------------------

To add a new provider, create a module under
``src/osprey/models/providers/`` that subclasses ``BaseProvider``.

1. **Create the adapter file** (e.g., ``my_provider.py``) under
   ``src/osprey/models/providers/``. Subclass ``BaseProvider`` and implement
   ``execute_completion`` and ``check_health`` by delegating to the helpers
   ``execute_litellm_completion`` and ``check_litellm_health`` from
   ``litellm_adapter``. See any existing provider (e.g., ``cborg.py``) as a
   template.

2. **Set routing attributes** on the class:

   - ``is_openai_compatible = True`` -- for OpenAI-compatible endpoints.
     LiteLLM routes as ``openai/{model_id}`` with a custom ``api_base``.
   - ``litellm_prefix`` -- overrides the LiteLLM provider prefix when it
     differs from the provider ``name`` (e.g., ``"gemini"`` for Google).

3. **Register the provider** in the provider registry (follow the pattern of
   existing entries).

4. **Test** with ``osprey health`` and a matching ``config.yml`` section.
