Configure LLM Providers
=======================

Osprey uses LLM providers in two contexts: **Claude Code** (the main agent)
communicates over the Anthropic Messages API, while **MCP tool servers** use
`LiteLLM <https://docs.litellm.ai/>`_ to call any provider. This guide covers
how to configure providers for both.

Available Providers
-------------------

.. list-table::
   :header-rows: 1
   :widths: 15 35 15 25

   * - Name
     - Description
     - API Key Env Var
     - Protocol
   * - ``anthropic``
     - Anthropic direct API
     - ``ANTHROPIC_API_KEY``
     - Anthropic (native)
   * - ``cborg``
     - LBNL CBorg proxy
     - ``CBORG_API_KEY``
     - Anthropic (native)
   * - ``als-apg``
     - ALS Accelerator Physics Group AWS proxy
     - ``ALS_APG_API_KEY``
     - Anthropic (native)
   * - ``stanford``
     - Stanford AI Playground
     - ``STANFORD_API_KEY``
     - OpenAI (proxied)
   * - ``amsc``
     - American Science Cloud proxy
     - ``AMSC_I2_API_KEY``
     - OpenAI (proxied)
   * - ``argo``
     - ANL Argo proxy
     - ``ARGO_API_KEY``
     - OpenAI (proxied)
   * - ``asksage``
     - AskSage proxy
     - ``ASKSAGE_API_KEY``
     - OpenAI (proxied)
   * - ``openai``
     - OpenAI (GPT models)
     - ``OPENAI_API_KEY``
     - OpenAI (proxied)
   * - ``google``
     - Google (Gemini models)
     - ``GOOGLE_API_KEY``
     - OpenAI (proxied)
   * - ``ollama``
     - Ollama (local models)
     - *(none)*
     - OpenAI (proxied)
   * - ``vllm``
     - vLLM inference server
     - *(none)*
     - OpenAI (proxied)

**Protocol** indicates how the provider communicates with Claude Code:

- **Anthropic (native)**: Speaks the Anthropic Messages API directly. No
  translation needed.
- **OpenAI (proxied)**: Speaks the OpenAI Chat Completions API. Osprey
  automatically starts a local translation proxy to bridge the protocols.

Setting Up API Keys
-------------------

Set the API key as an environment variable before running Osprey:

.. code-block:: bash

   # Direct vendors
   export ANTHROPIC_API_KEY="sk-ant-..."
   export OPENAI_API_KEY="sk-..."
   export GOOGLE_API_KEY="AIza..."

   # Institutional proxies
   export CBORG_API_KEY="..."
   export AMSC_I2_API_KEY="..."
   export ALS_APG_API_KEY="..."
   export ARGO_API_KEY="..."
   export STANFORD_API_KEY="..."

Ollama and vLLM run locally and do not require an API key.

Provider Configuration
----------------------

Providers are configured in two sections of ``config.yml``:

1. ``api.providers`` — declares available providers with their endpoints and
   model IDs.
2. ``claude_code`` — selects which provider Claude Code uses and at which
   model tier.

**Declare providers** under ``api.providers``:

.. note::

   Model IDs change every few months as new Claude, GPT, and Gemini releases
   ship. The IDs below were current at the time of writing — always check your
   provider's documentation (Anthropic, OpenAI, Google, CBORG, etc.) for the
   latest available model names before copying these values verbatim.

.. code-block:: yaml

   api:
     providers:
       anthropic:
         api_key: ${ANTHROPIC_API_KEY}
         base_url: https://api.anthropic.com
         models:
           haiku: claude-haiku-4-5-20251001
           sonnet: claude-sonnet-4-6
           opus: claude-opus-4-7

       cborg:
         api_key: ${CBORG_API_KEY}
         base_url: https://api.cborg.lbl.gov/v1
         models:
           haiku: anthropic/claude-haiku
           sonnet: anthropic/claude-sonnet
           opus: anthropic/claude-opus

       stanford:
         api_key: ${STANFORD_API_KEY}
         base_url: https://aiapi-prod.stanford.edu/v1
         models:
           haiku: claude-3-haiku
           sonnet: claude-4-sonnet
           opus: claude-4-sonnet

Each provider entry needs ``api_key``, ``base_url``, and a ``models`` mapping
that assigns provider-specific model IDs to tiers (``haiku``, ``sonnet``,
``opus``).

**Select the active provider** under ``claude_code``:

.. code-block:: yaml

   claude_code:
     provider: cborg
     default_model: sonnet

``provider`` picks one of the entries in ``api.providers``.
``default_model`` selects the tier for the main conversation (defaults to
``sonnet`` for most providers).

Model Tier Mapping
------------------

Claude Code uses three model tiers — ``haiku`` (fast/cheap), ``sonnet``
(balanced), and ``opus`` (powerful). Each provider maps these to its own model
IDs via the ``models`` block in ``api.providers``.

The resolver applies model IDs in this priority order:

1. ``claude_code.models`` — explicit per-tier overrides (highest priority).
2. ``api.providers.<name>.models`` — the provider's own model naming.
3. Built-in defaults — Anthropic direct model IDs (fallback).

For example, to override the opus tier for a specific project:

.. code-block:: yaml

   claude_code:
     provider: cborg
     default_model: sonnet
     models:
       opus: anthropic/claude-sonnet   # use sonnet even for opus-tier agents

Agents can also be pinned to specific tiers:

.. code-block:: yaml

   claude_code:
     agent_models:
       channel-finder: haiku
       logbook-search: sonnet

Protocol Translation
--------------------

Claude Code speaks the Anthropic Messages API. Providers that only offer an
OpenAI-compatible endpoint (marked *OpenAI (proxied)* above) need protocol
translation.

Osprey handles this automatically: when an OpenAI-only provider is selected,
a local translation proxy starts on a random port before Claude Code launches.
No manual configuration is required.

If you run a custom gateway that speaks Anthropic natively (e.g., a LiteLLM
proxy in Anthropic mode), add ``api_protocol: anthropic`` to skip the
translation proxy:

.. code-block:: yaml

   api:
     providers:
       my-litellm-gateway:
         api_key: ${MY_GATEWAY_KEY}
         base_url: https://my-gateway.example.com/v1
         api_protocol: anthropic
         models:
           haiku: claude-haiku-4-5-20251001
           sonnet: claude-sonnet-4-5-20250929

Verifying Connectivity
----------------------

After configuring a provider, check that the API key and endpoint work:

.. code-block:: bash

   osprey health

Adding a New Provider
---------------------

To add a new OpenAI-compatible provider, add an entry to ``api.providers``
in ``config.yml`` — no code changes required:

.. code-block:: yaml

   api:
     providers:
       my-provider:
         api_key: ${MY_PROVIDER_API_KEY}
         base_url: https://api.my-provider.com/v1
         models:
           haiku: claude-3-haiku
           sonnet: claude-3-sonnet
           opus: claude-3-opus

   claude_code:
     provider: my-provider
     default_model: sonnet

The framework automatically:

- Detects that ``my-provider`` is not a built-in Anthropic-native provider.
- Starts the translation proxy to bridge Anthropic → OpenAI protocols.
- Maps ``${MY_PROVIDER_API_KEY}`` to the auth token Claude Code expects.
- Injects the resolved model IDs into Claude Code's environment.
