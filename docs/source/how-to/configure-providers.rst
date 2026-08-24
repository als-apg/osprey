Configure LLM Providers
=======================

Osprey uses LLM providers in two contexts: **the Osprey agent** (the main agent)
communicates over the Anthropic Messages API, while **MCP tool servers** use
`LiteLLM <https://docs.litellm.ai/>`_ to call any provider. This guide covers
how to configure providers for both.

.. _provider-routing-diagram:

.. raw:: html
   :file: ../_diagrams/provider-routing.html

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
   * - ``amsc-i2``
     - American Science Cloud proxy
     - ``AMSC_I2_API_KEY``
     - OpenAI (proxied)
   * - ``argo``
     - ANL Argo proxy
     - ``ARGO_API_KEY``
     - OpenAI (proxied)
   * - ``asksage``
     - AskSage proxy
     - *(custom auth)*
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
   * - ``ds4``
     - DwarfStar local server
     - *(none)*
     - OpenAI (proxied)

**Protocol** indicates how the provider communicates with the Osprey agent:

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

.. note::

   A shell export reaches a **deployment** only once, when ``osprey init``
   creates the repository, and only for providers that profile references. After
   that, put the key in the repository's ``.env``: it is the deployment's one
   secret store, and nothing else re-reads your shell. See
   :ref:`profile-secrets`.

Provider Configuration
----------------------

Providers are configured in two sections of ``config.yml``:

1. ``api.providers`` — declares available providers with their endpoints and
   model IDs.
2. ``claude_code`` — selects which provider the Osprey agent uses and at which
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
           sonnet: claude-sonnet-4-5-20250929
           opus: claude-opus-4-6

       cborg:
         api_key: ${CBORG_API_KEY}
         base_url: https://api.cborg.lbl.gov/v1
         models:
           # Use pinned versions here — unversioned aliases like
           # anthropic/claude-sonnet break the agent's capability detection.
           haiku: claude-haiku-4-5
           sonnet: claude-sonnet-4-6
           opus: claude-opus-4-7

       stanford:
         api_key: ${STANFORD_API_KEY}
         base_url: https://aiapi-prod.stanford.edu/v1
         models:
           # A gateway need not serve Claude models at all — map its own IDs
           # onto the tiers by capability and cost.
           haiku: gpt-4o-mini
           sonnet: gpt-4o
           opus: o3-mini

Each provider entry needs ``api_key``, ``base_url``, and a ``models`` mapping
that assigns provider-specific model IDs to tiers (``haiku``, ``sonnet``,
``opus``). A full tier map is recommended — subagent tier routing uses it —
but it is not required: a tier that no source maps falls back to the default
model, with a build warning naming each substitution. The framework never
substitutes another provider's model IDs; a provider with no ``models``
mapping *and* no ``default_model`` to fall back on is refused.

``base_url`` is the endpoint the agent itself talks to. ``cborg`` and
``als-apg`` ship a built-in URL that a value here overrides; omit it to keep
the built-in one. ``anthropic`` ships none, so omitting it sends requests to
Anthropic's own API. Keep the trailing
``/v1`` on OpenAI-compatible gateways — the translation proxy needs it, and the
agent's own requests have it stripped automatically.

**Select the active provider** under ``claude_code``:

.. code-block:: yaml

   claude_code:
     provider: cborg
     default_model: sonnet

``provider`` picks one of the entries in ``api.providers``.
``default_model`` selects the model for the main conversation. Give it a tier
name, or any model ID the selected provider serves. An ID found in the
provider's ``models`` block resolves to that tier; any other ID is passed
through verbatim to the provider — a newly released model or a gateway-only
alias works without waiting for the tier map to catch up, and a misspelt ID
fails at the provider (an error naming the ID), not at resolution. If omitted,
``default_model`` falls back to the provider's own default tier — ``sonnet``
for ``anthropic``, ``haiku`` for ``cborg`` and ``als-apg``, and ``opus`` for
custom providers.

Model Tier Mapping
------------------

The Osprey agent uses three model tiers — ``haiku`` (fast/cheap), ``sonnet``
(balanced), and ``opus`` (most capable). Each provider maps these to its own model
IDs via the ``models`` block in ``api.providers``.

The resolver applies model IDs in this priority order:

1. ``claude_code.models`` — explicit per-tier overrides (highest priority).
2. ``api.providers.<name>.models`` — the provider's own model naming.
3. Built-in defaults — the bundled fallback model IDs the framework ships for
   ``anthropic``, ``cborg``, and ``als-apg``.

A tier that no source maps falls back to the default model, with a build
warning naming each substitution — agents pinned to an unmapped tier then run
the default model, not a tier-appropriate one, so a full map is recommended.
No provider ever inherits another provider's model IDs; only a provider with
no models and no ``default_model`` at all is refused.

For example, to override the opus tier for a specific project:

.. code-block:: yaml

   claude_code:
     provider: cborg
     default_model: sonnet
     models:
       opus: claude-sonnet-4-6   # use sonnet even for opus-tier agents

Agents can also be pinned to specific tiers:

.. code-block:: yaml

   claude_code:
     agent_models:
       channel-finder: haiku
       logbook-search: sonnet

Protocol Translation
--------------------

The Osprey agent speaks the Anthropic Messages API. Providers that only offer an
OpenAI-compatible endpoint (marked *OpenAI (proxied)* above) need protocol
translation.

Osprey handles this automatically: when an OpenAI-only provider is selected,
a local translation proxy starts on a random port before the Osprey agent launches.
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

(The unmapped ``opus`` tier here falls back to the default model at build
time, with a warning — map it to silence the substitution.)

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
- Maps ``${MY_PROVIDER_API_KEY}`` to the auth token the Osprey agent expects.
- Injects the resolved model IDs into the Osprey agent's environment.
