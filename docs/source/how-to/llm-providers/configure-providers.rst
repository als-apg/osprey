.. _how-to-configure-providers:

Configure LLM Providers
=======================

Osprey uses LLM providers in two contexts: **the OSPREY agent** (the main agent)
communicates over the Anthropic Messages API, while **MCP tool servers** call
the same named providers directly through `LiteLLM <https://docs.litellm.ai/>`_.
This guide covers how to configure providers for both.

.. _provider-routing-diagram:

.. raw:: html
   :file: ../../_diagrams/provider-routing.html

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
     - ALS Accelerator Physics Group gateway
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
   * - ``ds4``
     - DwarfStar local server
     - *(none)*
     - OpenAI (proxied)

**Protocol** indicates how the provider communicates with the OSPREY agent:

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

``als-apg`` ships the endpoint of the gateway it fronts,
``https://llm.als.lbl.gov``, so the key is all a deployment needs. A site that
reaches the gateway somewhere else names that host instead — in the shell, or
straight in ``providers.yml`` under ``api.providers.als-apg.base_url``:

.. code-block:: bash

   export ALS_APG_BASE_URL="https://your-gateway.example.org/v1"

The variable beats a value in the config, which is what makes it a runtime
redirect for a deployment whose endpoint is already baked into an image.

A provider that requires an endpoint and ships none behaves differently: every
path that would place a call refuses rather than sending the gateway's token to
another host. A launch — ``osprey chat``, ``osprey web``, an agent run — stops
with a message naming the variable and the config key that would settle it, and
a direct model call and ``osprey health`` report the same thing more briefly, as
``Base URL required for <provider>``.

On a multi-user deployment such an endpoint has to be in the repository's env
chain rather than only in your shell. Each per-user terminal runs with
``.env.users``, a generated file the deploy copies the provider's key and
endpoint into, and it is copied from what is on disk — never from the
environment ``osprey up`` runs in. A chain that sets neither is refused before
any container starts, naming the variable.

.. note::

   A shell export reaches a **deployment** only once, when ``osprey init``
   creates the repository, and only for providers that profile references. After
   that, put the key in the repository's ``.env``: it is the deployment's one
   secret store, and nothing else re-reads your shell. See
   :ref:`profile-secrets`.


.. _provider-configuration:

Provider Configuration
----------------------

Two files in the deployment repository configure providers, and they answer
different questions. ``providers.yml`` is the catalog: every provider this
deployment can name, with its endpoint and the model ids it serves. ``profile.yml`` selects
one of them, with its top-level ``provider:`` and ``model:`` keys
(``osprey set provider=… model=…`` writes them for you).

A gateway that OSPREY does not ship is an entry appended to ``providers.yml``:

.. code-block:: yaml

   providers:
     my-gateway:
       api_key: ${MY_GATEWAY_API_KEY}
       base_url: https://my-gateway.example.com/v1
       default_model: claude-sonnet-5
       models:
         - claude-opus-5
         - claude-sonnet-5
         - claude-haiku-4-5

and one line in ``profile.yml`` naming it:

.. code-block:: yaml

   provider: my-gateway

The catalog is the one home for these facts, so a ``config: api.providers.*``
key in ``profile.yml`` is refused, naming ``providers.yml`` as the place to put
it; and a ``provider:`` the catalog does not declare is refused with the list of
names it does declare. ``osprey profile expand --providers`` refreshes the
entries OSPREY ships and keeps yours.

``osprey build`` renders the profile and the catalog into ``build/config.yml``,
which every build wipes and re-renders; edit the two source files, never the
rendered one. The rendered file has two relevant sections:

1. ``api.providers`` — declares available providers with their endpoints and
   the model ids each serves.
2. ``claude_code`` — selects which provider the OSPREY agent uses and, when the
   profile names one, which model.

The YAML blocks below show that **rendered** ``build/config.yml``, so you can
see what the two source files become. The whole catalog appears under
``api.providers``:

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
         default_model: claude-sonnet-5
         models:
           - claude-fable-5-1
           - claude-opus-5-5
           - claude-opus-5
           - claude-sonnet-5
           - claude-haiku-4-5

       cborg:
         api_key: ${CBORG_API_KEY}
         base_url: https://api.cborg.lbl.gov/v1
         default_model: claude-haiku-4-5
         health_model: claude-haiku-4-5
         models:
           - claude-opus-5
           - claude-sonnet-5
           - claude-haiku-4-5

       stanford:
         api_key: ${STANFORD_API_KEY}
         base_url: https://aiapi-prod.stanford.edu/v1
         default_model: gpt-4o
         models:
           - gpt-4o-mini
           - gpt-4o
           - o3-mini

Each entry in ``providers.yml`` takes ``base_url``, ``default_model`` and
``models`` — a list of the model ids the gateway serves, spelled as the gateway
spells them, which must contain ``default_model`` — plus optional ``api_key``,
``health_model`` (the cheapest served id, used by ``osprey health``) and
``claude_code_aliases`` (see :ref:`claude-code-alias-names` below). Use the
versioned ids a gateway serves: an unversioned alias such as
``anthropic/claude-sonnet`` carries no version for the agent's capability
detection to match.

``base_url`` is the endpoint the agent itself talks to. Every entry the shipped
catalog carries names one, and a value here replaces it: the institutional
gateways (``cborg``, ``amsc-i2``, ``stanford``, ``als-apg``) and the vendors'
own APIs are spelled out, while ``ollama`` and ``argo`` are spelled as a
variable with the well-known host as its default. For ``als-apg``,
``ALS_APG_BASE_URL`` overrides the entry in turn, as above. A gateway a
deployment adds itself has no shipped entry, so it names its endpoint here or
refuses to start. Keep the trailing
``/v1`` on OpenAI-compatible gateways — the translation proxy needs it, and the
agent's own requests have it stripped automatically.

``extra_body`` is an optional mapping sent in the request body of every
completion. A LiteLLM gateway that uses client-side auth reads a per-user
upstream key from it, so ``api_key`` stays the gateway credential while
``extra_body: {api_key: ${UPSTREAM_KEY}}`` carries the user's own.

**Select the active provider** with the profile's two top-level fields:

.. code-block:: yaml

   provider: cborg
   model: claude-sonnet-5

which render as ``claude_code.provider`` and ``claude_code.default_model``:

.. code-block:: yaml

   claude_code:
     provider: cborg
     default_model: claude-sonnet-5

Both rendered keys are the build's to write, so spelling either under
``config:`` is refused, naming the field to set instead.

Which model answers
-------------------

``provider`` picks one of the entries in ``providers.yml``. ``model`` is the
deployment's main model: a model id the provider serves. Omit it and the
provider entry's ``default_model`` answers. An id the entry's ``models`` list
does not carry is still used, and the build names it in one warning line. A
newly released model works before the catalog lists it, and a misspelt id fails
at the provider (an error naming the id). A bare ``haiku``, ``sonnet`` or
``opus`` is refused, with the ids the provider serves: those words are Claude
Code's alias names, not model ids.

Every job that calls a model names its own id or runs on the main model:

* ``claude_code.agent_models.<agent>`` — the model one agent runs; omitted, the
  main model.
* ``channel_finder.channel_name_generation.llm_model.model_id`` — omitted, the
  main model.
* ``logbook.composition.model`` — the compose panel's model when the operator
  picks none; omitted, the main model. The panel offers the ids the provider
  serves.

.. code-block:: yaml

   provider: cborg
   model: claude-sonnet-5
   config:
     claude_code.agent_models.channel-finder: claude-haiku-4-5
     claude_code.agent_models.logbook-deep-research: claude-opus-5

.. _claude-code-alias-names:

Claude Code's alias names
-------------------------

Claude Code has three alias names of its own — ``haiku``, ``sonnet`` and
``opus`` — which it reads from ``ANTHROPIC_DEFAULT_HAIKU_MODEL``,
``ANTHROPIC_DEFAULT_SONNET_MODEL`` and ``ANTHROPIC_DEFAULT_OPUS_MODEL``, and its
own background calls ask for ``haiku``. OSPREY fills all three at build:

1. Derived from the served list: each alias takes the newest served id of that
   Claude family (``claude-opus-5`` over ``claude-opus-4-6``).
2. A gateway's ``claude_code_aliases`` in its catalog entry wins over
   derivation.
3. ``claude_code.aliases.<name>`` in a deployment's ``profile.yml`` wins over
   both.
4. An alias nothing resolves points at the main model, and the build prints
   one line naming the substitution — on a gateway that serves no Claude
   models, all three.

.. code-block:: yaml

   config:
     claude_code.aliases.haiku: claude-haiku-4-5

``osprey status --agents`` lists each alias with its model and where it came
from, and each agent's model with its origin.

Protocol Translation
--------------------

The OSPREY agent speaks the Anthropic Messages API. Providers that only offer an
OpenAI-compatible endpoint (marked *OpenAI (proxied)* above) need protocol
translation.

Osprey handles this automatically: when an OpenAI-only provider is selected,
a local translation proxy starts on a random port before the OSPREY agent launches.
No manual configuration is required — you never invoke the proxy yourself.

The path is identical whether the endpoint is self-hosted (``ollama``, ``vllm``
— local, so no API key) or a remote service that speaks only the OpenAI
protocol.

If you run a custom gateway that speaks Anthropic natively (e.g., a LiteLLM
proxy in Anthropic mode), add ``api_protocol: anthropic`` to its
``providers.yml`` entry to skip the translation proxy:

.. code-block:: yaml

   providers:
     my-litellm-gateway:
       api_key: ${MY_GATEWAY_KEY}
       base_url: https://my-gateway.example.com/v1
       api_protocol: anthropic
       default_model: claude-sonnet-5
       models:
         - claude-sonnet-5
         - claude-haiku-4-5-20251001

``api_protocol`` takes exactly two values, ``anthropic`` and ``openai``.
Anything else — including a capitalised ``Anthropic`` — is refused when the
provider is resolved, naming the provider and the two accepted values. Leave
the key out and the provider is treated as OpenAI, which is what all but the
Anthropic-native built-ins are.

Spend Attribution on a LiteLLM Gateway
--------------------------------------

A deployment authenticates to its gateway with one key, so without help the
gateway's spend logs book every terminal, the dispatch worker and every
headless run to that single key. When the provider is a `LiteLLM proxy
<https://docs.litellm.ai/docs/proxy/cost_tracking>`_, OSPREY stamps the acting
identity onto every request instead:

* ``x-litellm-end-user-id`` — who asked: the terminal's roster user
  (``OSPREY_TERMINAL_USER``), a service container's framework identity such as
  ``dispatch-worker-0``, or the local account of an ``osprey chat`` session.
* ``x-litellm-tags`` — ``osprey,surface:<terminal|dispatch|service|local>``.

The agent carries them through Claude Code's ``ANTHROPIC_CUSTOM_HEADERS``
(merged into any corporate-proxy headers you already set there), and the
LiteLLM SDK path used by MCP servers sets the same identity as the OpenAI
``user`` field. Nothing is sent to a direct vendor.

The built-in ``als-apg`` and ``cborg`` providers are LiteLLM proxies and get
this automatically; a ``gateway:`` key on one of those names overrides that
default, and ``gateway: none`` turns attribution off for an entry that points
the name at a direct endpoint. A custom gateway declares it in its
``providers.yml`` entry:

.. code-block:: yaml

   providers:
     my-litellm-gateway:
       api_key: ${MY_GATEWAY_KEY}
       base_url: https://my-gateway.example.com/v1
       api_protocol: anthropic   # or omit it for the OpenAI route
       gateway: litellm

On the gateway, read the result per person with ``/customer/info?end_user_id=``
or from the ``end_user`` and ``request_tags`` columns of ``/spend/logs``. Both
need the gateway to run with a database (virtual keys enabled); a stateless
LiteLLM ignores the headers.

Verifying Connectivity
----------------------

After configuring a provider, check that the API key and endpoint work:

.. code-block:: bash

   osprey health

Adding a New Provider
---------------------

To add a new OpenAI-compatible provider, append an entry to ``providers.yml``
beside ``profile.yml`` (see
:ref:`Provider Configuration <provider-configuration>` above) and run
``osprey build`` — no code changes required. ``osprey health`` reports such a
config-only provider as *skipped*, not failed: there is no adapter class to
probe it with, so the run says it went unverified rather than grading the
deployment unhealthy. The rendered result in ``build/config.yml``:

.. code-block:: yaml

   api:
     providers:
       my-provider:
         api_key: ${MY_PROVIDER_API_KEY}
         base_url: https://api.my-provider.com/v1
         default_model: my-model-large
         models:
           - my-model-small
           - my-model-large

   claude_code:
     provider: my-provider

The framework automatically:

- Detects that ``my-provider`` is not a built-in Anthropic-native provider.
- Starts the translation proxy to bridge Anthropic → OpenAI protocols.
- Reads the OSPREY agent's auth token from ``MY_PROVIDER_API_KEY``. The launcher
  derives that variable name from the provider's own name — uppercased, dashes to
  underscores — and never reads the entry's ``api_key`` value, so the name here
  lines up only because the provider is called ``my-provider``.
- Injects the resolved model ids into the OSPREY agent's environment; Claude
  Code's three alias names point at ``my-model-large``, the main model, since
  the provider serves no Claude models.

.. note::

   **This no-code entry serves the OSPREY agent.** MCP tool servers resolve a
   provider by *name* against the built-in table in this guide, so a
   config-only entry means nothing to them and a tool call that asks for it
   fails with ``Unknown provider``. Giving an MCP tool server a new provider
   takes code: a provider class registered under that name through a
   ``ProviderRegistration`` in your application's registry. That registry file
   is the one named by ``registry_path`` in the project's ``config.yml`` (set
   it in your profile's ``config:`` block) or by the ``REGISTRY_PATH``
   environment variable; see :doc:`/contributing/extending-osprey`.
