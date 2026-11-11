=============
LLM Providers
=============

Use these pages to choose which provider and which model each OSPREY model
call uses. You make two separate decisions:

- **Which provider.** A provider is an endpoint and its API key. It can be an
  Anthropic-native endpoint (Anthropic itself or an institutional proxy), an
  OpenAI-compatible endpoint, or a model served on your own hardware. You
  declare providers in one catalog, ``providers.yml`` beside the deployment's
  ``profile.yml``, which is rendered into ``api.providers``.
- **Which model handles which task.** This is routing. OSPREY routes for two
  consumers, and you choose the provider and model for each one separately:

  - **The OSPREY agent** speaks the Anthropic Messages API. Set its provider
    and model in the profile's top-level ``provider:`` and ``model:`` fields.
  - **The MCP (Model Context Protocol) tool servers** call the same named
    providers directly through LiteLLM. Set each tool server's own provider
    setting. Some of these default to the agent's.

Both consumers draw endpoints and keys from the same catalog, so a key you set
there serves both.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Configure LLM Providers
      :link: configure-providers
      :link-type: doc
      :shadow: md

      The available providers, the API key each one expects, the
      ``providers.yml`` catalog that declares them, and the profile fields that
      point the agent at one.

   .. grid-item-card:: Run Open & Local Models
      :link: run-open-models
      :link-type: doc
      :shadow: md

      Run the agent on open-weight or self-hosted models. Covers how the local
      translation proxy bridges an OpenAI-protocol endpoint to the Anthropic
      Messages API, remote or on your own machine. Also lists the open models
      known to sustain the full end-to-end (e2e) test suite, and how to benchmark one yourself.

.. toctree::
   :hidden:

   configure-providers
   run-open-models
