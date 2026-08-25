=============
LLM Providers
=============

Two separate decisions sit behind every model call OSPREY makes. *Which
provider* you use -- an Anthropic-native endpoint (Anthropic itself or an
institutional proxy), an OpenAI-compatible endpoint, or a model served on your
own hardware -- is a matter of endpoints and API keys. *Which model handles
which task* is routing, and OSPREY has two consumers to route for: the OSPREY
agent speaks the Anthropic Messages API, while the MCP tool servers call the
same named providers directly through LiteLLM. Both draw endpoints and keys
from one ``api.providers`` block, so a key set there serves both -- but which
provider and model tier each one runs on is chosen separately:
``claude_code.provider`` for the agent, and each tool server's own provider
setting, some of which default to the agent's.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Configure LLM Providers
      :link: configure-providers
      :link-type: doc
      :shadow: md

      The available providers, the API key each one expects, and the
      ``api.providers`` and ``claude_code`` blocks that declare them and point
      the agent at one.

   .. grid-item-card:: Run Open & Local Models
      :link: run-open-models
      :link-type: doc
      :shadow: md

      Running the agent on open-weight or self-hosted models: how the local
      translation proxy bridges an OpenAI-protocol endpoint to the Anthropic
      Messages API, the same way whether the model is remote or on your own
      machine, which open models are known to sustain the full e2e suite, and
      how to benchmark one yourself.

.. toctree::
   :hidden:

   configure-providers
   run-open-models
