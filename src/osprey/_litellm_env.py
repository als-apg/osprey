"""LiteLLM's import-time environment, set before any OSPREY module imports it.

LiteLLM reads two settings from ``os.environ`` while it is being imported, so
they have to be in place before the first ``import litellm`` in the process.
Every OSPREY module that imports litellm imports this module first (a test
holds that), rather than ``osprey/__init__.py`` setting them: importing the
package itself configures nothing, and processes that never touch LiteLLM --
the image-render worker runs with a scrubbed environment -- stay untouched.

Both are ``setdefault``, not assignment: an operator who set either one
explicitly keeps their value.
"""

import os

# LiteLLM calls a bare `dotenv.load_dotenv()` at import when LITELLM_MODE is
# "DEV" (its default), which searches *upward* from the process for a `.env`
# and publishes it into os.environ. A git worktree with no `.env` of its own
# therefore inherits the parent checkout's -- importing an OSPREY provider
# module would absorb whatever `.env` sits above it on disk. LITELLM_MODE gates
# nothing else in the package, so this disables that one side effect.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

# LiteLLM fetches its model map from GitHub at import unless told to use the
# copy it ships. On an air-gapped host that fetch stalls every importing
# process until it times out, and which model ids LiteLLM knows would differ
# by host. The bundled copy pins both to the installed LiteLLM version.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
