"""The retired layer directory and its successor, as tests name them.

Read from :data:`osprey.utils.workspace.RETIRED_LAYER_DIRS`, which holds
exactly one entry for as long as the deprecation window is open. Closing the
window deletes that entry, and the unpacking below then fails at import in every
test that still exercises the retired directory — the list of what to delete.
"""

from osprey.utils.workspace import RETIRED_LAYER_DIRS

((RETIRED_LAYER_DIRNAME, RETIRED_LAYER_SUCCESSOR),) = RETIRED_LAYER_DIRS.items()
