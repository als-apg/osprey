"""The package's command judges the installed CANOPY; ``--pins`` prints the pins.

Stdlib-only, like its siblings: it runs where neither CANOPY nor the terminal's
dependencies are importable.
"""

import sys

from osprey.integrations.canopy.proven import main

if __name__ == "__main__":
    sys.exit(main())
