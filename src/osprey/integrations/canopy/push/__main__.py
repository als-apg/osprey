"""``python -m osprey.integrations.canopy.push``: run the push review CLI."""

import sys

from osprey.integrations.canopy.push.verbs import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
