**Breaking change:** `osprey facility panels write|check|classes` is removed,
and `osprey.facility.panels` with it, with no re-export: the panel engine is
the new `osprey-panels` package, import root `osprey_panels`, with its own
command `osprey-panels write|check|classes`. The framework does not depend on
it. Every input is named on the command line: `--facility` and `--tables` are
required, `--repo` is gone, and `--theme` names the theme file the drawing
colours are held against. The package is not published on PyPI: install it
from a checkout with `pip install ./packages/osprey-panels`, then
`osprey-panels write --facility build/facility.json --tables data/facility/panels --out <dir>`.
