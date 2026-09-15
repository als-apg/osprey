`osprey up` from a pre-release install no longer fails every image build with
`No matching distribution found for osprey-connectors`. A beta framework exists
only beside a beta `osprey-connectors`, which plain `pip` never picks for a
requirement that names none; every image build now passes `OSPREY_PIP_PRE=1`
for a pre-release pin and the recipes resolve with `--pre`, the same admission
`osprey build` already gave its `uv` resolve. The `pip` fallback of that venv
install does the same.
