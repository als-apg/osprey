**Breaking change:** OSPREY now runs CANOPY rebased onto SLAC's current main,
where the formula module is `formula-engine` (was `fox`), served at
`/formula-engine/...`. Build the CANOPY payload from the rebased commits
`proven.py` names; a payload built at the old commits is refused.
