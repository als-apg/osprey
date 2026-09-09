A mediated channel write now names the limits check it applies instead of
picking it at run time. The tool imports its validator from the same installed
OSPREY, so the older-validator fallback it used to select could never be taken,
and it obscured which checks a refusal had actually made.
