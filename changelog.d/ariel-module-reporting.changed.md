`osprey ariel status` now reports every registered enhancement and search
module, including `qmd_export` and any a deployment registered itself, instead of
a fixed list of five. The per-module enhancement counts behind it are aggregated
by module key rather than by name, so a module the framework does not ship is
counted too.
