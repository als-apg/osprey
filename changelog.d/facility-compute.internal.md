The pyat engine's `prepare` also reports the deck's length, the end of the s
axis `locate` reports on, taken as written and never multiplied by the deck's
periodicity.

The facility build computes each wired device's model, position and length
from its model's deck (the shortest arc on a periodic deck, first entrance to
last exit on a single-pass one), places devices by the deepest span of their
model, numbers them per class in each place and each model, and lists each
device's groups. It stops on spans that do not resolve or overlap, an imported
place that contradicts its span, a device or address wired by two models, a
repeated wired element, a declared `texture` model or a channel on a status
address, and a nominal outside its limits band. `build_facility` runs every
stage in memory and returns the facility file, and `validate` now runs the same
stages. No command writes the file yet.
