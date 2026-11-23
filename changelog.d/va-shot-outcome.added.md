A turn-by-turn ring says what became of its last shot. Three new wiring
signals beside `turns_tracked`: `shot_status` (`1` tracked, `2` the bunch could
not be set up, `3` an unexpected error), `turns_survived` (the last turn the
beam reached the last armed monitor) and `monitors_reached` (the armed monitors
the beam reached on the first turn). A shot that is not tracked is now logged
with its reason, an unexpected error with its traceback, instead of reading as
a lost beam.
