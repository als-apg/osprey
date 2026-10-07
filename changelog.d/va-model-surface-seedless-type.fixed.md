The virtual accelerator boots a model whose model-only writable variables
include a type with no `default_value` field, such as a particle-group
variable. Such a variable has no boot seed, and `reset` leaves it alone.
