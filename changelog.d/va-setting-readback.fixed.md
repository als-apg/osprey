A virtual-accelerator write of a physics setting now reads back only the
setting, its readback and what its model computes from its settings, instead
of the whole served roster, so an apply over every corrector no longer runs
past `action_timeout_s`.
