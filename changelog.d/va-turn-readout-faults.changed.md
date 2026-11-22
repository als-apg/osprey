A Virtual Accelerator monitor's read-out faults now act on its turn-by-turn
arrays as they act on its orbit reading. The `offset`, `gain`, `polarity`,
`noise` and `roll` a scenario states on a monitor's orbit reading reach
`turn_x` and `turn_y` in the same order, so first-turn steering is rehearsed
on the imperfect monitors the closed-orbit tools already see.
