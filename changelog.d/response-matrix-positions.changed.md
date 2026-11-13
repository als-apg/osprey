`model_response_matrix` places every row and column along the ring: the tool's
row and column objects gain `position`, the SDK's `ResponseMatrix` gains
`row_positions` and `column_positions`, the lattice element index, counted for
`kind="trajectory"` from the injection element. The trajectory response's rows
are now the monitors' `turn_x` and `turn_y` records, in metres, instead of the
orbit readbacks.
