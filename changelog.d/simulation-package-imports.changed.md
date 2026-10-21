`osprey_connectors.simulation` no longer re-exports the engine, machine and expression names; import them from `osprey_connectors.simulation.engine`, `.machine` and `.expressions`.
The package root now imports neither numpy nor lume and holds the tick period (`resolve_tick_s`), `decode_char_waveform`, `coerce` and the active-scenario state helpers.
