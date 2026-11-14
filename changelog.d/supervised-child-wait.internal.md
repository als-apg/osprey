The web terminal's child supervisor takes its backoff wait as a constructor argument
beside its clock, the real wait by default, so its schedule can be driven without
wall time.
