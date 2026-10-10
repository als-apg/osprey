Check the combined facility file in fixed stages: schema, references, and the
pair, value, limits and seed rules. Each stage runs only when the earlier ones
are clean, and every error of the first failing stage is gathered. limits.yaml
holds records only: a channel without a record is locked, a setpoint record with
both bounds is writable within them, and every record confirms its writes unless
it sets `confirm: false`. `writable: true` on a non-setpoint and a non-integral
bound on an int channel are `limit-invalid`. An absent limits.yaml is never an
error, and neither is one holding only comments or a header. No command runs the
checks yet.
