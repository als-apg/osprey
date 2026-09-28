Check the combined facility file in fixed stages: schema, references, and the
pair, value, limits and seed rules. Each stage runs only when the earlier ones
are clean, and every error of the first failing stage is gathered. An absent
limits.yaml is never an error, and neither is one holding only comments or a
header. No command runs the checks yet.
