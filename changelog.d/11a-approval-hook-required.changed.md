A profile whose approval policy asks for `execute`, `execute_file` or
`pyaml_measure` and does not select the `approval` hook stops at build
(`profile-invalid` on `approval.enabled`). Add `approval` to `hooks:`, or set
`config: approval.enabled: false` (or `skip` for the guarded tools). An
unstated `approval.enabled` counts as asking.
