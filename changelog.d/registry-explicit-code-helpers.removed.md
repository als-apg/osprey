`osprey.registry` no longer exports `get_framework_defaults()` or
`generate_explicit_registry_code()`; nothing in OSPREY called them. Use
`extend_framework_registry()` to build on the framework registry.
