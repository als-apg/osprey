`SimulatorView` serves the records of `osprey_connectors.simulation.records`
themselves: `channel()` answers a `ChannelRecord`, `models()`, `model()` and
`physics_models()` a `ModelRecord` (was `ModelEntry`), and `bindings()` and
`binding()` `WiringRecord` items. `Channel`, `Binding` and `Model` are gone. Read
a model's wiring as `model.wiring` (was `bindings`), its deck as
`view.deck_path(model)` (was the absolute `deck`), what an engine is built from
as `record.engine_entry()` and `model.engine_settings()` (was `binding.record`
and `settings`), and a wiring record's model from the loop over `view.models()`
or `view.channel(address).owner` (was `binding.model`). Every wiring record
states `id`, `direction`, `engine_role`, `plane` (`null` for none) and
`refresh`, and a channel states `options` only on a `bool` or `enum` channel,
`shape` exactly on a `waveform` one and `precision` only on a `float` one, with
both sides of a `value_range`: a view that breaks one is refused at read with
`ViewSchemaError` naming the key ("rebuild with osprey build").

`SimulatorView(path)` reads nothing: open a view with `SimulatorView.open()`
(or `of_render`, `of_project`, `find`). `document()` takes the document class,
`view.document(VariablesDocument)` instead of `view.document(VARIABLES_FILE)`;
each document names its file and schema as `FILE` and `SCHEMA`, and
`VIEW_DOCUMENTS` lists the five. `view.check()` reads every file at once, and
`document_bytes(document)` writes a view file's bytes. `view_path(project_dir)`
is where a project's view is, and `sim_address(code, model, signal)` spells a
status or IOC health address.

`ChannelRecord.value_type` is a `ValueType` (`osprey_connectors.simulation.values`),
and `channel.coerce(value)`, `channel.zero()` and `channel.labels` replace
`values.channel_labels(...)`, `texture.channel_value_type(...)` and
`texture.is_float_channel(...)`, which are gone; the free `coerce` and `zero`
stay for raw values. `seed_fingerprint()` returns a `SeedFingerprint` and
`compare_fingerprint()` takes one; `FingerprintComparison.differences` holds
`KnobChange(key, stored, expected)` records instead of tuples. The stored
manifest is unchanged.
