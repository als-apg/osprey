A registered kind's `settings_from_measurement(model_entry, measurement_record)`
returns `{<engine>: {<setting>: value}}`; the build applies the patches of the
kinds a measurement file lists to the model's settings once, before the engine
checks them, so the simulator view and every build stage read one settings map.
A patch of a setting the model states by hand, or that another listed kind
patches, stops the build naming both.
