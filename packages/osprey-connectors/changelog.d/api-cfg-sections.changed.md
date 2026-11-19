**Breaking change:** the functions that read the `control_system:` section
(`resolve_target`, `baseline_target`, `configured_targets`, `switch_capable`,
`target_configured`, `machine_kind`, `resolve_control_system_type`,
`resolve_serving`, `connector_transport`, `speaks_channel_access`,
`talks_to_network`, `tier_writes`, `tier_allows`, the five limits-posture
readers, `owner_tier_term`, `write_ceiling`, `write_verdict`,
`backstop_verdict`, `execution_verdict` and the four write-authorization
verdicts) take a `osprey_connectors.config_sections.ControlSystemSection`
instead of the raw mapping: write `ControlSystemSection.read(config["control_system"])`
once and pass it. `read` never raises, and every answer is the one the raw
mapping gave. `ConnectorFactory.build_control_system_connector` and
`create_control_system_connector` take the section too; `None` still reads the
loaded config's.
`TierWord`, `LimitsMode`, `GatewayRole`, `SelectedRole`, `GatewayMode` and
`GatewaysDeclared` are now `osprey_connectors.config_sections`' (`gateways`
still re-exports the gateway words).
`ConnectorContext.settings` is a `ControlSystemSettings` (`osprey_connectors.control_system`):
the factory reads the type's block into the connector class's `settings_model`
(`EPICSConnector`, `TangoConnector` and the in-process simulator declare one),
and `_open(settings)` receives it. An out-of-tree connector reads its own keys
from `settings.model_extra`, or declares a `ControlSystemSettings` subclass as its
`settings_model`. A mapping handed to `ConnectorContext(settings=...)` is still
accepted and read for the context's type.
A block the settings model cannot read fails the build with a
`ConfigurationError` naming the key, where it used to fail at `connect()` or not
at all: a bad `timeout_s`, a `fresh_reads` that spells neither true nor false,
`pva_channels` that is not a glob or a list of them, and a gateway whose
`address` is not a non-blank string, whose `port` is not an integer (a numeric
string is read as its number) or whose `use_name_server` spells neither true nor
false. `derive_endpoints` refuses the selected role's gateway the same way (a malformed
row of the other role derives no endpoint), and
`Endpoint.host` is a `str` and `Endpoint.port` an `int | None`.
`write_authorization.Login` is now `osprey_connectors.model.Login`, and a
connection block's `auth.username` with space around it is refused rather than
sent as written.
The loaded config carries the section, read once when it is loaded:
`current_config().control_system`. The write-path readers (the reference
monitor, the EPICS read-only refusal, `LimitsValidator.from_config` and
`from_config_most_restrictive`, the connector host's write report, and the
factory with no section given) read it there, so a test that fakes the section
installs a config (`install_config(testing.config_of({...}))` or
`using_config(...)`).
