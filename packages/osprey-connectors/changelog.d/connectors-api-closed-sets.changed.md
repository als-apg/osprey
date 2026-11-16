Every closed vocabulary in the connectors API has one named `Literal` type, and
the fields, parameters and returns that hold one of its words are typed with it;
the words themselves are unchanged. New names: `types.MachineKind`, `Transport`,
`Serving`, `LimitsMode` and `is_control_target()`; `posture_store.SwitchPosition`,
`ArmedPosition`, `Posture`, `OwnerTierTerm` and `WriteCeiling`;
`control_context.OwnerKind`; `write_authorization.ApprovalSurface`,
`ApproverSurface` and `RunCompletion`; `errors.LimitsViolationKind`;
`service_answers.ServiceKind` and `SourceKind`; `directory.DirectoryStatus`;
`archiver.ProcessingMode`; the gateway's `GatewayRole` (now including `pva`),
`SelectedRole`, `GatewayMode` and `GatewaysDeclared`; the connector host's
`StartPhase`, `HostStartStage` and `HostLossCause`; and the simulation view's
`ChannelRole`, `Direction`, `EngineRole`, `Plane`, `Refresh`, `Mark`,
`ChannelFaultKind`, `IocHealthSignal` and `ServingInstance`.

Records now refuse a word outside their set: `ControlContext.target`,
`Grant.target`, `WriteRequest.target`, `ServerReport.applied_target`,
`TargetDerivation.target` and `ConnectorHostError.target` are a `ControlTarget`;
`DirectoryQuery.status` is `Active` or `Inactive`; `Owner.kind`,
`LimitsPosture.mode`, `WriteVerdict.kind` and `.position`, `Endpoint.mode` and the
simulation view's wiring and channel fields take only their words, and a view or
scenario that states another word is refused when it is read. A `ControlContext`
built with a `switch` value that arms nothing reads it as off, as a parsed record
does. `get_data(processing=...)` is typed `ProcessingMode`.
`write_request()` refuses a target that is not a control target with
`ValueError`.

The tuple constants (`TIER_WORDS`, `SWITCH_POSITIONS`, `LIMITS_MODES`,
`SERVING_MODES`, `PROCESSING_MODES`, `CONTROL_TARGETS`, `START_PHASES`,
`ENGINE_ROLES`, `PLANES`, `REFRESH`, `MARKS`, `IOC_HEALTH_SIGNALS`) are derived from
their type with the same values in the same order. `simulation.archive.SeedState`
is a `StrEnum`. `channel_faults.FROZEN` and `DISCONNECTED` are public beside
`STUCK`. The texture model's name is one constant,
`osprey_connectors.simulation.TEXTURE`; `texture.TEXTURE_OWNER` is removed.
