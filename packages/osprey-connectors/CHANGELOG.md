# Changelog

All notable changes to `osprey-connectors` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this package adheres to [Calendar Versioning](https://calver.org/).

The package has its own calendar stream, released from its own
`osprey-connectors-vYYYY.M.P` tags by
`.github/workflows/release-connectors.yml`, independently of the framework's
`v*` tags. Releases are stable by default. Each release stays backward-compatible with every supported
framework release. Changes from before this stream began are recorded in the
repository's root `CHANGELOG.md`, under the framework releases that shipped
them.

Unreleased changes are written as fragments in `changelog.d/` beside this file
and folded into the `[Unreleased]` section below when a release is cut:

    python scripts/changelog_fragments.py apply --root packages/osprey-connectors

## [Unreleased]

## [2026.11.9] - 2026-11-10

### Added

- A `Composite` built with `evolve=True` drives engines that state the time
  contract, the simulator view accepts the `fast` and `event` refresh words, and
  `resolve_fast_tick_s` reads `simulation.fast_tick_s` (default 0.1 s).
- `TextureModel.reach()` and `Composite.texture_reach()` name, for each setting no
  model binds, the channels a write to it can change.

### Removed

- Every name a public module lists now has a user outside that module or is a
  declared extension point. The names nothing used are gone or private:

  - `osprey_connectors.archiver`: `channel_address()` is gone; nothing decodes a
    field name back. `to_utc()`, `require_datetime()`, `resolve_processing()`,
    `long_frame()`, `decimate_raw()`, `aggregate_series()`, `reject_non_numeric()`
    and `Processing` are no longer offered as a helper kit for connector authors;
    they stay private to the in-tree archivers.
  - `osprey_connectors.archive_targets`: `ADDRESS_SETTINGS`, `APPLIANCE_TARGETS`,
    `ARCHIVER_SECTION_KEY`, `DEFAULT_ADDRESS_SETTING`, `NO_TARGETS`,
    `STORE_ARCHIVERS` and `STORE_TARGETS` are private; `archive_targets()` and
    `archive_withheld()` answer the questions.
  - `osprey_connectors.control_context`: `live_report_payloads()` and
    `live_statement_reports()` are gone; a view that shows the machine reads each
    of `report_paths()` with `read_report()` and keeps the reports whose
    `display_only` is true. `state_dir()` is gone: use
    `osprey_connectors.posture_store.state_dir()`. `parse_posture()` is gone: use
    `osprey_connectors.posture_store.parse_posture_value()`. `parse_report()`,
    `posture_mirror()`, `report_path_under()`, `is_foreign_process()`,
    `REACHABLE_STATES`, `SCHEMA_VERSION` and `WRITES_STATEMENT_KEYS` are private.
  - `osprey_connectors.posture_store`: `invalidate_cache()` is gone: use
    `osprey_connectors.control_context.invalidate_cache()`, which holds the cache.
    `LOGIN_TIERS_KEY` is gone; nothing read it. `agent_data_root()`,
    `AGENT_DATA_ROOT_ENV_VAR`, `LAUNCH_POSTURE_ALL_TARGETS` and `VALID_POSTURES`
    are private; `stamped_agent_data_root()` and `state_dir()` stay.
  - `osprey_connectors.standin`: `archive_belongs_to_standin()`,
    `DEPLOYED_SERVICES_KEY` and `ARCHIVER_RECORDER_SERVICE` are gone;
    `osprey_connectors.archive_targets` decides whose history a store holds.
  - `osprey_connectors.session_hold`: `SCHEMA_VERSION` and `EXCERPT_LIMIT` are
    private.
  - `osprey_connectors.settle`: `POLL_INTERVAL_S` is private.
  - `osprey_connectors.recovery`: `Origin` is private.
  - `osprey_connectors.write_target`: `FALLBACK_WRITE_TOOLS` is private.
  - `osprey_connectors.types`: `CLI_ARCHIVER_TYPES`, `TRANSPORT_DOOCS` and
    `TRANSPORT_TANGO` are private; `factory.BUILTIN_CONNECTORS` lists the
    built-in archivers.
  - `osprey_connectors.ipc`: `DEFAULT_PVA_PORT`, `MODE_ADDR_LIST`, `ROLE_PVA`,
    `SETTLE_TIMEOUT_S` and `verify_child_report()` are no longer in the
    namespace.
  - `osprey_connectors.simulation`: the package root no longer re-exports the
    active-scenario state helpers, `STATUS_NO_DECK` or `coerce`; import them from
    `simulation.state`, `simulation.status` and `simulation.values`.
  - `osprey_connectors.simulation.composite`: `ENGINE_GROUP`, `INSTANCES` and
    `UDF` are private, and `STATUS_NO_DECK` is no longer re-listed (it is
    `simulation.status.STATUS_NO_DECK`).
  - `osprey_connectors.simulation.pulse`: every shape constant,
    `SHAPE_CONSTANT_NAMES` and `shape_version()` are private; `trace()` and
    `SHAPE_VERSION` stay, and `SHAPE_VERSION` is unchanged, so no stored archive
    re-seeds.
  - `osprey_connectors.simulation.topoff`: `shape()` is gone (`tau_period()` gives
    its `tau` and period). `DECAY_FORM`, `LIFETIME_KEY`, `REFILL_PHASE`,
    `SECONDS_PER_HOUR`, `TOPOFF_KEY`, `SHAPE_CONSTANT_NAMES` and
    `shape_version()` are private; `SHAPE_VERSION` is unchanged.
  - `osprey_connectors.simulation.channel_faults`: `DISCONNECTED` and `FROZEN`
    are private; `CHANNEL_FAULTS` lists every fault.
  - `osprey_connectors.simulation.values`: `VALUE_TYPES` and `WAVEFORM` are
    private.
  - `osprey_connectors.simulation.view`: `NO_VIEW_MESSAGE` is private.
- `osprey_connectors.write_authorization` no longer lists the helpers nothing
  outside the module calls. `authorization_verdict()`, `spend_step()` and
  `read_ledger()` are gone: a write path takes `authorized_execution_verdict()`,
  and a listing takes `grant_standing()`. `auth_overlap()`, `grant_in_force()`,
  `issue_grant()`, `parse_grant()`, `parse_ledger()`, `parse_request()`,
  `pid_namespace()`, `process_start_time()`, `self_issued_grants()`,
  `SCHEMA_VERSION`, `SURFACES`, `APPROVAL_SURFACES`, `COMPLETIONS` and
  `PRUNE_AFTER_S` are private. No framework release uses any of them.

## [2026.11.8] - 2026-11-10

### Added

- `ChannelError` carries `commands_fired`: the commands, in order, that may have gone out when a batch of commands raised it; `None` otherwise. It is not a wire field.
- `osprey_connectors.ipc.declared_gateways(block)` classifies how a connector
  block spells its gateways table (`GATEWAYS_NONE_DECLARED` for an explicit
  `gateways: []`, `GATEWAYS_MISSING`, `GATEWAYS_ROWS`), and the derivation carries
  it as `gateways_declared`; `declares_no_endpoints` now reads that verdict.
- A connector block can say its target has nothing to probe with an explicit
  `gateways: []`. The derivation carries that as `declares_no_endpoints`, and
  child verification names it in plain words; a missing table is not read as
  that statement.
- The simulator view names its record roles (`SETPOINT`, `READBACK`, `MONITOR`, `OUTPUT`), refresh `PERIODIC`, the `COMMAND_MARK` and `MARKS`, and a `Binding` carries the `marks` its wiring record states; a view without marks reads as before.

### Changed

- `write_target_for_tool` places an application tool (keyed in the render's
  `app_approvals`) on the session's control target, as the hooks do.
- `Binding.engine_role`, `SimulatorView.bindings(engine_role=...)` and `ENGINE_ROLES` name what a model's engine says a wiring record is, apart from a channel's own `role`; the view file is schema `osprey.facility.simulator/3`.

### Fixed

- `osprey_connectors.ipc.encode_error` now tags a subclass of a vocabulary error
  with its registered ancestor, so the far side rebuilds that error instead of a
  `ConnectorInternalError`.
- A write authorization whose holder stopped renewing its lease no longer blocks
  another program's approval on the same channels until it expires: the
  `auth_overlap` check reads the holder's ledgers and skips a grant only when
  every one it finds shows its lease time passed more than a minute ago. A
  request or grant is also refused past three ceilings -- 100000 steps, an
  expiry 24 hours out and a 600-second lease -- whether a person approves it or
  the program issues it itself.

## [2026.11.7] - 2026-11-08

### Fixed

- The archive seed no longer fails a simulator view that has a model with no
  deck. `osprey_connectors.simulation.status.is_error()` is the one rule for
  which model statuses are errors: `ok` and `no deck` are not.

## [2026.11.6] - 2026-11-08

### Added

- Archiver scenarios have two new event shapes. A `burst` jumps by `amplitude` at its instant and decays exponentially over `decay_s`, optionally with a slower tail (`tail_share` of the amplitude decaying over `tail_decay_s`), as a vacuum gauge does after a gas burst. A `loss` drops a channel by `amount` in one step, and only the channel's top-off seed refills it: one shot a second from the first top-off refill after the loss. Neither moves a sample before its instant.

## [2026.11.5] - 2026-11-06

### Added

- The simulator view names its record roles (`SETPOINT`, `READBACK`, `MONITOR`, `OUTPUT`) and refresh `PERIODIC`.

## [2026.11.4] - 2026-11-05

### Changed

- Archiver events on a shared channel no longer block a scenario set: they compose in set order in the archive, so `[vacuum-burst, rf-thermal]` now activates together. Overrides, faults, channel faults and motion still compose only on disjoint targets.
- The package's PyPI page now lists every built-in connector by type name and the
  client library each one needs, groups the public modules by what they are for
  (adding the write-authorization, refusal, session-hold, recovery and
  deployment-machine modules it left out), and links the documentation, repository,
  issue tracker and this changelog.
- A scenario's `still` in the simulator view is always a list of addresses; a view stating anything else is refused.

### Removed

- **Breaking change:** the one-current-job slots for supervised children are
  removed: `process.JobSlots`, `ChildJob`, `ChildExit` and `ExitCause` had no
  user left once the LATTICE panel was retired. `process.terminate` and
  `reap_exit_code` stay. `simulation.view.TEXTURE` is now private.

### Fixed

- Simulation: a model that names no deck and wires no channel is left unbuilt with
  the status `no deck` instead of failing in its engine. (#2235)

### Security

- `osprey_connectors.config.apply_env_chain` sets the env chain into a process
  environment without the shell-only launch variables: a `.env` line that names
  `PATH`, `HOME`, a runtime's start-up or loader variable, or a tool's launch
  variable is skipped, and the names it skipped are returned.

## [2026.11.3] - 2026-11-03

### Added

- `osprey_connectors.write_authorization` defines a program's bounded write
  authorization: a grant naming one machine, its owner and holder, a range per
  channel, a step budget, an expiry and a renewed lease, with a ledger the
  holder and its connector child keep, one step per write call. Its readers fail
  closed, and `authorization_verdict()`, `spend_step()` and
  `authorized_execution_verdict()` judge a whole batch against it. `REASONS`
  gains `auth_unreadable`, `auth_holder`, `auth_channel`, `auth_range`,
  `auth_exhausted`, `auth_lease_lapsed`, `auth_expired`, `auth_revoked` and
  `auth_overlap`, and `WriteVerdict` gains `authorization`.

  A process that names an authorization (`OSPREY_WRITE_AUTHORIZATION`) has every
  write call checked against it by the connector's write guard: listed channels,
  values in range, a step left, a live lease, no stop. One call is one step,
  spent before anything is put; a batch — the base class's, which a connector's
  own `write_multiple_channels` gives way to under an authorization — is checked
  whole first, so a policy refusal writes nothing, and its per-channel writes
  are not checked again. The machine's switch may stay off, and turning it on
  does not lift the bounds. A process that names no authorization is checked
  exactly as before. The bounds hold a program that runs as written: the budget
  and the lease are the holder's own record, and the gateway is the boundary.

  `ConnectorHostPool.connector()` takes `owner=` and `authorization=`; a key that
  names either is `(target, execution_mode, owner, authorization)`, its child gets
  `OSPREY_CONTROL_OWNER` and `OSPREY_WRITE_AUTHORIZATION`, selects the write
  gateway when the authorization allows, and reports it as `authorization`. A
  child of such a key that came up before its authorization could write refuses
  `switching`, spending nothing, and that refusal comes back like any other.
  `authorized_write_verdict()` is the answer a gateway is chosen on.
- A stop through `control_context.set_switch()` (named machines, or `"all"`) now
  marks every program authorization on the stopped machines revoked inside the
  record lock, so the holder's next write refuses `auth_revoked`; `set_switch()`
  takes `by=` to name whose stop it was. `osprey_connectors.write_authorization`
  gains the holder's request (`write_request()`, `parse_request()`,
  `list_requests()`, `withdraw_request()`, `OSPREY_WRITE_REQUESTS_DIR`), the
  person's approval (`approve_request()`, which issues the grant under the
  request's id, and `approved_grant()`, which a holder waits on), `prune_grants()`,
  `grant_in_force()`, `read_ledger()`, and `listed_grants()` and
  `grant_standing()`, which every listing shares: a grant whose lease lapsed or
  whose steps are spent is shown ended, with its steps used. An approval or a
  self-issue refuses `auth_overlap` against another holder's live grant in the
  owner's own directory and the holders' self-issued grants, seen by an
  approver through its read-only view of their ledger directory
  (`OSPREY_WRITE_LEDGER_VIEW_DIR`, `self_issued_directories()`,
  `self_issued_grants()`); a grant file there that cannot be read is logged and
  left out, never a veto. `grant_in_force()` holds a self-issued grant in force
  only while its switch stands at `allowed`. `control_context.held_epoch()`
  holds the record lock around a writer that stamps the record's epoch. A stop
  whose revocation marks cannot be written moves the record's epoch instead,
  ending every grant of that owner, and the authorizations locks are created
  group-writable (0660) so every uid sharing a grants directory can take them.
- `ChannelWriteResult.verdict_reason` and `ChannelWriteBlockedError.verdict_reason`
  name the write verdict's reason for a `WRITES_DISABLED` refusal — `switch_off`,
  `switching`, `auth_exhausted`, `auth_range` and the rest — across the
  connector-host pipe as well, so a program can tell them apart without parsing
  the message. The pipe's frame protocol version is 2: a version-1 peer, which
  would drop the reason, is refused.

## [2026.11.2] - 2026-11-02

### Removed

- **Breaking change:** four more names nothing outside their own module used are
  no longer public API: `posture_store.launch_pin_term`, and
  `write_target.is_write_call`, `is_write_tool` and `short_tool_name`. Each is now
  underscore-prefixed and out of `__all__`.

### Security

- `osprey_connectors.config.apply_env_chain` sets the env chain into a process
  environment without the shell-only launch variables: a `.env` line that names
  `PATH`, `HOME`, a runtime's start-up or loader variable, or a tool's launch
  variable is skipped, and the names it skipped are returned.

## [2026.11.1] - 2026-11-02

### Added

- `archive_targets()` says which control targets the deployment's `archiver:`
  block serves: the deployment's own MongoDB store serves the virtual
  accelerator on every deployment that builds one; every other archiver but the
  mock (the EPICS appliance, MYA, DOOCS) is the facility's and serves the live
  machine and the stand-in that rehearses it, when its block names an address.
- `ControlSystemConnector.wait_stale()` raises `ConnectorUnavailableError` — the
  lost connector, which its supervisor replaces — once the connector's transport
  has gone stale: the IOC serves a channel again but the client library keeps it
  down. EPICS CA with a name server configured watches the channels that
  connected and went down: a channel counts once another channel connected
  after it went down, or once it has been down for 30 s with nothing
  connecting. Only channels that connected in this process count, so in a
  replacement a channel that never comes back cannot drive a restart loop.
  Every other connector waits for ever.

  Known limit: a process in which no channel has ever connected never reports
  stale.
- Connectors answer `channel_key(address)`, the one key every spelling of a
  channel shares: whitespace stripped, and on EPICS `ca://X`, `X.VAL` and `X`
  are one channel.
- A connector subscription can now say whether its channel is connected:
  `subscribe()` takes a keyword-only `on_connection` callback, which EPICS CA,
  PVA and the in-process simulator call with `True` before the first value and
  `False` on a drop (DOOCS and Tango accept it and never call it).
  `ChannelMetadata` now carries the control, warning and alarm limit pairs and
  whether this client may write the channel; a bound that is not a finite
  number, or a pair whose low is not below its high, reads as `None`.
- The simulator runs top-off: a `topoff` seed serves a stored
  current that decays with a lifetime and refills on a period, and a
  `topoff_lifetime` seed serves a readback of that lifetime, so current and
  lifetime agree live and in the archive.
- A control-context report may carry a machine `statement`: `reachable`
  (`yes` until the connector is lost) and `writes`, the execution verdict's
  `{allowed, position, reason, kind}` as `control_context.write_statement()`
  builds it. Such a report is a display process's: `live_reports` leaves it
  out and `live_statement_reports` reads it. A `statement` of any other shape
  states nothing.
- A simulated `step` series on a float channel may end at a given time, after
  which the channel returns to its baseline.
- `session_hold_refusal()` and `execution_verdict()` take `session_keys=`, the
  sessions a write belongs to, and refuse `session_held` while any one of them
  has an open flag, with that session's sources; a process that serves several
  sessions (a panel's data process) names them instead of the environment's
  one key. Without the keyword nothing changes.
- `mark_long_lived_session()` marks a process that holds one connector for
  hours, such as a panel backend, so a readonly refusal there uses the
  deployment-wide wording, as in an MCP server.
- `subscribe(..., on_connection=...)` works on every connector, including one a
  facility wrote before the keyword existed: when a subclass's `subscribe` does
  not take `on_connection`, the base class passes it a wrapped callback and
  reports `True` before the channel's first value (never `False`, since such a
  connector never says a channel dropped). A `subscribe` that takes the keyword
  receives it unchanged. The wrapper is a deprecated compatibility path: defining
  such a subclass emits a `DeprecationWarning` naming it, and the wrapper will be
  removed in osprey-connectors 2027.3.0.

### Changed

- EPICS CA subscriptions are served from one record per channel address,
  however many subscribers it has (`X` and `ca://X` share one). An IOC restart
  no longer delivers duplicate values, every subscription carries the channel's
  limits, and `get_metadata` of a subscribed address answers from its control
  data. `warm()` starts the CA context ahead of the first subscription.
- The simulated septum pulse's flat top now rings with a few damped components
  whose amplitude, phase and edge timing change from shot to shot. Archives
  seeded with the old pulse shape re-seed.

### Deprecated

- `standin.archive_belongs_to_standin`, `standin.ARCHIVER_RECORDER_SERVICE` and
  `standin.DEPLOYED_SERVICES_KEY` are kept only for the framework releases that
  import them and go in a later release: the deployment's store is the virtual
  accelerator's and no store belongs to the stand-in. Ask `archive_targets()`
  which targets an archive serves.

### Fixed

- A web terminal no longer follows a control-context record whose owner PID now belongs to another user's process. After an upgrade that put an init process at PID 1 of the container, the record left by the previous container still named PID 1 as its owning terminal; the new terminal read root's init as a live owner and refused every target switch with `context_owned_elsewhere`.
- An EPICS write now ends within `S + 2T` (`step_read_timeout_s` and `timeout_s`
  of its connector block): the `max_step` read, the put and the confirming read
  each get one deadline that covers their connection as well as their call, the
  confirming read reads the value and alarm state only (no lazy units, precision
  or limit fetches), and a string written to an enum channel is put as its state
  index with the labels fetched inside the put's deadline. A write runs on the
  connector's own executor, which starts a thread per phase, so neither one-shot
  reads filling the default executor nor other writes in flight can delay it.
  That executor holds at most 64 threads at once; a write submitted past the cap
  raises `ConnectorInternalError` before anything is sent.
- The in-process simulator accepts `on_connection` on `subscribe` and reports its channels connected, as every connector's subscription contract promises.
- A PVA subscription whose server goes away no longer goes silent on its last
  value: the drop arrives as a connection-down report on the same monitor, never
  as a value. `get_metadata` of a channel already subscribed answers from that
  subscription instead of opening a second request, and raises
  `ChannelUnreachableError` when no update arrives within `timeout_s`.

## [2026.11.0] - 2026-11-01

### Added

- Writes can be paused for one session: while a session has an unanswered hold,
  every write it executes is refused with the reason `session_held`, naming what
  raised the hold, and `execution_verdict` orders that refusal into the write
  verdict. The connection and gateway stay as they were across hold and release.

## [2026.10.2] - 2026-10-31

### Changed

- The real machine's banner now reads `Live — writes move hardware` and its refusal word is `the live machine`.

## [2026.10.1] - 2026-10-31

### Changed

- **Three-position write switch.** The login tier is three-valued
  (`never | asked | allowed`) and caps the switch. The switch record stores a
  position per machine, `asked` or `allowed` (absent is off); a stored `on`
  reads as `asked`, and an unknown position refuses. `set_switch` takes
  `asked`, `allowed`, and still `on` (as `asked`). An older package refuses
  every login against a login-tier map holding `asked`, so release this
  package before the framework.
- `osprey_connectors.errors.RefusalReason` is the one refusal-reason
  vocabulary; each member names who refuses and what to do next. A refused
  write result must name one; only a refused result may carry a reason, and
  only a refused, failed or unconfirmed one a message.
- `ChannelError` is the base of every per-channel failure and names the
  channel the same way in process and across the connector host; the IPC
  exception registry derives from it.
- Two failure roots: `ChannelUnreachableError` for a channel a connector
  cannot reach, `ConnectorUnavailableError` when its transport is gone (an
  unreachable archive service counts as the latter). The connector-host
  boundary carries both, and the pool replaces a child whose connector is
  unavailable.
- Connector recovery is decided by one policy, `osprey_connectors.recovery`:
  failure class and origin in, at most one act per child.
- Composite: a model failing while an active scenario faults it or overrides
  one of its setpoints names those scenarios first in its status. (#2188)
- `identity` names the spool directory a record is written into.

### Fixed

- `is_process_alive` no longer reads a Linux thread id as a running process.
  (#2172)
- The archive composite closes its state directory, and the mock archiver
  closes a composite it replaces or drops.
- A connector bug inside the connector-host child crosses the boundary as
  `ConnectorInternalError`, not as an unreachable control system; a reply the
  parent cannot rebuild fails only that call, and the connection stays up.
- `settle` raises `SettleReadError` only when the readback read failed with a
  `ChannelError` or answered no value; a lost connector propagates as its own
  error.
- Stopping the Channel Access loop closes the loop once its thread has ended.

## [2026.10.0] - 2026-10-29

The first release on the package's own calendar stream, cut independently of the
framework. Its public API is what the framework and external tools may rely on
from here on: names only the framework used have moved into the framework, and
the read-only run has one owner in `posture_env`. Framework 2026.9.0b1–b3 are
superseded and not supported with this release.

### Added

- `osprey_connectors.posture_env` is the one home of the read-only run:
  `POSTURE_ENV_VAR`, `SANDBOX_MODE` and `is_readonly_run()`. The module imports
  nothing from the package, so the posture store no longer imports the
  control-system connectors to ask it, and import-linter now holds every
  top-level module of the package in layers.

### Changed

- **Breaking change:** the EPICS connector and the connector-host pool choose
  the write gateway from the write verdict: a connector whose writes are allowed routes through
  `gateways.write_access`, otherwise it stays on `read_only`. A write allowed
  after the gateway was chosen is refused as `switching` until the connector
  reconnects. A machine with only a write gateway reads through it while its
  writes are off, refused in software; a read-only run on such a machine
  refuses to connect.

### Removed

- **Breaking change:** names only the framework used leave the public API
  before the first stable release. From `workspace`: the layer directories (`TIERS_DIRNAME`,
  `APPS_DIRNAME`, `LAYER_DIR_KINDS`, `LAYER_DIRNAMES`, `RETIRED_LAYER_DIRS`,
  `layer_dir_kind`), `PROFILE_FILENAME`, `STATE_DIR_NAME`, `IMAGE_DIR_NAME`,
  `container_image_context`, `RUNTIME_DATA_DIR_NAME`, `AUDIT_DIR_RELPATH`,
  `ARCHIVE_DIR_RELPATH`, `STATE_ZONE_DIRS`, `CHAT_UPLOADS_SUBDIR`,
  `SIMULATOR_STANDIN_LOG_SUBDIR`, `deployment_env_chain` and
  `repo_root_for_agent_data`. From `dotenv`: every `.env` writer and the
  compose-safety checks (`append_profile_env`, `write_env_merged`,
  `env_file_lock`, `env_lock_path`, `atomic_write`, `format_env_line`,
  `dotenv_line_var`, `compose_unsafe_vars`, `compose_unsafe_vars_in_chain`,
  `EnvConflict`, `ProfileEnvAppendResult` and the file banners); the chain
  reader stays. From `config`: `resolve_execution_method`,
  `EXECUTION_METHOD_SUBPROCESS`, `DEFAULT_EXECUTION_TIMEOUT_SECONDS`,
  `to_facility_iso`, `RUNTIME_WRITE_PATH_KEYS` and
  `find_runtime_write_paths_under_data`. The config load's runtime-write
  advisory now checks only `simulation.state_dir`: the other key it named,
  `services.channel_finder.pipelines.hierarchical.feedback.store_path`, is a
  spelling no reader uses, so it never matched a real config.
- `osprey_connectors.control_system` no longer exports `is_readonly_run`; import
  it from `osprey_connectors.posture_env`.
- **Breaking change:** `osprey_connectors.posture_store` no longer exports the
  old narrowing and store readers: `effective_writes`, `store_permits`,
  `store_verdict`, `store_verdict_detail`, `record_verdict_detail`,
  `StoreVerdict`, `StoreVerdictDetail`, `recorded_posture`, `target_posture`,
  `launch_permits`, `launch_narrowed_target` and `parse_launch_posture`. Ask
  `write_verdict(section, target)` instead — its `allowed` and `reason` replace
  the bool and the verdict member, and a run whose launch pin refuses answers
  `reason == "launch_pinned"`.
- **Breaking change:** names nothing outside their own module used are no longer
  public API. `config.get_framework_service_config` and `log_filter.LoggerFilter`
  are removed. These are now private: `config.RUNTIME_STATE_DIR`,
  `control_context.OWNER_KINDS`, `live_report_payloads`, `record_path_under`,
  `report_path` and `report_path_under`, `logger.QUIET_THIRD_PARTY_LOGGERS`,
  `log_filter.suppress_logger_level`,
  `process.REAP_WINDOW_S`, `standin.ARCHIVER_RECORDER_BLOCK_KEY` and
  `archive_recorder_configured`, `types.ARCHIVER_SETTINGS_LEAF`,
  `LIMITS_BLOCK_LEAVES` and `ONE_REAL_MACHINE`, `yaml_loader.safe_dumper`,
  `simulation.archive.DEFAULT_CHUNK_SIZE`, `DEFAULT_WORKERS`,
  `SEED_SCHEMA_VERSION`, `prepare_collection`, `seed_grid`,
  `synthesize_documents` and `write_manifest`,
  `simulation.composite.LOG_RECORD_MAX_BYTES`, `STATUS_MAX_BYTES` and
  `cap_status`, and `simulation.envelope.MOTION_SIGMAS` and `noise_sigma`.
  `factory` no longer exports the test-only registry helpers
  (`isolated_connector_registries`, `snapshot_connector_registries`,
  `restore_connector_registries`, `ConnectorRegistrySnapshot`).
  `control_system` no longer exports `METADATA_FIELDS`, `archiver` no longer
  exports `HOST_OVERRIDE_ENV` and `PORT_OVERRIDE_ENV`, and `ipc` no longer
  exports the wire codec and exception registry (`encode_request`,
  `FrameReader`, `ResultFrame`, `EXCEPTION_SPECS`, ...) or `CHILD`,
  `CHILD_MODULE`, `DEFAULT_CA_PORT`, `NEVER_LIVE_TYPES` and `same_endpoint`;
  `ipc` still exports `FrameDecodeError` and `FrameEncodeError`.
- **Breaking change:** `osprey_connectors.ipc` no longer exports
  `config_writes_enabled`. Ask `posture_store.write_verdict(section, target)`
  instead.
- **Breaking change:** `osprey_connectors.types` no longer exports the
  `writes_enabled` readers: `WRITES_ENABLED_KEY`, `TYPE_WRITES_ENABLED_LEAF`,
  `type_writes_enabled`, `target_writes_enabled`, `target_writes_enabled_key`,
  `session_posture`, `any_target_writes_enabled`,
  `any_armed_target_checks_limits`, `writes_enabled_key`,
  `writes_enabled_answering_key` and `writes_enabled_remedy`. Use `WRITES_KEY`
  with `tier_writes(section)` or `tier_allows(section)`, which no longer read
  the retired keys.

### Fixed

- Limits checking in `optional` mode with no `database_path` no longer needs a limits database: every channel is written as unlisted. A configured path must still exist.
