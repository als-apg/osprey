# osprey-connectors

Lean control-system and archiver connectors for OSPREY: EPICS, DOOCS, and
mock/virtual-accelerator connectors for reading and writing control-system
channels, plus matching archiver connectors (EPICS Archiver Appliance,
DOOCS local history, MongoDB, mock) for historical data. The package also
carries the small set of support modules those connectors depend on —
configuration loading, logging setup, the error/exception taxonomy, and the
data-driven simulation core used by the mock connectors — without pulling in
the rest of the OSPREY framework (agents, capabilities, web UI, etc.).

## Install

```
pip install osprey-connectors
```

## Public API

Every public module lists its public names in `__all__`, and a module whose
name starts with an underscore is internal and can change in any release.

Most modules are their own API, each imported by its own path: `config`,
`workspace`, `dotenv`, `yaml_loader`, `connection`, `types`, `factory`,
`call_timeout`, `errors`, `logger`, `log_filter`, `identity`, `relative_time`,
`control_context`, `posture_env`, `posture_store`, `safety_file`, `settle`,
`process`, `standin` and `honesty`. So are the `simulation` package and its modules
(`simulation.state`, `simulation.view`, `simulation.archive`, ...): its root
imports neither numpy nor lume, and the heavier modules are imported by their
own names.

Three subpackages gather their implementation modules behind one namespace,
and their names are imported from the subpackage itself:

| Namespace | What it holds |
| --- | --- |
| `osprey_connectors.control_system` | The control-system connector contract (`ControlSystemConnector`, `ChannelValue`, `ChannelWriteResult`, `WriteOutcome`, `values_match`, ...) and `LimitsValidator` |
| `osprey_connectors.archiver` | The archiver connector contract (`ArchiverConnector`, `ArchiverMetadata`), the shared time-range helpers, the field name a channel is stored under, the MongoDB client arguments |
| `osprey_connectors.ipc` | What a connector-host supervisor needs: launching and stopping a child, the proxy and the pool, endpoint derivation and verification, and the errors a frame raises (the wire codec itself is internal) |

`osprey_connectors.control_system.write_door`, the door a connector opens around
its own write, is a public module of its own.

The read-only run — `POSTURE_ENV_VAR`, `SANDBOX_MODE` and `is_readonly_run()` —
lives in `osprey_connectors.posture_env`, a leaf that imports nothing from the
package. `osprey_connectors.control_system` exported `is_readonly_run` before
2026.10.0 and no longer does.

The concrete connectors are internal (`control_system._epics_connector`,
`archiver._mongodb_archiver_connector`, ...): a caller builds one through
`ConnectorFactory` from its registered type name.

`osprey-framework` imports these subpackages only through their namespaces, and
takes from every module only names its `__all__` lists. `lint-imports`
(import-linter, configured in the repository's root `pyproject.toml`) holds the
first, and ruff's `PLC2701` and `tests/connectors/test_public_api.py` the
second. import-linter also holds this package to never importing `osprey`, and
holds every top-level module of it in layers: the connector host above the
factory, the factory above the archivers, the archivers above the
control-system connectors, those above the simulation core, and the posture
store, config loading and the bottom layer (`posture_env`, `types`,
`identity`, ...) below them all.

## Import map

These modules used to live under `osprey.*`. The old paths have been
removed; the table maps each old path to the public spelling that replaces it.

| Old path (`osprey.*`) | Import from (`osprey_connectors.*`) |
| --- | --- |
| `osprey.connectors` | `osprey_connectors` |
| `osprey.connectors.factory` | `osprey_connectors.factory` |
| `osprey.connectors.types` | `osprey_connectors.types` |
| `osprey.connectors.honesty` | `osprey_connectors.honesty` |
| `osprey.connectors.control_system`, `.control_system.base`, `.control_system.limits_validator` | `osprey_connectors.control_system` |
| `osprey.connectors.control_system.*_connector` | built through `osprey_connectors.factory.ConnectorFactory` |
| `osprey.connectors.archiver`, `.archiver.base`, `.archiver._timerange` | `osprey_connectors.archiver` |
| `osprey.connectors.archiver.*_archiver_connector` | built through `osprey_connectors.factory.ConnectorFactory` |
| `osprey.errors` | `osprey_connectors.errors` |
| `osprey.utils.config` | `osprey_connectors.config` |
| `osprey.utils.dotenv` (reading the env chain) | `osprey_connectors.dotenv` |
| `osprey.utils.identity` | `osprey_connectors.identity` |
| `osprey.utils.workspace` (config and agent-data resolution) | `osprey_connectors.workspace` |
| `osprey.utils.logger` | `osprey_connectors.logger` |
| `osprey.utils.log_filter` | `osprey_connectors.log_filter` |
| `osprey.utils.relative_time` | `osprey_connectors.relative_time` |

Not everything those modules held moved here. The deployment repo's layout
(layer directories, the image staging area, the audit and archive zones)
belongs to the framework, in a new `osprey.utils.workspace` that holds only
those names. The executor's vocabulary and the
facility timestamp formatter are the framework's `osprey.utils.execution_method`
and `osprey.utils.facility`, and writing `.env` files is
`osprey.deployment.dotenv`.

## Optional runtime dependencies

Only the connectors you actually use need their backing libraries installed;
none of the following are declared as hard dependencies of this package, so
importing `osprey_connectors` never requires them:

- **EPICS Channel Access** — `EPICSConnector` speaks Channel Access through
  `aioca`, a declared dependency, which loads the per-architecture `libca`
  that `epicscorelibs` ships. Nothing else is needed to talk to real IOCs.
- **`pymongo`** — required by `MongoDBArchiverConnector`. Installed with this
  package; the connector imports it inside `connect()` so registration stays
  cheap, not because it is optional.
- **`doocs4py`** — required by `DOOCSConnector` and `DOOCSArchiverConnector`.
  Install separately (it is not on PyPI in all environments) if you connect
  to a DOOCS control system.

## Stability

`osprey-connectors` versions by calendar, `YYYY.M.P` (with `bN` / `rcN` for
pre-releases), on its own release tags: `osprey-connectors-vYYYY.M.P`, cut when
the package changes rather than with every framework release. Its numbers are
not the framework's, and no framework release requires a matching one. (The
`0.1.0` release on PyPI predates this scheme; its tag is retired.)

Each release stays backward-compatible with every supported `osprey-framework`
release, and CI enforces that in its `connectors-compat` and
`framework-at-connectors-floor` lanes. The framework therefore declares only a
floor on this package, never a pin.

The exception taxonomy in `osprey_connectors.errors` is public API: the class
names `ChannelWriteBlockedError`, `ChannelWriteFailedError`,
`ChannelLimitsViolationError` and `ChannelNotFoundError`, and their `reason`
codes, are load-bearing for callers that branch on them.

- `ChannelWriteBlockedError` — the write was refused and no value was written.
  `reason` is one of `WRITES_DISABLED`, `LIMITS`, `VALIDATION_ERROR`,
  `CONTROL_SYSTEM_REFUSED`, `RAW_CLIENT_WRITE`, or `CHANNEL_NOT_FOUND`. The
  first three are refused before any `caput`; `CONTROL_SYSTEM_REFUSED` means
  the control system was asked and denied the write itself (EPICS access
  security, for instance); `RAW_CLIENT_WRITE` means user code called a client
  library's put directly instead of `osprey.runtime.write_channel`; and
  `CHANNEL_NOT_FOUND` means the address is not a channel of the facility.
  Nothing was sent in any of these cases except `CONTROL_SYSTEM_REFUSED`.
- `ChannelWriteFailedError` — the write was attempted but did not verifiably
  succeed. `reason` is one of `FAILED` (the control system did not take the
  value), `MISMATCH` (the channel holds a different value), or `UNCONFIRMED`
  (sent, but not acknowledged in time or the confirming re-read itself
  failed). The codes are the `WriteOutcome` words that
  raise; the exception also carries `outcome`, `value_written` and
  `observed_value`.
- `ChannelLimitsViolationError` — a channel write violated configured safety
  limits (range, read-only, step size, or unlisted channel).
- `ChannelNotFoundError` — a read named an address that is not a channel of the
  facility. Only a connector that holds the facility's whole address set raises
  it; `reason` is `CHANNEL_NOT_FOUND`, the refusal reason a write to such an
  address carries.

Adding classes or reason codes is compatible. Removing or renaming one breaks
the compatibility guarantee above, so it waits until no supported framework
release relies on it.
