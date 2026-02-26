# Osprey Framework - Latest Release (v0.11.4)

**ARIEL Write Support, Machine State Reader & Google Sheets Integration**

## What's New in v0.11.4

### Highlights

- **ARIEL bidirectional facility adapters** - `FacilityAdapter` interface now supports writes with `supports_write` and `create_entry()`. `GenericJSONAdapter` provides atomic local JSON append; `ALSLogbookAdapter` posts via olog RPC XML with retry. `ARIELSearchService.create_entry()` orchestrates facility-first writes with optimistic local upsert and re-ingestion sync.
- **Machine state reader** - New `MachineStateReader` service for bulk channel snapshots from the control system connector. Pipeline-aware Jinja2 template selects demo channels matching the active channel finder pipeline.
- **Google Sheets channel database** - `GoogleSheetsChannelDatabase` reads/writes channel data from Google Sheets via `gspread`, integrating with the `in_context` pipeline via `source: google_sheets` config option.

### Added
- **ARIEL**: Bidirectional facility adapter write support with new models (`FacilityEntryCreateRequest`, `FacilityEntryCreateResult`, `SyncStatus`, `WriteConfig`) (#174)
- **Machine State**: `MachineStateReader` service with pipeline-aware channel templates (#173)
- **Channel Finder**: Google Sheets channel database backend with `gspread` integration (#171)

### Fixed
- **CI**: Make E2E test failures non-blocking in gate job
- **Tests**: Mark flaky caproto test as `xfail` on macOS CI

---

## Installation

```bash
pip install --upgrade osprey-framework
```

Or install with all optional dependencies:

```bash
pip install --upgrade "osprey-framework[all]"
```

---

## What's Next?

Check out our [documentation](https://als-apg.github.io/osprey) for:
- [ARIEL Quick Start](https://als-apg.github.io/osprey/developer-guides/05_production-systems/07_logbook-search-service/index.html) -- get a working logbook search in minutes
- [Migration assistance](https://als-apg.github.io/osprey/contributing/03_ai-assisted-development.html) -- upgrade existing agents to v0.11
- Native capabilities and `osprey eject` guide
- Complete tutorial series

## Contributors

Thank you to everyone who contributed to this release!

---

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
