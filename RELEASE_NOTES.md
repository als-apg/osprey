# Osprey Framework - Latest Release (v0.11.5)

**Timezone Support & Open WebUI Fix**

## What's New in v0.11.5

### Highlights

- **Timezone-aware datetime pipeline** - All datetimes are now normalized to the system's local timezone with a human-readable `timezone_name` field (e.g. "EST", "PST"). This fixes UTC/local confusion in responses, generated code, and plots. The `timezone_name` propagates from `TimeRangeContext` through `ArchiverDataContext` to the respond node, ensuring consistent timezone labels throughout.
- **Open WebUI artifacts fix** - Artifacts now display correctly in the Open WebUI interface.

### Fixed
- **Timezone**: Normalize all datetimes to local timezone with `timezone_name` field, fixing UTC offset confusion in LLM responses and generated plots (#189, #187)
- **Open WebUI**: Fix artifacts not showing up in Open WebUI interface (#179)

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
