# Osprey Framework - Latest Release (v0.10.4)

**Dependency Fix** - litellm/aiohttp compatibility

## What's New in v0.10.4

### Bug Fix

- **Dependencies**: Pin `aiohttp>=3.10` for litellm compatibility (#87)
  - Fixes `AttributeError: module aiohttp has no attribute ConnectionTimeoutError`
  - `aiohttp.ConnectionTimeoutError` was added in aiohttp 3.10; litellm requires it but doesn't pin the version
  - This was causing Docker container deployment failures

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
- TUI mode guide
- Artifact system API reference
- Complete tutorial series

## Contributors

Thank you to everyone who contributed to this release!

---

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
