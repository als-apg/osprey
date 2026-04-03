# APS Archiver Connector: Design & Implementation

**Date:** 2026-04-03
**File:** `aps-control-assistant/aps_archiver_connector.py`

---

## 1. Problem

The OSPREY framework's built-in EPICS archiver connector uses the `archivertools` Python library (`als-archiver-client`), which was developed for the ALS archiver. When querying the APS archiver at `https://pvarchiver.aps.anl.gov`, it fails with:

```
list index out of range
```

The `archivertools` library's `match_data()` method internally constructs URLs, parses responses, and aligns timestamps in a way that is incompatible with the APS Archiver Appliance response format. The error occurs inside the library, not in OSPREY code.

### Root Cause

| Aspect | archivertools (ALS) | APS Archiver |
|--------|-------------------|--------------|
| URL construction | Library-internal, ALS-specific | Standard `/retrieval/data/getData.json` |
| Response parsing | Expects DataFrame with `secs`/`nanos` columns | Returns JSON array of `{meta, data}` objects |
| Empty data handling | May fail on `data[0]` access | Returns `[]` for no-data PVs |
| Error recovery | All-or-nothing (one PV failure breaks all) | Per-PV error isolation needed |

## 2. Design Decision: Local Plugin, Not Framework Patch

OSPREY is designed for **multi-facility** use (ALS, APS, NSLS-II, etc.). Modifying the framework's `epics_archiver_connector.py` to fix APS-specific behavior would risk breaking other facilities.

Instead, OSPREY's `ConnectorFactory` supports **dynamic connector loading** via dotted module paths:

```python
# factory.py line 188-198
if "." in connector_type:
    module_path, class_name = connector_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    connector_class = getattr(module, class_name)
```

This allows any project to register a custom connector by:
1. Placing a Python file in the project root
2. Setting `archiver.type` to the dotted class path in `config.yml`

No framework source code is modified.

## 3. Implementation

### 3.1 Connector File

**`aps_archiver_connector.py`** — a single file in the project root implementing `ArchiverConnector` (the OSPREY abstract base class).

#### Key Functions

| Function | Purpose |
|----------|---------|
| `_build_retrieval_url(base_url)` | Appends `/retrieval/data/getData.json` if not present |
| `_to_iso8601_utc(dt)` | Converts datetime to ISO8601 UTC with trailing `Z` |
| `_parse_series(payload)` | Parses the archiver JSON response into `{pv: pd.Series}` |
| `_fetch_data(...)` | Batch + per-PV fallback REST calls to the archiver |

#### Data Flow

```
User query: "plot SR beam current last 2 hours"
    │
    ▼
MCP tool: archiver_read(channels=["S-DCCT:CurrentM"], start="2h ago")
    │
    ▼
ConnectorFactory.create_archiver_connector()
    │  reads config.yml → type = "aps_archiver_connector.APSArchiverConnector"
    │  imports aps_archiver_connector module from project root
    │  instantiates APSArchiverConnector
    │  calls connector.connect(config) with {url, timeout}
    │
    ▼
APSArchiverConnector.get_data(pv_list, start, end)
    │
    ▼
_fetch_data() → GET https://pvarchiver.aps.anl.gov/retrieval/data/getData.json
    │              ?from=2026-04-03T09:23:00Z&to=2026-04-03T11:23:00Z&pv=S-DCCT:CurrentM
    │
    ▼
_parse_series() → extracts {secs, nanos, val} → pd.Series with UTC DatetimeIndex
    │
    ▼
Returns pd.DataFrame → MCP tool saves to artifact store → plot rendered
```

#### Batch + Fallback Strategy

```
Step 1: Batch request (all PVs in one HTTP call)
    GET ...?pv=PV1&pv=PV2&pv=PV3

Step 2: For any PVs missing from batch response, try individually
    GET ...?pv=PV2  (if PV2 had no data in batch)

This handles cases where one PV has issues without failing the entire request.
```

#### Response Parsing

The APS archiver returns JSON in this format:

```json
[
  {
    "meta": {"name": "S-DCCT:CurrentM", "PREC": "1"},
    "data": [
      {"secs": 1743676980, "nanos": 0, "val": 127.78, "severity": 0, "status": 0},
      {"secs": 1743676985, "nanos": 0, "val": 127.75, "severity": 0, "status": 0}
    ]
  }
]
```

The parser:
- Iterates each entry in the array
- Extracts PV name from `meta.name`
- Converts `secs + nanos/1e9` to UTC datetime
- Unwraps single-item list values (some PVs return `"val": [127.78]`)
- Skips entries with empty `data` arrays gracefully

### 3.2 Configuration

In `config.yml`:

```yaml
archiver:
  type: aps_archiver_connector.APSArchiverConnector

  "aps_archiver_connector.APSArchiverConnector":
    url: https://pvarchiver.aps.anl.gov
    timeout: 60
```

The quoted key `"aps_archiver_connector.APSArchiverConnector"` is required because YAML interprets unquoted dots as nested keys. The factory looks up `config["archiver"][connector_type]` where `connector_type` is the literal dotted string.

### 3.3 Processing Modes

The connector supports archiver server-side processing operators:

| Mode | Example PV | Description |
|------|-----------|-------------|
| `raw` | `S-DCCT:CurrentM` | All archived samples (default) |
| `mean` | `mean_600(S-DCCT:CurrentM)` | 10-minute averaged data |
| `max` | `max_3600(S-DCCT:CurrentM)` | Hourly maximum values |

When `processing` and `bin_size` are provided, the PV name is wrapped: `{processing}_{bin_size}({pv})`.

## 4. Comparison with Old ITS Connector

The old `its-control-assistant` had a 514-line `aps_archiver.py` with additional features:

| Feature | Old (aps_archiver.py) | New (aps_archiver_connector.py) |
|---------|----------------------|-------------------------------|
| Direct REST API | Yes | Yes |
| Batch + per-PV fallback | Yes | Yes |
| SDDS logger fallback | Yes | No (not needed for SR PVs) |
| Local timezone conversion | Yes (TIME_PARSING_LOCAL) | No (OSPREY handles TZ) |
| Time range filtering | Manual check per point | Trusts archiver response |
| Lines of code | 514 | 200 |

The new connector is simpler because OSPREY's MCP layer handles timezone conversion, time range parsing, and data formatting. The connector only needs to fetch and parse.

## 5. Switching Back

To revert to the built-in `archivertools`-based connector:

```yaml
archiver:
  type: epics_archiver
  epics_archiver:
    url: https://pvarchiver.aps.anl.gov
    timeout: 60
```

To use mock data for development:

```yaml
archiver:
  type: mock_archiver
```
