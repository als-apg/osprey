# Local Timezone Support for Time Range Parsing

## Summary

Added support for local timezone time parsing in the `time_range_parsing` capability. This allows applications to configure whether time ranges should be parsed in UTC (default) or local timezone.

## Changes

### `src/osprey/capabilities/time_range_parsing.py`

#### New Environment Variables

- `TZ`: Specifies the local timezone (default: `America/Chicago`)
- `TIME_PARSING_LOCAL`: Set to `true` to use local timezone instead of UTC (default: `false`)

#### New Imports

```python
import os
from zoneinfo import ZoneInfo
```

#### New Module-Level Constants

```python
# Get local timezone from TZ environment variable
LOCAL_TIMEZONE = os.environ.get("TZ", "America/Chicago")

# Get local time parsing preference from environment variable
TIME_PARSING_LOCAL = os.environ.get("TIME_PARSING_LOCAL", "false").lower() in ("true", "1", "yes")
```

#### Updated `_get_time_parsing_system_prompt()`

- Added `local: bool = False` parameter
- When `local=True`, uses local timezone for current time reference
- Updated prompt to display timezone information (e.g., "LOCAL TIME, America/Chicago timezone")

#### Updated `TimeRangeParsingCapability` Class

- Added `local: bool = TIME_PARSING_LOCAL` class attribute
- Reads default value from `TIME_PARSING_LOCAL` environment variable
- Passes `local` parameter to `_get_time_parsing_system_prompt()`

#### Updated Validation Logic

- Changed from checking "future years" to checking "future dates"
- Now compares against actual current time with appropriate timezone
- Allows past years (e.g., 2025, 2024) as long as they're before current time
- Added 1-hour tolerance for edge cases
- Improved error messages to show actual current time

## Configuration

### Application `.env` File

Add the following to your application's `.env` file to enable local timezone parsing:

```bash
# Timezone
TZ=America/Chicago

# Time Parsing Configuration
# Set to true to use local timezone (from TZ) instead of UTC for time range parsing
TIME_PARSING_LOCAL=true
```

### Docker Compose

If running in Docker, add the environment variables:

```yaml
environment:
  - TZ=America/Chicago
  - TIME_PARSING_LOCAL=true
```

## Backward Compatibility

- Default behavior unchanged (UTC timezone)
- Existing applications continue to work without modification
- Only applications that explicitly set `TIME_PARSING_LOCAL=true` will use local timezone

## Use Cases

1. **Control System Applications**: Operators typically work in local timezone
2. **Log Analysis**: Correlating with local time-stamped events
3. **Shift-Based Operations**: When queries reference "today", "yesterday", etc.

## Related Changes

Applications using local time parsing may also need to update their data retrieval modules to handle the timezone conversion appropriately:

- Archiver API typically requires UTC times
- SDDS logger uses local time internally
- Data visualization should display in local timezone
