# OSPREY Fixes — Porting from next_osprey

Changes made to `/home/oxygen/SHANG/osprey` to match the working
`/home/oxygen/SHANG/next_osprey`. **12 files modified** (89 insertions,
142 deletions) plus vendor asset files copied.

---

## 1. Archiver Connector: urllib → archivertools

**File:** `src/osprey/connectors/archiver/epics_archiver_connector.py`

Replaced the raw `urllib` HTTP implementation with the `archivertools`
library. The old code manually constructed HTTP requests, parsed JSON
responses, and built pandas Series one PV at a time. The new code uses
`ArchiverClient.match_data()` and properly converts the `secs`/`nanos`
columns returned by archivertools into a DatetimeIndex.

**Before (broken):**
```python
import urllib.request, urllib.parse, urllib.error
# Manual HTTP fetch per PV, lastSample_{n_secs}() downsampling syntax
params = urllib.parse.urlencode({"pv": pv, "from": start_str, "to": end_str})
url = f"{self._url}/retrieval/data/getData.json?{params}"
with urllib.request.urlopen(req, timeout=self._timeout) as resp:
    payload = json.loads(resp.read().decode())
```

**After (working):**
```python
from archivertools import ArchiverClient
self._archiver_client = ArchiverClient(archiver_url=archiver_url)
data = self._archiver_client.match_data(
    pv_list=pv_list, precision=precision_ms, start=start_date, end=end_date
)
# Convert secs/nanos to DatetimeIndex
if "secs" in data.columns and "nanos" in data.columns:
    timestamps = pd.to_datetime(data["secs"], unit="s") + pd.to_timedelta(data["nanos"], unit="ns")
    data = data.drop(columns=["secs", "nanos"])
    data.index = timestamps
```

---

## 2. Base Store File Locking Fix

**File:** `src/osprey/stores/base_store.py`

Fixed the `_with_index_lock()` method that caused `[Errno 9] Bad file
descriptor` when the archiver tried to save results to the artifact
store. The `os.open()` + `os.fdopen()` two-step with `O_RDONLY | O_CREAT`
flags produced an invalid fd for `fcntl.flock()`.

**Before (broken):**
```python
fd_num = os.open(self._lock_file, os.O_RDONLY | os.O_CREAT, 0o664)
fd = os.fdopen(fd_num, "rb")
fcntl.flock(fd.fileno(), fcntl.LOCK_EX)  # [Errno 9] Bad file descriptor
```

**After (working):**
```python
fd = open(self._lock_file, "w")
fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
```

---

## 3. initialize_workspace_singletons Signature Fix

**Files:**
- `src/osprey/mcp_server/startup.py` — function definition
- `src/osprey/mcp_server/control_system/server.py` — call site
- `src/osprey/mcp_server/workspace/server.py` — call site
- `src/osprey/mcp_server/python_executor/server.py` — call site
- `src/osprey/mcp_server/ariel/server.py` — call site
- `src/osprey/mcp_server/channel_finder_hierarchical/server.py` — call site + added import
- `src/osprey/mcp_server/channel_finder_middle_layer/server.py` — call site + added import
- `src/osprey/mcp_server/channel_finder_in_context/server.py` — call site + added import

Changed `initialize_workspace_singletons()` to accept a `workspace_root: Path`
parameter instead of internally calling `resolve_shared_data_root()`. All 7
call sites updated to pass `workspace_root`. Three channel-finder servers also
needed `from osprey.utils.workspace import resolve_workspace_root` added.

**Before:**
```python
# startup.py
def initialize_workspace_singletons() -> None:
    from osprey.utils.workspace import resolve_shared_data_root
    initialize_artifact_store(workspace_root=resolve_shared_data_root())

# server.py (callers)
initialize_workspace_singletons()
```

**After:**
```python
# startup.py
def initialize_workspace_singletons(workspace_root: Path) -> None:
    initialize_artifact_store(workspace_root=workspace_root)

# server.py (callers)
workspace_root = resolve_workspace_root()
initialize_workspace_singletons(workspace_root)
```

---

## 4. Web Terminal: Bundled Vendor Assets + FileResponse

**Files:**
- `src/osprey/interfaces/web_terminal/static/vendor/*` — copied from next_osprey
- `src/osprey/interfaces/web_terminal/static/index.html` — updated paths
- `src/osprey/interfaces/web_terminal/app.py` — switched to FileResponse

The web terminal showed a **black empty page** because:
1. `static/vendor/` was empty (no xterm.js, highlight.js, marked.js, fonts)
2. `index.html` used Jinja2 `{{ vendor_url(...) }}` template syntax to load
   these files with CDN fallback, but the CDN URLs weren't resolving
3. `app.py` served `index.html` via `TemplateResponse` requiring Jinja2 processing

**Vendor files copied:**
```
static/vendor/
├── addon-fit.min.js
├── addon-web-links.min.js
├── atom-one-dark.min.css
├── atom-one-light.min.css
├── fonts/
│   └── (font files)
├── highlight.min.js
├── marked.min.js
├── xterm.min.css
└── xterm.min.js
```

**index.html — before:**
```html
<link rel="stylesheet" href="{{ vendor_url('xterm.css', '/static/vendor/xterm.min.css') }}">
<script src="{{ vendor_url('xterm.js', '/static/vendor/xterm.min.js') }}"></script>
```

**index.html — after:**
```html
<link rel="stylesheet" href="/static/vendor/xterm.min.css">
<script src="/static/vendor/xterm.min.js"></script>
```

**app.py — before:**
```python
from fastapi.templating import Jinja2Templates
from osprey.interfaces.vendor import vendor_url
templates = Jinja2Templates(directory=str(STATIC_DIR))
templates.env.globals["vendor_url"] = vendor_url

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})
```

**app.py — after:**
```python
from fastapi.responses import FileResponse

@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")
```

---

## .env Fix (aps-control-assistant)

**File:** `aps-control-assistant/.env`

Commented out `ANTHROPIC_API_KEY` to avoid conflict with the Argo
provider's `apiKeyHelper` which sets `ANTHROPIC_AUTH_TOKEN`.

```diff
- ANTHROPIC_API_KEY=${USER}
+ # ANTHROPIC_API_KEY=${USER}  # Commented out to avoid conflict with apiKeyHelper (Argo provider)
```
