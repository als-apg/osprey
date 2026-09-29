"""MCP-server text names what the agent should do, never one client's tool.

These servers answer any MCP client, so a tool description or a result hint that
tells the agent to use a specific coding-agent client's tool is wrong for every
other client. Hints name the capability instead ("open <path> with your
file-reading tool"). ``dispatch_worker/`` is skipped: it drives the harness and
names harness tools as policy.
"""

import re
from pathlib import Path

import osprey.mcp_server

_HARNESS_TOOL = re.compile(
    r"\b(Read|Write|Edit|MultiEdit|Bash|Glob|Grep|WebFetch|WebSearch|NotebookEdit)\s+tool\b"
)


def test_mcp_server_text_names_no_harness_tool():
    root = Path(osprey.mcp_server.__file__).parent
    scanned = 0
    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "dispatch_worker" in path.relative_to(root).parts:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8")
        for match in _HARNESS_TOOL.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{path.relative_to(root.parent)}:{line}: {match.group(0)!r}")
    assert scanned >= 50, f"only {scanned} files scanned under {root}"
    assert not hits, "MCP-server text names a harness tool:\n" + "\n".join(hits)
