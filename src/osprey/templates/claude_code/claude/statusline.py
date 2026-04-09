"""OSPREY deployed-agent statusline.

Reads Claude Code state JSON from stdin and emits a single colored line:

    Sonnet | 45% 90k/200K | my-project (main) | v2.0.0 | osprey-v0.11.5

Design notes:
    - Matches the style of .claude/hooks/osprey_*.py (stdin JSON, graceful
      fallbacks, no external deps beyond the stdlib + the `osprey` package
      which is always installed in deployed agents).
    - Framework version is looked up via `import osprey` at runtime so the
      statusline tracks the actual deployed OSPREY version, not the version
      that shipped with the template.
"""

import json
import sys
from pathlib import Path


def _read_input() -> dict:
    try:
        return json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return {}


def _model_short(data: dict) -> str:
    raw = (data.get("model") or {}).get("display_name") or ""
    for name in ("Opus", "Sonnet", "Haiku"):
        if name.lower() in raw.lower():
            return name
    return raw or "?"


def _context(data: dict) -> tuple[int, int, int]:
    cw = data.get("context_window") or {}
    pct = int(cw.get("used_percentage") or 0)
    size = int(cw.get("context_window_size") or 0)
    used_k = int(pct * size / 100 / 1000)
    max_k = int(size / 1000)
    return pct, used_k, max_k


def _dir_and_branch(data: dict) -> tuple[str, str]:
    cwd_full = (data.get("workspace") or {}).get("current_dir") or ""
    cwd = Path(cwd_full).name or cwd_full
    branch = ""
    if cwd_full:
        import subprocess

        try:
            r = subprocess.run(
                ["git", "-C", cwd_full, "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if r.returncode == 0:
                branch = r.stdout.strip()
        except Exception:
            pass
    return cwd, branch


def _osprey_version() -> str:
    try:
        import osprey

        return getattr(osprey, "__version__", "") or ""
    except Exception:
        return ""


# ANSI colors (match ~/.claude/statusline-command.sh palette)
CYAN, YELLOW, WHITE, MAGENTA, GREEN, BLUE, DIM, RESET = (
    "\033[36m",
    "\033[33m",
    "\033[1;37m",
    "\033[35m",
    "\033[32m",
    "\033[34m",
    "\033[2m",
    "\033[0m",
)


def main() -> None:
    data = _read_input()
    model = _model_short(data)
    pct, used_k, max_k = _context(data)
    cwd, branch = _dir_and_branch(data)
    cc_ver = data.get("version") or ""
    osprey_ver = _osprey_version()

    parts = [f"{CYAN}{model}{RESET}"]
    parts.append(f"{YELLOW}{pct}% {used_k}k/{max_k}K{RESET}")
    dir_part = f"{WHITE}{cwd}{RESET}"
    if branch:
        dir_part += f" {MAGENTA}({branch}){RESET}"
    parts.append(dir_part)
    if cc_ver:
        parts.append(f"{GREEN}v{cc_ver}{RESET}")
    if osprey_ver:
        parts.append(f"{BLUE}osprey-v{osprey_ver}{RESET}")

    sys.stdout.write(f" {DIM}|{RESET} ".join(parts))


if __name__ == "__main__":
    main()
