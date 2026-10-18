# The gate environment: sourced by every gate script, never executed.
#
# It puts the tree's pins in place before any check runs:
#   - node_modules: `npm ci` runs only when scripts/js_deps_current.py finds the
#     installed tree differs from package-lock.json; without npm it is skipped.
#   - Claude Code: scripts/pinned_claude.py prints a directory whose `claude`
#     reports the pinned version, and that directory goes first on PATH; when
#     none does, it prints the install command and this step fails.
#
# It sets no shell options, is safe under `set -u`, and keeps the caller's cwd.
# Every variable it uses starts with _gate_env_ and is unset again on success.

_gate_env_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1

_gate_env_js="unchecked (no npm)"
if command -v npm >/dev/null 2>&1; then
    _gate_env_js="current"
    if ! python3 "$_gate_env_root/scripts/js_deps_current.py"; then
        npm ci --prefix "$_gate_env_root" || return 1
        _gate_env_js="installed"
    fi
fi

_gate_env_pin="$(uv run --project "$_gate_env_root" python "$_gate_env_root/scripts/pinned_claude.py" --pin)" || return 1
_gate_env_dir="$(uv run --project "$_gate_env_root" python "$_gate_env_root/scripts/pinned_claude.py")" || return 1
export PATH="$_gate_env_dir:$PATH"

echo "gate env: claude $_gate_env_pin ($_gate_env_dir); node_modules $_gate_env_js"
unset _gate_env_root _gate_env_js _gate_env_pin _gate_env_dir
