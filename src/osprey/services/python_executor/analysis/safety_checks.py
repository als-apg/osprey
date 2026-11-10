"""Standalone code safety checks — no third-party dependencies.

Usable by the MCP execute tool for pre-execution validation.
"""

import ast

from osprey.services.python_executor.write_surface import CLIENT_PACKAGES


def check_syntax(code: str) -> list[str]:
    """Check Python syntax validity. Returns list of issues (empty = valid)."""
    issues = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
    except Exception as e:
        issues.append(f"Syntax parsing error: {str(e)}")
    return issues


# Builtins that run code the caller supplies as data. Matched on the AST, as a
# CALL of a bare name — never as a substring. ``"eval(" in code`` refuses
# ``retrieval(...)``, ``df.eval("a + b")`` (a pandas expression) and
# ``model.eval()`` (put a torch module in inference mode); none of those runs
# caller-supplied code, and all three are ordinary analysis an operator has
# every reason to submit. An attribute call such as ``builtins.eval(...)`` is
# deliberately not matched here either: it is not a bare-name call, and the
# layer that answers obfuscation is the runtime guard, not this text scan.
_DANGEROUS_CALLS = {
    "exec": "Use of exec() function",
    "eval": "Use of eval() function",
    "__import__": "Dynamic import usage",
    "compile": "Dynamic code compilation",
}

_PROHIBITED_IMPORTS = ["subprocess", "os.system", "eval", "exec"]


def _dangerous_call_issues(code: str) -> list[str]:
    """One issue per :data:`_DANGEROUS_CALLS` builtin actually *called*.

    Reported once per builtin, however often it appears. Syntax errors are the
    business of :func:`check_syntax`; this walker stays quiet on them.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        warning = _DANGEROUS_CALLS.get(node.func.id)
        if warning is not None and warning not in found:
            found.append(warning)
    return found


# Control-system client libraries a readonly run may not import. Reads go
# through ``osprey.runtime.read_channel``; importing a client directly in a
# readonly script is only ever a route around the write gates. Checking at the
# import statement is immune to the aliasing that defeats call-site regexes
# (``from epics import caput as _w``), and an import never appears inside a
# comment or string, so it has none of the regex false positives either.
#
# The set is ``write_surface.CLIENT_PACKAGES``, derived from the client half of
# the write surface rather than spelled again here: one producer means a client
# added to the runtime guard is denied at import in the same change.
# Acquisition frameworks (ophyd-async, Bluesky) are deliberately NOT in it —
# they are document and analysis libraries too, so only their write entry
# points refuse, and that happens at runtime.
_READONLY_DENIED_IMPORTS = CLIENT_PACKAGES


def check_readonly_imports(code: str) -> list[str]:
    """Return an issue per control-system client import. Empty = none found.

    Matches on the top-level package, so ``p4p.client.thread`` and
    ``from epics.ca import put`` are both caught. Syntax errors are the
    business of :func:`check_syntax`; this walker stays quiet on them.
    """
    issues: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return issues
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        issues.extend(
            f"Import of '{name}' is not allowed in readonly mode"
            for name in names
            if name.split(".")[0] in _READONLY_DENIED_IMPORTS
        )
    return issues


def quick_safety_check(code: str) -> tuple[bool, list[str]]:
    """Run all safety checks. Returns (passed, issues).

    This is the main public API for MCP tools.
    Combines syntax, security, and import checks.
    """
    all_issues = check_syntax(code)

    # Only run security/import checks if syntax is valid
    if not any("Syntax error" in i for i in all_issues):
        all_issues.extend(f"Security risk: {warning}" for warning in _dangerous_call_issues(code))
        # A substring, not a call: ``os.system`` names a dotted attribute, so
        # there is no bare-name call to key on.
        if "os.system" in code:
            all_issues.append("Security risk: System command execution")

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    all_issues.extend(
                        f"Prohibited import detected: {alias.name}"
                        for alias in node.names
                        if alias.name in _PROHIBITED_IMPORTS
                    )
                elif isinstance(node, ast.ImportFrom) and node.module in _PROHIBITED_IMPORTS:
                    all_issues.append(f"Prohibited import detected: {node.module}")
        except SyntaxError:
            pass  # Syntax errors handled by check_syntax

    return (len(all_issues) == 0, all_issues)
