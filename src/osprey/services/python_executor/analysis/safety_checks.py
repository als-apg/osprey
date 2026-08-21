"""Standalone code safety checks — no framework dependencies.

Usable by the MCP execute tool for pre-execution validation.
"""

import ast


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


# Patterns that indicate security risks in user-submitted code
_DANGEROUS_PATTERNS = [
    ("exec(", "Use of exec() function"),
    ("eval(", "Use of eval() function"),
    ("__import__", "Dynamic import usage"),
    ("open(", "File operations - ensure proper handling"),
    ("subprocess", "Subprocess usage - potential security risk"),
    ("os.system", "System command execution"),
]

_PROHIBITED_IMPORTS = ["subprocess", "os.system", "eval", "exec"]


def check_security(code: str) -> list[str]:
    """Check for dangerous code patterns. Returns list of issues."""
    issues = []
    for pattern, warning in _DANGEROUS_PATTERNS:
        if pattern in code:
            if pattern in ["open(", "subprocess"]:
                issues.append(f"Warning: {warning}")
            else:
                issues.append(f"Security risk: {warning}")
    return issues


def check_imports(code: str) -> list[str]:
    """Check for prohibited imports via AST. Returns list of issues."""
    issues = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _PROHIBITED_IMPORTS:
                        issues.append(f"Prohibited import detected: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module in _PROHIBITED_IMPORTS:
                    issues.append(f"Prohibited import detected: {node.module}")
    except SyntaxError:
        pass  # Syntax errors handled by check_syntax
    return issues


# Control-system client libraries a readonly run may not import. Reads go
# through ``osprey.runtime.read_channel``; importing a client directly in a
# readonly script is only ever a route around the write gates. Checking at the
# import statement is immune to the aliasing that defeats call-site regexes
# (``from epics import caput as _w``), and an import never appears inside a
# comment or string, so it has none of the regex false positives either.
_READONLY_DENIED_IMPORTS = frozenset({"epics", "p4p", "caproto", "pvaccess", "tango", "PyTango"})


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
        for name in names:
            if name.split(".")[0] in _READONLY_DENIED_IMPORTS:
                issues.append(f"Import of '{name}' is not allowed in readonly mode")
    return issues


def quick_safety_check(code: str) -> tuple[bool, list[str]]:
    """Run all safety checks. Returns (passed, issues).

    This is the main public API for MCP tools.
    Combines syntax, security, and import checks.
    """
    all_issues = []
    all_issues.extend(check_syntax(code))

    # Only run security/import checks if syntax is valid
    if not any("Syntax error" in i for i in all_issues):
        security_issues = check_security(code)
        import_issues = check_imports(code)

        # Only block on high-risk patterns (exec/eval/__import__)
        high_risk = [i for i in security_issues if "Security risk:" in i]
        all_issues.extend(high_risk)
        all_issues.extend(import_issues)

    return (len(all_issues) == 0, all_issues)
