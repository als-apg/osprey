#!/usr/bin/env python3
"""Check the build-app skill against the OSPREY in reach, and print that OSPREY's app contract.

The skill was reviewed against one state of the files that define what an application
author writes: the manifest schema, the rules, the import boundary, the SDK's names, the
``osprey app`` command and the scaffold. ``contract-fingerprint.json`` beside ``scripts/``
records a SHA-256 of each. This script hashes the same files of the OSPREY it finds and
says which differ, then prints the contract as that OSPREY's source states it: every
manifest key with its type and default, every rule name with its message, the modules an
application may import, the SDK's names, the command's options and the scaffold's files.

It reads source and never imports OSPREY, so it runs on any Python 3.9+ and reports the
same way for a source checkout and an installed wheel. The OSPREY it reads is, in order:
``--root``, a checkout the working directory is inside, the package this interpreter
finds, the package the ``osprey`` command on ``PATH`` runs from.

Exit status: 0 when the report was printed (``--strict``: and nothing differs), 1 under
``--strict`` when a watched file differs, 2 when no OSPREY with an app contract is found.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
FINGERPRINT = SKILL_DIR / "contract-fingerprint.json"

#: The files that define what an application author writes, relative to the ``osprey``
#: package directory, each with what it decides and the parts of SKILL.md that rest on it.
#: An entry ending in ``/`` is a directory, hashed file by file.
WATCHED: tuple[tuple[str, str, str], ...] = (
    (
        "apps/manifest.py",
        "the app.yml schema: keys, types, defaults",
        "Contract snapshot; step 3 (manifest); Writes and approvals",
    ),
    (
        "apps/rules.py",
        "the rules across a manifest, the other apps and the profile",
        "Contract snapshot; Writes and approvals; Common mistakes",
    ),
    (
        "apps/loader.py",
        "what build and check read of a deployment: skills, agents, config keys, services",
        "Product rules (the app's skill); step 2; step 7",
    ),
    (
        "apps/agents.py",
        "an app's subagent file: frontmatter keys, refused keys, the delegation note",
        "Product rules (the app's skill); step 7",
    ),
    (
        "apps/boundary.py",
        "the modules application code may import",
        "Product rules (hardware writes); Contract snapshot",
    ),
    (
        "apps/sdk/__init__.py",
        "the SDK's names",
        "step 4 (backend); step 5 (tool server)",
    ),
    (
        "cli/app_cmd.py",
        "osprey app new / check: options, the design-system and JS checks",
        "step 2; step 6 (panel); step 8 (verify)",
    ),
    (
        "apps/scaffold-app/",
        "the scaffold osprey app new copies",
        "step 2; the file layout and every pattern steps 3-7 build on",
    ),
)

_SKIP_DIRS = frozenset({"__pycache__", "node_modules", ".pytest_cache"})
_SKIP_SUFFIXES = (".pyc", ".pyo")
_RELEASE = re.compile(r"v?(\d+(?:\.\d+)*)")


# --------------------------------------------------------------------------- locate


def _has_contract(package: Path) -> bool:
    return (package / "apps" / "manifest.py").is_file()


def _checkout_package(start: Path) -> Path | None:
    """The ``src/osprey`` of an OSPREY checkout that *start* is inside."""
    for directory in (start, *start.parents):
        package = directory / "src" / "osprey"
        if (package / "__init__.py").is_file() and (directory / "plugins" / "osprey").is_dir():
            return package
    return None


def _importable_package() -> Path | None:
    """The ``osprey`` package this interpreter would import, without importing it."""
    try:
        spec = importlib.util.find_spec("osprey")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _command_package() -> Path | None:
    """The package the ``osprey`` command on ``PATH`` runs from, asked of its interpreter."""
    command = shutil.which("osprey")
    if command is None:
        return None
    try:
        first = Path(command).read_bytes().splitlines()[0].decode("utf-8", "replace")
    except (OSError, IndexError):
        return None
    if not first.startswith("#!"):
        return None
    interpreter = first[2:].strip().split(" ")[0]
    probe = (
        "import importlib.util as u; s = u.find_spec('osprey'); "
        "print(next(iter(s.submodule_search_locations)))"
    )
    try:
        done = subprocess.run(
            [interpreter, "-c", probe], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = done.stdout.strip().splitlines()[-1:] if done.returncode == 0 else []
    return Path(line[0]) if line else None


def locate(root: str | None) -> tuple[Path | None, str]:
    """Find the ``osprey`` package directory to read, and say how it was found."""
    if root is not None:
        given = Path(root).expanduser().resolve()
        for candidate in (given, given / "src" / "osprey", given / "osprey"):
            if (candidate / "__init__.py").is_file() and _has_contract(candidate):
                return candidate, "--root"
        return None, f"--root {given} holds no osprey package with apps/manifest.py"
    found: list[str] = []
    for how, package in (
        ("the checkout around the working directory", _checkout_package(Path.cwd())),
        ("this interpreter's import path", _importable_package()),
        ("the osprey command on PATH", _command_package()),
    ):
        if package is None:
            continue
        if _has_contract(package):
            return package, how
        found.append(f"{package} (from {how}) has no apps/manifest.py")
    return None, "; ".join(found) or "no osprey package found"


def osprey_version(package: Path) -> str:
    """The version of the OSPREY at *package*, or ``unknown``."""
    checkout = package.parent.parent
    if package.parent.name == "src" and (checkout / ".git").exists():
        try:
            done = subprocess.run(
                ["git", "-C", str(checkout), "describe", "--tags", "--match", "v[0-9]*"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            done = None
        if done is not None and done.returncode == 0 and done.stdout.strip():
            return done.stdout.strip().removeprefix("v")
    stamp = package / "_version.py"
    if stamp.is_file():
        match = re.search(
            r"^(?:__version__|version)\s*=\s*['\"]([^'\"]+)['\"]",
            stamp.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    for info in sorted(package.parent.glob("osprey_framework-*.dist-info")):
        return info.name.removeprefix("osprey_framework-").removesuffix(".dist-info")
    return "unknown"


def _release(version: str) -> tuple[int, ...] | None:
    match = _RELEASE.match(version)
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


# ---------------------------------------------------------------------- fingerprint


def _digest(path: Path) -> str:
    """SHA-256 of a file's bytes with line endings normalized, so a checkout mode cannot differ."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _files_under(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not path.name.endswith(_SKIP_SUFFIXES)
        and not _SKIP_DIRS.intersection(path.relative_to(directory).parts)
    )


def fingerprint(package: Path) -> dict[str, str]:
    """Hash every watched file present under *package*, keyed by its POSIX relative path."""
    hashes: dict[str, str] = {}
    for entry, _defines, _sections in WATCHED:
        target = package / entry
        if entry.endswith("/"):
            if target.is_dir():
                for path in _files_under(target):
                    hashes[path.relative_to(package).as_posix()] = _digest(path)
        elif target.is_file():
            hashes[entry] = _digest(target)
    return hashes


def _entry_of(relative: str) -> tuple[str, str, str]:
    for entry in WATCHED:
        if relative == entry[0] or (entry[0].endswith("/") and relative.startswith(entry[0])):
            return entry
    return (relative, "no longer watched", "-")


def drift(package: Path) -> tuple[list[str], dict[str, object]]:
    """Compare *package* with the recorded fingerprint.

    Returns:
        The differing files, each as ``<path>  (<changed|added|missing>)``, and the record.
    """
    record = json.loads(FINGERPRINT.read_text(encoding="utf-8"))
    recorded: dict[str, str] = record["files"]
    current = fingerprint(package)
    differing = [
        f"{path}  (changed)"
        for path in recorded
        if current.get(path, recorded[path]) != recorded[path]
    ]
    differing += [f"{path}  (missing here)" for path in recorded if path not in current]
    differing += [f"{path}  (not in the reviewed set)" for path in current if path not in recorded]
    return sorted(differing), record


def report_drift(package: Path, how: str) -> bool:
    """Print the drift report; return whether anything differs."""
    version = osprey_version(package)
    print("== Skill versus this OSPREY ==")
    print(f"OSPREY read: {package}  (version {version}; found by {how})")
    if not FINGERPRINT.is_file():
        print(f"No fingerprint at {FINGERPRINT}: nothing to compare. Treat every contract")
        print("fact in SKILL.md as unverified and use the summary below.")
        return True
    differing, record = drift(package)
    reviewed = str(record.get("reviewed_against", "unknown"))
    print(f"Skill reviewed against: OSPREY {reviewed}")
    if not differing:
        print("RESULT: contract unchanged since review. SKILL.md's snapshot holds for this OSPREY.")
        return False
    mine, theirs = _release(version), _release(reviewed)
    if mine is not None and theirs is not None and mine != theirs:
        if mine < theirs:
            print(
                "This OSPREY is OLDER than the skill: the skill may name keys, rules or SDK\n"
                "names this OSPREY lacks. Update OSPREY, or build to the summary below."
            )
        else:
            print(
                "This OSPREY is NEWER than the skill: the contract moved after the skill was\n"
                "reviewed. Update the plugin, or build to the summary below."
            )
    print(f"RESULT: contract CHANGED since review in {len(differing)} file(s).")
    print("The summary below is this OSPREY's truth; where SKILL.md disagrees, the summary and")
    print("the source win. Re-read these files before relying on the named parts of SKILL.md:")
    by_entry: dict[tuple[str, str, str], list[str]] = {}
    for line in differing:
        by_entry.setdefault(_entry_of(line.split("  ")[0]), []).append(line)
    for (entry, defines, sections), lines in by_entry.items():
        print(f"  {entry} -- {defines}")
        print(f"      distrust in SKILL.md: {sections}")
        for line in lines:
            print(f"      {package / line.split('  ')[0]}  {line.split('  ')[1]}")
    return True


def refresh(package: Path) -> None:
    """Record *package*'s fingerprint as the reviewed one."""
    record = {
        "_comment": (
            "Written by scripts/app_contract.py --refresh after SKILL.md was re-read against "
            "the contract. Do not edit by hand."
        ),
        "reviewed_against": osprey_version(package),
        "files": fingerprint(package),
    }
    FINGERPRINT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(record['files'])} files at OSPREY {record['reviewed_against']}")
    print(f"in {FINGERPRINT}")


# -------------------------------------------------------------------------- summary


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None


def _text(node: ast.AST | None) -> str:
    """A message expression as the text it produces, with ``{expr}`` for what is computed."""
    if node is None:
        return ""
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.JoinedStr):
        return "".join(
            _text(part)
            if isinstance(part, ast.Constant)
            else "{" + ast.unparse(part.value).split("(")[0] + "}"
            for part in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _text(node.left) + _text(node.right)
    if isinstance(node, ast.IfExp):
        return _text(node.body)
    return "{" + ast.unparse(node) + "}"


def _wrap(text: str, indent: str, width: int = 100) -> str:
    return textwrap.fill(
        " ".join(text.split()),
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_on_hyphens=False,
        break_long_words=False,
    )


def _doc_line(node: ast.AST) -> str:
    doc = ast.get_docstring(node) if isinstance(node, (ast.ClassDef, ast.FunctionDef)) else None
    return doc.split("\n\n")[0].replace("\n", " ") if doc else ""


def _constants(tree: ast.Module, names: tuple[str, ...]) -> list[str]:
    lines = []
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
        if isinstance(target, ast.Name) and target.id in names and node.value is not None:
            lines.append(f"  {target.id} = {ast.unparse(node.value)}")
    return lines


def summarize_manifest(package: Path) -> None:
    tree = _parse(package / "apps" / "manifest.py")
    print("\n== app.yml schema (apps/manifest.py) ==")
    if tree is None:
        print("  not readable")
        return
    print("Unknown keys are refused at every level. `key: type = default`; no default = required.")
    for line in _constants(tree, ("APP_NAME_PATTERN", "READ_METHODS", "PACKAGE_URL_TEMPLATE")):
        print(line)
    aliases = {
        node.targets[0].id: ast.unparse(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Subscript)
        and ast.unparse(node.value).startswith("Annotated[")
    }
    for name, value in aliases.items():
        print(_wrap(f"type {name} = {value}", "  "))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        if not any(isinstance(base, ast.Name) and base.id == "_Strict" for base in node.bases):
            continue
        print(f"\n  {node.name}: {_doc_line(node)}".rstrip(": ").rstrip())
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                default = f" = {ast.unparse(item.value)}" if item.value is not None else ""
                print(_wrap(f"{item.target.id}: {ast.unparse(item.annotation)}{default}", "    "))


def _rules_in(tree: ast.Module) -> dict[str, str]:
    """Each rule name a module can report, with the first message found for it."""
    rules: dict[str, str] = {}

    def add(rule: ast.AST | None, message: ast.AST | None) -> None:
        if isinstance(rule, ast.Constant) and isinstance(rule.value, str):
            rules.setdefault(rule.value, _text(message))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            args = node.args
            if name == "refuse" and len(args) >= 3:
                add(args[1], args[2])
            elif name in ("Violation", "AppManifestError") and len(args) >= 4:
                add(args[2], args[3])
            elif name == "PydanticCustomError" and len(args) >= 2:
                add(args[0], args[1])
        elif isinstance(node, ast.Assign):
            target, value = node.targets[0], node.value
            if isinstance(target, ast.Name) and target.id == "rule":
                add(value, None)
            elif (
                isinstance(target, ast.Tuple)
                and isinstance(value, ast.Tuple)
                and len(target.elts) == len(value.elts)
            ):
                named = {
                    element.id: item
                    for element, item in zip(target.elts, value.elts)  # noqa: B905 - lengths checked
                    if isinstance(element, ast.Name)
                }
                if "rule" in named:
                    add(named["rule"], named.get("message"))
    return rules


def summarize_rules(package: Path) -> None:
    print("\n== Rules build and `osprey app check` report (rule name: message) ==")
    for relative in (
        "apps/manifest.py",
        "apps/loader.py",
        "apps/rules.py",
        "apps/boundary.py",
        "cli/app_cmd.py",
    ):
        tree = _parse(package / relative)
        if tree is None:
            print(f"  {relative}: not readable")
            continue
        print(f"  {relative}")
        for rule, message in _rules_in(tree).items():
            print(_wrap(f"{rule}: {message}" if message else rule, "    ")[:330])
    tree = _parse(package / "apps" / "rules.py")
    if tree is not None:
        print("  names an application may not take (apps/rules.py):")
        names = ("HOOK_DISPATCHED_TOOLS", "QUEUE_TOOL_PREFIX", "APPROVED_PREFIX", "APPROVED_TARGET")
        for line in _constants(tree, names):
            print("  " + line)


def summarize_code_surface(package: Path) -> None:
    print("\n== What application code may import ==")
    tree = _parse(package / "apps" / "boundary.py")
    for line in _constants(tree, ("ALLOWED_MODULES",)) if tree is not None else []:
        print(line)
    print("  A name with a leading underscore anywhere in the dotted path is refused.")
    tree = _parse(package / "apps" / "sdk" / "__init__.py")
    exported: list[str] = []
    if tree is not None:
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                exported = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    print(f"\n== osprey.apps.sdk: {len(exported)} names (apps/sdk/__init__.py) ==")
    print(_wrap(", ".join(exported), "  "))
    print("  Signature and docstring of one: python -m pydoc osprey.apps.sdk.<name>")


def summarize_cli(package: Path) -> None:
    tree = _parse(package / "cli" / "app_cmd.py")
    print("\n== osprey app (cli/app_cmd.py) ==")
    if tree is None:
        print("  not readable; run `osprey app --help`")
        return
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        command, parameters = None, []
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                if isinstance(decorator, ast.Name) and decorator.id == "repo_option":
                    parameters.append("--repo")
                continue
            called = ast.unparse(decorator.func)
            if called == "app.command" and decorator.args:
                command = _text(decorator.args[0])
            elif called == "click.argument" and decorator.args:
                parameters.append(_text(decorator.args[0]).upper())
            elif called == "click.option" and decorator.args:
                parameters.append(_text(decorator.args[0]))
        if command is None:
            continue
        shown = " ".join(sorted(parameters, key=lambda p: p.startswith("--")))
        print(f"  osprey app {command} {shown}".rstrip())
        print(_wrap(_doc_line(node), "      "))
    for line in _constants(tree, ("PANEL_STYLESHEETS", "_JS_TEST_SUFFIXES", "_NOT_AUTHORED_DIRS")):
        print(line)
    sheet = package / "interfaces" / "design_system" / "static" / "css" / "primitives.css"
    if sheet.is_file():
        classes = sorted(set(re.findall(r"\.(osprey-[a-z0-9-]+)", sheet.read_text("utf-8"))))
        print(f"\n== Design-system primitives: {len(classes)} osprey-* classes ==")
        print(f"  {sheet}")
        print("  each prefixed osprey-:")
        print(_wrap(" ".join(c.removeprefix("osprey-") for c in classes), "    "))


def summarize_scaffold(package: Path) -> None:
    scaffold = package / "apps" / "scaffold-app"
    print("\n== The scaffold `osprey app new` copies ==")
    print(f"  {scaffold}")
    if not scaffold.is_dir():
        print("  not present in this OSPREY")
        return
    for path in _files_under(scaffold):
        lines = len(path.read_bytes().splitlines())
        print(f"    {path.relative_to(scaffold).as_posix():44s} {lines:4d} lines")
    example = package / "profiles" / "presets" / "apps" / "orbit-bump"
    if example.is_dir():
        print(f"  Worked example (several routes and tools, a writing panel): {example}")


def summarize(package: Path) -> None:
    summarize_manifest(package)
    summarize_rules(package)
    summarize_code_surface(package)
    summarize_cli(package)
    summarize_scaffold(package)


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--root", help="an osprey package directory, or a checkout or site-packages holding one"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--drift-only", action="store_true", help="print only the drift report")
    mode.add_argument("--summary-only", action="store_true", help="print only the contract summary")
    mode.add_argument(
        "--refresh",
        action="store_true",
        help="record this OSPREY's fingerprint as reviewed (maintainers, in an OSPREY checkout)",
    )
    parser.add_argument("--strict", action="store_true", help="exit 1 when a watched file differs")
    args = parser.parse_args(argv)

    package, how = locate(args.root)
    if package is None:
        print("== Skill versus this OSPREY ==")
        print(f"RESULT: no OSPREY with an app contract found ({how}).")
        print("Run this with the Python that has OSPREY installed (for a uv project:")
        print("`uv run python <this script>`), or pass --root <checkout or site-packages>.")
        print("Until it runs, SKILL.md's contract snapshot is unverified.")
        return 2
    if args.refresh:
        refresh(package)
        return 0
    differs = False
    if not args.summary_only:
        differs = report_drift(package, how)
    if not args.drift_only:
        summarize(package)
    return 1 if args.strict and differs else 0


if __name__ == "__main__":
    sys.exit(main())
