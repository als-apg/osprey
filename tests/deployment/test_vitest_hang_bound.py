"""Vitest's per-test and per-hook timeout is one hang bound, declared once.

``vitest.config.js`` sets ``testTimeout`` and ``hookTimeout`` to one named
constant sized like the Python lane's per-test cap (``pytest --timeout=600``).
It detects a hang; it is never a speed budget, so nothing may lower it: not a
test's own timeout argument, not ``vi.setConfig``, not a CLI flag in an npm
script, a CI step or a shell script.

Each check is a pure function over source text, run once on the real tree and
once per mutation on an in-memory edited copy that must make it fail. Nothing
here runs Node or a subprocess other than ``git ls-files``; the workflow files
are read with ``yaml.safe_load``, never by a text match over the YAML.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
VITEST_CONFIG = REPO_ROOT / "vitest.config.js"
PACKAGE_JSON = REPO_ROOT / "package.json"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
SCRIPTS_DIR = REPO_ROOT / "scripts"
SETUP_FILE = "tests/vitest.setup.mjs"

#: The bound's floor, in milliseconds: pytest's per-test cap, ``--timeout=600``.
HANG_BOUND_FLOOR_MS = 600_000
BOUND_NAME = "HANG_BOUND_MS"
TIMEOUT_KEYS = ("testTimeout", "hookTimeout")

TEST_CALLEES = ("test", "it", "describe", "bench")
HOOK_CALLEES = ("beforeEach", "afterEach", "beforeAll", "afterAll")

#: Every CLI spelling of the two options (cac camel-cases option names).
TIMEOUT_FLAG_RE = re.compile(r"^--(testTimeout|hookTimeout|test-timeout|hook-timeout)(=|$)")


# --------------------------------------------------------------------------- #
# A JS masker: blanks comments, string, template and regex literal contents so
# that brackets and commas are only counted in code.
# --------------------------------------------------------------------------- #

_REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_KEYWORDS = ("return", "typeof", "case", "do", "else", "in", "of", "void", "yield", "await")


def _regex_may_start(out: list[str]) -> bool:
    j = len(out) - 1
    while j >= 0 and out[j].isspace():
        j -= 1
    if j < 0:
        return True
    if out[j] in _REGEX_PRECEDERS:
        return True
    tail = "".join(out[max(0, j - 6) : j + 1])
    return any(re.search(rf"(^|[^\w$]){kw}$", tail) for kw in _REGEX_KEYWORDS)


def mask_js(src: str) -> str:
    """Return ``src`` with literal and comment contents replaced by spaces.

    Delimiters stay, offsets stay, newlines stay; code is untouched.
    """
    out: list[str] = []
    i, n = 0, len(src)

    def blank(chunk: str) -> str:
        return "".join(c if c == "\n" else " " for c in chunk)

    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            end = src.find("\n", i)
            end = n if end == -1 else end
            out.append(blank(src[i:end]))
            i = end
        elif c == "/" and nxt == "*":
            end = src.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append(blank(src[i:end]))
            i = end
        elif c in "'\"`" or (c == "/" and _regex_may_start(out)):
            j = i + 1
            in_class = False
            while j < n:
                d = src[j]
                if d == "\\":
                    j += 2
                    continue
                if c == "/" and d == "[":
                    in_class = True
                elif c == "/" and d == "]":
                    in_class = False
                elif d == c and not in_class:
                    break
                elif d == "\n" and c in "'\"/":
                    break
                j += 1
            out.append(c + blank(src[i + 1 : j]) + (src[j] if j < n else ""))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


_OPEN = {"(": ")", "[": "]", "{": "}"}


def _close(masked: str, start: int) -> int:
    """Index of the bracket closing the one at ``start``; -1 if unbalanced."""
    depth = 0
    for k in range(start, len(masked)):
        ch = masked[k]
        if ch in _OPEN:
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return k
    return -1


def _split_top(masked: str, start: int, end: int) -> list[tuple[int, int]]:
    """Spans of the depth-0, comma-separated items strictly inside start..end."""
    spans: list[tuple[int, int]] = []
    depth, item = 0, start + 1
    for k in range(start + 1, end):
        ch = masked[k]
        if ch in _OPEN:
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            spans.append((item, k))
            item = k + 1
    spans.append((item, end))
    return [(a, b) for a, b in spans if masked[a:b].strip()]


# --------------------------------------------------------------------------- #
# Check 1: the config declares the bound once.
# --------------------------------------------------------------------------- #

_NUMBER_RE = re.compile(r"^\d[\d_]*(?:\.\d+)?(?:[eE]\d+)?$")


def _number(text: str) -> int | None:
    text = text.strip()
    if not _NUMBER_RE.match(text):
        return None
    return int(float(text.replace("_", "")))


def config_problems(src: str) -> list[str]:
    masked = mask_js(src)
    problems: list[str] = []
    decls = list(re.finditer(rf"\b(?:const|let|var)\s+{BOUND_NAME}\s*=\s*([^;\n]+)", masked))
    if len(decls) != 1:
        return [f"{BOUND_NAME} must be declared exactly once, found {len(decls)}"]
    value = _number(decls[0].group(1))
    if value is None:
        problems.append(f"{BOUND_NAME} must be a numeric literal")
    elif value < HANG_BOUND_FLOOR_MS:
        problems.append(f"{BOUND_NAME} = {value} is below the floor {HANG_BOUND_FLOOR_MS}")
    block = re.search(r"\btest\s*:\s*\{", masked)
    if block is None:
        return [*problems, "vitest.config.js has no test: block"]
    body = masked[block.end() - 1 : _close(masked, block.end() - 1) + 1]
    for key in TIMEOUT_KEYS:
        found = re.findall(rf"\b{key}\s*:\s*([^,\n}}]+)", body)
        if found != [BOUND_NAME]:
            problems.append(f"test.{key} must be set once to {BOUND_NAME}, found {found}")
    return problems


def hang_bound_ms(src: str) -> int:
    match = re.search(rf"\b{BOUND_NAME}\s*=\s*([^;\n]+)", mask_js(src))
    assert match is not None
    value = _number(match.group(1))
    assert value is not None
    return value


# --------------------------------------------------------------------------- #
# Check 2: no test, hook or setConfig call lowers it.
# --------------------------------------------------------------------------- #

_MODIFIERS = r"(?:\s*\.\s*(?:only|skip|concurrent|sequential|todo|fails|skipIf|runIf|each|for))*"
_CALLEE_RE = re.compile(rf"(?<![\w$.])({'|'.join(TEST_CALLEES + HOOK_CALLEES)}){_MODIFIERS}\s*[(`]")
_SET_CONFIG_RE = re.compile(r"(?<![\w$])vi\s*\.\s*setConfig\s*\(")
_FUNCTION_RE = re.compile(r"^(?:async\s+)?(?:function\b|\(|[\w$]+\s*=>)")


def _call_args(masked: str, match: re.Match[str]) -> tuple[int, int] | None:
    """The (open, close) of the call that receives the test arguments.

    A modifier taking its own arguments (``.each(table)``, ``.skipIf(cond)``) or a
    tagged template is skipped; the next parenthesised list is the call.
    """
    pos = match.end() - 1
    while True:
        if masked[pos] == "`":
            pos = masked.index("`", pos + 1) + 1
        else:
            close = _close(masked, pos)
            if close == -1:
                return None
            head = masked[match.start() : pos]
            follows = masked[close + 1 :].lstrip()
            takes_args = re.search(r"\.\s*(?:each|for|skipIf|runIf)\s*$", head)
            if not (takes_args and follows.startswith(("(", "."))):
                return pos, close
            pos = close + 1
        rest = re.match(rf"{_MODIFIERS}\s*([(`])", masked[pos:])
        if rest is None:
            return None
        pos += rest.end() - 1


def _timeout_value_problem(text: str, bound: int, where: str) -> str | None:
    value = _number(text)
    if value is None:
        return f"{where}: timeout {text.strip()!r} cannot be evaluated"
    if value < bound:
        return f"{where}: timeout {value} is below the hang bound {bound}"
    return None


def _object_timeout(src: str, masked: str, span: tuple[int, int]) -> list[str | None]:
    """Values of the ``timeout`` keys of an object literal (shorthand → name)."""
    a, b = span
    text = masked[a:b].strip()
    if not text.startswith("{"):
        return []
    open_at = masked.index("{", a)
    close = _close(masked, open_at)
    values: list[str | None] = []
    for ia, ib in _split_top(masked, open_at, close):
        entry = src[ia:ib].strip()
        key = re.match(r"^['\"]?timeout['\"]?\s*(:|$)", entry)
        if key:
            values.append(entry[key.end() :] if key.group(1) == ":" else "timeout")
    return values


def source_problems(name: str, src: str, bound: int) -> list[str]:
    masked = mask_js(src)
    problems: list[str] = []

    def line(at: int) -> str:
        return f"{name}:{src.count(chr(10), 0, at) + 1}"

    for match in _CALLEE_RE.finditer(masked):
        call = _call_args(masked, match)
        if call is None:
            problems.append(f"{line(match.start())}: cannot read the arguments of this call")
            continue
        args = _split_top(masked, *call)
        for span in args:
            for value in _object_timeout(src, masked, span):
                problem = _timeout_value_problem(value or "", bound, line(span[0]))
                if problem:
                    problems.append(problem)
        is_hook = match.group(1) in HOOK_CALLEES
        trailing = args[1:] if is_hook else (args[2:] if len(args) >= 3 else [])
        for a, b in trailing[-1:]:
            text = src[a:b].strip()
            if text.startswith("{") or _FUNCTION_RE.match(text):
                continue
            problem = _timeout_value_problem(text, bound, line(a))
            if problem:
                problems.append(problem)

    for match in _SET_CONFIG_RE.finditer(masked):
        close = _close(masked, match.end() - 1)
        if close == -1:
            problems.append(f"{line(match.start())}: cannot read the arguments of vi.setConfig")
            continue
        for a, b in _split_top(masked, match.end() - 1, close):
            if not masked[a:b].strip().startswith("{"):
                problems.append(f"{line(a)}: vi.setConfig argument cannot be evaluated")
                continue
            open_at = masked.index("{", a)
            for ia, ib in _split_top(masked, open_at, _close(masked, open_at)):
                entry = src[ia:ib].strip()
                key = re.match(r"^['\"]?(testTimeout|hookTimeout)['\"]?\s*(:|$)", entry)
                if key is None:
                    continue
                value = entry[key.end() :] if key.group(2) == ":" else key.group(1)
                problem = _timeout_value_problem(value, bound, line(ia))
                if problem:
                    problems.append(problem)
    return problems


def _tracked_js_sources() -> dict[str, str]:
    listed = subprocess.run(
        ["git", "ls-files", "*.test.js", "*.test.mjs", SETUP_FILE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return {path: (REPO_ROOT / path).read_text(encoding="utf-8") for path in sorted(listed)}


# --------------------------------------------------------------------------- #
# Check 3: no invocation lowers it.
# --------------------------------------------------------------------------- #


def _flag_problems(where: str, command: str) -> list[str]:
    try:
        tokens = shlex.split(command, comments=True)
    except ValueError:
        tokens = command.split()
    return [f"{where}: {tok}" for tok in tokens if TIMEOUT_FLAG_RE.match(tok)]


def _invokes_vitest(run: str) -> bool:
    return "vitest" in run or "test:js" in run


def invocation_problems(
    package: dict[str, Any], workflows: dict[str, Any], scripts: dict[str, str]
) -> list[str]:
    problems: list[str] = []
    for script, command in sorted(package.get("scripts", {}).items()):
        problems += _flag_problems(f"package.json scripts.{script}", command)
    for wf_name, workflow in sorted(workflows.items()):
        for job_name, job in sorted((workflow or {}).get("jobs", {}).items()):
            for index, step in enumerate(job.get("steps", []) or []):
                run = step.get("run")
                if isinstance(run, str) and _invokes_vitest(run):
                    where = f"{wf_name} jobs.{job_name}.steps[{index}]"
                    problems += _flag_problems(where, run)
    for script_name, text in sorted(scripts.items()):
        for number, script_line in enumerate(text.splitlines(), start=1):
            problems += _flag_problems(f"scripts/{script_name}:{number}", script_line)
    return problems


def _package() -> dict[str, Any]:
    return json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))


def _workflows() -> dict[str, Any]:
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
    }


def _shell_scripts() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8") for path in sorted(SCRIPTS_DIR.glob("*.sh"))
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def _config_src() -> str:
    return VITEST_CONFIG.read_text(encoding="utf-8")


def test_config_declares_one_hang_bound() -> None:
    assert config_problems(_config_src()) == []


@pytest.mark.parametrize(
    ("label", "old", "new"),
    [
        ("drop testTimeout", f"testTimeout: {BOUND_NAME},", ""),
        ("drop hookTimeout", f"hookTimeout: {BOUND_NAME},", ""),
        ("literal testTimeout", f"testTimeout: {BOUND_NAME},", "testTimeout: 5000,"),
        ("constant below floor", f"{BOUND_NAME} = 600_000", f"{BOUND_NAME} = 599_999"),
    ],
)
def test_config_declares_one_hang_bound_refuses(label: str, old: str, new: str) -> None:
    src = _config_src()
    assert old in src, label
    assert config_problems(src.replace(old, new)) != [], label


def test_no_js_test_lowers_the_hang_bound() -> None:
    bound = hang_bound_ms(_config_src())
    problems = [
        problem
        for name, src in _tracked_js_sources().items()
        for problem in source_problems(name, src, bound)
    ]
    assert problems == []


_SYNTHETIC = "test('runs', async () => {\n  expect(1).toBe(1);\n});\n"


@pytest.mark.parametrize(
    ("label", "src"),
    [
        ("trailing numeric", _SYNTHETIC.replace("});\n", "}, 2000);\n")),
        ("options object", "test('runs', { timeout: 100 }, () => {});\n"),
        ("options object after the callback", "it.only('runs', () => {}, { timeout: 100 });\n"),
        ("each form", "test.each([1, 2])('case %i', (n) => {}, 1000);\n"),
        ("hook timeout", "beforeEach(async () => {}, 50);\n"),
        ("describe options", "describe.concurrent('suite', { timeout: 10 }, () => {});\n"),
        ("unresolvable name", "test('runs', () => {}, SLOW_MS);\n"),
        ("setConfig", "vi.setConfig({ testTimeout: 1000 });\n"),
    ],
)
def test_no_js_test_lowers_the_hang_bound_refuses(label: str, src: str) -> None:
    assert source_problems("synthetic.test.mjs", src, HANG_BOUND_FLOOR_MS) != [], label


@pytest.mark.parametrize(
    ("label", "src"),
    [
        ("equal to the bound", _SYNTHETIC.replace("});\n", "}, 600_000);\n")),
        ("options object at the bound", "test('runs', { timeout: 600000 }, () => {});\n"),
        ("setConfig above the bound", "vi.setConfig({ hookTimeout: 900000 });\n"),
        ("condition wait", "await vi.waitFor(cb, { timeout: 50 });\n"),
        ("regex test method", "expect(/x/.test(s)).toBe(true);\n"),
    ],
)
def test_no_js_test_lowers_the_hang_bound_accepts(label: str, src: str) -> None:
    assert source_problems("synthetic.test.mjs", src, HANG_BOUND_FLOOR_MS) == [], label


@pytest.mark.parametrize(
    ("path", "snippet"),
    [
        (
            SETUP_FILE,
            "nativeWaitFor(callback, { timeout: options, interval: WAIT_FOR_INTERVAL_MS })",
        ),
        ("tests/interfaces/artifacts/preview-content.test.mjs", "renderJsonHtml({}, 0)"),
        (
            "tests/interfaces/lattice_dashboard/net.test.mjs",
            "stubFigure({ status: 'ready', key: 'k', figure }, 200)",
        ),
        (
            "tests/interfaces/web_terminal/activity-strip.test.mjs",
            "pastFrame({ kind: 'channel', detail: 'SR01:HCM1:SP' }, 5)",
        ),
    ],
)
def test_no_js_test_lowers_the_hang_bound_reads_only_test_calls(path: str, snippet: str) -> None:
    """Real-tree negative controls: a timeout or trailing number outside a test call."""
    src = (REPO_ROOT / path).read_text(encoding="utf-8")
    assert snippet in src
    assert source_problems(path, snippet + ";\n", HANG_BOUND_FLOOR_MS) == []


def test_no_invocation_lowers_the_hang_bound() -> None:
    assert invocation_problems(_package(), _workflows(), _shell_scripts()) == []


def test_no_invocation_lowers_the_hang_bound_refuses_an_npm_script() -> None:
    package = copy.deepcopy(_package())
    package["scripts"]["test:js"] = "vitest run --testTimeout=5000"
    assert invocation_problems(package, _workflows(), _shell_scripts()) != []


def test_no_invocation_lowers_the_hang_bound_refuses_a_ci_step() -> None:
    workflows = copy.deepcopy(_workflows())
    steps = [
        step
        for job in workflows["ci.yml"]["jobs"].values()
        for step in job.get("steps", []) or []
        if step.get("name") == "Run JS tests (vitest)"
    ]
    assert len(steps) == 1
    steps[0]["run"] += " -- --hook-timeout 5000"
    assert invocation_problems(_package(), workflows, _shell_scripts()) != []


def test_no_invocation_lowers_the_hang_bound_refuses_a_shell_script() -> None:
    scripts = {**_shell_scripts(), "gate.sh": "npx vitest run --test-timeout=1000\n"}
    assert invocation_problems(_package(), _workflows(), scripts) != []
