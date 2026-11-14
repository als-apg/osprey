"""Text a deployment repo or an agent session is handed names things that exist.

The templates under ``src/osprey/templates`` and the strings ``osprey init``
writes are read by an operator in their own repository, or by the agent in
every web-terminal session. Neither reader can check a sentence against the
code, so a file to edit that the next build clears, a variable nothing reads or
a retired UI mode is followed as written. Each test below holds one such
class.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.docs.test_environment_variable_page import _REPO_ROOT

_SRC = _REPO_ROOT / "src" / "osprey"
_TEMPLATES = _SRC / "templates"
_PRESETS = _SRC / "profiles" / "presets"

#: ``config.yml`` named as the place a setting lives. The only one in a
#: deployment repo is ``build/config.yml``, which every build re-renders.
_IN_CONFIG_YML = re.compile(r"\bin\s+(?:the\s+)?`?(?<!build/)config\.yml")

#: Web-terminal vocabulary that left the product: the mock backend became the
#: simulator served in process, and the Expert/Simple views became one surface.
_RETIRED_WORDS = re.compile(r"\bmock backend\b|\bexpert\b|\bsimple mode\b|\bSimple may\b", re.I)


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _operator_files() -> list[Path]:
    """Prose that lands in a deployment repo outside ``build/``."""
    files = [_SRC / "cli" / "init_cmd.py"]
    files += sorted(_TEMPLATES.glob("apps/*/data/**/*.md"))
    files += sorted(_TEMPLATES.glob("**/web-terminal-context/*.md"))
    return files


def _agent_files() -> list[Path]:
    """Text seeded into an agent session: the context baselines and every skill."""
    files = sorted(_TEMPLATES.glob("**/web-terminal-context/*.md"))
    files += sorted(_TEMPLATES.glob("claude_code/claude/skills/**/SKILL.md"))
    files += sorted(_PRESETS.glob("**/SKILL.md"))
    return files


def test_the_generated_config_never_asks_to_be_edited() -> None:
    """``build/config.yml`` is cleared by every build; nothing in it is uncommented."""
    template = _TEMPLATES / "project" / "config.yml.j2"
    offences = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(template.read_text(encoding="utf-8").splitlines(), 1)
        if line.startswith("#") and re.search(r"\buncomment\b", line, re.I)
    ]
    assert not offences, (
        f"{_rel(template)} renders into build/config.yml, which the next build discards; "
        "point the reader at profile.yml instead:\n" + "\n".join(offences)
    )


def test_operator_prose_never_places_a_key_in_config_yml() -> None:
    offences = []
    for path in _operator_files():
        text = path.read_text(encoding="utf-8")
        offences += [
            f"{_rel(path)}:{text.count(chr(10), 0, match.start()) + 1}"
            for match in _IN_CONFIG_YML.finditer(text)
        ]
    assert not offences, (
        "a deployment repo has no config.yml of its own — a key is set in profile.yml's "
        "`config:` block:\n" + "\n".join(offences)
    )


def test_every_variable_the_env_example_offers_is_read() -> None:
    """A ``NAME=`` line the template spells itself names a variable some module reads."""
    template = _TEMPLATES / "project" / "env.example.j2"
    names = re.findall(r"^#?\s?([A-Z][A-Z0-9_]+)=", template.read_text(encoding="utf-8"), re.M)
    readers = "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for root in (_SRC, _REPO_ROOT / "packages")
        for path in root.rglob("*")
        if path.is_file() and path != template and path.suffix in {".py", ".j2", ".sh", ".yml"}
    )
    unread = [name for name in names if not re.search(rf"\b{name}\b", readers)]
    assert not unread, f"{_rel(template)} offers variables nothing reads: {unread}"


def test_the_context_tree_is_named_where_it_is_authored() -> None:
    """``docker/web-terminal-context`` exists only under ``build/``, as a copy."""
    offences = [
        _rel(path)
        for path in (*_operator_files(), *_agent_files())
        if re.search(r"(?<!build/)docker/web-terminal-context", path.read_text(encoding="utf-8"))
    ]
    assert not offences, (
        "per-user context is authored in `web-terminal-context/<user>/` at the repo root:\n"
        + "\n".join(sorted(set(offences)))
    )


def test_agent_text_uses_no_retired_web_terminal_words() -> None:
    from osprey.agent_runner.build_artifacts.catalog import _get_default_artifacts

    offences = []
    for path in _agent_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _RETIRED_WORDS.search(line):
                offences.append(f"{_rel(path)}:{number}: {line.strip()}")
    for artifact in _get_default_artifacts():
        if _RETIRED_WORDS.search(artifact.description):
            offences.append(f"catalog {artifact.canonical_name}: {artifact.description}")
    assert not offences, "retired web-terminal vocabulary:\n" + "\n".join(offences)


def test_the_demo_context_names_every_tier_that_can_write() -> None:
    """The sentence telling the agent who may write lists the write-armed tiers."""
    context = _TEMPLATES / "apps" / "demo" / "web-terminal-context" / "base.md"
    prose = " ".join(context.read_text(encoding="utf-8").split()).replace("-", "")
    sentence = re.search(r"[^.]*\bcan execute writes\b", prose)
    assert sentence, f"{_rel(context)} no longer says which tiers can execute writes"
    armed = [
        tier.stem
        for tier in sorted((_PRESETS / "tiers").glob("*.yml"))
        if re.search(r"^\s+control_system\.writes: (?:asked|allowed)\b", tier.read_text(), re.M)
    ]
    assert armed, "no tier preset is write-armed"
    unnamed = [tier for tier in armed if tier.replace("-", "") not in sentence.group(0)]
    assert not unnamed, f"{_rel(context)} leaves out write-armed tiers: {unnamed}"
