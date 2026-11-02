"""The ``code_review`` config section: reading it, validating it, finding its prompt.

The section is validated in two stages. A build goes through
:func:`parse_review_section`, which refuses a malformed section whether or not it
is enabled. A running session goes through :func:`load_review_settings`, which
reads ``enabled`` first and treats a section that is off as off, validating
nothing further. The build refuses a superset of what runtime refuses, so a
malformed section never reaches a session.

The file is read here, uncached, rather than through ``ConfigBuilder``: an edit
must reach the next review, and a defect in some other section is not this
feature's concern. It is not read through ``load_osprey_config`` either, which
would turn a broken file into "off".
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from osprey.utils.workspace import STATE_DIR_NAME
from osprey_connectors.workspace import resolve_config_path

#: The reviewer's prompt file, relative to the repo root.
DEFAULT_PROMPT = "data/code_review/reviewer.md"

#: Seconds one review may take, on either surface.
DEFAULT_TIME_LIMIT_S = 60.0

#: Seconds the notebook hook's deadline allows beyond the review's own time limit.
HOOK_OVERHEAD_S = 30

#: Parent of each review's fresh temporary working directory. A sibling of the
#: state directory, not inside it.
REVIEW_STATE_DIR = f"{STATE_DIR_NAME}/code_review"

#: What a review's finding may do: only report it, or also stop the code.
LEVELS = ("flag", "block")

#: The config file's top-level key for this feature.
_SECTION_KEY = "code_review"


class ReviewConfigError(Exception):
    """The ``code_review`` section, its config file or its prompt cannot be used."""


@dataclass(frozen=True)
class ReviewSettings:
    """A validated ``code_review`` section.

    Attributes:
        enabled: Whether code is reviewed at all.
        level: One of :data:`LEVELS`.
        prompt: Path of the reviewer's prompt; a relative one is anchored at the repo root.
        modes: Execution modes reviewed on the execute surface; the notebook surface
            has no mode.
        time_limit: Seconds one review may take, on either surface.
    """

    enabled: bool = False
    level: str = "block"
    prompt: str = DEFAULT_PROMPT
    modes: frozenset[str] = frozenset({"readwrite"})
    time_limit: float = DEFAULT_TIME_LIMIT_S


def read_review_config(path: Path) -> Mapping:
    """Read the config file at *path*, uncached on every call.

    No ``${VAR}`` expansion happens: the section's values are literals, and a
    placeholder fails validation as a config error.

    Args:
        path: The config file.

    Returns:
        The file's top-level mapping, or an empty one when the file does not exist.

    Raises:
        ReviewConfigError: The file exists but cannot be read, does not parse, or
            does not hold a mapping.
    """
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ReviewConfigError(f"cannot read {path}: {exc}") from exc
    # A ValueError is non-UTF-8 text or a date YAML accepts and Python cannot build.
    except (yaml.YAMLError, ValueError, RecursionError) as exc:
        raise ReviewConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ReviewConfigError(f"{path} does not contain a YAML mapping")
    return document


def load_review_settings(config: Mapping | None = None) -> ReviewSettings:
    """The settings a running session acts on.

    A missing section, or one whose ``enabled`` is absent or exactly ``False``, is
    off with no further validation of the section. Only the ``code_review`` section
    is read; a defect elsewhere in the config is ignored.

    Args:
        config: A loaded config mapping. ``None`` reads the file
            :func:`~osprey_connectors.workspace.resolve_config_path` names.

    Returns:
        The settings, or the off defaults.

    Raises:
        ReviewConfigError: The file cannot be used, the section is not a mapping,
            ``enabled`` is not a bool, or an enabled section is malformed.
    """
    if config is None:
        config = read_review_config(resolve_config_path())
    if _SECTION_KEY not in config:
        return ReviewSettings()
    section = config[_SECTION_KEY]
    if isinstance(section, Mapping) and section.get("enabled", False) is False:
        return ReviewSettings()
    return parse_review_section(section)


def parse_review_section(section: object) -> ReviewSettings:
    """Validate every key of a present section, whatever ``enabled`` says.

    Args:
        section: The value of the config's ``code_review`` key.

    Returns:
        The validated settings.

    Raises:
        ReviewConfigError: Naming the offending key.
    """
    if not isinstance(section, Mapping):
        raise ReviewConfigError(f"code_review must be a mapping, got {section!r}")

    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ReviewConfigError(f"code_review.enabled must be true or false, got {enabled!r}")

    level = section.get("level", ReviewSettings.level)
    if level not in LEVELS:
        raise ReviewConfigError(f"code_review.level must be one of {LEVELS}, got {level!r}")

    prompt = section.get("prompt", DEFAULT_PROMPT)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ReviewConfigError(f"code_review.prompt must be a file path, got {prompt!r}")

    return ReviewSettings(
        enabled=enabled,
        level=level,
        prompt=prompt,
        modes=_parse_modes(section.get("modes", ReviewSettings.modes)),
        time_limit=_parse_time_limit(section.get("time_limit", DEFAULT_TIME_LIMIT_S)),
    )


def _parse_modes(value: object) -> frozenset[str]:
    """Validate ``code_review.modes``: a collection of execution mode names."""
    from osprey.mcp_server.python_executor.tools._execution_gates import VALID_EXECUTION_MODES

    allowed = sorted(VALID_EXECUTION_MODES)
    if not isinstance(value, list | tuple | set | frozenset):
        raise ReviewConfigError(f"code_review.modes must be a list from {allowed}, got {value!r}")
    for mode in value:
        if not isinstance(mode, str) or mode not in VALID_EXECUTION_MODES:
            raise ReviewConfigError(f"code_review.modes must be from {allowed}, got {mode!r}")
    return frozenset(value)


def _parse_time_limit(value: object) -> float:
    """Validate ``code_review.time_limit``: a positive, finite number of seconds."""
    refusal = ReviewConfigError(
        f"code_review.time_limit must be a positive number of seconds, got {value!r}"
    )
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise refusal
    try:
        seconds = float(value)
    except OverflowError as exc:
        raise refusal from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise refusal
    return seconds


def resolve_prompt(settings: ReviewSettings, repo_root: Path, config: Mapping | None = None) -> str:
    """Read the reviewer's prompt, refusing one the agent could have written.

    The prompt must sit in the profile's source zone, which a build does not
    overwrite and the agent cannot write to. Being protected is not enough:
    ``build/`` and ``var/*`` are protected too, and hold what a run generated. A
    relocated agent-data root can overlap the source zone, so a prompt under a
    permitted root is refused as well.

    Args:
        settings: The settings naming the prompt.
        repo_root: Anchor of a relative prompt path and of the source zone.
        config: Loaded config mapping, for the permitted roots. ``None`` leaves
            them to :func:`~osprey.mcp_server.python_executor.executor.resolve_permitted_roots`.

    Returns:
        The prompt's text.

    Raises:
        ReviewConfigError: The prompt is missing, unreadable or not UTF-8, is
            outside the source zone, or is under a permitted root.
    """
    from osprey.mcp_server.python_executor.executor import (
        PROFILE_SOURCE_ENTRIES,
        resolve_permitted_roots,
    )

    root = Path(repo_root)
    label = f"code_review.prompt {settings.prompt!r}"
    try:
        path = (root / settings.prompt).resolve()
        source_zone = [(root / entry).resolve() for entry in PROFILE_SOURCE_ENTRIES]
    except (OSError, ValueError) as exc:
        raise ReviewConfigError(f"{label} is not a usable path: {exc}") from exc

    if not any(path.is_relative_to(entry) for entry in source_zone):
        raise ReviewConfigError(f"{label} is not inside the profile's source zone under {root}")
    if any(path.is_relative_to(permitted) for permitted in resolve_permitted_roots(root, config)):
        raise ReviewConfigError(f"{label} is under the agent's own data root")

    try:
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise ReviewConfigError(f"{label} cannot be read: {exc}") from exc


def hook_deadline_s(settings: ReviewSettings) -> int:
    """Seconds the notebook hook is given: the time limit rounded up, plus the overhead."""
    return math.ceil(settings.time_limit) + HOOK_OVERHEAD_S
