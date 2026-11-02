"""The ``osprey set`` verb — the one sanctioned CLI write into ``profile.yml``.

Every other lifecycle verb reads the deployment repo's source zone; this one
writes it. The write lands in ``profile.yml`` and nowhere else: ``build/`` is
100% derived, so a CLI edit of the rendered ``build/config.yml`` would be
undone by the next build without ever having reached the file that describes
the deployment. That failure mode — a setting that works until someone
rebuilds — is what having exactly one writer prevents.

The machinery is the build's own write-back
(:func:`~osprey.cli.build_profile_resolve.write_back_cli_overrides`), promoted
rather than reimplemented: comment-preserving round-trip YAML, one atomic
replace of the whole document, ``--set`` key spellings unchanged, and the same
refusal of an ``extends:`` write that materialization makes. A second writer
with its own spelling rules would mean ``osprey set connector=epics`` and
``osprey init --set connector=epics`` landing differently in the same file.

One shorthand is accepted as a key spelling: ``connector=``, folded into
``config.control_system.type`` by the shared layering step. It writes the
literal dotted key a reader would otherwise type by hand, so nothing lands in
the profile that only this command can understand.

``epics_gateway=<facility>`` was a second one. It expanded a table of named
facilities' gateway hostnames that core no longer carries — a gateway address
is site infrastructure, not something a framework can know — so it is refused
here with the dotted keys that do the job.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import click

from osprey_connectors.types import TIER_WORDS

from .output import note, report, warn
from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option
from .styles import Styles

#: Retired shorthand. It stood for a facility out of a table of named
#: institutions' gateway hostnames that core no longer ships; refused rather
#: than passed through, because as a profile key it is one the next
#: ``osprey build`` rejects.
RETIRED_GATEWAY_KEY = "epics_gateway"

#: Where the EPICS connector's gateway table lives in the rendered config, as
#: the ``config.``-prefixed key path ``--set`` addresses it by.
GATEWAY_KEY_PREFIX = "config.control_system.connector.epics.gateways"


#: The two spellings that state a control-system type on this command line.
_TYPE_KEYS = ("connector", "config.control_system.type")


def _refuse_retired_shorthands(pairs: tuple[str, ...]) -> None:
    """Refuse retired spellings before anything is written.

    Two refusals: the ``epics_gateway=`` shorthand, and a retired control-system
    type name stated through ``connector=`` or
    ``config.control_system.type=`` — the latter names the new spelling. Runs
    before the shared parser sees anything, so the refusal costs no partial
    write: the whole command line is rejected the way a malformed pair is.
    """
    from osprey_connectors.types import RETIRED_CONTROL_SYSTEM_TYPES, retired_type_message

    for pair in pairs:
        key, separator, value = pair.partition("=")
        if (
            separator
            and key.strip() in _TYPE_KEYS
            and value.strip() in RETIRED_CONTROL_SYSTEM_TYPES
        ):
            raise click.UsageError(retired_type_message(value.strip()))
        if separator and key.strip() == RETIRED_GATEWAY_KEY:
            raise click.UsageError(
                f"`{RETIRED_GATEWAY_KEY}=` named a facility out of a gateway table "
                "OSPREY no longer ships. Write the gateway your control network "
                "actually has:\n"
                f"  osprey set {GATEWAY_KEY_PREFIX}.read_only.address=gw.example.org "
                f"{GATEWAY_KEY_PREFIX}.read_only.port=5064"
            )


#: The write ceiling, as the rendered-config path ``config.``-prefixed keys address.
_WRITES_PATH = "control_system.writes"

#: The value this command may write from any ceiling.
_WRITES_NARROW = "never"

#: The value this command may write from the widest ceiling alone.
_WRITES_ASKED = "asked"

#: The section of the rendered config the ceiling is read from.
_CONTROL_SYSTEM = "control_system"


def _config_leaves(path: str, value: object) -> list[tuple[str, object]]:
    """Every ``(dotted path, value)`` leaf *value* writes when stated at *path*.

    A mapping value states the whole block at its key, so a posture key can sit
    anywhere inside one (``config.control_system={writes: allowed}``). The
    ceiling itself is not descended into: whatever is stated at it is its value.
    """
    if isinstance(value, dict) and value and path != _WRITES_PATH:
        leaves: list[tuple[str, object]] = []
        for key, inner in value.items():
            leaves.extend(_config_leaves(f"{path}.{key}" if path else str(key), inner))
        return leaves
    return [(path, value)]


def _rendered_ceiling(repo_root: Path) -> str:
    """The write ceiling the root renders, read through the render.

    The root may be silent and inherit from what it extends, and a ``config:``
    block holds dotted keys applied in order, so neither the file nor a plain key
    lookup answers this.
    """
    from osprey_connectors.types import tier_writes

    from .build_profile_merge import _rendered_control_system
    from .build_profile_resolve import resolve_build_profile

    profile, _ = resolve_build_profile(repo_root / PROFILE_FILENAME, None)
    return tier_writes(_rendered_control_system(profile.config))


def _read_ceiling(repo_root: Path) -> tuple[str | None, Exception | None]:
    """The rendered ceiling, or ``None`` and why it could not be read.

    Any failure counts: a ceiling that cannot be read is unknown, whatever raised.
    """
    try:
        return _rendered_ceiling(repo_root), None
    except Exception as e:  # every failure leaves the ceiling unknown
        return None, e


def _root_ceiling(repo_root: Path) -> str:
    """The ceiling a move is judged against: unknown refuses it, nothing written yet."""
    ceiling, error = _read_ceiling(repo_root)
    if ceiling is None:
        raise click.UsageError(str(error)) from error
    return ceiling


def _load_profile(data: bytes) -> object:
    """*data* parsed by the parser the write uses; raises its error on a bad document."""
    from ruamel.yaml import YAML

    return YAML(typ="safe", pure=True).load(data)


def _refuse_unparsable(profile_path: Path) -> bytes:
    """The profile's bytes, refused before anything is written unless they are a YAML mapping.

    An empty file is a mapping with nothing in it yet.
    """
    from ruamel.yaml.error import YAMLError

    data = profile_path.read_bytes()
    try:
        document = _load_profile(data)
    except YAMLError as e:
        raise click.UsageError(
            f"{profile_path.name} is not valid YAML, so nothing was written. Fix the file "
            f"first; `osprey set` edits a document it can read.\n{e}"
        ) from e
    if document is not None and not isinstance(document, dict):
        raise click.UsageError(
            f"{profile_path.name} holds a {type(document).__name__}, not a mapping of profile "
            "keys, so nothing was written. Fix the file first."
        )
    return data


def _names_writes(value: object) -> bool:
    """Whether a mapping in *value*, at any depth, has a key spelling ``writes``."""
    if not isinstance(value, dict):
        return False
    return any(
        not isinstance(k, str) or "writes" in k or _names_writes(v) for k, v in value.items()
    )


def _can_reach_ceiling(key: object, value: object) -> bool:
    """Whether a ``config:`` entry can bear on the rendered ceiling.

    The ceiling is the value at ``control_system.writes``, and an entry assigns the
    value at its own path, so only a path that holds it or lies beneath it reaches
    it. Anything that could be read otherwise is taken to reach it too: an odd
    spelling, and a ``writes`` word anywhere in the path or the value (the retired
    ``writes_enabled`` leaves among them).
    """
    if not isinstance(key, str):
        return True
    segments = [segment.strip() for segment in key.split(".")]
    target = _WRITES_PATH.split(".")
    shared = min(len(segments), len(target))
    return (
        not all(segments)
        or segments[:shared] == target[:shared]
        or any("writes" in segment for segment in segments)
        or _names_writes(value)
    )


#: Stands in the entries for a ``config:`` that is not a mapping.
_NOT_A_MAPPING = object()


def _ceiling_inputs(
    document: object, legal: frozenset[str]
) -> tuple[object, list[tuple[object, object]]]:
    """What in the root file the rendered ceiling is computed from.

    The render reads the merged ``config:``, which the root's known keys shape
    (what it inherits, and from where) and its entries write into, in order. A
    key the schema does not know feeds nothing: the build refuses it. An entry
    that cannot reach the ceiling (:func:`_can_reach_ceiling`) is left out.
    """
    if document is None:
        document = {}
    if not isinstance(document, dict):
        return ("document", document), []
    top = {k: v for k, v in document.items() if k != "config" and k in legal}
    if "config" not in document:
        return top, []
    config = document["config"]
    if not isinstance(config, dict):
        # It holds no entry, but replacing it with one changes what the render
        # reads, so its shape is itself an input. A bare `config:` (null) too:
        # the render refuses it where an absent key renders.
        return top, [(_NOT_A_MAPPING, config)]
    return top, [(k, v) for k, v in config.items() if _can_reach_ceiling(k, v)]


def _ceiling_change(before: bytes, after: bytes, profile_path: Path, word: str | None) -> str:
    """How the write changed what the ceiling is computed from, judged on the file.

    ``"none"`` when nothing that feeds the ceiling changed; ``"never"`` when the one
    change is the root's dotted ceiling line, now ``never`` (*word*); else
    ``"changed"``, which a line that cannot be judged also is.
    """
    try:
        after_document = _load_profile(after)
    except Exception:  # a document that cannot be read has changed
        return "changed"
    legal = _legal_keys(profile_path)
    top_before, entries_before = _ceiling_inputs(_load_profile(before), legal)
    top_after, entries_after = _ceiling_inputs(after_document, legal)
    if top_before != top_after:
        return "changed"
    if entries_before == entries_after:
        return "none"
    if word == _WRITES_NARROW:
        # A `config:` that was not a mapping becomes the mapping holding the line.
        rest_before = [e for e in entries_before if e[0] not in (_WRITES_PATH, _NOT_A_MAPPING)]
        rest_after = [e for e in entries_after if e[0] != _WRITES_PATH]
        if (_WRITES_PATH, _WRITES_NARROW) in entries_after and rest_before == rest_after:
            return _WRITES_NARROW
    return "changed"


def _broken_by_this_line(repo_root: Path, before: bytes, error: Exception) -> click.UsageError:
    """Take back a write that broke a root that resolved, and the refusal saying why."""
    from .build_profile_resolve import _atomic_write_bytes

    _atomic_write_bytes(repo_root / PROFILE_FILENAME, before)
    return click.UsageError(str(error))


def _land_ceiling(repo_root: Path, word: str, before: bytes, resolved_before: bool) -> None:
    """Make an accepted move true of the render, or take it back and refuse.

    The render is the oracle: a dotted write can sit behind a ``control_system:``
    mapping, because a merge keeps an inherited key at its inherited position and
    the mapping is then assigned whole over it. So the word is also written into
    that mapping, which stays where the merge put it. A move that still does not
    land restores *before*: exit 0 on this key tells the operator writes are off.

    A line that breaks a root which resolved (*resolved_before*) is taken back too.
    A root that already did not resolve was judged on the file instead: the line
    wrote ``never`` and nothing else the ceiling is computed from, so it is reported.
    """
    from .build_profile_resolve import _atomic_write_bytes, _write_profile_values

    profile_path = repo_root / PROFILE_FILENAME
    ceiling, error = _read_ceiling(repo_root)
    if ceiling is not None and ceiling != word:
        _write_profile_values(profile_path, [(["config", *_WRITES_PATH.split(".")], word)])
        ceiling, error = _read_ceiling(repo_root)
    if ceiling is None:
        if resolved_before:
            raise _broken_by_this_line(repo_root, before, cast(Exception, error)) from error
        note(f"{profile_path.name} does not resolve; `{_WRITES_PATH}` not checked in the render.")
        return
    if ceiling != word:
        _atomic_write_bytes(profile_path, before)
        raise click.UsageError(
            f"`{_WRITES_PATH}: {word}` does not reach the render, which stays `{ceiling}`. "
            "Nothing was written; state it in the profile or tier file that sets the block."
        )


def _hold_ceiling(repo_root: Path, ceiling: str, before: bytes) -> None:
    """Take back a line that raised the ceiling without naming it, and refuse.

    Judged by the render, like an accepted move: *ceiling* is what the root rendered
    before the write. A render that no longer reads cannot be judged, so the line is
    taken back.
    """
    from .build_profile_resolve import _atomic_write_bytes

    after, error = _read_ceiling(repo_root)
    if after is None:
        raise _broken_by_this_line(repo_root, before, cast(Exception, error)) from error
    if TIER_WORDS.index(after) > TIER_WORDS.index(ceiling):
        _atomic_write_bytes(repo_root / PROFILE_FILENAME, before)
        raise click.UsageError(
            f"This line would remove the word that holds `{_WRITES_PATH}` at `{ceiling}`, and "
            f"the render would rise to `{after}`. Nothing was written. "
            + _widening_refusal(after, f", which is `{ceiling}`")
        )


def _judge_line(
    repo_root: Path,
    before: bytes,
    ceiling: str | None,
    error: Exception | None,
    narrowed_to: str | None,
) -> None:
    """Keep a written line only when it is known not to raise the ceiling.

    Every line is judged, not only the keys expected to reach the ceiling. The
    render decides when it read before the write (*ceiling*). When it did not, the
    file decides: the line is kept only when it changed nothing the ceiling is
    computed from, or wrote ``never`` and nothing else. Any other line on such a
    root cannot be judged, so it is taken back and refused.
    """
    from .build_profile_resolve import _atomic_write_bytes

    profile_path = repo_root / PROFILE_FILENAME
    change = _ceiling_change(before, profile_path.read_bytes(), profile_path, narrowed_to)
    if ceiling is None and change == "changed":
        _atomic_write_bytes(profile_path, before)
        raise click.UsageError(
            f"{profile_path.name} does not resolve, so what this line does to "
            f"`{_WRITES_PATH}` cannot be judged. Nothing was written. Fix the profile "
            f"first; `osprey set config.{_WRITES_PATH}={_WRITES_NARROW}` stays writable.\n"
            f"{error}"
        ) from error
    if narrowed_to:
        _land_ceiling(repo_root, narrowed_to, before, resolved_before=ceiling is not None)
    elif ceiling is not None and change != "none":
        _hold_ceiling(repo_root, ceiling, before)


def _widening_refusal(word: str, stands_at: str = "") -> str:
    """The refusal of a move ``osprey set`` may not make, naming the edit a reviewer reads."""
    return (
        f"`osprey set` only lowers the write ceiling{stands_at}: `{_WRITES_PATH}: {word}` is "
        "stated by editing the profile or tier file, where it is reviewed. No CLI path turns "
        f"writes on; `config.{_WRITES_PATH}={_WRITES_NARROW}` lowers it from any ceiling, "
        f"`={_WRITES_ASKED}` only from `{TIER_WORDS[-1]}`."
    )


def _refuse_write_posture(pairs: tuple[str, ...], repo_root: Path) -> str | None:
    """Refuse every pair that would raise or misstate the write ceiling.

    No CLI path turns writes on. ``config.control_system.writes=never`` narrows
    from any ceiling and is accepted; ``=asked`` narrows only from ``allowed``,
    so it is judged against the ceiling the root renders today; ``=allowed`` and
    ``=asked`` from ``never`` or ``asked`` are refused — raising the ceiling is an
    edit to the profile or tier file a reviewer reads, not a command — and any
    other value is refused naming the three words. The retired ``writes_enabled``
    leaves (global or per connector type, at any depth of the path) are refused
    pointing at the new key, and so is a ``config`` value that is not a mapping.
    Runs before anything is written, so one refused pair rejects the whole line.
    A line that moves the ceiling without stating it is :func:`_judge_line`'s.

    Returns:
        The word the line narrows the ceiling to; ``None`` when it does not name it.
    """
    import yaml

    narrowed_to: str | None = None
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        key = key.strip()
        if not separator or not (key == "config" or key.startswith("config.")):
            continue
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        if key == "config" and not isinstance(value, dict):
            raise click.UsageError(
                f"`config` takes a mapping of rendered-config keys, not `{raw.strip()}`. "
                "Nothing was written; to set one key, address it as `config.<path>=<value>`."
            )
        for path, leaf in _config_leaves(key.removeprefix("config").lstrip("."), value):
            if path.startswith(f"{_CONTROL_SYSTEM}.") and "writes_enabled" in path.split("."):
                raise click.UsageError(
                    f"`config.{path}` is retired. The write ceiling is "
                    f"`{_WRITES_PATH}` ({' | '.join(TIER_WORDS)}). `osprey set` can only "
                    f"lower it — `osprey set config.{_WRITES_PATH}={_WRITES_NARROW}`; to allow "
                    "writes, edit the profile or tier file."
                )
            if path == _WRITES_PATH or path.startswith(f"{_WRITES_PATH}."):
                if path == _WRITES_PATH and leaf == _WRITES_NARROW:
                    narrowed_to = cast(str, leaf)
                    continue
                if path == _WRITES_PATH and leaf == _WRITES_ASKED:
                    ceiling = _root_ceiling(repo_root)
                    if TIER_WORDS.index(leaf) < TIER_WORDS.index(ceiling):
                        narrowed_to = cast(str, leaf)
                        continue
                    raise click.UsageError(
                        _widening_refusal(cast(str, leaf), f", which is `{ceiling}`")
                    )
                if path == _WRITES_PATH and leaf in TIER_WORDS:
                    raise click.UsageError(_widening_refusal(cast(str, leaf)))
                raise click.UsageError(
                    f"`{_WRITES_PATH}` takes one of {' | '.join(TIER_WORDS)}; `osprey set` "
                    f"writes only `{_WRITES_NARROW}`, or `{_WRITES_ASKED}` from "
                    f"`{TIER_WORDS[-1]}`."
                )
    return narrowed_to


def _unrecognized_top_level_keys(pairs: tuple[str, ...], profile_path: Path) -> list[str]:
    """Top-level keys in *pairs* not legal in the profile file at *profile_path*.

    Legal means the known-key set of that file's kind
    (:func:`~osprey.cli.build_profile_load.known_profile_keys`), decided by
    where the file sits — the repo's ``profile.yml`` is a root profile.

    The write itself is happy to put any key into the file — it is a YAML edit,
    not a schema-aware one — but the profile schema is CLOSED, so an unknown
    top-level key makes the very next ``osprey build`` refuse the whole profile.
    Without this the operator learns that at the build, from a message about
    ``profile.yml`` rather than about the command that just edited it.

    A warning rather than a refusal: the value did reach the file, saying so is
    the honest report, and an operator writing a key ahead of the release that
    reads it should not be blocked by this command. The build still refuses.

    ``config.``-prefixed keys are excluded — they address the rendered config,
    whose key space is open except the closed blocks
    (:func:`~osprey.cli.build_profile_load.reject_unknown_closed_config_keys`,
    which the build applies), and the profile's ``config:`` block holds them as
    the dotted paths they are.
    """
    legal = _legal_keys(profile_path)
    # Deduplicated through a dict, not a `set()`: the command in this module is
    # NAMED `set`, so at module scope that name is a click Command and calling
    # it re-enters the CLI instead of building a set.
    unknown: dict[str, None] = {}
    for pair in pairs:
        key, separator, _ = pair.partition("=")
        if not separator:
            continue
        head = key.strip().split(".", 1)[0]
        if head and head != "config" and head not in legal:
            unknown[head] = None
    return sorted(unknown)


def _legal_keys(profile_path: Path) -> frozenset[str]:
    """The top-level keys legal in the profile file at *profile_path*."""
    from .build_profile_load import known_profile_keys
    from .profile_root import resolve_profile_root

    return known_profile_keys(resolve_profile_root(profile_path)[1])


#: Claude Code's own alias names. They name no model, so ``model=`` refuses them.
_CLAUDE_CODE_ALIAS_WORDS = ("haiku", "sonnet", "opus")


#: The key a pin on one agent's model is written under, as ``--set`` spells it.
AGENT_MODELS_KEY = "config.claude_code.agent_models"


def _read_profile(repo_root: Path) -> dict:
    """The parsed ``profile.yml``, or ``{}`` when it cannot be read as a mapping."""
    import yaml

    try:
        profile = yaml.safe_load((repo_root / PROFILE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return profile if isinstance(profile, dict) else {}


def _served_models(repo_root: Path, provider: str) -> list[str]:
    """The model ids the catalog entry for *provider* lists; empty when it lists none."""
    from osprey.profiles.providers import load_provider_catalog
    from osprey_connectors.errors import BuildProfileError

    try:
        entry = load_provider_catalog(repo_root).entries.get(provider) or {}
    except BuildProfileError:
        entry = {}
    return [str(m) for m in entry.get("models") or []]


def _refuse_alias_word(key: str, model: str, provider: str, served: list[str]) -> None:
    """Refuse *model* when it is one of Claude Code's alias names rather than a model id."""
    if model in _CLAUDE_CODE_ALIAS_WORDS:
        raise click.UsageError(
            f"`{key}={model}` is not a model id. Provider '{provider}' serves: "
            f"{', '.join(served) or 'no listed models'}."
        )


def _configured_agents(profile: dict) -> list[str]:
    """The agent names the profile's ``config:`` block defines under ``claude_code.agents``."""
    config = profile.get("config")
    if not isinstance(config, dict):
        return []
    names: dict[str, None] = {}
    prefix = "claude_code.agents."
    for key, value in config.items():
        key = str(key)
        if key.startswith(prefix):
            names[key[len(prefix) :].split(".", 1)[0]] = None
        elif key == "claude_code.agents" and isinstance(value, dict):
            names.update(dict.fromkeys(str(name) for name in value))
        elif key == "claude_code" and isinstance(value, dict):
            agents = value.get("agents")
            if isinstance(agents, dict):
                names.update(dict.fromkeys(str(name) for name in agents))
    return sorted(names)


def _model_check(repo_root: Path, pairs: tuple[str, ...]) -> list[tuple[str, str, list[str]]]:
    """Each model id *pairs* writes, the provider it runs on, and what that provider serves.

    Covers ``model=`` and every ``config.claude_code.agent_models`` pin, written
    one agent at a time or as a whole mapping. The provider is a ``provider=``
    pair on the same command line, else the profile's own ``provider:``. Before
    anything is written, a bare alias word is refused with the ids the provider
    serves, and so is every pin ``osprey build`` refuses.

    Returns:
        One ``(model, provider, served)`` per id written, ``model=`` first and
        then the pins sorted by agent; empty when *pairs* sets no model and no
        pin, or no provider can be named.
    """
    import yaml

    from osprey.registry.mcp import FRAMEWORK_AGENTS

    from .validate_claude_artifacts import agent_model_pin_errors

    values: dict[str, str] = {}
    pins: dict[str, str] = {}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = yaml.safe_load(raw) if raw.strip() else None
        if key in ("model", "provider"):
            values[key] = "" if value is None else str(value)
        elif key == AGENT_MODELS_KEY and isinstance(value, dict):
            pins.update({str(a): str(m) for a, m in value.items() if m is not None})
        elif key.startswith(AGENT_MODELS_KEY + ".") and value is not None:
            pins[key[len(AGENT_MODELS_KEY) + 1 :]] = str(value)
    model = values.get("model")
    if not model and not pins:
        return []
    profile = _read_profile(repo_root)
    provider = values.get("provider") or profile.get("provider")
    if not provider:
        return []
    provider = str(provider)
    served = _served_models(repo_root, provider)

    if model:
        _refuse_alias_word("model", model, provider, served)
    for agent in sorted(pins):
        _refuse_alias_word(f"{AGENT_MODELS_KEY}.{agent}", pins[agent], provider, served)

    errors = agent_model_pin_errors(
        repo_root / "agents", pins, {*FRAMEWORK_AGENTS, *_configured_agents(profile)}
    )
    if errors:
        raise click.UsageError("\n".join(errors))

    checked = [(model, provider, served)] if model else []
    checked.extend((pins[agent], provider, served) for agent in sorted(pins))
    return checked


@click.command(name="set")
@click.argument("pairs", nargs=-1, metavar="KEY=VALUE...")
@repo_option
def set(pairs: tuple[str, ...], repo: Path | None) -> None:
    """Write settings into the deployment profile.

    Each KEY=VALUE is written into this repo's profile.yml in place, comments
    intact. That file is the source of truth, so this is the only command that
    edits configuration for you — the rendered build/config.yml is generated
    from it and is never hand-edited or CLI-edited. Run `osprey build` to carry
    a setting through to build/, then `osprey up` to deploy it.

    KEY is a top-level profile key (provider, model, channel_finder_mode,
    connector) or a dotted path. `model` is a model id the provider serves, and so is
    each `config.claude_code.agent_models.<agent>` pin. Keys under `config.` address the rendered
    config: `config.control_system.type=epics` writes that literal dotted entry
    into the profile's config: block, replacing the value already there. A
    mapping value states the whole block at that key: `config.approval.tools={…}`
    replaces every `approval.tools.*` entry, `config.approval.tools.execute=skip`
    changes that one leaf.

    VALUE is read as YAML — true/false become booleans, bare numbers become
    numbers, everything else is text.

    One shorthand stands in for a longer key path: `connector=` writes
    config.control_system.type.

    The write ceiling only narrows here: `config.control_system.writes=never`
    is accepted from any ceiling, `=asked` only from `allowed`, and `=allowed`
    is refused — allowing writes is an edit to the profile or tier file. Any line
    that would raise the ceiling, or whose effect on it cannot be judged because
    the profile does not resolve, is refused and nothing is written.

    Examples:

    \b
      $ osprey set model=claude-sonnet-5
      $ osprey set config.claude_code.agent_models.logbook-deep-research=claude-opus-5-5
      $ osprey set connector=epics
      $ osprey set channel_finder_mode=in_context
      $ osprey set config.system.timezone=America/Los_Angeles
      $ osprey set --repo ~/my-assistant config.control_system.writes=never
    """
    from ruamel.yaml.error import YAMLError

    from osprey.deployment.staleness import check_drift
    from osprey_connectors.errors import BuildProfileError

    from .build_profile import write_back_cli_overrides

    if not pairs:
        raise click.UsageError(
            "Nothing to set. Name at least one KEY=VALUE, e.g. `osprey set model=claude-sonnet-5`.\n\n"
            "To see what the profile currently says, run `osprey config`."
        )

    # Resolved before anything is parsed: standing outside a deployment repo is
    # the operator's first problem, and a message about a malformed pair would
    # send them looking in the wrong place.
    from .build_profile_presets import bundled_layer_dir_refusal

    layer_refusal = bundled_layer_dir_refusal(Path.cwd() if repo is None else repo)
    if layer_refusal is not None:
        raise click.UsageError(layer_refusal)
    repo_root = find_repo_root(repo)
    profile_path = repo_root / PROFILE_FILENAME

    before = _refuse_unparsable(profile_path)
    _refuse_retired_shorthands(pairs)
    narrowed_to = _refuse_write_posture(pairs, repo_root)
    model_check = _model_check(repo_root, pairs)

    # The leaves judged above are the keys that name the ceiling; a line can still
    # move it without naming it, so every line is judged after the write as well.
    ceiling, unread = _read_ceiling(repo_root)
    try:
        # Every pair is merged into one layer before any of it is written, so a
        # command line with one bad pair in it changes nothing at all.
        written = write_back_cli_overrides(profile_path, set_pairs=pairs)
    except (BuildProfileError, YAMLError) as e:
        raise click.UsageError(str(e)) from e
    _judge_line(repo_root, before, ceiling, unread, narrowed_to)

    report(f"✓ Wrote {len(written)} setting(s) into {profile_path}", style=Styles.SUCCESS)
    for key in written:
        note(key)
    for model, provider, served in model_check:
        if served and model not in served:
            note(
                f"{model} is not in provider '{provider}''s served list; the build trusts the gateway."
            )

    unrecognized = _unrecognized_top_level_keys(pairs, profile_path)
    # The retired app-template key is unknown for a REASON the generic advice
    # gets wrong: prefixing it with `config.` writes a key nothing reads, and
    # what the operator actually wants is the verb that fills the profile in.
    # Its own message says so, and it is the same one the loader refuses with.
    from .build_profile_load import _RETIRED_APP_TEMPLATE_REFUSAL
    from .build_profile_presets import PRESET_DATA_BUNDLE_KEY

    if PRESET_DATA_BUNDLE_KEY in unrecognized:
        unrecognized.remove(PRESET_DATA_BUNDLE_KEY)
        warn(f"Not a profile key: {PRESET_DATA_BUNDLE_KEY}", _RETIRED_APP_TEMPLATE_REFUSAL)
    if unrecognized:
        warn(
            f"Not a profile key: {', '.join(unrecognized)}",
            "It was written, but the profile schema is closed, so `osprey build` will "
            "refuse this profile until it is corrected or removed. To address the "
            "rendered config instead, prefix the key with `config.`. Keys legal in "
            f"{profile_path.name}: {', '.join(sorted(_legal_keys(profile_path)))}.",
        )

    # The build this edit just invalidated, named while the operator is still
    # here. `check_drift` never raises, so a repo with no build/ — the common
    # case right after `init` — reports that rather than failing the write.
    report(check_drift(repo_root).status_line)
