"""The ``osprey config`` verb — read a deployment's configuration.

Three views of the same subject, one flag apart, because "the configuration" is
three different files depending on what is being asked. The default view is the
SOURCE: ``profile.yml``, what the facility wrote and what git tracks — printed
verbatim, comments and all, since the comments are half of why an operator
opens it. ``--rendered`` is the OUTPUT view, ``build/config.yml`` as the last
build produced it, which is what a running container actually reads and the
place to look when the deployment behaves unlike the source suggests.
``--defaults`` is neither a source nor an output but a LEDGER: every key the
framework reads, with what happens when a profile leaves that key out. It is
rendered from the config-key manifest's ``default:`` column, each entry of
which was read out of the reader that key's evidence cites — so the answer is
the reader's answer rather than a template's example value. It is the one view
that needs no deployment repo, and the only place that answers "what if I
delete this line?", which a preset cannot: a preset shows a key's VALUE.

The verb only reads. Writes go through ``osprey set``, which edits the source
and leaves the render to ``osprey build`` — a config the CLI mutates in place
is a config the next build silently discards.
"""

import importlib.resources
import textwrap
from pathlib import Path

import click
import yaml
from rich.syntax import Syntax

from osprey.cli import output, styles
from osprey.cli.styles import Styles

from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option

#: Filename of the config-key manifest, whose ``default:`` column this module
#: renders. It ships inside ``osprey.profiles`` rather than beside the
#: repository's other tooling because this view is the one that must work from
#: a plain ``pip install`` with no checkout in sight — the wheel packages
#: ``src/osprey`` and nothing else, so a manifest outside the package is a
#: manifest ``--defaults`` cannot open. Same arrangement, and the same reason,
#: as the packaged ``providers.yml`` beside it.
MANIFEST_FILENAME = "config_key_manifest.yml"


def packaged_manifest_path() -> Path:
    """Path of the config-key manifest shipped inside ``osprey.profiles``."""
    return Path(str(importlib.resources.files("osprey.profiles"))) / MANIFEST_FILENAME


#: Markers standing in for the three reserved words that name an absent
#: literal. Looked up rather than reformatted so an unrecognised word raises
#: instead of being printed as though it were a value the reader falls back to.
#: ``n/a`` is the fourth reserved word and is not here: it prints the key that
#: covers the leaf, which is per-entry rather than fixed.
_ABSENT_LITERAL_MARKERS = {
    "required": "<required>",
    "derived": "<derived>",
    "no-fallback": "<no fallback>",
}

#: The reserved word for a leaf with no reader of its own.
_COVERED = "n/a"

#: Width the ledger's comment prose wraps to. Two under the 80th column, so a
#: wrapped line plus its ``# `` prefix still fits an unresized terminal.
_PROSE_WIDTH = 78

_LEDGER_PREAMBLE = """\
# OSPREY configuration keys and their defaults.
#
# Every key the framework reads, with the value it falls back to when
# profile.yml does not spell that key. Each answer was read out of the code
# that reads the key, so this is what actually happens when a line is absent
# — the question a preset cannot answer, because a preset shows a value.
#
# A LEDGER, NOT A TEMPLATE. Keys are printed in full dotted form, grouped by
# the top-level block they belong to. A key with a literal default prints it
# as YAML. Where there is no literal, a marker stands in:
#
#   <required>          No fallback exists and none may be invented. A
#                       deployment's profile.yml has to spell this key.
#   <derived>           Computed when the key is read. The comment above the
#                       key names what it is computed from.
#   <no fallback>       The reader supplies none: it raises, or nothing reads
#                       the key yet, or "unstated" is deliberately not
#                       "false". The comment above the key says which.
#   <covered by KEY>    The leaf has no reader of its own, so KEY carries the
#                       default that covers it. Where no enclosing key is in
#                       the ledger this reads <no separate default> instead.
"""


def _load_key_manifest() -> dict[str, dict]:
    """The manifest's ``keys`` block: dotted path → entry.

    Raises:
        click.ClickException: If the manifest is missing or carries no keys.
            Both mean a broken installation rather than anything wrong with
            the operator's deployment, and the message says so.
    """
    path = packaged_manifest_path()
    if not path.is_file():
        raise click.ClickException(
            f"Could not locate the configuration key manifest at {path}. "
            "This points at a broken installation, not at anything in your deployment."
        )
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise click.ClickException(
            f"The configuration key manifest at {path} is not readable YAML: {exc}. "
            "This points at a broken installation, not at anything in your deployment."
        ) from exc
    keys = document.get("keys")
    if not keys:
        raise click.ClickException(
            f"The configuration key manifest at {path} declares no keys. "
            "This points at a broken installation, not at anything in your deployment."
        )
    return keys


def _literal(value: object) -> str:
    """*value* as the YAML scalar or flow collection it round-trips through.

    The manifest's literals are YAML values, so they go back out as YAML:
    ``null`` stays ``null``, an empty string stays quoted, and ``{}`` stays a
    mapping rather than becoming the word "None" or an empty line. Emitted in
    flow style on one line, and with wrapping switched off, because a default
    that wrapped would read as two keys.
    """
    dumped = yaml.safe_dump(
        value, default_flow_style=True, allow_unicode=True, width=10**6, sort_keys=True
    ).splitlines()
    # safe_dump closes a bare scalar document with `...` on its own line.
    return "\n".join(line for line in dumped if line != "...").strip()


def _covering_key(key: str, entry: dict, manifest: dict[str, dict]) -> str:
    """The key whose default covers *key*, as a marker.

    The manifest states it outright on the leaves that have a single reader
    higher up (``covered-by``). The rest are block keys whose own leaves carry
    the answers, and there the enclosing block is the honest citation — but
    only if the manifest actually declares it, so the view never invents a key
    that is not in the ledger it is printing.
    """
    covering = entry.get("covered-by")
    if not covering:
        parent = key.rpartition(".")[0]
        covering = parent if parent in manifest else None
    return f"<covered by {covering}>" if covering else "<no separate default>"


def _default_marker(key: str, entry: dict, manifest: dict[str, dict]) -> str:
    """What *key*'s ``default:`` column prints as, right of the colon.

    Raises:
        click.ClickException: If the entry states no ``default:`` at all. That
            is a hole in the manifest rather than an answer, and printing it
            would be worse than refusing: ``entry.get("default")`` is ``None``
            for a missing column and for ``default: null`` alike, so the hole
            would render as ``null`` — an operator reading "this key falls back
            to null" where the truth is that nobody wrote the column down.
            ``scripts/check_config_keys.py`` fails the same shape in CI, so a
            manifest reaching a reader with one is a broken installation.
    """
    if "default" not in entry:
        raise click.ClickException(
            f"The configuration key manifest entry for {key} states no default. "
            "This points at a broken installation, not at anything in your deployment."
        )
    default = entry.get("default")
    if isinstance(default, str):
        if default in _ABSENT_LITERAL_MARKERS:
            return _ABSENT_LITERAL_MARKERS[default]
        if default == _COVERED:
            return _covering_key(key, entry, manifest)
    return _literal(default)


def _render_defaults_ledger() -> str:
    """The manifest's ``default:`` column, grouped and annotated.

    Rendered as text rather than as data because half of the answer is prose:
    a ``derived`` key without the note saying what it derives from has told
    the reader nothing. Notes ride above their key as comments rather than
    beside it — the longest dotted key here is 84 characters, so a trailing
    comment would put the prose off the right edge of any terminal.
    """
    manifest = _load_key_manifest()

    sections: dict[str, list[str]] = {}
    for key in manifest:
        sections.setdefault(key.partition(".")[0], []).append(key)

    lines = [
        _LEDGER_PREAMBLE.rstrip("\n"),
        "#",
        f"# {len(manifest)} keys in {len(sections)} sections.",
    ]
    for section, keys in sections.items():
        lines.append("")
        banner = f"# ── {section} "
        lines.append(banner + "─" * max(0, _PROSE_WIDTH - len(banner)))
        for key in keys:
            entry = manifest[key] or {}
            note = entry.get("default_note")
            if note:
                lines.extend(f"# {line}" for line in textwrap.wrap(str(note), _PROSE_WIDTH - 2))
            lines.append(f"{key}: {_default_marker(key, entry, manifest)}")
    return "\n".join(lines) + "\n"


# UNGUARDED: copy passed to this wrapper is not seen by the house-style guard in
# `tests/cli/test_printed_copy_style.py`, which matches printer names exactly. The
# name cannot join its set, because `cli/deploy_scaffold.py` has an `_emit` that
# writes a FILE, and registering the bare name would judge that one's arguments as
# prose. What reaches a person from here is a label and a config file's own bytes
# rather than sentences, so the gap costs little; a sentence added here is review's
# to catch.
def _emit(text: str, *, label: str, source: Path | None) -> None:
    """Print configuration *text*, highlighted for a human, raw for a pipe.

    ``osprey config > deployment.yml`` has to produce the file it showed, so
    when stdout is not a terminal the content goes out unchanged and unadorned:
    no header, no rewrapping, no highlight escapes. The header and the syntax
    colouring are for the interactive reader, who needs to know which of the
    three views they are looking at and where it lives on disk.

    The off-terminal branch is a machine seam, the same class as a ``--json``
    payload: it writes a file's bytes to stdout rather than copy at a person,
    so it goes out through :func:`click.echo` rather than through the renderer.
    Rich expands tabs and can pad a line, and a config the CLI showed must be
    the config it writes.
    """
    if not styles.console.is_terminal:
        click.echo(text)
        return

    output.report("")
    output.report(label, style=Styles.BOLD)
    if source is not None:
        output.note(str(source))
    output.report("")
    output.table(Syntax(text, "yaml", theme="monokai", line_numbers=False, word_wrap=True))


@click.command(name="config")
@click.option(
    "--rendered",
    is_flag=True,
    help="Show the built config.yml the deployment actually runs on.",
)
@click.option(
    "--defaults",
    is_flag=True,
    help="Show every key the framework reads and its default. Needs no deployment repo.",
)
@repo_option
def config(rendered: bool, defaults: bool, repo: Path | None):
    """Show the deployment configuration.

    With no flag, prints the source profile.yml — the tracked, hand-edited
    manifest — exactly as it is on disk, comments included. Output is piped
    through unchanged when stdout is not a terminal.

    Examples:

    \b
      # The source manifest for the repo you are standing in
      osprey config
    \b
      # What the last build produced, which is what containers read
      osprey config --rendered
    \b
      # Every key the framework reads, with what it falls back to
      osprey config --defaults | less
    """
    if rendered and defaults:
        raise click.UsageError(
            "--rendered and --defaults show different things. Pass at most one: "
            "--rendered is this deployment's build output, --defaults is the framework's "
            "key ledger."
        )

    if defaults:
        # Deliberately before any repo discovery: the key ledger is a property
        # of the installation, so this view works from anywhere.
        _emit(_render_defaults_ledger(), label="Framework default configuration", source=None)
        return

    repo_root = find_repo_root(repo)

    if rendered:
        from .profile_conventions import BUILD_OUTPUT_DIR

        rendered_path = repo_root / BUILD_OUTPUT_DIR / "config.yml"
        if not rendered_path.is_file():
            raise click.ClickException(
                f"No rendered configuration at {rendered_path}.\n\n"
                "The build output is disposable and this repo has not been built yet "
                "(or was reset). Run 'osprey build' to render it."
            )
        _emit(
            rendered_path.read_text(encoding="utf-8"),
            label="Rendered configuration (as built)",
            source=rendered_path,
        )
        return

    profile_path = repo_root / PROFILE_FILENAME
    _emit(
        profile_path.read_text(encoding="utf-8"),
        label="Source profile",
        source=profile_path,
    )
