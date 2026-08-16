"""The ``osprey config`` verb — read a deployment's configuration.

Three views of the same subject, one flag apart, because "the configuration" is
three different files depending on what is being asked. The default view is the
SOURCE: ``profile.yml``, what the facility wrote and what git tracks — printed
verbatim, comments and all, since the comments are half of why an operator
opens it. ``--rendered`` is the OUTPUT view, ``build/config.yml`` as the last
build produced it, which is what a running container actually reads and the
place to look when the deployment behaves unlike the source suggests.
``--defaults`` is neither: the framework's own template, the answer to "what
keys exist at all", and the one view that needs no deployment repo.

The verb only reads. Writes go through ``osprey set``, which edits the source
and leaves the render to ``osprey build`` — a config the CLI mutates in place
is a config the next build silently discards.
"""

from pathlib import Path

import click
from jinja2 import Template
from rich.syntax import Syntax

from osprey.cli import output, styles
from osprey.cli.styles import Styles

from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option

#: Values the framework template is rendered with for the ``--defaults`` view.
#: Placeholders, deliberately: the view answers "what keys exist and what do
#: they default to", so a name in it must read as an example rather than as
#: something the operator's deployment is called.
_DEFAULTS_EXAMPLE_CONTEXT = {
    "project_name": "example_project",
    "package_name": "example_project",
    "project_root": "/path/to/example_project",
    "hostname": "localhost",
    "default_provider": "anthropic",
    "default_model": "haiku",
}


def _render_framework_defaults() -> str:
    """The framework's configuration template, rendered with example values.

    Returned as text rather than parsed data so the template's comments survive
    — they are what turns a wall of keys into documentation of the options.
    """
    template_path = Path(__file__).parent.parent / "templates" / "project" / "config.yml.j2"
    if not template_path.is_file():
        raise click.ClickException(
            f"Could not locate the framework configuration template at {template_path}. "
            "This points at a broken installation, not at anything in your deployment."
        )
    return Template(template_path.read_text(encoding="utf-8")).render(**_DEFAULTS_EXAMPLE_CONTEXT)


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
    help="Show the framework's default template. Needs no deployment repo.",
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
      # Every key the framework understands, with its default
      osprey config --defaults > defaults.yml
    """
    if rendered and defaults:
        raise click.UsageError(
            "--rendered and --defaults show different files. Pass at most one: "
            "--rendered is this deployment's build output, --defaults is the framework template."
        )

    if defaults:
        # Deliberately before any repo discovery: the framework template is a
        # property of the installation, so this view works from anywhere.
        _emit(_render_framework_defaults(), label="Framework default configuration", source=None)
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
