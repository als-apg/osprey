"""Print what the facility checks found: errors, warnings and validation lines.

The facility description prints nothing; it returns its errors, warnings and
lines. These helpers write them, one per line, to stderr unless a stream is
given.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import IO, Any

import click

from osprey_facility.errors import FacilityBuildError, FacilityBuildWarning
from osprey_facility.validate import Stage
from osprey_facility.validate import validate as validate_lines

__all__ = ["report_errors", "report_lines", "report_warnings", "validate_facility"]


def report_lines(lines: Iterable[str], file: IO[Any] | None = None) -> None:
    """Print each line, to stderr unless a stream is given.

    Args:
        lines: The lines, in print order.
        file: The stream to write to.
    """
    for line in lines:
        if file is None:
            click.echo(line, err=True)
        else:
            click.echo(line, file=file)


def report_errors(errors: Iterable[FacilityBuildError], file: IO[Any] | None = None) -> None:
    """Print each error's line, to stderr unless a stream is given.

    Args:
        errors: The errors, in print order.
        file: The stream to write to.
    """
    report_lines((error.format_message() for error in errors), file)


def report_warnings(warnings: Iterable[FacilityBuildWarning], file: IO[Any] | None = None) -> None:
    """Print each warning's line, to stderr unless a stream is given.

    Args:
        warnings: The warnings, in print order.
        file: The stream to write to.
    """
    report_lines((warning.line for warning in warnings), file)


def validate_facility(
    facility_dir: Path,
    *,
    project_name: str,
    extra_checks: Sequence[Stage],
    file: IO[Any] | None = None,
) -> int:
    """Check a facility directory and print every error of the first failing stage.

    A clean directory prints the warnings of its build instead.

    Args:
        facility_dir: The ``data/facility`` directory.
        project_name: The project's name.
        extra_checks: Checks the caller supplies
            (:func:`osprey_facility.validate.run_stages`).
        file: The stream the lines go to; stderr when omitted.

    Returns:
        The exit code: 0 when every stage is clean, else 1.
    """
    code, lines = validate_lines(facility_dir, project_name=project_name, extra_checks=extra_checks)
    report_lines(lines, file)
    return code
