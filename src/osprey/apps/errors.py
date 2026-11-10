"""Errors raised while reading or checking a deployment's applications."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Violation:
    """One broken rule: where it is, which rule, and what to do instead.

    Attributes:
        app: The application (its directory name) the violation belongs to;
            empty for one that belongs to a login.
        field: The dotted path of the refused key; empty when the violation
            concerns the file as a whole.
        rule: A short name for the rule, e.g. ``extra_forbidden``.
        message: What is wrong and where the value belongs instead.
        source: The file the field lives in when it is not the application's
            ``app.yml`` (a ``pyproject.toml``); ``None`` means ``app.yml``, or
            no file for a login's violation.
    """

    app: str
    field: str
    rule: str
    message: str
    source: str | None = None

    @property
    def where(self) -> str:
        """The file and field the violation points at."""
        if self.source is not None:
            base = self.source
        elif self.app:
            base = f"apps/{self.app}/app.yml"
        else:
            base = ""
        if base and self.field:
            return f"{base}: {self.field}"
        return base or self.field

    def __str__(self) -> str:
        return f"{self.where}: {self.message}" if self.where else self.message


class AppManifestError(ValueError):
    """An application, or the set of them, was refused.

    Carries every violation found; :attr:`app`, :attr:`field`, :attr:`rule` and
    :attr:`message` are the first one's.

    Attributes:
        violations: Every violation, in the order found.
        app: The first violation's application.
        field: The first violation's dotted field path.
        rule: The first violation's rule name.
        message: The first violation's message.
    """

    def __init__(
        self,
        app: str,
        field: str,
        rule: str,
        message: str,
        *,
        violations: Sequence[Violation] = (),
    ) -> None:
        self.violations: tuple[Violation, ...] = tuple(violations) or (
            Violation(app, field, rule, message),
        )
        first = self.violations[0]
        self.app = first.app
        self.field = first.field
        self.rule = first.rule
        self.message = first.message
        if len(self.violations) == 1:
            super().__init__(str(first))
        else:
            lines = "\n".join(f"  - {v}" for v in self.violations)
            super().__init__(f"{len(self.violations)} violations:\n{lines}")

    @classmethod
    def from_violations(cls, violations: Sequence[Violation]) -> AppManifestError:
        """Build one refusal carrying every violation in *violations* (at least one)."""
        if not violations:
            raise ValueError("a refusal needs at least one violation")
        first = violations[0]
        return cls(first.app, first.field, first.rule, first.message, violations=violations)
