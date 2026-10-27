"""Errors raised while reading an application's manifest."""

from __future__ import annotations


class AppManifestError(ValueError):
    """An application's ``app.yml`` was refused.

    Attributes:
        app: The application (its directory name) whose manifest was refused.
        field: The dotted path of the refused key inside ``app.yml``; empty when
            the refusal concerns the manifest file as a whole.
        rule: A short name for the rule that refused it, e.g. ``extra_forbidden``
            or ``compose_key``.
        message: What is wrong and where the value belongs instead.
    """

    def __init__(self, app: str, field: str, rule: str, message: str) -> None:
        self.app = app
        self.field = field
        self.rule = rule
        self.message = message
        where = f"apps/{app}/app.yml" + (f": {field}" if field else "")
        super().__init__(f"{where}: {message}")
