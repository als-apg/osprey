"""Doubles the image-module suites (caption, embedding) share."""

from __future__ import annotations


class FakeCursor:
    """Canned result rows, and the row count a write reports."""

    def __init__(self, rows: list[tuple], rowcount: int = 0) -> None:
        self._rows = rows
        self.rowcount = rowcount

    async def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None

    async def fetchall(self) -> list[tuple]:
        return list(self._rows)


class FakeGate:
    """A recording :class:`PictureGate`."""

    def __init__(self, *, deterministic: bool = True, pictures: int | None = None) -> None:
        self.allow_deterministic = deterministic
        self.pictures_left = pictures
        self.started = 0
        self.successes = 0
        self.signatures: list[str] = []

    def may_start_picture(self) -> bool:
        if self.pictures_left is not None:
            if self.pictures_left <= 0:
                return False
            self.pictures_left -= 1
        self.started += 1
        return True

    def succeeded(self) -> None:
        self.successes += 1

    def deterministic(self, signature: str) -> bool:
        self.signatures.append(signature)
        return self.allow_deterministic
