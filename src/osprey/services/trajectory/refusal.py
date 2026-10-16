"""The one shape every trajectory refusal takes: a code, its status, a sentence.

Each layer of the service — admission, the action worker, the sessions —
refuses with its own subclass of :class:`Refused` and its own code table, so a
door answers every refusal alike: ``HTTPException(status_code=exc.status,
detail=exc.as_detail())``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

__all__ = ["Refused"]


class Refused(Exception):
    """A refusal a door answers with.

    Attributes:
        code: The refusal's code, a key of the subclass's :attr:`statuses`.
        status: The HTTP status the door answers with.
        detail: The sentence an operator reads.
    """

    #: The HTTP status each code of this subclass answers with.
    statuses: ClassVar[Mapping[str, int]] = {}

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.status = self.statuses[code]
        self.detail = detail

    def as_detail(self) -> dict[str, str]:
        """The ``HTTPException`` detail the service's refusals carry."""
        return {"code": self.code, "detail": self.detail}
