"""How the CLI shows a same-name-different-checkout refusal.

One renderer, shared by ``osprey reset`` and ``osprey init --reset``, because
the two verbs meet the identical guard and an operator who has read one refusal
should recognise the other. What differs between them is the verb in the opening
line and whether there is a second way out, so both are arguments.

The split between what prints and what does not is the point of this module.
:attr:`~osprey.deployment.reset.ForeignCheckoutError.inventory` is the evidence,
one line per resource, and on a real deployment it is dozens of lines that repeat
one path between them. Printed by default it pushes the sentence that explains
the refusal below the fold, which is how a guard doing exactly its job reads as a
crash. So the conclusion, the path and the way out print always, and the evidence
prints under ``--verbose`` with a line saying it is there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osprey.cli import output
from osprey.cli.phase_reporter import is_verbose

if TYPE_CHECKING:
    from osprey.deployment.reset import ForeignCheckoutError

__all__ = ["render_foreign_refusal"]


def render_foreign_refusal(
    error: ForeignCheckoutError, actor: str, *, extra_remedy: str | None = None, mark: bool = True
) -> None:
    """Print ``error`` as a refusal, with its evidence only where it was asked for.

    :param error: The refusal, in parts.
    :param actor: The verb the operator typed, for the opening line. ``reset``
        and ``--reset`` are the two, and each names itself so neither reports
        the other's verb at somebody who did not run it.
    :param extra_remedy: A second way out this verb can offer and the other
        cannot. Printed under the shared one, because the shared one is the
        destructive option and an operator should read the alternative before
        acting on it.
    :param mark: Passed to :func:`osprey.cli.output.fail`. ``False`` where a
        phase has already printed the ✗ this refusal leaves through.
    """
    cause = error.cause
    if error.inventory:
        cause += (
            f"\n\nRun again with --verbose to list all {len(error.inventory)} of them."
            if not is_verbose()
            else "\n\nRead off the resources' own labels, none of it inferred:\n"
            + "\n".join(error.inventory)
        )
    output.fail(error.summary(actor), cause, error.remedy, mark=mark)
    if extra_remedy:
        output.note(f"or {extra_remedy}")
