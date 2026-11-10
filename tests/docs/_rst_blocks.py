"""Code blocks of a reST page, as the doc tests read them."""

from __future__ import annotations

import textwrap


def code_blocks(text: str, language: str) -> list[str]:
    """The bodies of every ``.. code-block:: <language>`` in *text*, dedented.

    Directive options (``:caption: .env``) are not part of a block's body.
    """
    lines = text.splitlines()
    blocks: list[str] = []
    for i, line in enumerate(lines):
        if line.strip() != f".. code-block:: {language}":
            continue
        indent = len(line) - len(line.lstrip())
        body: list[str] = []
        for following in lines[i + 1 :]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            body.append(following)
        dedented = textwrap.dedent("\n".join(body)).splitlines()
        blocks.append("\n".join(ln for ln in dedented if not ln.startswith(":")).strip("\n"))
    return blocks
