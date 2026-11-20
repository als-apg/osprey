"""The Dockerfile syntax every shipped recipe may use: the engine's builtin frontend.

OSPREY's images build with whatever Dockerfile frontend the engine carries, so a
recipe never names a frontend image (``# syntax=``) that a site registry mirror
cannot serve. The price is that a recipe may use only what that builtin frontend
parses at the oldest Docker Engine OSPREY supports
(``RUNTIME_FLOORS[RuntimeProduct.DOCKER_ENGINE]``). The tables below are that
allowlist, each entry with the first Docker Engine release whose builtin
frontend builds it; :func:`builtin_frontend_problems` holds a recipe to them.
It is applied to every shipped recipe's raw text
(:mod:`tests.deployment.test_service_dockerfiles`) and to the rendered project
recipe (:mod:`tests.cli.test_dockerfile_template`).
"""

from __future__ import annotations

import re

#: Instruction flags, keyed ``(instruction, flag)``, with the first Docker
#: Engine release whose builtin frontend parses them (Docker Engine release
#: notes: multi-stage builds 17.05, ``COPY --chown`` 17.09, the ``FROM``
#: platform flag 18.04, ``RUN --mount`` without a syntax directive 20.10).
FLAG_FLOORS: dict[tuple[str, str], tuple[int, ...]] = {
    ("COPY", "--chown"): (17, 9),
    ("COPY", "--from"): (17, 5),
    ("FROM", "--platform"): (18, 4),
    ("RUN", "--mount"): (20, 10),
}

#: ``RUN --mount`` types, with the first Docker Engine release that builds them.
MOUNT_TYPE_FLOORS: dict[str, tuple[int, ...]] = {"cache": (20, 10)}

#: ``RUN --mount`` keys, with the first Docker Engine release that parses them.
MOUNT_KEY_FLOORS: dict[str, tuple[int, ...]] = {
    "id": (20, 10),
    "sharing": (20, 10),
    "target": (20, 10),
    "type": (20, 10),
}

#: The instructions whose flags and forms the frontend version decides.
CHECKED_INSTRUCTIONS = frozenset({"ADD", "COPY", "FROM", "RUN"})

_PARSER_DIRECTIVE = re.compile(r"^#\s*(syntax|check)\s*=", flags=re.IGNORECASE | re.MULTILINE)
_HEREDOC = re.compile(r"<<-?\s*[\"']?[A-Za-z_]")


def _instructions(text: str) -> list[tuple[str, str]]:
    """``(instruction, body)`` per instruction, continuations joined.

    Comment lines and template lines (``{% ... %}``) are not instructions.
    """
    joined = re.sub(r"\\\n", " ", text)
    out = []
    for raw in joined.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, body = line.partition(" ")
        if keyword.isupper() and len(keyword) > 1:
            out.append((keyword, body.strip()))
    return out


def _mount_problems(flag: str, value: str) -> list[str]:
    options = dict(option.partition("=")[::2] for option in value.split(","))
    problems = [f"RUN {flag}: mount key {key!r}" for key in options if key not in MOUNT_KEY_FLOORS]
    if options.get("type") not in MOUNT_TYPE_FLOORS:
        problems.append(f"RUN {flag}: mount type {options.get('type')!r}")
    return problems


def builtin_frontend_problems(text: str) -> list[str]:
    """Every use in *text* of syntax the builtin frontend's allowlist lacks.

    Three things are refused: a parser directive (``# syntax=``, ``# check=``),
    an instruction flag or ``RUN --mount`` type or key the tables above do not
    hold, and a heredoc on ``FROM``/``COPY``/``ADD``/``RUN``.
    """
    problems = [f"parser directive: {m.group(0)}" for m in _PARSER_DIRECTIVE.finditer(text)]
    for keyword, body in _instructions(text):
        if keyword not in CHECKED_INSTRUCTIONS:
            continue
        tokens = body.split()
        while tokens and tokens[0].startswith("--"):
            flag = tokens.pop(0)
            name, _, value = flag.partition("=")
            if (keyword, name) not in FLAG_FLOORS:
                problems.append(f"{keyword} {name}: flag")
            elif name == "--mount":
                problems += _mount_problems(flag, value)
        if _HEREDOC.search(" ".join(tokens)):
            problems.append(f"{keyword}: heredoc in {body[:60]!r}")
    return problems
