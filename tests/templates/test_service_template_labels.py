"""Which labels a shipped service template writes, read off the template source.

The build writes ``build/osprey-labels.override.yml`` and passes it to every
compose invocation. It gives every rendered service the project name, the
checkout identity, the project root and the config digest, so it is the only
producer of those four labels: a template that spelled one again would be a
second producer, and nothing would show that a template omitting them still
comes up labelled.

Two labels stay the template's own. The checkout label on every named volume,
because the override labels services, not volumes, and ``osprey reset`` needs it
before it removes one. And the env digest, on exactly the services that read the
env chain, because carrying it anywhere else would recreate a service on every
``.env`` edit.

Read at source level rather than off the goldens, so the branches no golden
scenario renders (more bluesky lanes, more VA instances, more dispatch workers)
are covered too.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

from osprey.deployment.compose_generator import (
    CONFIG_DIGEST_LABEL,
    PROJECT_LABEL,
    PROJECT_ROOT_LABEL,
    REPO_ID_LABEL,
)

#: The labels the generated labels override gives every rendered service.
_GENERATED_LABELS = (PROJECT_LABEL, REPO_ID_LABEL, PROJECT_ROOT_LABEL, CONFIG_DIGEST_LABEL)

#: The env-chain digest label, which stays on the services that read the chain.
_ENV_DIGEST_LABEL = "osprey.env.digest"

#: The templates whose services read the env chain.
_ENV_CHAIN_TEMPLATES = {"archive", "ariel_sync", "dispatch_worker", "event_dispatcher", "qmd"}

#: A top-level compose key on a line of its own (``services:``, ``volumes:``).
_TOP_LEVEL_KEY = re.compile(r"^([a-z_]+):\s*$")

#: A mapping key on an indented line, with its indentation.
_INDENTED_KEY = re.compile(r"^( +)([A-Za-z0-9_.{}\- ]+?):(\s|$)")


def _service_templates() -> list[Path]:
    """Every shipped service compose template."""
    root = Path(str(resources.files("osprey").joinpath("templates/services")))
    templates = sorted(root.glob("*/docker-compose.yml.j2"))
    assert len(templates) >= 16, f"found {len(templates)} service templates under {root}"
    return templates


def _keyed_lines(path: Path) -> list[tuple[int, str, int, str]]:
    """``(line number, top-level key, indent, key)`` for every indented key line.

    Comment lines and Jinja lines carry no key and are skipped; the top-level
    key is the last ``name:`` seen at column zero.
    """
    keyed = []
    section = ""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        top = _TOP_LEVEL_KEY.match(line)
        if top:
            section = top.group(1)
            continue
        if line.lstrip().startswith(("#", "{")):
            continue
        match = _INDENTED_KEY.match(line)
        if match:
            keyed.append((number, section, len(match.group(1)), match.group(2)))
    return keyed


def test_no_service_template_writes_a_generated_label() -> None:
    """The labels override is the only producer of the four project labels."""
    hits = [
        f"{path.parent.name}/{path.name}:{number}: {key}"
        for path in _service_templates()
        for number, section, _indent, key in _keyed_lines(path)
        if section == "services" and key in _GENERATED_LABELS
    ]

    assert not hits, (
        f"these service template lines write a label the labels override generates: {hits}. "
        "Remove them; the build gives every rendered service these labels."
    )


def test_every_named_volume_in_a_service_template_keeps_the_checkout_label() -> None:
    """Every labelled named volume carries the checkout label ``osprey reset`` reads."""
    missing = []
    labelled = 0
    for path in _service_templates():
        volume = None
        in_labels = False
        carries = True
        for number, section, indent, key in _keyed_lines(path):
            if section != "volumes":
                continue
            if indent == 2:
                if in_labels and not carries:
                    missing.append(f"{path.parent.name}: {volume}")
                volume, in_labels, carries = f"{key} (line {number})", False, True
            elif indent == 4:
                if in_labels and not carries:
                    missing.append(f"{path.parent.name}: {volume}")
                in_labels = key == "labels"
                carries = not in_labels
            elif in_labels and key == REPO_ID_LABEL:
                carries = True
                labelled += 1
        if in_labels and not carries:
            missing.append(f"{path.parent.name}: {volume}")

    assert not missing, (
        f"these named volumes have a labels block without {REPO_ID_LABEL}: {missing}. "
        "The labels override does not reach volumes, so the template writes it."
    )
    assert labelled >= 15, f"only {labelled} named-volume checkout labels found"


def test_env_digest_stays_on_the_services_that_read_the_env_chain() -> None:
    """The env digest is written by exactly the templates whose services read the chain."""
    writers = {
        path.parent.name
        for path in _service_templates()
        for _number, section, _indent, key in _keyed_lines(path)
        if section == "services" and key == _ENV_DIGEST_LABEL
    }

    assert writers == _ENV_CHAIN_TEMPLATES
