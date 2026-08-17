"""Byte-for-byte baseline for every bundled service template at its defaults.

Each template under the packaged ``templates/services/`` tree is rendered with
the axes UNSET — no ``network:``, no per-service overrides, an empty
``services.<name>`` block — through the same ``_inject_project_metadata``
context ``osprey up`` builds, and compared against a committed golden under
``render_defaults/``. The suite discovers templates by glob, so a newly bundled
service arrives here without anyone remembering to add it: it fails for want of
a golden rather than shipping unpinned.

What this pins that the substring tests around it cannot: those prove a
template's blocks piecewise, one assertion per block. Byte equality proves the
render is *only* those blocks — that a template edit did not also shift a
comment, drop a blank line, or reorder a stanza somewhere no assertion was
looking. The default render is the shape essentially every deployment gets, so
an unnoticed drift here reaches every adopter at once.

The baseline deliberately carries the deltas this feature introduced — the
``osprey.env.digest`` label on the two dispatch-pair services, and the worker's
``env_file:`` comment rewritten for the env chain. Those are pinned, not
tolerated: :func:`test_goldens_pin_the_env_chain_deltas` fails if a regeneration
quietly drops them.

**Update discipline** — read this before touching anything under
``render_defaults/``:

- A failure here means a template's default render changed. That is not
  automatically a bug, but it is never a reason to edit a golden by hand.
- Regenerate the whole set from the templates, in the SAME reviewed change as
  the template edit that moved them, so a reviewer sees the template diff and
  its rendered consequence side by side — never as a separate "make the test
  pass again" commit::

      PYTHONPATH=src ./.venv/bin/python tests/templates/test_render_defaults_golden.py

- Then read the golden diff. Every changed byte has to trace back to the
  template edit you just made. A byte you cannot account for is the bug this
  suite exists to catch.

The render-time values that differ per checkout — the deployment repo's path,
its identity hash, and the running framework version — are pinned to fixed
literals in the context (see the ``_PINNED_*`` constants) rather than
substituted at comparison time, so the goldens stay literal, readable files.
What derives those values in production is covered by the generator's own
tests; what this suite covers is what the templates do with them.

Nothing here is pinned per *run*: a render is a function of its context alone,
which is what :mod:`test_render_reproducible` asserts directly.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path, PurePosixPath

import pytest
import yaml

_GOLDEN_DIR = Path(__file__).parent / "render_defaults"

#: Project name the whole baseline renders under. Short and obviously synthetic
#: — it lands in image tags, container names and mount paths throughout.
_PROJECT_NAME = "golden-project"

#: The render-time values that would otherwise differ from one machine to the
#: next. Pinned into the context after injection so the committed files are
#: literal: the deployment repo's resolved path, the hash of that path, and the
#: framework version the renderer happens to be running.
_PINNED_PROJECT_ROOT = f"/deploy/{_PROJECT_NAME}"
_PINNED_REPO_ID = "0123456789ab"
_PINNED_OSPREY_VERSION = "0.0.0"


def _templates_root() -> Path:
    """The packaged ``templates/`` directory the loader is rooted at."""
    return Path(str(resources.files("osprey").joinpath("templates")))


def _discover_templates() -> list[str]:
    """Every bundled service compose template, templates-root relative.

    A glob rather than a list, so a service template added later cannot slip in
    unpinned — it shows up in :func:`test_every_bundled_template_has_a_golden`
    as a golden nobody wrote.
    """
    root = _templates_root()
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.glob("services/**/docker-compose*.yml.j2")
    )


#: Templates-root-relative paths of every bundled service template.
TEMPLATES = _discover_templates()


def _golden_name(rel_path: str) -> str:
    """The golden filename pinning *rel_path*.

    A per-service template is named for its service directory; the one template
    at the services root (the shared network declaration) is ``services-root``.
    """
    parent = PurePosixPath(rel_path).parent.name
    return "services-root.yml" if parent == "services" else f"{parent}.yml"


def _service_keys() -> list[str]:
    """The ``services.<key>`` blocks a template render needs to be present.

    Every per-service template reads its own block with no fallback, so a
    context missing one raises ``UndefinedError`` before any default applies.
    Derived from the discovered directory names rather than listed, for the
    same reason the templates are.
    """
    parents = (PurePosixPath(rel).parent.name for rel in TEMPLATES)
    return [name for name in parents if name != "services"]


@contextmanager
def _probe_repo() -> Iterator[Path]:
    """A deployment repo in the shape the default render assumes.

    A local ``.env`` and no ``.env.shared`` — the overwhelmingly common chain,
    and the one whose ``env_file:`` output the pre-chain templates already
    carried. An empty repo would render no ``env_file:`` block at all and pin
    none of the chain comment the worker template now emits.
    """
    with tempfile.TemporaryDirectory() as directory:
        repo_root = Path(directory)
        (repo_root / ".env").write_text("OSPREY_GOLDEN_FIXTURE=1\n", encoding="utf-8")
        yield repo_root


def _pinned_context(config: dict) -> dict:
    """Inject production render metadata into *config*, then pin the volatiles.

    The injection is ``_inject_project_metadata`` — the same call
    ``render_template`` makes — so what a golden pins is the generator's own
    context, not one the test assembled. Only the three values that would
    otherwise differ from one checkout to the next are overwritten afterwards.

    Shared with the axis-shape suite (``test_render_axis_shapes.py``), which
    hands in the same raw config with a per-scenario overlay applied: one
    context builder, so a scenario golden and a default golden can only differ
    where the scenario differs.
    """
    from osprey.deployment.compose_generator import _inject_project_metadata

    config = _inject_project_metadata(config)
    config["project_root"] = _PINNED_PROJECT_ROOT
    config["osprey_version"] = _PINNED_OSPREY_VERSION
    config["osprey_labels"] = {
        **config["osprey_labels"],
        "project_root": _PINNED_PROJECT_ROOT,
        "repo_id": _PINNED_REPO_ID,
    }
    return config


def _raw_default_config(repo_root: Path) -> dict:
    """The un-injected config a deployment with every axis unset renders from.

    The per-service blocks are empty dicts: that is how a deployment which never
    heard of the network axis renders, and spelling defaults here would prove
    only that the test and the template agree on a value the test supplied.

    Split out from :func:`_default_context` so the reproducibility suite
    (``test_render_reproducible.py``) can inject it itself, unpinned — what it
    asserts is a property of the production injection, which a pinned context
    would hide.
    """
    return {
        "project_name": _PROJECT_NAME,
        "project_root": str(repo_root),
        "services": {key: {} for key in _service_keys()},
        "deployment": {},
        "system": {"timezone": "UTC"},
        "deployed_services": [],
    }


def _default_context(repo_root: Path) -> dict:
    """The production render context for a deployment with every axis unset."""
    return _pinned_context(_raw_default_config(repo_root))


def _render_templates(context: dict, rel_paths: Iterable[str]) -> dict[str, str]:
    """Render *rel_paths* against *context*, keyed by golden name.

    The render goes through an Environment rooted at the packaged templates
    directory, matching ``_packaged_compose_template`` in
    ``tests/deployment/test_compose_generator.py``: the adopting templates
    import ``services/_network_axis.j2``, and an import needs a loader.

    Shared with the axis-shape suite, which renders the subset each scenario
    pins rather than the whole bundled set.
    """
    from jinja2 import Environment, FileSystemLoader

    environment = Environment(loader=FileSystemLoader(str(_templates_root())), autoescape=False)
    return {_golden_name(rel): environment.get_template(rel).render(**context) for rel in rel_paths}


def _render_defaults() -> dict[str, str]:
    """Render every bundled template at its defaults, keyed by golden name."""
    with _probe_repo() as repo_root:
        return _render_templates(_default_context(repo_root), TEMPLATES)


@pytest.fixture(scope="module")
def rendered() -> dict[str, str]:
    """Every default render, once for the module."""
    return _render_defaults()


def test_every_bundled_template_has_a_golden() -> None:
    """The committed set and the bundled set are the same set, both ways.

    A new service template with no golden is unpinned; a golden with no
    template is a file nothing renders any more and nothing would ever fail on.
    """
    committed = {path.name for path in _GOLDEN_DIR.glob("*.yml")}
    expected = {_golden_name(rel) for rel in TEMPLATES}

    assert committed == expected, (
        "goldens and bundled service templates have drifted apart — regenerate "
        "with: PYTHONPATH=src ./.venv/bin/python "
        "tests/templates/test_render_defaults_golden.py"
    )


@pytest.mark.parametrize("rel_path", TEMPLATES, ids=_golden_name)
def test_default_render_matches_golden(rel_path: str, rendered: dict[str, str]) -> None:
    """The default render is byte-identical to its committed baseline."""
    name = _golden_name(rel_path)
    golden = _GOLDEN_DIR / name
    assert golden.is_file(), f"missing golden for {rel_path}: {name}"

    assert rendered[name] == golden.read_text(encoding="utf-8"), (
        f"{rel_path} no longer renders its committed default. Every changed "
        "byte must trace to a deliberate template edit — see this module's "
        "update discipline before regenerating."
    )


def test_goldens_are_parseable_compose_documents() -> None:
    """The baselines are real YAML, so a truncated one cannot pass as a match.

    Byte equality alone would happily hold two identically broken files to each
    other. Parsing the committed side is what makes an empty or half-written
    golden fail loudly.
    """
    for path in sorted(_GOLDEN_DIR.glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(document, dict) and document, f"{path.name} is not a compose document"


def test_goldens_pin_the_env_chain_deltas() -> None:
    """The deltas this feature introduced are IN the baseline, not merely allowed.

    The dispatch pair carries the env-chain digest label — the thing that makes
    an edit to a chain file a compose-document change, and so a container
    recreation — and the worker lists the chain it found. A regeneration that
    dropped either would otherwise sail through as "the new default".
    """
    dispatcher = (_GOLDEN_DIR / "event_dispatcher.yml").read_text(encoding="utf-8")
    worker = (_GOLDEN_DIR / "dispatch_worker.yml").read_text(encoding="utf-8")

    digest_label = 'osprey.env.digest: "${OSPREY_ENV_DIGEST:-}"'
    assert digest_label in dispatcher
    assert digest_label in worker

    assert "env_file:\n      - ./.env\n" in worker, (
        "the default chain is the local .env alone — a golden without it was "
        "rendered against the wrong repo shape"
    )
    assert "./.env.shared" not in worker, "the default shape has no shared chain file"


def _regenerate() -> None:
    """Overwrite every golden from today's templates. See the update discipline."""
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in sorted(_render_defaults().items()):
        (_GOLDEN_DIR / name).write_text(text, encoding="utf-8")
        print(f"wrote {_GOLDEN_DIR / name}")


if __name__ == "__main__":
    _regenerate()
