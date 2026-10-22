"""The one way the trajectory acceptance stack is rendered.

Every consumer of the acceptance stack -- the CI job that builds and pushes its
images, the walk that deploys it on the acceptance host, the deployed e2e lane
-- renders it through :func:`render`, so the images CI pushes and the stack the
walk runs come from the same source zone. A second render path, however close,
would let the two drift apart while each still passed.

The render is three steps and nothing else: ``osprey init`` from the control-
assistant preset with the edits ``profile.yml`` states, the post-init edits the
deployment needs (one roster user on one persona delta, corrector-only limits),
and ``osprey build --dev``. Nothing in it reads the host: the timeouts are
fixed in the profile, every published port lies in the profile's port block,
and the image axes name the registry the acceptance images live in.

Importable with no pytest fixture -- the CI job calls :func:`render` from a
plain ``uv run python -`` script.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from tests.e2e import _orm_stack
from tests.e2e.profile_edits import set_pairs

#: The committed acceptance profile: the edits :func:`render` makes at init.
PROFILE_PATH = Path(__file__).with_name("profile.yml")

#: The bundled preset the edits apply to.
PRESET = "control-assistant"

#: The compose project the acceptance stack deploys under; every container,
#: image and volume of it is named from this.
PROJECT_NAME = "trajectory-acceptance"

#: The one roster user, and the persona delta she is on: the ``trajectory``
#: panel and nothing else.
USER = "alice"
PERSONA = "operator"

#: Every vertical corrector's setpoint; the bound keeps each one below the
#: ~11.5 A at which the cold solve loses the closed orbit.
VCM_SETPOINT = re.compile(r"SR:MAG:VCM:\d+:CURRENT:SP")
VCM_BOUND_A = 11.0

#: A model key the terminal's agent starts with. The agent is started only so
#: the panel token it carries exists; nothing in a render contacts a model.
PLACEHOLDER_MODEL_KEY_LINE = "ANTHROPIC_API_KEY=placeholder-model-key\n"

#: How long ``osprey init`` and ``osprey build`` may each take.
RENDER_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC

#: The environment variable naming the acceptance images' tag (the checkout
#: sha they were built from). It outranks the profile's ``images.tag``.
IMAGE_TAG_ENV = "OSPREY_IMAGE_TAG"
#: The environment variable holding the token that may read the acceptance
#: images from :data:`IMAGE_REGISTRY`. Its value is never written to a file.
REGISTRY_READ_TOKEN_ENV = "OSPREY_ACCEPTANCE_REGISTRY_TOKEN"
#: The registry the acceptance images are pushed to and pulled from.
IMAGE_REGISTRY = "ghcr.io/als-apg"

#: The most shots one acquisition of the walk asks for. The profile's
#: ``services.trajectory.action_timeout_s`` admits this many worst-case shots,
#: and the service's published shot maximum must cover it.
N_SHOTS_MAX = 10

#: The first hundred ports of the profile's block: the window the acceptance
#: host forwards, and the one the walk checks free before ``osprey up``.
PORT_WINDOW = range(26200, 26300)

#: The scenario set the walk applies (``osprey sim apply --no-seed``) before
#: its first shot.
ACTIVE_SET: tuple[str, ...] = ("txt-injection",)

#: The calibrated kick: one vertical corrector's setpoint raised by 0.5 A
#: (1e-4 rad/A, so 5e-5 rad) with RF on, which loses no particle in turn 1 and
#: moves the downstream BPMs' turn-1 vertical reading well above the readout
#: noise. Copied from the trajectory walk's CALIBRATION.json, key ``kick``.
KICK: dict[str, Any] = {
    "corrector": "VCM01",
    "record": "SR:MAG:VCM:01:CURRENT:SP",
    "value_a": 0.5,
    "kick_rad": 5e-05,
    "plane": "y",
    "rf": "on",
}


class RenderError(RuntimeError):
    """``osprey init`` or ``osprey build`` failed; the message carries its output."""


def profile() -> dict[str, Any]:
    """The committed acceptance profile edits, as a mapping."""
    return yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))


def init_argv(out_dir: Path) -> list[str]:
    """``osprey init`` arguments (sans the ``init`` token) rendering into *out_dir*."""
    return [
        str(Path(out_dir) / PROJECT_NAME),
        "--preset",
        PRESET,
        "--no-git",
        *set_pairs(profile()),
    ]


def render(out_dir: Path | str) -> Path:
    """Render the acceptance deployment repo under *out_dir*, ready for ``osprey up``.

    Args:
        out_dir: An existing directory; the repo is created as
            ``<out_dir>/<PROJECT_NAME>`` and must not exist yet.

    Returns:
        The rendered repo's root.

    Raises:
        RenderError: If ``osprey init`` or ``osprey build`` exits non-zero.
    """
    out_dir = Path(out_dir)
    repo = out_dir / PROJECT_NAME
    osprey_bin = _orm_stack.find_osprey_console_script()
    _run([str(osprey_bin), "init", *init_argv(out_dir)], cwd=out_dir, step="init")
    shape_repo(repo)
    author_corrector_limits(repo)
    _run(
        [str(osprey_bin), "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        cwd=out_dir,
        step="build",
    )
    _orm_stack.seed_repo_env(repo)
    env_path = repo / ".env"
    with env_path.open("a", encoding="utf-8") as handle:
        handle.write(PLACEHOLDER_MODEL_KEY_LINE)
    os.chmod(env_path, 0o600)
    return repo


def _run(cmd: list[str], *, cwd: Path, step: str) -> None:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=RENDER_TIMEOUT_SEC,
        env={**os.environ, "CLAUDECODE": ""},
    )
    if result.returncode != 0:
        raise RenderError(
            f"osprey {step} failed (rc={result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def shape_repo(repo: Path) -> None:
    """One roster user on one persona delta, after ``osprey init`` materialised the preset's.

    The delta arms writes on the stand-in and on the virtual accelerator, lists
    the ``trajectory`` panel and no other write-oriented one (the stack deploys
    no event dispatcher), and declares no ``extends`` -- a file in
    ``personas/`` inherits from the profile beside it.
    """
    profile_path = repo / "profile.yml"
    materialised = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    terminals = materialised["config"]["modules.web_terminals"]
    terminals["users"] = [{"name": USER, "index": 0, "persona": PERSONA}]
    terminals["personas"] = {PERSONA: {"build_profile": f"personas/{PERSONA}.yml"}}
    terminals["default_persona"] = PERSONA
    profile_path.write_text(yaml.safe_dump(materialised, sort_keys=False), encoding="utf-8")

    personas = repo / "personas"
    for delta in personas.glob("*.yml"):
        delta.unlink()
    (personas / f"{PERSONA}.yml").write_text(
        f"name: {PROJECT_NAME} ({PERSONA})\n"
        "deploy_services: false\n"
        "web_panels:\n"
        "  - trajectory\n"
        "config:\n"
        "  control_system.writes_enabled: false\n"
        "  control_system.connector.epics.writes_enabled: false\n"
        "  control_system.connector.virtual_accelerator.writes_enabled: true\n"
        "  control_system.connector.live_standin.writes_enabled: true\n"
        "  web.ui_mode: expert\n"
        "  modules.web_terminals.enabled: false\n",
        encoding="utf-8",
    )


def author_corrector_limits(repo: Path) -> None:
    """Replace the repo's ``limits.yaml`` records with corrector-only ones.

    Written between ``osprey init`` and ``osprey build`` so the render
    enforces them. The shipped corrector records stay; every vertical
    corrector the facility wires gains a symmetric bound; nothing names a
    command channel, so the preset's ``optional`` mode lets commands pass.
    """
    facility = repo / "data" / "facility"
    limits_file = facility / "limits.yaml"
    limits = yaml.safe_load(limits_file.read_text(encoding="utf-8"))
    vcms = sorted(
        {
            match
            for path in facility.rglob("*.yaml")
            if path != limits_file
            for match in VCM_SETPOINT.findall(path.read_text(encoding="utf-8"))
        }
    )
    if not vcms:
        raise RenderError(f"no vertical corrector setpoint found under {facility}")
    kept = [r for r in limits["records"] if ":MAG:" in r["address"] and r["address"] not in vcms]
    limits["records"] = kept + [
        {"address": vcm, "min_value": -VCM_BOUND_A, "max_value": VCM_BOUND_A} for vcm in vcms
    ]
    limits_file.write_text(yaml.safe_dump(limits, sort_keys=False), encoding="utf-8")
