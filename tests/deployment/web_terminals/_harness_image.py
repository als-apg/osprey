"""The small image the served-sidecar tests run the auth and bluesky-web sidecars from.

The deployed sidecar images install the whole ``osprey-framework`` distribution,
a multi-minute, multi-hundred-MB build no unit lane can afford. The harness
image installs only the part of that dependency set the sidecars' import
closure needs, with versions read from the installed distribution's own
metadata, and the tests bind-mount this checkout's ``src/`` over it. Both
``test_auth_serving.py`` (and everything that drives its ``serving_stack``) and
``test_bluesky_sidecar_roster_proof.py`` run from it, under one tag.

Importing this module starts nothing and probes no container engine; only
:func:`build_harness_image` talks to docker.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import textwrap
from importlib.metadata import requires as distribution_requires
from pathlib import Path

from packaging.requirements import Requirement

from osprey.deployment.upstream_images import UPSTREAM_IMAGES

#: The nginx the rendered perimeter runs: the deployment's own default.
NGINX_IMAGE = UPSTREAM_IMAGES["nginx"]

#: The base the harness image is built on.
BASE_IMAGE = "python:3.11-slim"

#: The harness image's repository; its tag is a digest of the Dockerfile.
HARNESS_REPOSITORY = "osprey-web-sidecar-harness"

PACKAGES = (
    "fastapi",
    "uvicorn",
    "authlib",
    "itsdangerous",
    "python-multipart",
    "httpx",
    "jinja2",
    "ruamel.yaml",
    "pyyaml",
    "click",
    "rich",
)
"""The sidecars' import closure, by distribution name.

The auth sidecar's app needs ``fastapi``, ``uvicorn``, ``authlib``,
``itsdangerous`` and ``python-multipart``; its route modules add ``httpx`` (the
OIDC client) and ``jinja2`` (the login page); ``pyyaml``, ``ruamel.yaml`` and
``rich`` arrive through :mod:`osprey.utils`, which :mod:`osprey.interfaces.web_auth`
imports; ``click`` through ``httpx``'s and uvicorn's command-line entries.
``import osprey.interfaces.bluesky_web.app`` needs the same web stack and the
config/registry packages ``osprey.interfaces._app_setup`` drags in behind the
auth middleware. The *versions* are never named here -- see
:func:`declared_specs`.
"""


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_specs() -> tuple[str, ...]:
    """The requirement specifiers ``osprey-framework`` declares for :data:`PACKAGES`.

    Read from the *installed distribution's* metadata rather than written out
    here, so the harness image can only contain what a deployment's
    ``pip install osprey-framework`` would also install. That is what keeps
    ``test_login_post_parses_its_form_body_in_the_container`` honest about
    ``python-multipart``: the harness has it because the distribution declares
    it, not because this file asked for it.

    Raises:
        AssertionError: If the distribution does not declare one of them. That
            is a real defect rather than a harness problem -- the sidecars
            import the package, so the deployed images would be missing it too.
    """
    declared: dict[str, str] = {}
    for raw in distribution_requires("osprey-framework") or ():
        requirement = Requirement(raw)
        # Extras (``; extra == "dev"``) are not in a default install, so they
        # are not in the deployed image either.
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        declared[_canonical(requirement.name)] = str(requirement)

    missing = [name for name in PACKAGES if _canonical(name) not in declared]
    assert not missing, (
        f"osprey-framework does not declare {missing}, which the auth and bluesky-web "
        "sidecars import. The deployed sidecar images install the distribution's "
        "declared dependencies and would be missing them too."
    )
    return tuple(declared[_canonical(name)] for name in PACKAGES)


def harness_dockerfile() -> str:
    """The harness image's Dockerfile, its only build input."""
    specs = " ".join(f'"{spec}"' for spec in declared_specs())
    return textwrap.dedent(
        f"""\
        FROM {BASE_IMAGE}
        RUN pip install --no-cache-dir {specs}
        ENV PYTHONPATH=/src:/connectors-src PYTHONDONTWRITEBYTECODE=1
        """
    )


def harness_tag(dockerfile: str) -> str:
    """The tag of the image *dockerfile* builds.

    It carries a digest of the Dockerfile, so a dependency-declaration change
    produces a new tag instead of silently reusing a stale image, and two
    sessions share a tag only when they would have built identical content.
    """
    return f"{HARNESS_REPOSITORY}:{hashlib.sha256(dockerfile.encode()).hexdigest()[:12]}"


def build_harness_image(tmp_path: Path) -> str:
    """Build (or reuse) the harness image and return its tag.

    Reuses a tag already on the host -- an earlier session's -- and never
    removes it: every sidecar test of a session runs from it.
    """
    dockerfile = harness_dockerfile()
    tag = harness_tag(dockerfile)
    present = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    if present.returncode == 0:
        return tag

    context = tmp_path / "harness"
    context.mkdir()
    (context / "Dockerfile").write_text(dockerfile)
    result = subprocess.run(
        ["docker", "build", "-t", tag, str(context)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, f"harness image build failed:\n{result.stdout}\n{result.stderr}"
    return tag
