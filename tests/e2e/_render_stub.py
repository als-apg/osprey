"""Swap a rendered tier's image recipe for a stub, keeping everything else it rendered.

Some lanes put something other than the real terminal behind nginx — a page that
names the container that served it, or one that echoes the request's headers —
because their subject is the perimeter in front of the terminal, and a real
terminal image is a framework install per tier. The tier itself stays real: its
file is in the repo's ``tiers/``, ``osprey build`` renders it, and a start reads
that render's ``config.yml`` exactly as it would for any deployment. Only the
recipe the start builds the tier's image from is replaced, in both copies a
build writes: the host's flat render at ``build/<project>/`` and the container
repo at ``build/.image/<project>/`` whose ``build/Dockerfile`` the image is
built from, with the build context at that container repo.

A start renders nothing, so what the build wrote is what it runs; the swap is
made after ``osprey build`` and before ``osprey up``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from osprey.utils.workspace import IMAGE_DIR_NAME
from osprey_connectors.workspace import BUILD_DIR_NAME


def stub_render_image(
    repo: Path, project: str, dockerfile: str, files: Mapping[str, str] | None = None
) -> None:
    """Replace the image recipe of the rendered tier *project* with *dockerfile*.

    Args:
        repo: The deployment repo, already built.
        project: The tier's render project (``<project_name>-<tier>``).
        dockerfile: The stub recipe. It may ``COPY build/<name>`` any of *files*.
        files: Extra files written beside the container copy's ``Dockerfile``,
            by name.
    """
    flat = repo / BUILD_DIR_NAME / project
    image_build = repo / BUILD_DIR_NAME / IMAGE_DIR_NAME / project / BUILD_DIR_NAME
    for recipe in (flat / "Dockerfile", image_build / "Dockerfile"):
        assert recipe.is_file(), f"osprey build did not render {recipe}"
        recipe.write_text(dockerfile, encoding="utf-8")
    for name, text in (files or {}).items():
        (image_build / name).write_text(text, encoding="utf-8")
