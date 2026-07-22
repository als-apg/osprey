"""Dev-wheel artifact helpers: local-requirements manifest and staged-artifact discovery.

Pure, side-effect-light helpers that derive and locate the artifacts a ``--dev``
deploy stages alongside a locally-built osprey wheel. The wheel build itself, its
per-process cache, and the staging orchestration live in
:mod:`osprey.deployment.compose_generator` (which imports the manifest writer
below); these functions are split out because deriving the requirements manifest
and enumerating staged artifacts are deterministic operations with no dependency
on the build cache or template-rendering flow.
"""

import os
from pathlib import Path

# Staged next to every dev wheel (see compose_generator's
# ``_copy_local_framework_for_override``): the wheel's own base dependency list,
# which the Dockerfiles' deps layer installs with the build toolchain still
# present. Without it the deps layer primes only the RELEASED PyPI pin's
# dependencies, and any base dep the local wheel adds that the release lacks
# (e.g. a native sdist like softioc) would have to compile in the later
# toolchain-less wheel-install layer — and fail.
LOCAL_REQUIREMENTS_FILENAME = "osprey-local-requirements.txt"


def _wheel_base_requirements(wheel_path: str | Path) -> list[str]:
    """Extract the wheel's base (non-extra) ``Requires-Dist`` entries, sorted.

    Parses ``*.dist-info/METADATA`` inside the wheel and returns every
    ``Requires-Dist`` requirement string EXCEPT those gated behind an extra
    (environment marker referencing ``extra``). Non-extra markers such as
    ``python_version`` are kept verbatim so pip evaluates them in-container.

    The result is fully deterministic for identical wheels — sorted, with
    whitespace normalized — because the manifest built from it is COPY'd into
    a BuildKit-content-hashed layer: byte-identical input keeps the deps-layer
    cache warm across deploys.

    :param wheel_path: Path to the built wheel file
    :type wheel_path: str or pathlib.Path
    :return: Sorted requirement strings (extras excluded)
    :rtype: list[str]
    :raises ValueError: If the wheel contains no ``*.dist-info/METADATA``
    """
    import zipfile
    from email.parser import Parser

    from packaging.requirements import Requirement

    with zipfile.ZipFile(wheel_path) as whl:
        metadata_names = sorted(
            name for name in whl.namelist() if name.endswith(".dist-info/METADATA")
        )
        if not metadata_names:
            raise ValueError(f"no *.dist-info/METADATA found in wheel {wheel_path}")
        metadata = Parser().parsestr(whl.read(metadata_names[0]).decode("utf-8"))

    requirements = []
    for entry in metadata.get_all("Requires-Dist") or []:
        req = Requirement(entry)
        if req.marker is not None and "extra" in str(req.marker):
            continue  # extra-gated dep: stays out of the base install set
        # Collapse header-folding whitespace so the written line is stable
        # and single-line; the requirement text itself stays verbatim.
        requirements.append(" ".join(entry.split()))
    return sorted(requirements)


def _write_local_requirements_manifest(cached_wheel: Path, out_dir: str) -> None:
    """Write ``osprey-local-requirements.txt`` next to the staged dev wheel.

    Content contract (shared with the service Dockerfiles, which COPY the file
    and pip-install it in their toolchain-equipped deps layer): one requirement
    per line, sorted, trailing newline — byte-identical for identical wheels.

    :param cached_wheel: The cached local wheel the manifest derives from
    :type cached_wheel: pathlib.Path
    :param out_dir: Build context directory the wheel was staged into
    :type out_dir: str
    """
    manifest_path = os.path.join(out_dir, LOCAL_REQUIREMENTS_FILENAME)
    content = "".join(f"{line}\n" for line in _wheel_base_requirements(cached_wheel))
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)


def _staged_dev_artifact_paths(context_dir: str) -> set[Path]:
    """All dev-staging artifacts (wheels + requirements manifest) in a context.

    Callers that stage into a long-lived directory (project-image and persona
    builds) snapshot this before/after staging so their ``finally`` cleanup
    removes exactly what staging added — wheel AND manifest — and nothing that
    was already there.

    :param context_dir: Build context directory
    :type context_dir: str
    :return: Paths of staged dev artifacts currently present
    :rtype: set[pathlib.Path]
    """
    root = Path(context_dir)
    artifacts = set(root.glob("*.whl"))
    manifest = root / LOCAL_REQUIREMENTS_FILENAME
    if manifest.exists():
        artifacts.add(manifest)
    return artifacts
