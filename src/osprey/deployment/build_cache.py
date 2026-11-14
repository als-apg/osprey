"""The layer cache OSPREY's image builds read and write, named by the environment.

A host that builds the same images again and again on fresh builders, such as a
CI runner, can give every build a BuildKit layer cache through two variables:

``OSPREY_BUILD_CACHE_FROM``
    A ``--cache-from`` spec, e.g. ``type=registry,ref=ghcr.io/org/cache:{key}``.
``OSPREY_BUILD_CACHE_TO``
    A ``--cache-to`` spec of the same shape, e.g. ``...:{key},mode=max``.

``{key}`` is replaced by the cache key of the image being built, which names
its recipe and not the deployment, so two deployments share one cache:

* ``project`` for every image built from the project Dockerfile: the project
  image and each web-terminal render image. They have the same layers up to
  the project's own files, the expensive dependency install among them.
* ``auth-sidecar`` for the web terminals' login sidecar.
* the build context's directory name for a compose service: ``bluesky``,
  ``event_dispatcher``, ``qmd``, and so on.

The variables reach every build OSPREY runs: the direct ``docker build`` of the
project, render and sidecar images take them as flags, and every compose
invocation takes them as one more ``-f`` file
(:func:`compose_cache_override_file`). An unset or empty variable adds nothing,
so a host that sets neither builds exactly as before.

Docker only. Podman's ``--cache-from`` names a repository, not a spec, and
podman-compose reads neither key, so under podman nothing is added. A source
that does not exist yet is a warning to BuildKit, not an error, so a cache that
has never been written costs a build nothing. A sink needs a builder that can
export a cache: Docker's containerd image store or a ``docker-container``
builder. The classic overlay2 store refuses one.

A source that exists but cannot be read is another matter. When the registry
fails to serve a layer the source lists, BuildKit fails the whole build, even
though the cache only ever saves time. :func:`run_image_build` runs every image
build OSPREY runs, and rebuilds once without the source when that is how a
build failed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import yaml

from osprey.deployment.errors import CapturedProcessError

logger = logging.getLogger(__name__)

CACHE_FROM_ENV = "OSPREY_BUILD_CACHE_FROM"
CACHE_TO_ENV = "OSPREY_BUILD_CACHE_TO"

#: The placeholder each spec's image cache key replaces.
KEY_PLACEHOLDER = "{key}"

#: The key of every image built from ``templates/project/Dockerfile.j2``.
PROJECT_KEY = "project"
#: The key of the web terminals' login sidecar image.
AUTH_SIDECAR_KEY = "auth-sidecar"

_OVERRIDE_HEADER = """\
# osprey-build-cache.*.override.yml — written for one compose invocation. Do not edit.
#
# The layer cache named by OSPREY_BUILD_CACHE_FROM / OSPREY_BUILD_CACHE_TO, for
# every service of the invocation that compose builds.
"""


def _specs(key: str, environ: Mapping[str, str]) -> tuple[str | None, str | None]:
    """The (from, to) specs for *key*, each ``None`` when its variable is unset or empty."""
    source = environ.get(CACHE_FROM_ENV, "").strip()
    sink = environ.get(CACHE_TO_ENV, "").strip()
    return (
        source.replace(KEY_PLACEHOLDER, key) or None,
        sink.replace(KEY_PLACEHOLDER, key) or None,
    )


def _is_docker(runtime: str) -> bool:
    return Path(runtime).name == "docker"


def build_cache_args(runtime: str, key: str, environ: Mapping[str, str] | None = None) -> list[str]:
    """The ``--cache-from`` / ``--cache-to`` flags a direct ``<runtime> build`` takes.

    :param runtime: The container command the build runs under (``docker``, ``podman``).
    :param key: The image's cache key (:data:`PROJECT_KEY`, :data:`AUTH_SIDECAR_KEY`).
    :param environ: The environment to read; ``None`` reads the process's.
    :return: The flags, or ``[]`` when neither variable is set or the runtime is not Docker.
    """
    if not _is_docker(runtime):
        return []
    source, sink = _specs(key, os.environ if environ is None else environ)
    args: list[str] = []
    if source:
        args.extend(("--cache-from", source))
    if sink:
        args.extend(("--cache-to", sink))
    return args


def service_cache_key(build: object) -> str | None:
    """The cache key of a compose service from its ``build:`` value.

    The context directory's own name: ``./build/services/bluesky`` is
    ``bluesky`` in every deployment, whatever the project is called. A URL
    context names no directory, and ``.`` is the deployment's own directory,
    named after the project; neither gets a key.
    """
    context = build if isinstance(build, str) else None
    if isinstance(build, Mapping):
        context = str(build.get("context") or ".")
    if context is None or "://" in context:
        return None
    name = Path(os.path.normpath(context)).name
    return name or None


def compose_cache_override(
    root: Path, compose_files: Sequence[str | Path], environ: Mapping[str, str] | None = None
) -> dict | None:
    """The cache entries for every service in *compose_files* that compose builds.

    An entry only for a service with a ``build:``: compose would read
    a ``build:`` in an override as making an image-only service buildable.

    :param root: The deployment repo root, absolute; relative files resolve against it.
    :param compose_files: The invocation's compose files.
    :param environ: The environment to read; ``None`` reads the process's.
    :return: An override document, or ``None`` when there is nothing to add.
    """
    from osprey.deployment.compose_merge import ComposeMergeError, _load_compose_document

    environ = os.environ if environ is None else environ
    if not any(environ.get(name, "").strip() for name in (CACHE_FROM_ENV, CACHE_TO_ENV)):
        return None
    services: dict[str, dict] = {}
    for compose_file in compose_files:
        path = Path(compose_file)
        try:
            document = _load_compose_document(path if path.is_absolute() else root / path)
        except ComposeMergeError:
            # Compose reports the file on its own terms.
            continue
        entries = document.get("services")
        if not isinstance(entries, Mapping):
            continue
        for name, service in entries.items():
            if not isinstance(service, Mapping) or service.get("build") is None:
                continue
            key = service_cache_key(service["build"])
            if key is None:
                continue
            source, sink = _specs(key, environ)
            build: dict[str, list[str]] = {}
            if source:
                build["cache_from"] = [source]
            if sink:
                build["cache_to"] = [sink]
            services[str(name)] = {"build": build}
    return {"services": services} if services else None


def compose_cache_override_file(
    runtime: str, root: Path, compose_files: Sequence[str | Path]
) -> Path | None:
    """Write the invocation's cache override, and return its path for one more ``-f``.

    Content-addressed in the render zone, so two concurrent invocations over
    different service sets never write the same path.

    :param runtime: The container command compose runs under.
    :param root: The deployment repo root, absolute.
    :param compose_files: The invocation's compose files.
    :return: The file, or ``None`` when the runtime is not Docker or there is nothing to add.
    """
    if not _is_docker(runtime):
        return None
    document = compose_cache_override(root, compose_files)
    if document is None:
        return None
    from osprey_connectors.workspace import BUILD_DIR_NAME

    return _write_override(root / BUILD_DIR_NAME, document)


#: The file name every cache override is written under (:func:`_write_override`).
_OVERRIDE_NAME = re.compile(r"osprey-build-cache\.[0-9a-f]+\.override\.yml")


def _write_override(directory: Path, document: dict) -> Path:
    """Write *document* as a cache override in *directory*, named by its content."""
    from osprey.deployment.compose_merge import _FILE_MODE
    from osprey_connectors.safety_file import atomic_replace

    text = _OVERRIDE_HEADER + "\n" + yaml.safe_dump(document, sort_keys=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    target = directory / f"osprey-build-cache.{digest}.override.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace(target, text.encode("utf-8"), mode=_FILE_MODE, sync="best_effort")
    return target


# --- a source that cannot be read -------------------------------------------------

#: How BuildKit reports a layer it could not fetch from a registry, e.g.
#: ``could not fetch content descriptor sha256:243d… (application/vnd.oci.image
#: .layer.v1.tar+gzip) from remote: not found``. A local file a ``COPY`` names
#: and the context lacks fails with ``failed to compute cache key`` too, but never
#: with this, so a broken recipe is not rebuilt for nothing. The message does not
#: name the registry: a base-image layer the registry fails to serve matches as
#: well, and costs one more attempt.
_SOURCE_READ_FAILURE = re.compile(
    rb"could not fetch content descriptor sha256:[0-9a-f]+ .*from remote"
)


def without_cache_source(cmd: Sequence[str]) -> list[str] | None:
    """*cmd* with its cache source taken out, or ``None`` when it reads none.

    A direct build loses its ``--cache-from`` pair. A compose invocation's cache
    override is written again without its ``cache_from`` entries, or dropped
    when they were all it held. A sink stays: the build it writes is the one
    the next build reads.
    """
    out: list[str] = []
    dropped = False
    args = iter(cmd)
    for arg in args:
        value = next(args, None) if arg in ("--cache-from", "-f") else None
        if arg == "--cache-from":
            dropped = True
            continue
        if arg == "-f" and value is not None and _OVERRIDE_NAME.fullmatch(Path(value).name):
            override = Path(value)
            document = yaml.safe_load(override.read_text(encoding="utf-8")) or {}
            services = document.get("services") or {}
            for service in services.values():
                if service.get("build", {}).pop("cache_from", None) is not None:
                    dropped = True
            kept = {name: s for name, s in services.items() if s.get("build")}
            if kept:
                out.extend((arg, str(_write_override(override.parent, {"services": kept}))))
            continue
        out.append(arg)
        if value is not None:
            out.append(value)
    return out if dropped else None


def _source_read_failed(error: CapturedProcessError) -> bool:
    """Whether *error*'s captured output shows a registry failing to serve a layer."""
    if error.spool_path is None:
        # Verbose mode: the output went to the terminal, not to a file.
        return False
    try:
        output = error.spool_path.read_bytes()
    except OSError:
        return False
    return _SOURCE_READ_FAILURE.search(output) is not None


def run_image_build(cmd: Sequence[str], run: Callable[[list[str]], object]) -> None:
    """Run the image build *cmd* through *run*, once more without its source if that failed it.

    *run* runs one argv, raising :class:`CapturedProcessError` when it fails
    (:func:`osprey.deployment.subprocess_capture.run_captured`). A build that
    failed for any other reason, or that read no source, fails as it did. The
    rebuild builds the layers the source would have given, so it is slower,
    and its own failure is the one raised.
    """
    try:
        run(list(cmd))
    except CapturedProcessError as error:
        # The output first: rewriting a compose override is a file written for a
        # retry that a broken recipe never earns.
        fallback = without_cache_source(cmd) if _source_read_failed(error) else None
        if fallback is None:
            raise
        logger.warning(
            "The build cache could not be read (the registry did not serve a layer it lists); "
            "building again without it. The failed attempt's output: %s",
            error.spool_path,
        )
        run(fallback)
