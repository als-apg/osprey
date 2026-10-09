"""The BuildKit cache mounts every pip- and apt-installing image RUN carries.

A deploy builds several images, often at once, and each installs the framework
and a C toolchain. The recipes route those downloads through cache mounts so a
layer that has to rebuild fetches only what changed, and images building side
by side share one copy. The rule is spelled once here and applied to the
shipped service recipes (:mod:`tests.deployment.test_service_dockerfiles`) and
the rendered project template (:mod:`tests.cli.test_dockerfile_template`).
"""

from __future__ import annotations

from tests.deployment._proxy_idiom import raw_run_instructions

PIP_CACHE_MOUNT = "--mount=type=cache,target=/root/.cache/pip"
UV_CACHE_MOUNT = "--mount=type=cache,target=/root/.cache/uv"
APT_CACHE_MOUNTS = (
    "--mount=type=cache,target=/var/cache/apt,sharing=locked",
    "--mount=type=cache,target=/var/lib/apt/lists,sharing=locked",
)
DOCKER_CLEAN_REMOVAL = "rm -f /etc/apt/apt.conf.d/docker-clean"


def assert_build_caches_mounted(text: str, label: str) -> None:
    """Assert every installing RUN in *text* fetches through the shared caches.

    Four things hold together. Each pip RUN mounts pip's cache and does not
    opt out of it with ``--no-cache-dir``; a RUN that installs with uv mounts
    uv's cache too, and uninstalls uv before it ends so no image ships it. Each apt install mounts both apt
    caches, and no RUN deletes the lists, which live in the mount. The base
    image's docker-clean hook, which deletes every downloaded ``.deb``, is
    removed before the first install; with it gone, an install RUN without the
    mounts would leave its archives in the layer, which is why the mounts are
    required on every one rather than only the slow ones.
    """
    runs = raw_run_instructions(text)
    installs = [run for run in runs if "apt-get install" in run]
    if installs:
        assert DOCKER_CLEAN_REMOVAL in text, f"{label}: docker-clean is left in place"
        assert text.index(DOCKER_CLEAN_REMOVAL) < text.index("apt-get install"), (
            f"{label}: docker-clean must go before the first apt install"
        )
    for run in installs:
        for mount in APT_CACHE_MOUNTS:
            assert mount in run, f"{label}: apt install without `{mount}`:\n{run}"
    for run in runs:
        assert "rm -rf /var/lib/apt/lists" not in run, (
            f"{label}: a RUN deletes the apt lists the cache mount holds:\n{run}"
        )
        if "pip install" in run:
            assert PIP_CACHE_MOUNT in run, f"{label}: pip install without its cache:\n{run}"
            assert "--no-cache-dir" not in run, f"{label}: pip opts out of its cache:\n{run}"
        if "uv pip install" in run:
            assert UV_CACHE_MOUNT in run, f"{label}: uv install without its cache:\n{run}"
            assert "pip uninstall -y uv" in run, f"{label}: uv is left in the image:\n{run}"
