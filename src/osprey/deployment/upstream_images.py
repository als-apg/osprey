"""The images OSPREY pulls from an upstream, and the one way to name each.

:data:`UPSTREAM_IMAGES` holds the default reference of every such image, spelled
as its upstream names it. :func:`upstream_image` is how a reader names one: it
moves the reference onto the registry standing in for docker.io
(``images.base_registry``), so a declared mirror reaches every image OSPREY
pulls from Docker Hub.

This module imports nothing from ``osprey`` but the stdlib-only
:mod:`osprey.deployment.image_facts`, so every module that names an upstream
image can read it at import time.
"""

from osprey.deployment.image_facts import IMAGE_PYTHON

#: The registry Docker Hub references name, spelled out or implied.
DOCKER_HUB = "docker.io"

#: The default reference of every image OSPREY pulls from an upstream, keyed as
#: readers name it (the compose templates read the pulled services' entries
#: from ``upstream_images``).
UPSTREAM_IMAGES = {
    "postgres": "pgvector/pgvector:pg16",
    "mongodb": "mongo:7",
    # Pinned to the 5.26 LTS line rather than tracking a newer Neo4j:
    # neosemantics (n10s) — the plugin that imports RDF, without which the store
    # has no way to load the corpus at all — is installed at container start
    # from its plugin manifest
    # (https://neo4j-labs.github.io/neosemantics/versions.json), and 5.26 is the
    # newest server line that manifest covers; it does so patch-by-patch, where
    # older lines get at most a single frozen entry. The pin is therefore a
    # statement about the plugin's release cadence, not about Neo4j's, and it
    # can move as soon as n10s ships for a newer server. Check the manifest —
    # and the supported-set pin test beside the graphdb config tests — before
    # bumping.
    "graphdb": "neo4j:5.26-community",
    "redis": "redis:7.4-alpine",
    "tiled": "ghcr.io/bluesky/tiled:0.2.18",
    "clickhouse": "clickhouse/clickhouse-server:26.8.18.2",
    "nginx": "nginx:1.31-alpine",
    # Fully qualified: the transcripts copy runs as a bare ``docker|podman
    # run``, and podman resolves an unqualified name through short-name
    # aliases or a prompt.
    "transcripts_copy": "docker.io/library/alpine:3.20",
    # The pipeline's render job runs `osprey build` on the images' own Python,
    # which builds the deployment wheels the fit check judges against it.
    "ci_python": f"python:{IMAGE_PYTHON[0]}.{IMAGE_PYTHON[1]}-slim",
    "ci_docker": "docker:29",
    "ci_dind": "docker:29-dind",
    "ci_alpine": "alpine:3.23",
}


def hub_image(reference: str, base_registry: str) -> str:
    """A Docker Hub image reference, moved onto the registry standing in for docker.io.

    The mirror must keep Docker Hub's paths: an official image is served under
    ``library/`` and a namespaced one under its own namespace, exactly as
    Docker Hub files them. So ``redis:7.4-alpine`` becomes
    ``<mirror>/library/redis:7.4-alpine``, ``pgvector/pgvector:pg16`` becomes
    ``<mirror>/pgvector/pgvector:pg16``, and ``ghcr.io/bluesky/tiled:0.2.18``,
    which names another registry, is returned unchanged.

    :param reference: An image reference, short or fully qualified.
    :type reference: str
    :param base_registry: The registry standing in for docker.io, ``""`` when
        the deployment declares none.
    :type base_registry: str
    :return: *reference* on the mirror, or *reference* itself when there is no
        mirror or it names a registry other than Docker Hub.
    :rtype: str
    """
    if not base_registry:
        return reference
    first, slash, rest = reference.partition("/")
    if slash and first != DOCKER_HUB and ("." in first or ":" in first or first == "localhost"):
        return reference
    path = rest if slash and first == DOCKER_HUB else reference
    if "/" not in path:
        path = f"library/{path}"
    return f"{base_registry}/{path}"


def upstream_image(name: str, base_registry: str) -> str:
    """The :data:`UPSTREAM_IMAGES` entry *name*, moved onto *base_registry*.

    :param name: A key of :data:`UPSTREAM_IMAGES`.
    :type name: str
    :param base_registry: The registry standing in for docker.io, ``""`` when
        the deployment declares none.
    :type base_registry: str
    :return: The entry's reference, through :func:`hub_image`.
    :rtype: str
    :raises KeyError: If *name* is not a catalogue entry.
    """
    return hub_image(UPSTREAM_IMAGES[name], base_registry)
