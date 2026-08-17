"""Compose file generation and build directory setup.

Handles Jinja2 template rendering, service discovery, build directory
management, and Docker Compose file creation for container deployments.

Working-directory precondition: the render helpers here (:func:`setup_build_dir`,
:func:`render_kernel_templates`, :func:`_incremental_setup_build_dir`) resolve
``build_dir`` and each service's declared ``path`` RELATIVELY, against the
process working directory, and are called from the build with that directory at
the repo root. They are not safe to call from anywhere else. The one function
that answers a question rather than performing a render —
:func:`find_existing_compose_files` — takes an explicit ``base`` instead,
because its wrong answer is an empty list rather than an error.
"""

import json
import os
import re
import shutil
import stat
from pathlib import Path, PurePosixPath

import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.cli import output
from osprey.cli.phase_reporter import report_step
from osprey.deployment.runtime_helper import ComposeProvider, get_runtime_command, runtime_env
from osprey.deployment.subprocess_capture import run_captured

# The dev-wheel build, per-process cache, and staging-copy helpers live in
# wheel_build; the copy helper is invoked here by _stage_dev_wheel_for_context,
# and the cache reset is re-exported (redundant-alias) so the deployment tests
# and conftest keep patching / importing it from this module's namespace.
from osprey.deployment.wheel_build import (
    _copy_local_framework_for_override,
    preflight_dev_mode,
)
from osprey.deployment.wheel_build import (
    _reset_wheel_build_cache as _reset_wheel_build_cache,
)
from osprey.utils import dotenv
from osprey.utils.config import ConfigBuilder, load_project_config
from osprey.utils.log_filter import quiet_logger
from osprey.utils.logger import get_logger

logger = get_logger("deployment.compose")


def _report_fact(message: str) -> None:
    """Report ``message`` under this module's logger.

    The promotion contract lives in :func:`osprey.cli.output.report_fact`; this
    binds it to the compose logger so call sites pass the line alone.

    Args:
        message: The finished line, built by the caller.
    """
    output.report_fact(logger, message)


SERVICES_DIR = "services"
SRC_DIR = "src"
OUT_SRC_DIR = "repo_src"

TEMPLATE_FILENAME = "docker-compose.yml.j2"
COMPOSE_FILE_NAME = "docker-compose.yml"

#: The deployment's one secret store, at the repo root. Compose is pointed at it
#: explicitly (``--env-file``) on every invocation, and the containers mount it
#: from there — never from the render, which every build re-creates.
COMPOSE_ENV_FILENAME = ".env"

#: The committed, non-secret defaults file beside it — lower precedence than
#: :data:`COMPOSE_ENV_FILENAME` wherever both are delivered. Taken from
#: :mod:`osprey.utils.dotenv` rather than re-typed, so the renderer that lists
#: the chain and the helpers that merge it cannot disagree about the filename.
ENV_SHARED_FILENAME = dotenv.ENV_SHARED_FILENAME

#: Machine-readable record of WHICH env-chain files a render found, written into
#: the render's output base beside the compose files it explains. The render
#: fixes chain membership: whichever files existed then are the ones the
#: rendered ``env_file:`` lists name and the ones the invocation's ``--env-file``
#: flags point at. A deploy that finds a different set on disk is running
#: compose files that describe a different chain, which is why ``osprey up``
#: reads this back and refuses on a mismatch rather than discovering it as a
#: missing value inside a container.
#:
#: Deliberately a sidecar rather than a comment inside a rendered compose file:
#: the record has to exist for a deployment with no services at all, and every
#: rendered compose file stays byte-identical to what it was before the chain
#: existed. Deliberately JSON with plain filenames, not paths: a build renders
#: in a staging tree that is later moved, so an absolute path recorded here
#: would name a directory that no longer exists by the time anything reads it.
ENV_CHAIN_MARKER_FILENAME = "env-chain.json"

#: The key holding the ordered filename list inside that file.
ENV_CHAIN_MARKER_KEY = "env_chain"

#: Where a project image holds its deployment repo: ``templates/project/
#: Dockerfile.j2``'s ``WORKDIR /app/<project_name>``. Spelled here so the one
#: template that mounts a volume INTO that repo takes the path from the same
#: derivation the image was built with instead of re-typing the prefix.
_CONTAINER_APP_ROOT = "/app"

#: Directory the qmd sidecar's read-only corpus mounts live under, inside the
#: container. Each corpus lands at ``<root>/<collection>``, so the mount target
#: and the collection name cannot drift apart.
QMD_CORPUS_ROOT = "/corpus"

#: qmd collection the OKF facility-knowledge bundle is indexed under. This is a
#: contract, not a label: a query filters on the collection name, so this must
#: equal ``osprey.services.facility_knowledge.okf.bundle.OKF_COLLECTION``.
#: Restated here rather than imported, because rendering a compose file must not
#: drag the facility-knowledge service into the deployment import graph; a test
#: asserts the two spellings agree.
QMD_OKF_COLLECTION = "okf"

#: qmd collection ARIEL's markdown mirror is indexed under. Same contract: the
#: ARIEL search module filters on this name.
QMD_ARIEL_COLLECTION = "ariel"


def resolve_repo_root(config=None, config_path=None):
    """The deployment repo a compose invocation is pinned to.

    Every path a compose file spells relatively — bind-mount sources, build
    contexts, ``env_file`` entries — is resolved by compose against ONE
    directory, the compose project directory. This function answers what that
    directory is, and :func:`compose_base_cmd` pins it — with
    ``--project-directory``, or, for a provider that does not parse that flag,
    with the shape described there — so it is never inferred from the working
    directory the verb happened to be typed in.

    Resolution order, most authoritative first:

    1. The config *path*, when the caller has one: the repo is the directory
       holding it, or its parent when the config sits in the ``build/`` zone
       (:func:`osprey.utils.workspace.repo_root_for_config`). Filesystem truth,
       and the only answer immune to a rewritten ``project_root``.
    2. The config's ``project_root`` key — but only when it names a directory
       that exists here. A project built with ``--runtime-root`` records the
       path it will run at *inside a container* (``/app/<name>``), which is not
       a host path at all; taking it would pin compose to a directory the host
       does not have.
    3. The working directory. Every repo-scoped verb chdirs into the repo root
       before reaching a compose invocation, so this is correct — and it is what
       the whole deployment layer relied on implicitly before the contract
       existed.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.
        config_path: Path to that config file, when the caller has it.

    Returns:
        An absolute path to the deployment repo root.
    """
    from osprey.utils.workspace import repo_root_for_config

    if config_path:
        return repo_root_for_config(Path(config_path).expanduser().absolute())

    configured = (config or {}).get("project_root")
    if configured:
        candidate = Path(str(configured)).expanduser()
        if candidate.is_dir():
            return candidate.absolute()

    return Path.cwd().absolute()


#: Label key carrying WHICH CHECKOUT a container or volume belongs to.
#: ``COMPOSE_PROJECT_NAME`` is derived from the repo's directory name, so two
#: clones of one deployment on a single host share a project name and a volume
#: namespace. This is what tells them apart — and what lets a destructive verb
#: refuse to remove the other checkout's resources.
REPO_ID_LABEL = "com.osprey.repo-id"


def repo_identity(repo_root):
    """A short, stable identity for the deployment repo at *repo_root*.

    ``sha256`` of the resolved absolute path, truncated to 12 hex characters —
    enough to tell the checkouts on one host apart, short enough to read in a
    ``docker inspect``. Symlinks are resolved, so two spellings of one directory
    produce one identity rather than two.

    The PATH, not its contents: the question this answers is "which checkout is
    this?", which has to survive every edit to the repo and every rebuild of it.

    The single spelling of the rule, deliberately, because every consumer
    compares against it and a second derivation would eventually disagree with
    the first — at which point one verb would act on a set of containers another
    verb reports. The consumers today: :func:`_inject_project_metadata` renders
    it into every container's and volume's :data:`REPO_ID_LABEL`, ``osprey
    down`` uses it to find this deployment's containers when ``build/`` no
    longer holds the compose files that declared them, ``osprey reset`` uses it
    to refuse removing resources that belong to a different checkout of the same
    name, and ``osprey status`` uses it to sort every container on the host by
    which checkout it belongs to. Any new one derives it from here too.

    :param repo_root: The deployment repo root.
    :return: 12 lowercase hex characters.
    """
    import hashlib

    resolved = Path(repo_root).expanduser().resolve()
    return hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]


def env_chain_names(repo_root):
    """Filenames of the env-chain files present at *repo_root*, in chain order.

    The single probe every part of a render asks its chain question through —
    the per-file flags the templates gate on, the membership marker
    (:func:`record_env_chain_membership`) and the ``--env-file`` fragment
    (:func:`compose_env_file_args`) — so a render cannot list one chain in a
    compose file and record or pass a different one.

    Bare filenames rather than paths, because that is what all three want: the
    templates spell them relative to the pinned compose project directory, and
    the marker has to survive the render being moved.

    :param repo_root: The deployment repo root.
    :return: e.g. ``[".env.shared", ".env"]`` — lowest precedence first, missing
        members skipped.
    """
    return [path.name for path in dotenv.chain_files(Path(repo_root))]


def compose_env_file_args(repo_root):
    """One ``--env-file`` flag per env-chain file at *repo_root*, in chain order.

    ``--env-file`` decides which values compose substitutes into every
    ``${VAR}`` in every compose file, so each path is spelled absolute and
    anchored on the repo rather than on the working directory. Resolved against
    anything but the deployment's own chain, it starts the stack with empty
    credentials and no error at all.

    The flag is REPEATED — ``--env-file <repo>/.env.shared --env-file
    <repo>/.env`` — so both files are visible to interpolation, with ``.env``
    last and therefore winning on any key both set. That is the Docker Compose
    v2 reading of repeated ``--env-file``, and this fragment is the docker
    shape; a provider that instead keeps only the last file it is handed is
    given one pre-merged file by the invocation seam
    (:func:`osprey.deployment.container_lifecycle._env_file_args`) rather than
    this list, because passing this list to it would silently drop
    ``.env.shared``.

    A repo with no chain file at all gets the empty list rather than a missing
    file passed anyway — compose hard-fails on an ``--env-file`` that is not
    there. The warning is for that case only: a repo carrying just
    ``.env.shared`` has an env chain, and telling its operator the environment
    is empty would be false.

    :param repo_root: The deployment repo root.
    :return: The argv fragment, ready to splice into a compose invocation.
    """
    chain = dotenv.chain_files(Path(repo_root))
    if not chain:
        # The remedy rides on the warning rather than on a line of its own: it
        # is the second half of the same fact, and on its own it reads as an
        # instruction nobody asked for.
        logger.warning(
            "No .env file found - services will start with default/empty environment variables\n"
            "  → cp .env.example .env, then edit .env to fill in the API keys"
        )
        return []

    args = []
    for env_file in chain:
        args.extend(("--env-file", str(env_file)))
    return args


def record_env_chain_membership(out_base, chain_names):
    """Write the render's chain membership to *out_base*, and return the path.

    Called once per render, from :func:`prepare_compose_files`, so the record
    describes the same probe the rendered compose files were built from. See
    :data:`ENV_CHAIN_MARKER_FILENAME` for what a reader does with it.

    :param out_base: The render's output base — where the compose files land.
    :param chain_names: Chain filenames in chain order (:func:`env_chain_names`).
    :return: Path to the written marker.
    """
    base = Path(out_base)
    base.mkdir(parents=True, exist_ok=True)
    marker = base / ENV_CHAIN_MARKER_FILENAME
    marker.write_text(
        json.dumps({ENV_CHAIN_MARKER_KEY: list(chain_names)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return marker


def read_rendered_env_chain(repo_root, build_dir=None):
    """The chain membership a render recorded, or ``None`` when there is none.

    The read side of :data:`ENV_CHAIN_MARKER_FILENAME`, for the deploy-time
    check that the chain on disk still matches the one the compose files
    describe.

    ``None`` — no marker, an unreadable one, or a payload that is not a list of
    filenames — means "this render recorded nothing", which is what a render
    from before the marker existed looks like. The caller decides what to do
    with that; it is deliberately not reported as a mismatch, because a
    mismatch is a refusal and "no record" is not evidence of one.

    :param repo_root: The deployment repo root.
    :param build_dir: The render zone's name, when it is not the default
        ``build``.
    :return: Recorded chain filenames in chain order, or ``None``.
    """
    from osprey.utils.workspace import BUILD_DIR_NAME

    marker = Path(repo_root) / (build_dir or BUILD_DIR_NAME) / ENV_CHAIN_MARKER_FILENAME
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    recorded = payload.get(ENV_CHAIN_MARKER_KEY) if isinstance(payload, dict) else None
    if not isinstance(recorded, list) or not all(isinstance(name, str) for name in recorded):
        return None
    return recorded


def compose_base_cmd(runtime_cmd, compose_files, repo_root, env_file_args=None, provider=None):
    """Pin a compose argv to one deployment repo, before any subcommand.

    The single spelling of the invocation contract (TR-3). Every compose
    command OSPREY runs — ``up``, ``down``, ``restart``, ``rm``, ``build``,
    ``pull``, ``logs``, the web stack's own invocations — is built from here,
    so no path in any rendered compose file can resolve against a different
    base depending on which verb ran it.

    TWO SHAPES, ONE PROJECT DIRECTORY
    =================================

    Both providers OSPREY supports must end up resolving every relative path in
    every rendered compose file against the repo root. They are told so in
    different ways, because they parse different flags.

    Docker Compose v2 (and the default when no probe result is threaded down)::

        <runtime> compose [--progress plain]
            --project-directory <repo-root>
            -f <repo-root>/build/<file> ...
            [--env-file <repo-root>/.env.shared --env-file <repo-root>/.env]

    ``--project-directory`` is what makes the rest true. Without it compose
    derives the project directory from the FIRST ``-f`` file's location, so a
    services invocation resolved relative paths against ``build/services/``
    while the web stack's resolved them against wherever its file happened to
    be written — two bases for one deployment, and the reason the two stacks
    could never share an invocation. With the repo root pinned, one base holds
    for both, and every relative path in every template is spelled against it.

    podman-compose::

        <runtime> compose [--progress plain]
            -f <repo-root>/.osprey-compose.yml
            [--env-file <repo-root>/build/.env.merged]

    No podman-compose version parses ``--project-directory``, so the pin has to
    be made a property of the ``-f`` list instead: the rendered files are merged
    into ONE document at the repo root
    (:func:`~osprey.deployment.compose_merge.write_merged_compose`), which puts
    the only ``-f`` file's own directory at the root. That is where every
    version chdir's — the ones that follow ``COMPOSE_PROJECT_DIR`` and the ones
    that override it with the first compose file's dirname — so the same
    relative paths resolve to the same files on all of them. The environment
    half of the same pin is :func:`compose_provider_env`; a caller that builds
    this argv without it is handing the newer versions a project directory of
    wherever the operator happened to be standing.

    The ``--env-file`` fragment differs for the same kind of reason: repeated
    flags mean "merge these" to docker and "keep the last one" to
    podman-compose, so the podman shape is handed a single pre-merged file. Both
    fragments come from
    :func:`osprey.deployment.container_lifecycle._env_file_args`, which is also
    where the merged file is (re)generated.

    Neither shape uses compose ``profiles:``/``--profile``: podman-compose's
    support for them differs by version, and a service silently left out of a
    deploy is exactly the failure this seam exists to prevent.

    The runtime argv is taken as an argument rather than resolved here, for the
    same reason :func:`osprey.deployment.runtime_helper.with_plain_progress`
    takes one: each call site reaches its runtime through the
    ``get_runtime_command`` name bound in its own module, which is the seam the
    deploy tests patch. The probe result is passed in for the same reason.

    :param runtime_cmd: Resolved compose argv base, e.g. ``["docker", "compose"]``
        (optionally already carrying ``--progress plain``). Never mutated.
    :param compose_files: Compose files for this invocation, in ``-f`` order.
        For the podman-compose shape these are the inputs to the merge, in the
        same order — the merged document is rewritten on every invocation, so it
        always describes the files this invocation was given.
    :param repo_root: The deployment repo root — the compose project directory.
    :param env_file_args: An already-resolved ``--env-file`` fragment, for
        callers that thread one down. ``None`` resolves it here, provider-shaped.
    :param provider: The detected compose provider
        (:class:`~osprey.deployment.runtime_helper.ComposeProvider`), when the
        caller has probed for one. ``None`` selects the docker shape.
    :return: A new argv list, ready for its subcommand.
    :raises ComposeMergeError: The podman-compose shape's merge input is
        missing, unparseable, or not a compose document.
    """
    root = Path(repo_root).expanduser().absolute()
    cmd = list(runtime_cmd)
    if provider is ComposeProvider.PODMAN_COMPOSE:
        from osprey.deployment.compose_merge import write_merged_compose

        cmd.extend(("-f", str(write_merged_compose(root, compose_files))))
    else:
        cmd.extend(("--project-directory", str(root)))
        for compose_file in compose_files:
            path = Path(compose_file)
            cmd.extend(("-f", str(path if path.is_absolute() else root / path)))
    if env_file_args is None:
        # Deferred: the provider-shaped resolver lives with the merged file it
        # (re)writes, in the module that owns the lifecycle verbs, and that
        # module imports this one.
        from osprey.deployment.container_lifecycle import _env_file_args

        env_file_args = _env_file_args(root, provider)
    cmd.extend(env_file_args)
    return cmd


def compose_provider_env(provider, repo_root):
    """The environment half of the provider-shaped invocation, if it needs one.

    :func:`compose_base_cmd` pins the project directory in argv for the docker
    shape and cannot for the podman-compose one, which parses no
    ``--project-directory``. podman-compose 1.4.1+ read ``COMPOSE_PROJECT_DIR``
    instead, and older versions chdir to the merged document's own directory —
    the repo root either way, but only if the variable is actually set. Left
    unset, 1.4.1+ take the working directory as the project directory, and every
    relative path in the merged document resolves against wherever the operator
    typed the verb.

    Kept beside the argv builder, and taking the same two inputs, because the
    two halves are one contract: a call site that threads the probe result into
    one and not the other emits an invocation that is internally inconsistent.

    :param provider: The detected compose provider, or ``None`` for the default
        (docker) shape.
    :param repo_root: The deployment repo root — the compose project directory.
    :return: Variables to layer onto the compose subprocess environment. Empty
        for every shape that carries the pin in argv.
    """
    if provider is not ComposeProvider.PODMAN_COMPOSE:
        return {}
    return {"COMPOSE_PROJECT_DIR": str(Path(repo_root).expanduser().absolute())}


def find_service_config(config, service_name):
    """Locate service configuration and template path for deployment.

    Services are declared in the project config's top-level ``services:`` block,
    keyed by service name — the same short name that appears in
    ``deployed_services``. This function looks the name up in that block and
    returns both the service configuration object and the path to its Docker
    Compose template, so the caller can access service settings and initiate
    template rendering.

    :param config: Configuration containing service definitions
    :type config: dict
    :param service_name: Service identifier as it appears under ``services:``
        (and in ``deployed_services``)
    :type service_name: str
    :return: Tuple containing service configuration and template path, or
        ``(None, None)`` if the name is not declared under ``services:``
    :rtype: tuple[dict, str] or tuple[None, None]

    Examples:
        Service discovery::

            >>> config = {'services': {'openobserve': {'path': 'services/openobserve'}}}
            >>> service_config, template_path = find_service_config(config, 'openobserve')
            >>> print(template_path)  # 'services/openobserve/docker-compose.yml.j2'

    .. seealso::
       :func:`setup_build_dir` : Processes discovered services for deployment
    """
    services = config.get("services", {})
    service_config = services.get(service_name)
    if service_config:
        return service_config, os.path.join(service_config["path"], TEMPLATE_FILENAME)

    return None, None


def _normalize_compose_name(name):
    """Normalize a string into a valid docker-compose project name.

    Mirrors docker-compose's own normalization so the value OSPREY pins as
    ``COMPOSE_PROJECT_NAME`` matches what compose would derive on its own:
    lowercase, keep only ``[a-z0-9_-]`` (dropping everything else, including
    whitespace), then strip leading AND trailing ``_``/``-`` so the result
    begins and ends with a letter or number. Trailing separators must go too:
    the project name is used as an image-tag prefix (``<project>-dispatch:local``),
    and a trailing separator would produce invalid docker references like
    ``proj_-dispatch:local``. An input that is already a valid lowercase name
    passes through byte-unchanged.

    :param name: Raw candidate project name
    :type name: str
    :return: Compose-valid project name (may be empty if no valid characters
        remain)
    :rtype: str
    """
    kept = re.findall(r"[a-z0-9_-]", str(name).lower())
    return "".join(kept).strip("_-")


def resolve_project_name(config):
    """Derive the project name used for labels and the ``<project>:local`` image tag.

    The name is resolved with a fixed priority order so that every consumer
    (container labels here, plus the ``<project>:local`` image tag that
    ``osprey up`` builds for the dispatch worker) agrees on one value:

    1. Root-level ``project_name`` attribute (preferred, explicit)
    2. Last component of the ``project_root`` path (smart fallback)
    3. ``"unnamed-project"`` (final fallback)

    The resolved name is normalized to a valid docker-compose project name via
    :func:`_normalize_compose_name` so it matches the value compose itself would
    use for volume/network namespacing. If normalization empties the candidate
    (no valid characters), the ``"unnamed-project"`` fallback is used.

    :param config: Configuration dictionary
    :type config: dict
    :return: Resolved project name
    :rtype: str
    """
    project_name = config.get("project_name")

    if not project_name:
        # Fallback: Extract from project_root path
        project_root = config.get("project_root", "")
        if project_root:
            project_name = os.path.basename(str(project_root).rstrip("/"))

    project_name = _normalize_compose_name(project_name) if project_name else ""

    if not project_name:
        # Final fallback: Default
        project_name = "unnamed-project"

    return project_name


def resolve_user_volume_names(config, user):
    """Derive the real runtime names of a web terminal user's named volumes.

    The web terminal compose template declares each user's volumes bare
    (``<user>-claude-config``, ``<user>-agent-data``). Compose namespaces bare
    volume names with ``COMPOSE_PROJECT_NAME`` (defaulting to
    ``basename(cwd)`` when unset), which is independent of the project name
    :func:`resolve_project_name` derives for container labels. Left alone,
    those two can diverge — the labels say one project, the actual volumes
    live under another. :func:`runtime_helper.runtime_env` pins
    ``COMPOSE_PROJECT_NAME`` to :func:`resolve_project_name`'s result for
    every runtime invocation, so this function returns the same
    ``<project>_<name>`` form to give volume-targeting code (inspect/rm/
    archive) the real runtime volume names deterministically, without
    shelling out to discover them.

    :param config: Configuration dictionary (``None`` is treated as an empty
        config, resolving to the ``"unnamed-project"`` namespace — symmetric with
        :func:`runtime_helper.runtime_env`)
    :type config: dict
    :param user: Web terminal username the volumes belong to
    :type user: str
    :return: Tuple of ``(claude_config_volume, agent_data_volume)`` runtime names
    :rtype: tuple[str, str]
    """
    project = resolve_project_name(config or {})
    return f"{project}_{user}-claude-config", f"{project}_{user}-agent-data"


def repo_relative_mount_source(raw, repo_root=None):
    """Spell a configured host path as a compose bind source.

    The one spelling rule every renderer uses, shared so the base compose files
    and the web-terminal overlay cannot disagree about how the same configured
    directory is written. Every relative path in a rendered compose file
    resolves against the pinned compose project directory, which is the
    deployment repo root (see :func:`compose_base_cmd`), so a repo-relative path
    is emitted with an explicit ``./`` — a bind source that is neither absolute
    nor dot-prefixed is not reliably read as a path by every runtime.

    :param raw: Path as configured, absolute or repo-relative
    :type raw: str
    :param repo_root: The deployment repo root, when the caller has one. An
        absolute path inside it is then spelled relative to it, so the rendered
        file survives the repo being moved. ``None`` — for a renderer with no
        repo root to resolve against, such as
        :func:`osprey.deployment.web_terminals.render.render_web_terminals`,
        which reads no filesystem — leaves absolute paths absolute. Both spell
        the same directory; only relocatability differs.
    :type repo_root: str | None
    :return: A compose-safe bind source
    :rtype: str
    """
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return f"./{PurePosixPath(path)}"
    if repo_root is None:
        return path.as_posix()
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return path.as_posix()
    return f"./{PurePosixPath(relative)}"


def _resolve_qmd_corpora(config, repo_root):
    """List the corpora the qmd sidecar indexes, one entry per collection.

    This is the single derivation behind BOTH of the sidecar's rendered
    artifacts: ``docker-compose.yml.j2`` turns each entry into a read-only bind
    mount and ``index.yml.j2`` turns the same entry into a qmd collection. Doing
    it once here is the point — a corpus mounted without a collection indexes
    nothing, and a collection declared without a mount points at an empty
    directory, and both fail as "the search returns nothing" rather than as an
    error.

    Two corpora are recognized, each gated on the config that produces it:

    * the facility-knowledge bundle, when ``facility_knowledge.bundle_path``
      names one;
    * ARIEL's markdown mirror, when
      ``ariel.enhancement_modules.qmd_export`` is enabled AND names a
      ``mirror_path`` (an enabled export with no path is a config error the
      exporter itself refuses at runtime; there is nothing to mount here).

    :param config: Configuration dictionary
    :type config: dict
    :param repo_root: The deployment repo root, for relative bind sources
    :type repo_root: str
    :return: Corpus descriptors with ``collection``, ``source`` and ``target``
    :rtype: list[dict]
    """

    def corpus(collection, raw_path):
        return {
            "collection": collection,
            "source": repo_relative_mount_source(raw_path, repo_root),
            "target": f"{QMD_CORPUS_ROOT}/{collection}",
        }

    corpora = []

    knowledge = config.get("facility_knowledge")
    bundle_path = knowledge.get("bundle_path") if isinstance(knowledge, dict) else None
    if isinstance(bundle_path, str) and bundle_path.strip():
        corpora.append(corpus(QMD_OKF_COLLECTION, bundle_path.strip()))

    ariel = config.get("ariel")
    modules = ariel.get("enhancement_modules") if isinstance(ariel, dict) else None
    export = modules.get("qmd_export") if isinstance(modules, dict) else None
    if isinstance(export, dict) and export.get("enabled"):
        # `mirror_path` may be written directly on the module block or inside
        # its `settings:` mapping — ARIEL's own loader merges the two with
        # `settings` winning, and the mount must follow whichever the exporter
        # will actually write to.
        settings = export.get("settings")
        mirror_path = export.get("mirror_path")
        if isinstance(settings, dict) and settings.get("mirror_path"):
            mirror_path = settings["mirror_path"]
        if isinstance(mirror_path, str) and mirror_path.strip():
            corpora.append(corpus(QMD_ARIEL_COLLECTION, mirror_path.strip()))

    return corpora


def _resolve_qmd_render_context(config, repo_root):
    """Build the ``osprey_qmd`` render context for the sidecar's templates.

    The port, the publish interface and the sweep interval are resolved through
    :mod:`osprey.deployment.qmd_service` rather than re-spelled as
    ``| default(...)`` filters in the template, because the compose fragment,
    the sidecar entrypoint and the Python client all have to agree on those
    three numbers and that module is where they are defined.

    A deployment with no ``services.qmd`` block still gets a context built from
    the schema defaults. The sidecar's templates are only rendered when qmd is a
    deployed service, so the fallback is never what a real deploy uses; it
    exists so that a render cannot fail with an attribute error on a name the
    template legitimately expects to be there.

    :param config: Configuration dictionary
    :type config: dict
    :param repo_root: The deployment repo root, for relative bind sources
    :type repo_root: str
    :return: ``port``, ``bind_address``, ``interval_seconds`` and ``corpora``
    :rtype: dict
    """
    from osprey.deployment.qmd_service import (
        QMDServiceConfig,
        resolve_bind_address,
        resolve_qmd_service_config,
    )

    resolved = resolve_qmd_service_config(config) or QMDServiceConfig(
        bind_address=resolve_bind_address(config)
    )
    return {
        "port": resolved.port,
        "bind_address": resolved.bind_address,
        "interval_seconds": resolved.interval_seconds,
        "corpora": _resolve_qmd_corpora(config, repo_root),
    }


def _inject_project_metadata(config):
    """Add project tracking metadata for container labels.

    This function injects deployment metadata into the configuration that will
    be used as Docker labels in the rendered compose files. These labels enable
    tracking which project/agent owns each container.

    The project name is extracted via :func:`resolve_project_name` (explicit
    ``project_name`` > last component of ``project_root`` > ``unnamed-project``).

    It also defaults the dispatch worker's image to ``<project>:local`` — the
    tag ``osprey up`` builds from the project ``Dockerfile`` — unless the
    profile pinned an explicit ``services.dispatch_worker.image``. The
    event-dispatcher is left untouched (it builds ``<project>-dispatch:local``
    via its own compose ``build:`` block; the worker deliberately has none).

    :param config: Configuration dictionary
    :type config: dict
    :return: Configuration with added osprey_labels section
    :rtype: dict
    """
    import datetime

    project_name = resolve_project_name(config)

    # Resolve the running framework version so service Dockerfiles can pin the
    # PyPI install (`pip install osprey-framework==<version>`) for production
    # builds. Dev builds install a locally-built wheel instead (see --dev).
    #
    # This is deliberately the *running* version rather than the release lineage,
    # and deliberately ungated. The service Dockerfiles hard-fail on an empty
    # OSPREY_VERSION but tolerate an unreleased one under OSPREY_DEV=1, priming
    # the layer with the latest release and warning that the pin was unreleased.
    # Substituting the base release here would silently prime with released code
    # and suppress that warning. A production build from a development checkout is
    # refused earlier and more clearly by `_resolve_pip_spec`.
    from osprey.version import get_running_version

    osprey_version = get_running_version()

    # The deployment repo this render belongs to. Resolved once, here, because
    # three separate things below are derived from it.
    repo_root = resolve_repo_root(config)

    # Create enhanced config with label metadata
    config_with_labels = config.copy()
    config_with_labels["osprey_labels"] = {
        "project_name": project_name,
        "project_root": config.get("project_root", os.getcwd()),
        "deployed_at": datetime.datetime.now().isoformat(),
        # Which CHECKOUT this is (:func:`repo_identity`). Baked in as a literal
        # at render time rather than left as a `${VAR}` for compose to
        # interpolate: the label has to be trustworthy for a verb that reads it
        # back to decide what to DESTROY, and an interpolated one is silently
        # empty for anyone who runs `docker compose` by hand without the
        # variable exported. A baked literal is a property of the build,
        # readable with `docker inspect` and reproducible from the repo path.
        #
        # Applied at CREATE time, like every container label: containers that
        # were already running keep whatever label they were created with until
        # something recreates them, and a named volume takes its labels only
        # when it is first created — an existing volume is never relabelled by a
        # later deploy.
        "repo_id": repo_identity(repo_root),
    }
    config_with_labels["osprey_version"] = osprey_version

    # Which env-chain files this deployment repo has, for the templates that
    # deliver them to a container. A service reads the chain through an
    # ``env_file:`` list, and compose hard-fails on an entry that is not there,
    # so the list can only name files that exist — which fixes chain membership
    # at RENDER time and is why the same probe is recorded for the deploy to
    # check against (:data:`ENV_CHAIN_MARKER_FILENAME`).
    #
    # Probed at the repo root, not in the working directory: the chain at that
    # single root is the deployment's whole env (TR-3), and a cwd probe answered
    # "no" for any invocation made from a subdirectory — silently dropping
    # provider auth from the worker.
    #
    # Three views of one probe, because templates need different ones: the
    # ordered list to emit an ``env_file:`` block from (lowest precedence first,
    # so the later entry wins, matching the ``--env-file`` order in
    # :func:`compose_env_file_args`), and a per-file flag to gate anything that
    # is about one specific member.
    chain_names = env_chain_names(repo_root)
    config_with_labels["osprey_env_chain"] = [f"./{name}" for name in chain_names]
    config_with_labels["osprey_env_shared_present"] = ENV_SHARED_FILENAME in chain_names
    config_with_labels["osprey_env_present"] = COMPOSE_ENV_FILENAME in chain_names

    # Bind-mount source for the Virtual Accelerator's scenario state directory,
    # derived here rather than hardcoded in the template: the mount must follow
    # whatever directory `osprey sim apply` actually writes, whether that moved
    # because the project relocated its agent-data root or because it set
    # `simulation.state_dir` outright. Resolved through the one shared helper so
    # writer and mount cannot disagree.
    #
    # Emitted relative to the REPO ROOT, because that is the compose project
    # directory every invocation now pins (see :func:`compose_base_cmd`), and
    # with an explicit ``./`` — a bind source that is neither absolute nor
    # dot-prefixed is not reliably read as a path by every runtime. A state dir
    # outside the repo has no relative spelling and is emitted absolute.
    from osprey.utils.workspace import agent_data_base_dir, resolve_simulation_state_dir

    state_dir = resolve_simulation_state_dir(config, repo_root).expanduser().absolute()
    try:
        relative_state_dir = state_dir.relative_to(repo_root)
    except ValueError:
        config_with_labels["osprey_state_mount_source"] = state_dir.as_posix()
    else:
        config_with_labels["osprey_state_mount_source"] = f"./{PurePosixPath(relative_state_dir)}"

    # The agent-data root as the CONTAINER sees it, ABSOLUTE and complete —
    # rendered from the same ``agent_data.base_dir`` the container's own
    # config.yml declares. Templates that mount a workspace volume spell the
    # mount target through this rather than a literal, so a project that
    # relocates its agent-data root cannot end up with the volume mounted at one
    # path and the process writing at another.
    #
    # The join happens HERE rather than as `{{ project_dir }}/{{ base }}` in the
    # template, for the same reason it does in
    # ``web_terminals.render._container_agent_data_dir``: an absolute base_dir
    # names the same absolute path inside the container and must NOT be
    # re-anchored under the project directory, and a template concatenation
    # cannot make that distinction — it would mount the volume at
    # ``/app/<project>//data/...`` while the process wrote to ``/data/...``.
    #
    # Known limit, shared with the web-terminals sibling: a ``~``-relative
    # base_dir is anchored here like any other relative path, while the process
    # inside expands it against the container's HOME. Such a project's agent
    # data is misfiled rather than lost, and naming HOME here would duplicate a
    # value the image sets — the duplication this derivation exists to remove.
    container_project_dir = PurePosixPath(_CONTAINER_APP_ROOT) / project_name
    agent_data_base = PurePosixPath(agent_data_base_dir(config))
    config_with_labels["osprey_container_agent_data_dir"] = (
        agent_data_base
        if agent_data_base.is_absolute()
        else container_project_dir / agent_data_base
    ).as_posix()

    # The qmd sidecar's resolved settings and its corpus list, derived here for
    # the same reason ``osprey_state_mount_source`` is: a mount source has to
    # follow whatever the config actually points at, and the sidecar's two
    # rendered artifacts — the compose fragment's read-only corpus mounts and
    # index.yml's collections — must be generated from ONE list or they drift
    # into a mount with no collection (indexes nothing) or a collection with no
    # mount (an empty directory). Injected unconditionally, like every other
    # derived key here; templates that do not name it are unaffected.
    config_with_labels["osprey_qmd"] = _resolve_qmd_render_context(config, repo_root)

    # Default the dispatch worker's image to the project image that
    # ``osprey up`` builds (``<project>:local``). The worker compose
    # template renders ``${OSPREY_WORKER_IMAGE:-{{ services.dispatch_worker.image
    # | default(osprey_labels.project_name ~ ':local') }}}`` — the template-level
    # fallback now matches this same project image, so this setdefault and the
    # template agree even when it cannot fire (e.g. a null ``dispatch_worker:``
    # key). ``setdefault`` honors a profile that pinned its own image. Copy the
    # nested dicts we touch so the shared input config is never mutated (the
    # top-level copy() above is shallow).
    services = config_with_labels.get("services")
    if isinstance(services, dict) and isinstance(services.get("dispatch_worker"), dict):
        services = dict(services)
        worker_cfg = dict(services["dispatch_worker"])
        worker_cfg.setdefault("image", f"{project_name}:local")
        services["dispatch_worker"] = worker_cfg
        config_with_labels["services"] = services

    return config_with_labels


def render_template(template_path, config, out_dir):
    """Render Jinja2 template with configuration context to output directory.

    This function processes Jinja2 templates using the configuration
    as context, generating concrete configuration files for container deployment.
    The system supports multiple template types including Docker Compose files
    and Jupyter kernel configurations, with intelligent output filename detection.

    Template rendering uses the complete configuration dictionary as Jinja2 context,
    enabling templates to access any configuration value including environment
    variables, service settings, and application-specific parameters. Environment
    variables can be referenced directly in templates using ${VAR_NAME} syntax
    for deployment-specific configurations like proxy settings. The output
    directory is created automatically if it doesn't exist.

    :param template_path: Path to the Jinja2 template file to render
    :type template_path: str
    :param config: Configuration dictionary to use as template context
    :type config: dict
    :param out_dir: Output directory for the rendered file
    :type out_dir: str
    :return: Full path to the rendered output file
    :rtype: str

    Examples:
        Docker Compose template rendering::

            >>> config = {'database': {'host': 'localhost', 'port': 5432}}
            >>> output_path = render_template(
            ...     'services/mongo/docker-compose.yml.j2',
            ...     config,
            ...     'build/services/mongo'
            ... )
            >>> print(output_path)  # 'build/services/mongo/docker-compose.yml'

        Jupyter kernel template rendering::

            >>> config = {'project_root': '/home/user/project'}
            >>> output_path = render_template(
            ...     'services/jupyter/python3-epics/kernel.json.j2',
            ...     config,
            ...     'build/services/jupyter/python3-epics'
            ... )
            >>> print(output_path)  # 'build/services/jupyter/python3-epics/kernel.json'

    .. note::
       The function automatically determines output filenames based on template
       naming conventions: .j2 extension is removed, and specific patterns
       like docker-compose.yml.j2 and kernel.json.j2 are recognized.

    .. seealso::
       :func:`setup_build_dir` : Uses this function for service template processing
       :func:`render_kernel_templates` : Batch processing of kernel templates
    """
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(template_path)

    # Inject project metadata for container labels
    config_dict = _inject_project_metadata(config)
    rendered_content = template.render(config_dict)

    # Determine output filename based on template type
    if template_path.endswith("docker-compose.yml.j2"):
        output_filename = COMPOSE_FILE_NAME
    elif template_path.endswith("kernel.json.j2"):
        output_filename = "kernel.json"
    else:
        # Generic fallback: remove .j2 extension
        output_filename = os.path.basename(template_path)[:-3]

    output_filepath = os.path.join(out_dir, output_filename)
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filepath, "w") as f:
        f.write(rendered_content)
    return output_filepath


def _stage_dev_wheel_for_context(out_dir, dev_mode):
    """Stage the local dev wheel into a service build context; report success.

    The boolean return is the success signal :func:`setup_build_dir` (and its
    incremental sibling) key the rendered ``OSPREY_DEV`` build arg on: only a
    context that actually received a wheel may relax the Dockerfile's pinned
    ``osprey-framework`` install (fail-closed — a failed build/staging keeps
    the fail-loud pin instead of silently installing the latest published
    release under a flag that means "run my local code").

    Only build contexts that contain a ``Dockerfile`` get the wheel: pure-image
    services (postgresql, openobserve) never install it, and a stray wheel in
    their context is dead weight that also churns the build-context hash.

    :param out_dir: The service's build context directory (already populated
        with the service's files, including any ``Dockerfile``)
    :type out_dir: str
    :param dev_mode: Whether ``--dev`` was passed
    :type dev_mode: bool
    :return: True iff a wheel was successfully staged into ``out_dir``
    :rtype: bool
    """
    if not dev_mode:
        # DEBUG, not INFO — fires once per service build context on every
        # production build, restating a flag the operator already passed.
        logger.debug("Production mode: Containers will install osprey from PyPI")
        return False
    if not os.path.isfile(os.path.join(out_dir, "Dockerfile")):
        # Routine and correct: pure-image services (postgresql, openobserve)
        # never install osprey. DEBUG, not INFO — at INFO these lines outnumber
        # the ones that carry a decision and bury them.
        logger.debug(
            f"Development mode: build context {out_dir} has no Dockerfile, "
            "so no osprey wheel was staged for it"
        )
        return False
    # Raises DevModeUnavailableError rather than returning False when --dev
    # cannot be honored; the deploy stops instead of silently deploying the
    # pinned PyPI release under a flag that means "run my local code".
    _copy_local_framework_for_override(out_dir)
    logger.debug("Development mode: Osprey override prepared")
    return True


def render_kernel_templates(source_dir, config, out_dir):
    """Process all Jupyter kernel templates in a service directory.

    This function provides batch processing for Jupyter kernel configuration
    templates, automatically discovering all kernel.json.j2 files within a
    service directory and rendering them with the current configuration context.
    This is particularly useful for Jupyter services that provide multiple
    kernel environments with different configurations.

    The function recursively searches the source directory for kernel template
    files and processes each one, maintaining the relative directory structure
    in the output. This ensures that kernel configurations are placed in the
    correct locations for Jupyter to discover them.

    :param source_dir: Source directory to search for kernel templates
    :type source_dir: str
    :param config: Configuration dictionary for template rendering
    :type config: dict
    :param out_dir: Base output directory for rendered kernel files
    :type out_dir: str

    Examples:
        Kernel template processing for Jupyter service::

            >>> # Source structure:
            >>> # services/jupyter/
            >>> #   ├── python3-epics-readonly/kernel.json.j2
            >>> #   └── python3-epics-write/kernel.json.j2
            >>>
            >>> render_kernel_templates(
            ...     'services/jupyter',
            ...     {'project_root': '/home/user/project'},
            ...     'build/services/jupyter'
            ... )
            >>> # Output structure:
            >>> # build/services/jupyter/
            >>> #   ├── python3-epics-readonly/kernel.json
            >>> #   └── python3-epics-write/kernel.json

    .. note::
       This function is typically called automatically by setup_build_dir when
       a service configuration includes 'render_kernel_templates: true'.

    .. seealso::
       :func:`render_template` : Core template rendering used by this function
       :func:`setup_build_dir` : Calls this function for kernel template processing
    """
    kernel_templates = []

    # Look for kernel.json.j2 files in subdirectories
    for root, _dirs, files in os.walk(source_dir):
        for file in files:
            if file == "kernel.json.j2":
                template_path = os.path.relpath(os.path.join(root, file), os.getcwd())
                kernel_templates.append(template_path)

    # Render each kernel template
    for template_path in kernel_templates:
        # Calculate relative output directory
        rel_template_dir = os.path.dirname(os.path.relpath(template_path, source_dir))
        kernel_out_dir = (
            os.path.join(out_dir, rel_template_dir) if rel_template_dir != "." else out_dir
        )

        render_template(template_path, config, kernel_out_dir)
        logger.debug(f"Rendered kernel template: {template_path} -> {kernel_out_dir}/kernel.json")


#: Bits a shared corpus directory must carry: ``rwxrws---``, ORed onto whatever
#: mode the directory already has (see :func:`ensure_shared_corpus_dir` — this
#: is a floor, not a replacement, so an operator's narrower `other` triad is
#: preserved).
#:
#: The ``s`` is the whole point. A **setgid** directory hands every file and
#: subdirectory created inside it the DIRECTORY's group, not the creating
#: process's primary group — so a concept drafted from ``web-alice``, a concept
#: drafted from ``web-bob`` and the bundle the operator wrote on the host all
#: end up group-OWNED by one group, whatever uid each writer ran as.
#:
#: Ownership is only half of it: setgid does not make any container process a
#: MEMBER of that group. The other half is the ``group_add:`` the compose
#: renderers emit from the gid :func:`ensure_shared_corpus_dir` returns. Group
#: ownership without membership grants nothing.
#:
#: Group-write then lets a second container edit or replace what the first one
#: wrote, which is what "shared corpus" has to mean for a multi-user roster.
#: Nothing is granted to ``other``: the qmd sidecar reads as root and the web
#: terminals come in through their group, so a world-readable corpus would widen
#: access to a facility's knowledge for no consumer that needs it.
SHARED_CORPUS_DIR_MODE = 0o2770


def resolve_facility_bundle_dir(config, repo_root=None):
    """The deployment's facility-knowledge bundle on the host, or ``None``.

    One reader for ``facility_knowledge.bundle_path``, so the directory that is
    provisioned, the one the qmd sidecar indexes and the one bound into each
    entitled web terminal are provably the same directory. A relative value
    anchors on the deployment repo root, matching how every other configured
    path in a rendered compose file resolves.

    :param config: Configuration dictionary
    :type config: dict
    :param repo_root: Deployment repo root; ``None`` resolves it from *config*
    :type repo_root: str | pathlib.Path | None
    :return: Absolute path to the bundle, or ``None`` when none is configured
    :rtype: pathlib.Path | None
    """
    knowledge = (config or {}).get("facility_knowledge")
    raw = knowledge.get("bundle_path") if isinstance(knowledge, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw.strip()).expanduser()
    if path.is_absolute():
        return path
    root = Path(repo_root) if repo_root is not None else Path(resolve_repo_root(config))
    return root / path


def shared_corpus_gid(path):
    """The group a container must join to share *path*, or ``None``.

    A pure read, deliberately separate from :func:`ensure_shared_corpus_dir`:
    the renderers need this number to emit ``group_add:``, and they must not
    acquire a side effect on the filesystem to get it. The deploy path
    provisions the directory first; every later render just asks what group it
    ended up with.

    ``None`` — the directory does not exist yet, or the platform reports no gid
    — means no ``group_add`` is emitted. That is the honest render: joining a
    group that was never established would be a guess, and the mount still works
    wherever the container's own uid already has access.

    :param path: The corpus directory, or ``None``
    :type path: str | pathlib.Path | None
    :return: The directory's group id, or ``None``
    :rtype: int | None
    """
    if path is None:
        return None
    try:
        return Path(path).expanduser().stat().st_gid
    except (OSError, AttributeError):
        return None


def _display_path(target, relative_to):
    """Spell *target* for a log line: relative to *relative_to* when it can be.

    Absolute paths in the default INFO view are a hygiene problem, not a
    cosmetic one — each one wraps a normal terminal and pushes the line that
    matters off the screen, which is why the build guards its output view down
    to a single one. A directory outside *relative_to* (or an unset root) has no
    relative spelling and is named absolutely rather than mangled.

    :param target: Path to display
    :type target: pathlib.Path
    :param relative_to: Root to spell it against, or ``None``
    :type relative_to: str | pathlib.Path | None
    :return: The path as it should appear in a log line
    :rtype: str
    """
    if relative_to is None:
        return str(target)
    try:
        return Path(target).relative_to(Path(relative_to).expanduser()).as_posix()
    except ValueError:
        return str(target)


def ensure_shared_corpus_dir(path, relative_to=None):
    """Create a shared corpus directory group-writable, and report its GID.

    Called on the deploy path before any bind mount is created, for the
    directories several writers share: the facility-knowledge bundle (bound into
    every entitled web-terminal container read-write, and into the qmd sidecar
    read-only) and any other corpus the sidecar indexes.

    Two failures this prevents, in order of how often they bite:

    1. **Root-owned mount source.** Left to the container runtime, a rootful
       daemon creates a missing bind source owned by root. The host operator who
       authors the bundle can then no longer write it, and neither can a
       container that does not run as root. Pre-creating it here means the
       directory always exists, owned by whoever ran the deploy.
    2. **Cross-container invisibility.** See :data:`SHARED_CORPUS_DIR_MODE` —
       the setgid bit is what gives every writer's output one common group.

    GID STRATEGY, stated rather than implied, in two halves:

    * **Host side — the group is inherited, never invented.** This sets the
      setgid *mode*; it does not ``chown`` the directory to a numeric GID. A
      hardcoded GID is wrong on every host but the one it was chosen on, and
      changing a directory's group needs privileges the deploy path is not
      guaranteed to have. The shared group is therefore whatever group the
      deploying user's directory already carries, retargetable by an operator
      with one out-of-band ``chgrp`` and no code change.
    * **Container side — membership is granted explicitly.** setgid confers
      group OWNERSHIP of newly created files; it does NOT make any process a
      MEMBER of that group, and a container's process carries only the groups
      its image's ``/etc/group`` gives it. The returned gid is therefore emitted
      as ``group_add:`` on every service that mounts the corpus (see
      :func:`osprey.deployment.web_terminals.render.render_web_terminals`),
      which is what actually gives those containers access. Without that half,
      access would hold only where the deploying user's uid happens to match the
      image's — an accident, not a property.

    Known limit, not worked around here: setgid fixes group ownership, while the
    permission BITS on each new file still come from the writing process's
    umask. A container writing under the common ``022`` creates ``rw-r--r--``
    files — group-readable, which is all the sidecar's indexing and another
    container's reading need, but not group-writable. Making one container able
    to overwrite another's individual files would need ``umask 002`` inside the
    image, which is an image property and not this function's to set.

    :param path: The corpus directory to provision
    :type path: str | pathlib.Path
    :param relative_to: Root to spell the directory against in the INFO line
        below. The default view carries exactly one absolute path — the tree the
        build wrote — and a second one wraps a normal terminal and buries it, so
        this line names ``data/facility_knowledge`` rather than 90 characters of
        ``/private/var/folders/...``. Affects the message only; the directory
        acted on is always *path*.
    :type relative_to: str | pathlib.Path | None
    :return: The directory's group id after provisioning, or ``None`` when the
        platform does not report one (Windows) or the directory could not be
        provisioned. This is the gid the mounting services join.
    :rtype: int | None
    """
    target = Path(path).expanduser()
    existed = target.is_dir()
    try:
        target.mkdir(parents=True, exist_ok=True)
        current = stat.S_IMODE(target.stat().st_mode)
        # ADD the bits sharing needs; never replace the mode. An operator who
        # deliberately keeps a bundle at 0700 must not have it silently widened
        # to world-readable by a deploy — the corpus can hold sensitive facility
        # information, and neither consumer needs the `other` triad anyway (the
        # sidecar runs as root; the web containers come in through group_add).
        # A directory this call CREATED has no operator intent to preserve, so it
        # starts at exactly the bits sharing needs and nothing wider.
        wanted = (current | SHARED_CORPUS_DIR_MODE) if existed else SHARED_CORPUS_DIR_MODE
        if wanted != current:
            os.chmod(target, wanted)
            if existed:
                # INFO, not DEBUG: this changed permissions on a directory the
                # operator already had, which they are entitled to see reported
                # rather than discover later. Named RELATIVELY and kept to one
                # unwrapped line — the default view already carries the one
                # absolute path worth printing (where the build landed), and a
                # second one pushes it off the screen. The full reasoning lives
                # in this function's docstring, where someone investigating the
                # change will look.
                logger.info(
                    f"Shared corpus {_display_path(target, relative_to)}: "
                    f"{current:04o} -> {wanted:04o} (setgid, group-writable)"
                )
    except OSError as e:
        # Never fatal. A corpus on a filesystem that refuses setgid (many
        # network mounts) or a directory owned by someone else still deploys —
        # single-user sharing works regardless, and the multi-user case fails
        # visibly at read time rather than by refusing the whole deploy here.
        logger.warning(
            f"Could not provision shared corpus directory {target}: {e}. "
            "Files written from one container may not be readable by another "
            "or indexable by the qmd sidecar."
        )
        return None
    gid = getattr(target.stat(), "st_gid", None)
    logger.debug(f"Provisioned shared corpus directory {target} (gid {gid})")
    return gid


def render_service_templates(source_dir, config, out_dir):
    """Render a service's non-compose ``.j2`` siblings into its build context.

    The build directory copy deliberately skips every ``.j2`` file, because
    templates are inputs and a container must never receive one unrendered. That
    leaves a service whose container needs a *rendered* config file — the qmd
    sidecar's ``index.yml``, which declares the collections it indexes — with no
    way to produce it. This closes that: any ``.j2`` sitting beside the service's
    ``docker-compose.yml.j2`` is rendered with the same context, dropping the
    extension (``index.yml.j2`` -> ``index.yml``).

    Only the service's own directory is scanned, not its subdirectories:
    ``kernel.json.j2`` files live one level down and belong to
    :func:`render_kernel_templates`, which is opt-in per service and would
    otherwise render them twice.

    :param source_dir: The service's template directory
    :type source_dir: str
    :param config: Configuration dictionary for template rendering
    :type config: dict
    :param out_dir: The service's build context directory
    :type out_dir: str
    :return: Paths of the rendered files, in the order they were rendered
    :rtype: list[str]
    """
    rendered = []
    for name in sorted(os.listdir(source_dir)):
        if not name.endswith(".j2") or name in (TEMPLATE_FILENAME, "kernel.json.j2"):
            continue
        template_path = os.path.relpath(os.path.join(source_dir, name), os.getcwd())
        rendered.append(render_template(template_path, config, out_dir))
        logger.debug(f"Rendered service template: {template_path} -> {out_dir}/{name[:-3]}")
    return rendered


def _ensure_agent_data_structure(config):
    """Ensure the agent-data directory and subdirectories exist before deployment.

    This function creates the agent data directory structure based on the configuration
    to prevent Docker/Podman mount failures when containers try to mount non-existent
    directories. The root comes from ``agent_data.base_dir`` — ``var/agent_data``
    under the repo root unless a profile moved it; each subdirectory below
    is created only when ``file_paths`` declares its name, so the created structure
    matches what :func:`osprey.utils.config.get_agent_dir` resolves at runtime. The
    scenario state directory is the one addition outside ``file_paths``, created only
    for a Virtual Accelerator deployment — the one compose service that mounts it.

    :param config: Configuration dictionary containing agent_data and file_paths settings
    :type config: dict
    """
    from osprey.connectors.types import VIRTUAL_ACCELERATOR
    from osprey.utils.workspace import agent_data_base_dir, resolve_simulation_state_dir

    # Get file paths configuration
    file_paths = config.get("file_paths", {})
    project_root = config.get("project_root", ".")

    # Create main agent data directory
    agent_data_path = Path(project_root) / agent_data_base_dir(config)
    agent_data_path.mkdir(parents=True, exist_ok=True)

    # Create all configured subdirectories
    subdirs = [
        "registry_exports_dir",
    ]

    for subdir_key in subdirs:
        if subdir_key in file_paths:
            subdir_name = file_paths[subdir_key]
            subdir_path = agent_data_path / subdir_name
            subdir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created agent data subdirectory: {subdir_path}")

    # The Virtual Accelerator's compose service bind-mounts the scenario state
    # directory read-only. Pre-create it for the same reason as the root above:
    # left to the container runtime, a rootful daemon creates the missing mount
    # source root-owned, and `osprey sim apply` on the host — which owns that
    # file — can then no longer write it.
    # `or []` rather than a default: a present-but-null `deployed_services:` is a
    # normal hand-edited shape, and `get(..., [])` returns None for it.
    #
    # Resolved through the same helper the mount source is rendered from, not by
    # joining the agent-data root here: the two agree only while
    # `simulation.state_dir` is unset, and a project that sets it would have the
    # directory pre-created in one place and mounted from another — leaving the
    # real mount source to the container runtime, which is exactly the
    # root-owned-directory failure this block exists to prevent.
    if VIRTUAL_ACCELERATOR in {str(name) for name in config.get("deployed_services") or []}:
        state_path = resolve_simulation_state_dir(config, Path(project_root))
        state_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created scenario state directory: {state_path}")

    # The facility-knowledge bundle, for the same root-owned-mount-source reason
    # as the scenario state directory above — the qmd sidecar binds it READ-ONLY
    # and so could never create it — plus the shared-group mode the multi-user
    # case needs (see :func:`ensure_shared_corpus_dir`). Provisioned here rather
    # than only on the web-terminal path, because a single-user deployment
    # mounts the same directory into the sidecar and must not be left to the
    # container runtime either.
    bundle_dir = resolve_facility_bundle_dir(config, project_root)
    if bundle_dir is not None:
        ensure_shared_corpus_dir(bundle_dir, relative_to=project_root)

    logger.debug(f"Ensured agent data structure exists at: {agent_data_path}")


def setup_build_dir(template_path, config, container_cfg, dev_mode=False):
    """Create complete build environment for service deployment.

    This function orchestrates the complete build directory setup process for
    a service, including template rendering, source code copying, configuration
    flattening, and additional directory management. It creates a self-contained
    build environment that contains everything needed for container deployment.

    The build process follows these steps:
    1. Create clean build directory for the service
    2. Render the Docker Compose template with configuration context
    3. Copy service-specific files (excluding templates)
    4. Copy source code if requested (copy_src: true)
    5. Copy additional directories as specified
    6. Create flattened configuration file for container use
    7. Process kernel templates if specified

    Source code copying includes intelligent handling of requirements files,
    automatically copying global requirements.txt to the container source
    directory to ensure dependency management works correctly in containers.

    :param template_path: Path to the service's Docker Compose template
    :type template_path: str
    :param config: Complete configuration dictionary for template rendering
    :type config: dict
    :param container_cfg: Service-specific configuration settings
    :type container_cfg: dict
    :param dev_mode: Development mode - copy local framework to containers
    :type dev_mode: bool
    :return: Path to the rendered Docker Compose file
    :rtype: str

    Examples:
        Basic service build directory setup::

            >>> container_cfg = {
            ...     'copy_src': True,
            ...     'additional_dirs': ['docs', 'scripts'],
            ...     'render_kernel_templates': False
            ... }
            >>> compose_path = setup_build_dir(
            ...     'services/osprey/jupyter/docker-compose.yml.j2',
            ...     config,
            ...     container_cfg
            ... )
            >>> print(compose_path)  # 'build/services/osprey/jupyter/docker-compose.yml'

        Advanced service with custom directory mapping::

            >>> container_cfg = {
            ...     'copy_src': True,
            ...     'additional_dirs': [
            ...         'docs',  # Simple directory copy
            ...         {'src': 'external_data', 'dst': 'data'}  # Custom mapping
            ...     ],
            ...     'render_kernel_templates': True
            ... }
            >>> compose_path = setup_build_dir(template_path, config, container_cfg)

    .. note::
       The function automatically handles build directory cleanup, removing
       existing directories to ensure clean builds. Global requirements.txt
       is automatically copied to container source directories when present.

    .. warning::
       This function performs destructive operations on build directories.
       Ensure build_dir is properly configured to avoid data loss.

    .. seealso::
       :func:`render_template` : Template rendering used by this function
       :func:`render_kernel_templates` : Kernel template processing
       :class:`configs.config.ConfigBuilder` : Configuration flattening
    """
    # Create the build directory for this service
    source_dir = os.path.relpath(os.path.dirname(template_path), os.getcwd())

    # Extract service name from the path for container path resolution
    # e.g., "services/jupyter" -> "jupyter", "src/osprey/templates/services/pipelines" -> "pipelines"
    os.path.basename(source_dir)

    # Clear the directory if it exists
    build_dir = config.get("build_dir", "./build")
    out_dir = os.path.join(build_dir, source_dir)

    # ...unless the output IS the source. A deployment repo's build renders into
    # a staging tree that becomes `build/`, so its output base is that tree's
    # root and each service's rendered `docker-compose.yml` lands beside the
    # `docker-compose.yml.j2` it came from. Clearing the directory there would
    # delete the templates this render is about to read — and there is nothing
    # to clear in the first place, because the staged tree was created fresh by
    # this same build. The clear exists for the other shape, where a persistent
    # output directory can hold a previous render's stale files.
    renders_in_place = os.path.realpath(out_dir) == os.path.realpath(source_dir)

    if os.path.exists(out_dir) and not renders_in_place:
        try:
            shutil.rmtree(out_dir)
        except OSError as e:
            if (
                "Device or resource busy" in str(e) or "nfs" in str(e).lower() or e.errno == 39
            ):  # Directory not empty
                logger.warning(f"Directory in use, attempting incremental update for {out_dir}")
                import time

                time.sleep(1)
                try:
                    shutil.rmtree(out_dir)
                except OSError:
                    logger.warning(f"Could not remove {out_dir}, using incremental update approach")
                    # Use incremental update instead of full rebuild
                    return _incremental_setup_build_dir(
                        template_path, config, container_cfg, out_dir, dev_mode
                    )
            else:
                raise
    os.makedirs(out_dir, exist_ok=True)

    # Copy the service's own files (except templates) and stage the dev wheel
    # BEFORE rendering the compose template: the render context's dev_mode flag
    # is what service templates turn into the OSPREY_DEV=1 build arg, and that
    # pin-relaxing arg must only be emitted for a context that actually
    # received a wheel (see _stage_dev_wheel_for_context).
    wheel_staged = False
    if source_dir != SERVICES_DIR:  # ignore the top level dir
        # Deep copy everything in source directory except templates. Skipped
        # when the output IS the source (see `renders_in_place` above): the
        # files are already at the destination, and copying a file onto itself
        # raises rather than no-opping.
        for file in [] if renders_in_place else os.listdir(source_dir):
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(out_dir, file)
            # Skip template files (both docker-compose and kernel templates)
            if file != TEMPLATE_FILENAME and not file.endswith(".j2"):
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)

        # Copy local osprey for development override (only in dev mode).
        # This will override the PyPI osprey after standard installation.
        wheel_staged = _stage_dev_wheel_for_context(out_dir, dev_mode)

    # Create the docker compose file from the template. The render context
    # carries dev_mode so service templates can emit dev-only build args
    # (e.g. OSPREY_DEV) exactly when `osprey up --dev` runs AND the
    # local wheel actually landed in this context — never when staging failed
    # (fail-closed: the Dockerfile then keeps its fail-loud pinned install).
    render_config = {**config, "dev_mode": dev_mode and wheel_staged}
    compose_filepath = render_template(template_path, render_config, out_dir)

    if source_dir != SERVICES_DIR:  # ignore the top level dir
        # Rendered config files the container mounts (the qmd sidecar's
        # index.yml). Rendered from the same context as the compose file, so a
        # mount the compose fragment declares and the file it points at are
        # generated from one set of values.
        render_service_templates(source_dir, render_config, out_dir)

        # Copy the source directory
        if container_cfg.get("copy_src", False):
            shutil.copytree(SRC_DIR, os.path.join(out_dir, OUT_SRC_DIR))

            # Copy global requirements.txt to repo_src if it exists
            # This handles consolidated requirements files
            global_requirements = "requirements.txt"
            if os.path.exists(global_requirements):
                repo_src_requirements = os.path.join(out_dir, OUT_SRC_DIR, "requirements.txt")
                shutil.copy2(global_requirements, repo_src_requirements)
                logger.debug(f"Copied global requirements.txt to {repo_src_requirements}")

            # Copy project's pyproject.toml to repo_src
            # Note: This is the user's project pyproject.toml, not framework's
            global_pyproject = "pyproject.toml"
            if os.path.exists(global_pyproject):
                repo_src_pyproject = os.path.join(out_dir, OUT_SRC_DIR, "pyproject_user.toml")
                shutil.copy2(global_pyproject, repo_src_pyproject)
                logger.debug(f"Copied user pyproject.toml to {repo_src_pyproject}")

        # Copy additional directories if specified in service configuration
        additional_dirs = container_cfg.get("additional_dirs", [])
        if additional_dirs:
            for dir_spec in additional_dirs:
                if isinstance(dir_spec, str):
                    # Simple string: copy directory with same name
                    src_dir = dir_spec
                    dst_dir = os.path.join(out_dir, dir_spec)
                elif isinstance(dir_spec, dict):
                    # Dictionary: allows custom source -> destination mapping
                    src_dir = dir_spec.get("src")
                    dst_dir = os.path.join(out_dir, dir_spec.get("dst", src_dir))
                else:
                    continue

                if src_dir and os.path.exists(src_dir):
                    # Handle both files and directories
                    if os.path.isfile(src_dir):
                        # For files, create parent directory and copy file
                        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
                        shutil.copy2(src_dir, dst_dir)
                        logger.debug(f"Copied file {src_dir} to {dst_dir}")
                    elif os.path.isdir(src_dir):
                        # For directories, use copytree
                        shutil.copytree(src_dir, dst_dir)
                        logger.debug(f"Copied directory {src_dir} to {dst_dir}")
                elif src_dir:
                    logger.warning(f"Path {src_dir} does not exist, skipping")

        # Ensure the agent-data directory structure exists before container deployment
        # This prevents mount failures when containers try to mount non-existent directories
        _ensure_agent_data_structure(config)

        # Create flattened configuration file for container
        # This merges all imports and creates a complete config without import directives
        # SECURITY: Use unexpanded config to prevent API keys from being written to disk
        # The container will expand ${VAR} placeholders at runtime from environment variables
        try:
            with quiet_logger(["registry", "CONFIG"]):
                global_config = ConfigBuilder()
                # Preserves ${VAR} placeholders - secrets not written to disk
                flattened_config = global_config.get_unexpanded_config()

            # Adjust paths for container environment
            # In containers, src/ is copied to repo_src/, so config paths must be updated

            def adjust_src_paths_recursive(obj):
                """Recursively adjust all src/ paths in config for container environment.

                When deploying to containers, the deployment system copies src/ -> repo_src/.
                Any config values that are paths starting with 'src/' must be updated to
                'repo_src/' to work correctly in the container environment.

                Args:
                    obj: Config dictionary or list to process
                """
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, str):
                            if value.startswith("src/"):
                                obj[key] = f"repo_src/{value[4:]}"
                                logger.debug(f"Container path adjustment: {value} -> {obj[key]}")
                            elif value.startswith("./src/"):
                                obj[key] = f"./repo_src/{value[6:]}"
                                logger.debug(f"Container path adjustment: {value} -> {obj[key]}")
                        elif isinstance(value, (dict, list)):
                            adjust_src_paths_recursive(value)
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, (dict, list)):
                            adjust_src_paths_recursive(item)

            # Recursively adjust all src/ paths in the config
            adjust_src_paths_recursive(flattened_config)

            config_yml_dst = os.path.join(out_dir, "config.yml")
            with open(config_yml_dst, "w") as f:
                yaml.dump(flattened_config, f, default_flow_style=False, sort_keys=False)
            logger.debug(f"Created flattened config.yml at {config_yml_dst}")
        except Exception as e:
            logger.warning(f"Failed to create flattened config: {e}")
            # Fall back to copying the original config verbatim, so this
            # degraded path never blocks the build.
            config_yml_src = "config.yml"
            if os.path.exists(config_yml_src):
                config_yml_dst = os.path.join(out_dir, "config.yml")
                shutil.copy2(config_yml_src, config_yml_dst)
                logger.debug(f"Copied original config.yml to {config_yml_dst}")

        # Render kernel templates if specified in service configuration
        if container_cfg.get("render_kernel_templates", False):
            logger.debug(f"Processing kernel templates for {source_dir}")
            render_kernel_templates(source_dir, config, out_dir)

    return compose_filepath


def _incremental_setup_build_dir(template_path, config, service_config, out_dir, dev_mode=False):
    """Setup build directory using incremental updates when full cleanup fails.

    This fallback function handles cases where the build directory cannot be
    completely removed due to file locks or NFS issues. It updates files
    incrementally instead of doing a full rebuild.

    Args:
        template_path (str): Path to the docker-compose template file
        config (dict): Configuration dictionary
        service_config (dict): Service-specific configuration
        out_dir (str): Output directory path that couldn't be cleaned

    Returns:
        str: Path to the rendered docker-compose.yml file
    """
    source_dir = os.path.relpath(os.path.dirname(template_path), os.getcwd())

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Copy/update files from source directory (skip if top-level services dir),
    # then stage the dev wheel — BEFORE rendering, exactly as in
    # setup_build_dir: the rendered OSPREY_DEV build arg must reflect whether a
    # wheel actually landed in this context.
    wheel_staged = False
    if source_dir != SERVICES_DIR:
        for file in os.listdir(source_dir):
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(out_dir, file)

            # Skip template files
            if file != TEMPLATE_FILENAME and not file.endswith(".j2"):
                try:
                    if os.path.isdir(src_path):
                        # For directories, use copytree with dirs_exist_ok (Python 3.8+)
                        if (
                            hasattr(shutil, "copytree")
                            and "dirs_exist_ok" in shutil.copytree.__code__.co_varnames
                        ):
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        else:
                            # Fallback for older Python versions
                            if not os.path.exists(dst_path):
                                shutil.copytree(src_path, dst_path)
                    else:
                        shutil.copy2(src_path, dst_path)
                except (OSError, shutil.Error) as e:
                    logger.warning(f"Could not update {dst_path}: {e}")

        wheel_staged = _stage_dev_wheel_for_context(out_dir, dev_mode)

    # Create/update the docker compose file from the template (dev_mode gated
    # on staging success, exactly as in setup_build_dir).
    render_config = {**config, "dev_mode": dev_mode and wheel_staged}
    compose_filepath = render_template(template_path, render_config, out_dir)

    # Same rendered-config step as the full path (see setup_build_dir): a
    # service whose container mounts a rendered file must get it here too, or
    # the incremental fallback silently produces a build context the compose
    # fragment's bind mounts point into and find nothing.
    if source_dir != SERVICES_DIR:
        render_service_templates(source_dir, render_config, out_dir)

    # Handle source directory copying if needed
    if service_config.get("copy_src", False):
        src_dst_path = os.path.join(out_dir, OUT_SRC_DIR)
        try:
            if (
                hasattr(shutil, "copytree")
                and "dirs_exist_ok" in shutil.copytree.__code__.co_varnames
            ):
                shutil.copytree(SRC_DIR, src_dst_path, dirs_exist_ok=True)
            else:
                if not os.path.exists(src_dst_path):
                    shutil.copytree(SRC_DIR, src_dst_path)
        except (OSError, shutil.Error) as e:
            logger.warning(f"Could not update source directory {src_dst_path}: {e}")

    return compose_filepath


def find_existing_compose_files(config, deployed_services, quiet=False, base=None):
    """Find existing compose files without rebuilding directories.

    This function locates existing docker-compose.yml files in the build directory
    for the specified services without triggering any rebuild operations.

    Both the config's ``build_dir`` and each service's declared ``path`` are
    relative, so the answer depends entirely on what they are resolved against.
    That anchor is *base*, an explicit argument rather than the working
    directory, because from the wrong directory this function does not fail — it
    returns an empty list, and "no compose files here" is indistinguishable from
    "this deployment declares no services". Callers that hold a repo root pass
    it; the ``os.getcwd()`` default is for a caller standing in the repo already.

    Args:
        config (dict): Configuration dictionary containing build_dir
        deployed_services (list): List of service names to find compose files for
        quiet (bool): If True, suppress warning messages about missing files
        base (str | Path | None): Directory the relative paths resolve against
            — the repo root. Defaults to the current working directory.

    Returns:
        list: Paths to the existing compose files, RELATIVE to *base* (which is
        the anchor :func:`compose_base_cmd` resolves each ``-f`` against).

    Example:
        compose_files = find_existing_compose_files(config, ['osprey.jupyter'])
        # Returns: ['./build/services/docker-compose.yml',
        #          './build/services/osprey/jupyter/docker-compose.yml']
    """
    compose_files = []
    base = Path(base) if base is not None else Path.cwd()
    build_dir = config.get("build_dir", "./build")

    # Add top-level compose file if it exists
    top_compose = os.path.join(build_dir, SERVICES_DIR, "docker-compose.yml")
    if (base / top_compose).exists():
        compose_files.append(top_compose)

    # Add service-specific compose files
    for service_name in deployed_services:
        service_config, template_path = find_service_config(config, service_name)
        if template_path:
            # Construct expected compose file path. A declared service path is
            # normally relative and already base-anchored, so only an absolute
            # one has to be made relative to the base.
            template_dir = os.path.dirname(template_path)
            source_dir = (
                os.path.relpath(template_dir, base)
                if os.path.isabs(template_dir)
                else os.path.normpath(template_dir)
            )
            compose_path = os.path.join(build_dir, source_dir, "docker-compose.yml")
            if (base / compose_path).exists():
                compose_files.append(compose_path)
            elif not quiet:
                logger.warning(
                    f"Compose file not found for service '{service_name}' at {compose_path}"
                )

    return compose_files


def clean_deployment(compose_files, config=None, repo_root=None):
    """Clean up containers, images, volumes, and networks for a fresh deployment.

    This function provides comprehensive cleanup capabilities for container
    deployments, removing containers, images, volumes, and networks to enable
    fresh rebuilds. It's particularly useful when configuration changes require
    complete environment reconstruction.

    :param compose_files: List of Docker Compose file paths for the deployment
    :type compose_files: list[str]
    :param config: Optional configuration dictionary for runtime detection
    :type config: dict, optional
    :param repo_root: The deployment repo this clean acts on, when the caller
        already resolved one (it holds the config *path*, which this function is
        not given). ``None`` resolves it from ``config`` alone, whose last
        resort is the working directory.
    :type repo_root: str or pathlib.Path, optional
    :raises UnsupportedComposeProviderError: The host's compose provider is not
        one OSPREY can invoke correctly. A clean that guessed the shape would
        leave the containers and volumes it claimed to remove.
    """
    # No announcement here: the two steps below name what is being removed as
    # they remove it, which is the same fact with an outcome attached.

    # Deferred for the same reason compose_base_cmd defers it: the lifecycle
    # module owns the probe seam and the merged env file, and it imports this one.
    from osprey.deployment.container_lifecycle import _compose_provider, _env_file_args

    # Both invocations below share the one pinned base (TR-3): the same repo
    # root, the same repo-anchored -f list, and the same env chain — which is
    # also what stops a `--env-file .env` from being resolved against whatever
    # directory the operator happened to stand in. Resolved before the probe so
    # both halves of the provider-shaped invocation are anchored on it.
    repo_root = Path(repo_root) if repo_root is not None else resolve_repo_root(config)
    provider = _compose_provider(config)
    env_file_args = _env_file_args(repo_root, provider)

    # Pin COMPOSE_PROJECT_NAME so `down --volumes` targets THIS deploy's project
    # and volumes; unpinned it derives the shared "services" project and would
    # destroy a sibling deploy's data volumes. The provider overlay rides in on
    # the same env: for the shape that cannot take the project directory in
    # argv, this is where it is pinned instead — and a `down` addressed at the
    # wrong project directory removes nothing while reporting success.
    run_env = runtime_env(config, {**os.environ, **compose_provider_env(provider, repo_root)})

    # Stop and remove containers, networks, volumes
    cmd_down = compose_base_cmd(
        get_runtime_command(config), compose_files, repo_root, env_file_args, provider
    )
    cmd_down.extend(["down", "--volumes", "--remove-orphans"])

    logger.debug(f"Running: {' '.join(cmd_down)}")
    report_step("Removing containers and volumes")
    # No check: a clean over a deployment that was never up exits non-zero, and
    # that has always been tolerated here. Capturing must not turn it into a
    # failure — only move the output off the terminal.
    run_captured(
        cmd_down, env=run_env, spool_name="compose-clean-down", repo_root=repo_root, check=False
    )

    # Remove images built by the compose files
    cmd_rmi = compose_base_cmd(
        get_runtime_command(config), compose_files, repo_root, env_file_args, provider
    )
    cmd_rmi.extend(["down", "--rmi", "all"])

    logger.debug(f"Running: {' '.join(cmd_rmi)}")
    report_step("Removing images")
    run_captured(
        cmd_rmi, env=run_env, spool_name="compose-clean-rmi", repo_root=repo_root, check=False
    )

    logger.debug("Cleanup completed")


def prepare_compose_files(config_path, dev_mode=False, expose_network=False, output_root=None):
    """Prepare compose files from configuration.

    Loads configuration and generates all necessary compose files for deployment.

    :param config_path: Path to the configuration file
    :type config_path: str
    :param dev_mode: Development mode - copy local framework to containers
    :type dev_mode: bool
    :param expose_network: Expose services to all network interfaces (0.0.0.0)
    :type expose_network: bool
    :param output_root: Directory this run writes its output under, replacing
        the config's ``build_dir`` for the duration and for this run only. The
        config on disk is never rewritten.

        ``None`` — the deploy-time case — writes where the config says, which is
        ``build_dir`` resolved against the repo root.

        A deployment repo's ``osprey build`` needs the other case. It renders
        into a staging tree that BECOMES ``build/`` when the atomic swap lands,
        so the shipped config correctly says ``build_dir: ./build`` (read from
        the repo root, where compose will read it) while this render has to
        write at the staging root itself. Taking the config's value there
        appended a second ``build/`` to a path that was already the output zone,
        putting every compose file one directory below where its own rendered
        mount paths — spelled against the repo root — resolve.

        Deliberately an output-base override rather than a change to what
        ``build_dir`` MEANS: the shipped value stays relative and is still
        absolutized only at invocation time (TR-3), so nothing about the pinned
        compose contract moves.
    :type output_root: str or None
    :return: Tuple of (config dict, list of compose file paths)
    :rtype: tuple[dict, list[str]]
    :raises RuntimeError: If configuration loading fails
    """
    # Fail before any work when --dev cannot be honored: every precondition is
    # a path check away, so there is no reason to surface it seven services in.
    if dev_mode:
        preflight_dev_mode()

    config = load_project_config(config_path, wrap_errors=True)

    # Where THIS render writes, when that differs from where the finished
    # deployment will read. Applied to the in-memory config only — the config on
    # disk keeps its own ``build_dir``, so what is shipped still says where the
    # compose files live relative to the repo root, and only this run's output
    # base moves. See the parameter's docstring for the case that needs it.
    if output_root is not None:
        config["build_dir"] = output_root

    # Handle network exposure setting
    # Default to localhost-only binding for security (Issue #126)
    if "deployment" not in config:
        config["deployment"] = {}
    if expose_network:
        config["deployment"]["bind_address"] = "0.0.0.0"
        logger.warning(
            "Network exposure enabled: services will bind to 0.0.0.0 (all interfaces). "
            "Ensure proper authentication is configured!"
        )
    elif "bind_address" not in config.get("deployment", {}):
        config["deployment"]["bind_address"] = "127.0.0.1"
        _report_fact("services bound to localhost only (127.0.0.1)")

    # Get deployed services list
    deployed_services = config.get("deployed_services", [])
    if deployed_services:
        # Not announced: the render phase's `compose files` step already says
        # this pass ran, and `osprey status` is where the list is read.
        deployed_service_names = [str(service) for service in deployed_services]
    else:
        logger.warning("No deployed_services list found, no services will be processed")
        deployed_service_names = []

    # Record which env-chain files this render found, beside the compose files
    # it explains. Written for every render, including one that deploys no
    # services at all: what the deploy checks is that the chain has not changed
    # since the render, and a deployment with no services still delivers the
    # chain to compose as ``--env-file`` flags. Probed through the same
    # ``resolve_repo_root`` the render context uses, so the record and the
    # rendered ``env_file:`` lists cannot disagree about which files were there.
    # ``resolve_repo_root(config)`` without the config path, deliberately: a
    # build renders from a staging tree, where the path resolves to the staging
    # root while the config's ``project_root`` still names the repo whose chain
    # the render is about — and the repo is the answer both need.
    record_env_chain_membership(
        config.get("build_dir", "./build"), env_chain_names(resolve_repo_root(config))
    )

    compose_files = []

    # Create the top level compose file. Skipped when no services are deployed:
    # an attached project (deploy_services: false) scaffolds no services/
    # templates at all, so rendering would throw — and there is nothing for the
    # top-level file to describe. deploy_up's empty-services guard (or its
    # web-terminals branch, which brings its own compose file) handles the rest.
    if deployed_service_names:
        top_template = os.path.join(SERVICES_DIR, TEMPLATE_FILENAME)
        build_dir = config.get("build_dir", "./build")
        out_dir = os.path.join(build_dir, SERVICES_DIR)
        top_template = render_template(top_template, config, out_dir)
        compose_files.append(top_template)

    # Create the service build directory for deployed services only
    for service_name in deployed_service_names:
        service_config, template_path = find_service_config(config, service_name)
        if service_config and template_path:
            if not os.path.isfile(template_path):
                raise RuntimeError(
                    f"Template file {template_path} not found for service '{service_name}'"
                )

            out = setup_build_dir(template_path, config, service_config, dev_mode)
            compose_files.append(out)
        else:
            raise RuntimeError(f"Service '{service_name}' not found in configuration")

    return config, compose_files
