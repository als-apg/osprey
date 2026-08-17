"""The container project layout: ``/app/<project>`` is a deployment repo.

An OSPREY container image carries a deployment, and the framework running
inside it resolves paths exactly the way it does on a host — there is no
container branch. That makes the layout a contract between three modules that
never import each other, which is why it is pinned here rather than left to the
docker-build e2e (``tests/e2e/test_dockerfile_e2e.py``, which cannot run in the
fast suite):

- ``build_cmd._render_container_project`` assembles the context the image is
  built from, and ``_stage_source_zone`` decides what travels in it;
- ``templates/project/Dockerfile.j2`` copies that context verbatim to
  ``/app/<project>``, so the context's layout IS the image's layout;
- ``registry/mcp.py`` renders every MCP server's ``OSPREY_CONFIG`` /
  ``CONFIG_FILE`` as ``{project_root}/build/config.yml``;
- ``cli/repo_resolver.find_repo_root`` — which the image's own ``CMD``
  (``osprey web``) goes through — accepts a directory as a deployment only when
  a ``profile.yml`` sits at or above it.

So the context has to be a deployment repo: ``profile.yml`` at its root, the
render one level down in ``build/``. A flat context whose root *is* the render
satisfies neither of the last two — its MCP servers name a ``build/config.yml``
that does not exist, and its ``CMD`` cannot find a deployment to serve at all.

The end-to-end proof is the docker-build e2e (``tests/e2e/test_dockerfile_e2e``),
which cannot run in the fast suite; these are the cheap pins that fail first.
"""

from __future__ import annotations

import json
import os
import re
from importlib import resources
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from jinja2 import Template

from osprey.cli.build_cmd import (
    _CONTAINER_INTERPRETER,
    _stage_source_zone,
    _write_image_context_dockerignore,
)
from osprey.cli.build_cmd import build as build_command
from osprey.cli.profile_root import PERSONA_DIRNAME
from osprey.cli.repo_resolver import PROFILE_FILENAME, RepoNotFoundError, find_repo_root
from osprey.registry.mcp import RENDERED_CONFIG_ENV_VALUE
from osprey.utils.workspace import (
    BUILD_DIR_NAME,
    IMAGE_DIR_NAME,
    RENDERED_CONFIG_RELPATH,
    STATE_DIR_NAME,
)

PROJECT_NAME = "layout-fixture"

#: ``COPY <src> <dest>``, with the project copy identified by its ``.`` source —
#: the other two COPYs in the template stage build-context files into /tmp.
_COPY_RE = re.compile(r"^COPY\s+\.\s+(\S+)\s*$", re.M)
_WORKDIR_RE = re.compile(r"^WORKDIR\s+(\S+)\s*$", re.M)


def _rendered_dockerfile(project_name: str = PROJECT_NAME) -> str:
    """Render the packaged project Dockerfile the way ``osprey build`` does."""
    template = resources.files("osprey").joinpath("templates/project/Dockerfile.j2")
    return Template(template.read_text(encoding="utf-8")).render(
        project_name=project_name,
        claude_code_cli_version="1.2.3",
    )


def _project_copy_target(dockerfile: str) -> str:
    targets = _COPY_RE.findall(dockerfile)
    assert len(targets) == 1, f"expected exactly one `COPY . <dest>`, found {targets}"
    return str(targets[0]).rstrip("/")


def _workdir(dockerfile: str) -> str:
    workdirs = _WORKDIR_RE.findall(dockerfile)
    assert len(workdirs) == 1, f"expected exactly one WORKDIR, found {workdirs}"
    return str(workdirs[0]).rstrip("/")


def test_the_image_is_its_build_context_at_the_project_root() -> None:
    """The Dockerfile copies its context verbatim, so context layout IS image layout.

    This is the whole reason the container contract is satisfiable without a
    single container-shaped branch anywhere: the image does not assemble a
    layout, it inherits one. Whatever shape ``osprey build`` gives the context,
    ``/app/<project>`` has — and the build gives it a deployment repo.

    Pinned because the alternative designs all show up here as a diverging COPY
    destination: a copy landing anywhere but the working directory would mean
    the image was rearranging its context, and every path the render recorded
    against ``/app/<project>`` would be off by that rearrangement.
    """
    dockerfile = _rendered_dockerfile()

    assert _project_copy_target(dockerfile) == _workdir(dockerfile), (
        "the project COPY must land exactly on the working directory — the "
        "image's project root is its context, unrearranged"
    )


def test_the_container_context_is_a_repo_whose_render_is_where_mcp_looks(tmp_path: Path) -> None:
    """The context the build assembles has the marker and the render in it.

    Two claims, and they are the two the container needs. ``profile.yml`` at the
    root is what lets ``osprey web`` — the image's own ``CMD`` — resolve the
    deployment it serves, since repo discovery has no container exception. And
    the render sitting at :data:`RENDERED_CONFIG_RELPATH` below that root is what
    makes ``registry/mcp.py``'s ``{project_root}/build/config.yml`` name a real
    file once the context lands at ``/app/<project>``.

    Driven through the real :func:`_stage_source_zone`, against a repo with a
    render already in it, so it is the production copier that has to keep the
    marker — not a fixture asserting its own handiwork.
    """
    repo = tmp_path / "facility"
    (repo / BUILD_DIR_NAME).mkdir(parents=True)
    (repo / PROFILE_FILENAME).write_text("name: facility\n", encoding="utf-8")
    (repo / RENDERED_CONFIG_RELPATH).write_text("project_root: /app/facility\n", encoding="utf-8")

    image_root = tmp_path / "image"
    (image_root / BUILD_DIR_NAME).mkdir(parents=True)
    (image_root / RENDERED_CONFIG_RELPATH).write_text("rendered\n", encoding="utf-8")
    _stage_source_zone(repo, image_root)

    project_root = _workdir(_rendered_dockerfile())
    assert (image_root / PROFILE_FILENAME).is_file(), (
        "the container context carries no repo marker, so the image's CMD "
        "cannot resolve the deployment it ships"
    )
    assert (image_root / RENDERED_CONFIG_RELPATH).is_file()
    assert RENDERED_CONFIG_ENV_VALUE.format(project_root=project_root) == (
        f"{project_root}/{RENDERED_CONFIG_RELPATH}"
    ), "the MCP servers' config path and the repo's render path have diverged"


def test_the_container_context_carries_no_secret_and_no_host_state(tmp_path: Path) -> None:
    """What must NOT travel: secrets, git, and the two derived zones.

    ``.env`` is excluded here rather than by the image's ``.dockerignore``
    because a ``.dockerignore`` pattern is anchored at the context root, and this
    context has depth — a secret one directory down is one a root-anchored
    ``.env*`` silently stops matching. A file that never enters the context
    cannot be baked into an image by a pattern that failed to match it.
    """
    repo = tmp_path / "facility"
    (repo / BUILD_DIR_NAME).mkdir(parents=True)
    (repo / STATE_DIR_NAME / "agent_data").mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / PROFILE_FILENAME).write_text("name: facility\n", encoding="utf-8")
    (repo / ".env").write_text("ANTHROPIC_API_KEY=sk-secret\n", encoding="utf-8")
    (repo / ".env.users").write_text("ANTHROPIC_API_KEY=sk-secret\n", encoding="utf-8")
    (repo / "data").mkdir()
    (repo / "data" / ".env").write_text("NESTED=sk-secret\n", encoding="utf-8")
    (repo / "data" / "channels.yml").write_text("channels: []\n", encoding="utf-8")

    image_root = tmp_path / "image"
    image_root.mkdir()
    _stage_source_zone(repo, image_root)

    leaked = sorted(str(p.relative_to(image_root)) for p in image_root.rglob(".env*"))
    assert not leaked, f"secrets copied into the container context: {leaked}"
    for zone in (BUILD_DIR_NAME, STATE_DIR_NAME, ".git"):
        assert not (image_root / zone).exists(), f"{zone} must not travel into an image"
    # ...and the source the drift fingerprint folds DOES travel, or the
    # container reports its own untouched build as drifted.
    assert (image_root / "data" / "channels.yml").is_file()


def test_a_render_without_a_repo_marker_is_not_a_deployment(tmp_path: Path) -> None:
    """Why the marker is load-bearing, stated as the failure it prevents.

    This is the shape of a flat image whose root *is* the render: a complete,
    valid config, and no deployment as far as the framework is concerned.
    """
    flat = tmp_path / "app" / PROJECT_NAME
    flat.mkdir(parents=True)
    (flat / "config.yml").write_text("project_root: /app/x\n", encoding="utf-8")

    with pytest.raises(RepoNotFoundError):
        find_repo_root(flat)


def test_a_mirrored_repo_layout_resolves_from_the_container_workdir(tmp_path: Path) -> None:
    """And the shape the image must have instead, proven against the real resolver."""
    project_root = tmp_path / "app" / PROJECT_NAME
    (project_root / BUILD_DIR_NAME).mkdir(parents=True)
    (project_root / PROFILE_FILENAME).write_text("name: layout\n", encoding="utf-8")
    (project_root / RENDERED_CONFIG_RELPATH).write_text(
        f"project_root: {project_root}\n", encoding="utf-8"
    )

    assert find_repo_root(project_root) == project_root.resolve()
    assert (project_root / RENDERED_CONFIG_RELPATH).is_file()


def test_the_pinned_container_interpreter_is_the_base_images(tmp_path: Path) -> None:
    """The interpreter constant and the base image it was read off, held together.

    A container render cannot derive its interpreter: the render happens on this
    machine and the interpreter exists in the image, so the value is pinned
    (``_CONTAINER_INTERPRETER``). A pinned value is only true while its premise
    is — and the premise is one line of the Dockerfile. Changing the base image
    without changing the constant would leave every MCP server command and every
    framework hook in every image naming a path that is not there, and nothing
    would say so until an agent tried to start one.
    """
    assert re.search(r"^FROM python:3\.12-slim\b", _rendered_dockerfile(), flags=re.M), (
        "the project image's base changed; re-derive _CONTAINER_INTERPRETER "
        "against the new base and update this pin"
    )
    assert _CONTAINER_INTERPRETER == "/usr/local/bin/python"


def test_the_context_root_dockerignore_matches_at_container_depth(tmp_path: Path) -> None:
    """The emitted patterns are ``**/``-anchored, or they exclude nothing.

    A ``.dockerignore`` is read at the context ROOT only, and every pattern in
    it is anchored there. This context is a repo, so everything the shipped
    patterns name — the render's own ``.env`` above all — sits one level down
    under ``build/``, where a root-anchored pattern does not match. Measured, not
    reasoned: on a context of exactly this shape, a plain ``.env*`` line let
    ``build/.env`` be baked into the image and ``**/.env*`` did not.

    Driven through the real emitter against the real shipped list, so the two
    cannot drift apart: what the render says must never enter an image is what
    the context root excludes, spelled for the shape the context actually has.
    """
    image_root = tmp_path / "image"
    (image_root / BUILD_DIR_NAME).mkdir(parents=True)
    shipped = resources.files("osprey").joinpath("templates/project/dockerignore").read_text()
    (image_root / BUILD_DIR_NAME / ".dockerignore").write_text(shipped, encoding="utf-8")

    _write_image_context_dockerignore(image_root)

    emitted = [
        line.strip()
        for line in (image_root / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert emitted, "no patterns emitted at the context root"
    for pattern in emitted:
        assert pattern.lstrip("!").startswith("**/"), (
            f"{pattern!r} is anchored at the context root, where nothing it names lives"
        )

    names = {pattern.lstrip("!").removeprefix("**/") for pattern in emitted}
    shipped_names = {
        line.strip().lstrip("!")
        for line in shipped.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert ".env*" in names and "!**/.env.example" in emitted
    # Everything the render excludes, minus the one deliberate difference: at
    # this depth `Dockerfile` names the build's OWN recipe (`-f build/Dockerfile`)
    # and the service recipes under it, which belong to the repo the image ships.
    assert names == shipped_names - {"Dockerfile"}
    assert not any(pattern.rstrip("/").endswith(BUILD_DIR_NAME) for pattern in names), (
        f"excluding {BUILD_DIR_NAME}/ here would ship an image with no render in it"
    )


# ---------------------------------------------------------------------------
# The contexts a REAL build produces. Everything above pins one seam; this pins
# what an operator's `osprey build` actually leaves on disk for `docker build`
# to consume — the only place the seams have to agree with each other.
# ---------------------------------------------------------------------------

# Every test below renders the exemplar for real (seconds, not milliseconds), so
# each one carries `@pytest.mark.slow`.

#: The exemplar's personas and the ``control_system.writes_enabled`` each delta
#: pins. That key IS the tier boundary between them, so it is the one field that
#: proves an image context carries ITS OWN merged config rather than the
#: deployment's — the defect that made persona deltas invisible to their own
#: MCP servers. ``ariel`` is the standalone logbook terminal: not a control tier
#: at all, and it pins the key off for the same reason the read-only tier does.
PERSONA_WRITES = {"ariel": False, "readonly": False, "readwrite": True}


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory) -> Path:
    """The exemplar, built for real, with secrets seeded.

    ``seed_env=True`` because a repo with no ``.env`` cannot show that the
    contexts carry none: the secret has to exist to be kept out.
    """
    from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

    repo = build_exemplar_repo(tmp_path_factory.mktemp("layout") / EXEMPLAR_DIRNAME, seed_env=True)
    previous = Path.cwd()
    os.chdir(repo)
    try:
        result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    finally:
        os.chdir(previous)
    assert result.exit_code == 0, result.output
    return repo


def _contexts(repo: Path) -> dict[str, Path]:
    """Every image context the build produced, by image project name."""
    image_dir = repo / BUILD_DIR_NAME / IMAGE_DIR_NAME
    return {entry.name: entry for entry in sorted(image_dir.iterdir()) if entry.is_dir()}


def _staleness_reasons_as_the_container_sees_them(context: Path, scratch: Path) -> list[str]:
    """Run the staleness advisory over a context as if it were mounted at ``/app/<name>``.

    The advisory hashes the profile ``build_args.profile_path_abs`` names. In a
    context that path is the CONTAINER's, which on this machine is nothing at
    all — and "nothing at all" makes the advisory fail open and return no
    reasons, which would make a test of it vacuous. So the manifest's container
    paths are re-pointed at the copy of that same tree sitting here, which is
    exactly the file the container would be hashing.

    Only the manifest is copied because that is all
    :func:`~osprey.deployment.staleness.staleness_reasons` reads.
    """
    from osprey.deployment.staleness import build_manifest_path, staleness_reasons

    manifest_path = build_manifest_path(context)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_root = f"/app/{context.name}"
    build_args = manifest.get("build_args") or {}
    for key in ("profile_path", "profile_path_abs"):
        value = build_args.get(key)
        if isinstance(value, str) and value.startswith(f"{runtime_root}/"):
            build_args[key] = str(context / value[len(runtime_root) + 1 :])

    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / manifest_path.name).write_text(json.dumps(manifest), encoding="utf-8")
    return staleness_reasons(scratch)


@pytest.mark.slow
def test_a_real_build_produces_one_context_per_image(built_repo: Path) -> None:
    """One rule, applied once per image this repo has to build.

    The deployment gets a context and so does every persona, because a persona
    is deployed as its own image. A persona image built from the deployment's
    context would come up with the deployment's config and none of that
    persona's — the difference between a read-only terminal and one that writes.
    """
    contexts = _contexts(built_repo)
    assert set(contexts) == {built_repo.name} | {
        f"{built_repo.name}-{persona}" for persona in PERSONA_WRITES
    }


@pytest.mark.slow
@pytest.mark.parametrize("persona", [None, *PERSONA_WRITES])
def test_every_produced_context_is_a_container_repo(built_repo: Path, persona: str | None) -> None:
    """The contract, read off the tree the build actually wrote.

    ``profile.yml`` at the root is what lets the image's own ``CMD``
    (``osprey web``) resolve a deployment at all; the render below it at
    ``build/`` is what makes every MCP server's ``{project_root}/build/config.yml``
    name a real file once the context lands at ``/app/<project>``; and the
    ``project_root`` recorded in that config is the container's, not this
    machine's — otherwise the agent's data root resolves beside the mounted
    volume rather than into it.
    """
    name = built_repo.name if persona is None else f"{built_repo.name}-{persona}"
    context = _contexts(built_repo)[name]
    project_root = f"/app/{name}"

    assert (context / PROFILE_FILENAME).is_file()
    assert find_repo_root(context) == context.resolve()

    config = yaml.safe_load((context / RENDERED_CONFIG_RELPATH).read_text(encoding="utf-8"))
    assert config["project_root"] == project_root

    declared = {
        value
        for server in json.loads((context / BUILD_DIR_NAME / ".mcp.json").read_text())[
            "mcpServers"
        ].values()
        for key, value in (server.get("env") or {}).items()
        if "CONFIG" in key
    }
    assert declared == {RENDERED_CONFIG_ENV_VALUE.format(project_root=project_root)}
    assert declared == {f"{project_root}/{RENDERED_CONFIG_RELPATH}"}


@pytest.mark.slow
@pytest.mark.parametrize("persona,writes", sorted(PERSONA_WRITES.items()))
def test_a_persona_context_carries_its_own_merged_config(
    built_repo: Path, persona: str, writes: bool
) -> None:
    """A persona image's config is the delta merged over the root, not the deployment's.

    Its ``.mcp.json`` names ``/app/<repo>-<persona>/build/config.yml``, and only
    this context holds that file — so this is where the persona's own
    ``control_system.writes_enabled`` has to be, or the servers in its container
    read a config that never heard of the delta.
    """
    context = _contexts(built_repo)[f"{built_repo.name}-{persona}"]
    config = yaml.safe_load((context / RENDERED_CONFIG_RELPATH).read_text(encoding="utf-8"))
    assert config["control_system"]["writes_enabled"] is writes


@pytest.mark.slow
def test_no_produced_context_carries_a_secret_or_a_host_path(built_repo: Path) -> None:
    """Two properties an image is not allowed to have, asserted over every context.

    No ``.env`` (``.env.example`` carries none and documents what an operator
    must supply), and no string naming the machine that built it — the render is
    made against ``/app/<project>``, so a host path in one is a value that
    escaped the substitution.
    """
    image_dir = built_repo / BUILD_DIR_NAME / IMAGE_DIR_NAME
    leaked = sorted(
        str(path.relative_to(image_dir))
        for path in image_dir.rglob(".env*")
        if path.is_file() and path.name != ".env.example"
    )
    assert not leaked, f"secrets in an image build context: {leaked}"

    host = str(built_repo.resolve())
    offenders = [
        str(path.relative_to(image_dir))
        for path in image_dir.rglob("*")
        if path.is_file() and host in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not offenders, f"the building host's path is recorded in: {offenders}"


@pytest.mark.slow
@pytest.mark.parametrize("persona", [None, *PERSONA_WRITES])
def test_every_launched_process_in_a_context_names_the_image_interpreter(
    built_repo: Path, persona: str | None
) -> None:
    """Config paths that resolve are worth nothing if nothing can start.

    Every MCP server command and every framework hook command in a container
    render must name an interpreter the IMAGE has. The derivation cannot supply
    one — it can only answer from the filesystem it is standing on, and for a
    container render its honest answer is this host's venv — so the value is
    pinned and threaded in. Without it the config-path fix is defeated in full:
    the paths resolve, and then no server and no hook can launch, including the
    write-safety hooks.
    """
    name = built_repo.name if persona is None else f"{built_repo.name}-{persona}"
    render = _contexts(built_repo)[name] / BUILD_DIR_NAME

    commands = {
        server.get("command")
        for server in json.loads((render / ".mcp.json").read_text())["mcpServers"].values()
    }
    assert commands == {_CONTAINER_INTERPRETER}

    settings = (render / ".claude" / "settings.json").read_text(encoding="utf-8")
    interpreters = re.findall(r"/[\w./-]*/python[\d.]*", settings)
    assert interpreters, "no absolute interpreter found in the rendered hook commands"
    assert set(interpreters) == {_CONTAINER_INTERPRETER}


@pytest.mark.slow
@pytest.mark.parametrize("persona", [None, *PERSONA_WRITES])
def test_a_produced_context_reports_neither_drift_nor_staleness(
    built_repo: Path, persona: str | None, tmp_path: Path
) -> None:
    """An untouched image reports itself untouched — by BOTH readers, not just the gate.

    Two independent things read a build's provenance inside a container and an
    operator meets both: :func:`check_drift`, the gate every lifecycle verb goes
    through, and :func:`staleness_reasons`, the advisory behind ``osprey status``
    and the warning ``osprey up`` prints. They must agree, and on a freshly built
    image the agreement they owe is "nothing has changed".

    They agree only if they hash the SAME file. The gate reads ``profile.yml``
    off the repo root; the advisory reads whatever ``build_args.profile_path_abs``
    names — so for a persona image, pointing that at the persona DELTA (the file
    the render came from) hashes the delta against a stamp taken from the root
    profile, and every persona container prints "profile has changed" on a build
    nobody touched, forever. A false drift report is the one thing the whole
    relocation exists to prevent, and an advisory that cries wolf on every
    container is how operators learn to ignore the real one.
    """
    from osprey.deployment.staleness import DriftState, check_drift

    name = built_repo.name if persona is None else f"{built_repo.name}-{persona}"
    context = _contexts(built_repo)[name]

    report = check_drift(context)
    assert report.state is DriftState.CLEAN, report.message
    assert report.changed_keys == ()

    assert _staleness_reasons_as_the_container_sees_them(context, tmp_path / name) == []


@pytest.mark.slow
@pytest.mark.parametrize("persona", [None, *PERSONA_WRITES])
def test_the_hashed_profile_is_the_repos_own_and_the_recorded_one_is_the_render_source(
    built_repo: Path, persona: str | None
) -> None:
    """The two profile records in a container manifest, and why they differ.

    ``profile_path`` is provenance: the file this render came from, which for a
    persona image is its delta. ``profile_path_abs`` is not provenance — it is
    the advisory's INPUT, the file it hashes against ``creation.preset_hash`` —
    so it has to name the profile that stamp was computed from, which is the
    repo's own ``profile.yml``. Stated here as an equality rather than left to
    the behavioural pin above, because the day someone "consistently" points both
    at the same file, the failure it causes is a warning in a container nobody is
    running a test in.
    """
    from osprey.deployment.staleness import build_manifest_path

    name = built_repo.name if persona is None else f"{built_repo.name}-{persona}"
    context = _contexts(built_repo)[name]
    manifest = json.loads(build_manifest_path(context).read_text(encoding="utf-8"))
    build_args = manifest["build_args"]

    assert build_args["profile_path_abs"] == f"/app/{name}/{PROFILE_FILENAME}"
    assert (context / PROFILE_FILENAME).is_file(), "the file the advisory hashes must ship"

    expected_source = PROFILE_FILENAME if persona is None else f"{PERSONA_DIRNAME}/{persona}.yml"
    assert build_args["profile_path"] == f"/app/{name}/{expected_source}"


@pytest.mark.slow
def test_a_regenerable_file_under_a_fold_input_is_neither_folded_nor_shipped(
    tmp_path_factory,
) -> None:
    """What the ``.dockerignore`` excludes is gone from the context BEFORE the stamp.

    A ``.dockerignore`` decides what the image receives; it does not touch the
    context. Those patterns are ``**/``-anchored, so they reach inside ``data/``
    and the convention directories — which are exactly the trees the profile
    fingerprint folds. A ``data/ingest.log`` left in the context would therefore
    be folded into the stamp here and be missing from the image there, and the
    container would recompute a different hash and refuse to start on drift
    nobody caused.

    Two assertions, and the pair is the point: the file is not in the context
    (so the image and the context hold the same tree), and the stamp still
    matches that tree (so the fingerprint was taken after the removal, not
    before). Reversing the order passes the first and fails the second.
    """
    from osprey.deployment.staleness import DriftState, check_drift
    from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

    repo = build_exemplar_repo(tmp_path_factory.mktemp("regenerable") / EXEMPLAR_DIRNAME)
    data_root = repo / "data"
    assert data_root.is_dir(), "the exemplar must have a data tree for this pin to mean anything"
    (data_root / "ingest.log").write_text("2026-01-01 loaded\n", encoding="utf-8")
    (data_root / "__pycache__").mkdir(exist_ok=True)
    (data_root / "__pycache__" / "loader.cpython-312.pyc").write_bytes(b"\x00compiled")

    previous = Path.cwd()
    os.chdir(repo)
    try:
        result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    finally:
        os.chdir(previous)
    assert result.exit_code == 0, result.output

    for name, context in _contexts(repo).items():
        shipped = sorted(
            str(path.relative_to(context))
            for path in context.rglob("*")
            if path.is_file() and (path.suffix in {".log", ".pyc"} or "__pycache__" in path.parts)
        )
        assert not shipped, f"{name} ships files its own .dockerignore excludes: {shipped}"
        report = check_drift(context)
        assert report.state is DriftState.CLEAN, f"{name}: {report.message}"
