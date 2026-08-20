"""The qmd sidecar's two rendered artifacts, asserted on the rendered text.

The sidecar is a compose fragment plus a collection config, and the two only
work if they agree: the fragment mounts a corpus read-only at a path, and the
config declares a collection pointed at that same path. A mount with no
collection indexes nothing; a collection with no mount points at an empty
directory. Neither fails loudly — both surface as "search returns nothing" —
so the agreement is asserted here rather than assumed, and both renders go
through the production injection (``_inject_project_metadata``) so what is
tested is the derivation a deploy actually uses.

Four further properties carry the design and each has its own test:

* **The published port is 8180, never 8181.** qmd's own daemon hardcodes
  ``listen(port, "localhost")`` — no ``--host`` flag, no env override — so it
  answers only on a loopback address inside the container, unreachable from any
  other container. Which loopback family that is depends on the host: Node
  resolves ``localhost`` to ``[::1]`` on some image/host combinations and to
  ``127.0.0.1`` on others. So the entrypoint runs the daemon on an internal
  port, probes both families for ``/health``, and points a socat forwarder at
  whichever one answered; the forwarder owns the published port. Publishing
  8181 would publish nothing.
* **The publish interface is project-wide.** ``deployment.bind_address``
  decides it, and a per-service ``bind_address`` is deliberately inert — the
  endpoint is unauthenticated, so it must not be possible to expose it on an
  interface the rest of the stack is not on.
* **The index survives a recreate.** It is a named volume, not a bind and not
  container-local: rebuilding it costs ~41 minutes at ALS scale.
* **Pre-staged models are two edits or none.** ``services.qmd.models_dir`` gates
  a build arg (which tells the image build to skip the 2.1 GB of downloads) and
  a read-only bind mount (which supplies those same files at runtime). One
  without the other is the silent no-models failure: an image built with nothing
  baked in and nothing mounted. The last two sections assert that the two sites
  fire together, and that the literal strings they share with the image's
  Dockerfile still agree with it.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _packaged_template(rel_path: str):
    """Compile a packaged template, addressed from the templates root.

    The compose fragment imports the shared network-axis macros by a path
    relative to the PROJECT root (``services/_network_axis.j2``), which is where
    ``compose_generator``'s own loader is rooted and where ``osprey build``
    places the macro file. A bare ``jinja2.Template`` has no loader and that
    import would raise, so every render here goes through an Environment rooted
    at the packaged ``templates/`` directory. The default Undefined mode is
    production's, so ``| default(...)`` chains behave exactly as they do under
    ``osprey up``.
    """
    from importlib import resources

    from jinja2 import Environment, FileSystemLoader

    import osprey

    templates_root = resources.files(osprey).joinpath("templates")
    env = Environment(loader=FileSystemLoader(str(templates_root)), autoescape=False)
    return env.get_template(rel_path)


def _context(
    *,
    project_name: str = "demo",
    project_root: str | None = None,
    qmd: dict | None = None,
    deployment: dict | None = None,
    facility_knowledge: dict | None = None,
    ariel: dict | None = None,
) -> dict:
    """Build the render context the production injection produces.

    Goes through ``_inject_project_metadata`` rather than hand-assembling
    ``osprey_qmd``: the corpus list, the port and the publish interface are
    exactly what is under test, and a hand-built context would assert against
    values the test itself supplied.
    """
    from osprey.deployment.compose_generator import _inject_project_metadata

    config: dict = {
        "project_name": project_name,
        # ``resolve_repo_root`` honours ``project_root`` only when it names a
        # directory that exists here, and falls back to the working directory
        # otherwise — which is fine for every test that mounts repo-RELATIVE
        # corpora, since their spelling does not depend on the root. The one
        # test that needs the root to matter passes a real one.
        "project_root": project_root or f"/r/{project_name}",
        "services": {"qmd": dict(qmd) if qmd is not None else {}},
        "system": {"timezone": "UTC"},
    }
    if deployment is not None:
        config["deployment"] = deployment
    if facility_knowledge is not None:
        config["facility_knowledge"] = facility_knowledge
    if ariel is not None:
        config["ariel"] = ariel
    return _inject_project_metadata(config)


def render_compose(**kwargs) -> str:
    """Render the sidecar's compose fragment."""
    return _packaged_template("services/qmd/docker-compose.yml.j2").render(_context(**kwargs))


def render_index(**kwargs) -> str:
    """Render the sidecar's collection config."""
    return _packaged_template("services/qmd/index.yml.j2").render(_context(**kwargs))


def compose_service(**kwargs) -> dict:
    """The parsed ``qmd`` service block of a rendered fragment."""
    return yaml.safe_load(render_compose(**kwargs))["services"]["qmd"]


def _packaged_text(rel_path: str) -> str:
    """Read a packaged sidecar file that is NOT a template.

    The Dockerfile and the entrypoint carry no Jinja — they are copied into the
    build context verbatim — so the tests that pair the fragment with them read
    the shipped bytes rather than a render.
    """
    from importlib import resources

    import osprey

    return resources.files(osprey).joinpath("templates", rel_path).read_text()


#: The two corpora a fully-configured deployment mounts, as they appear in
#: config. Spelled once so every test that needs "both corpora" agrees.
BOTH_CORPORA = {
    "facility_knowledge": {"bundle_path": "data/facility_knowledge"},
    "ariel": {
        "enhancement_modules": {"qmd_export": {"enabled": True, "mirror_path": "data/ariel_mirror"}}
    },
}


# ---------------------------------------------------------------------------
# The published port
# ---------------------------------------------------------------------------


def test_publishes_8180_not_qmds_own_8181():
    """8180 is the forwarder's port; 8181 is the unreachable daemon's.

    qmd binds a loopback address — whichever family the host resolves
    ``localhost`` to — and offers no ``--host`` flag, so a fragment that
    published 8181 would publish a port nothing outside the container's own
    namespace can reach on either family.
    """
    service = compose_service()

    assert service["ports"] == ["127.0.0.1:8180:8180"]
    assert service["environment"]["OSPREY_QMD_PORT"] == "8180"
    # 8181 appears in the fragment's header comment, explaining why it is NOT
    # here; it must not appear in anything compose acts on.
    assert "8181" not in yaml.safe_dump(service)


def test_port_override_moves_publish_environment_and_healthcheck_together():
    """One config key moves all three, or the forwarder forwards to nothing."""
    service = compose_service(qmd={"port": 9999})

    assert service["ports"] == ["127.0.0.1:9999:9999"]
    assert service["environment"]["OSPREY_QMD_PORT"] == "9999"
    assert "127.0.0.1:9999/health" in service["healthcheck"]["test"][1]


def test_port_default_tracks_the_schema_module():
    """The template carries no port literal of its own.

    ``deployment/qmd_service.py`` is where the number is defined, and the
    entrypoint and the Python client read it from there too.
    """
    from osprey.deployment.qmd_service import DEFAULT_PORT

    assert compose_service()["ports"] == [f"127.0.0.1:{DEFAULT_PORT}:{DEFAULT_PORT}"]


# ---------------------------------------------------------------------------
# The publish interface
# ---------------------------------------------------------------------------


def test_bind_address_comes_from_the_project_wide_key():
    service = compose_service(deployment={"bind_address": "0.0.0.0"})

    assert service["ports"] == ["0.0.0.0:8180:8180"]


def test_per_service_bind_address_is_inert():
    """A ``services.qmd.bind_address`` must not move the publish.

    The endpoint is unauthenticated, so honouring a per-service key would let a
    deployment expose search over the whole corpus on an interface the rest of
    the stack is not on. The project-wide key stays in charge.
    """
    service = compose_service(
        qmd={"bind_address": "0.0.0.0"}, deployment={"bind_address": "127.0.0.1"}
    )

    assert service["ports"] == ["127.0.0.1:8180:8180"]


def test_bind_address_defaults_to_loopback_with_no_deployment_block():
    assert compose_service()["ports"] == ["127.0.0.1:8180:8180"]


# ---------------------------------------------------------------------------
# The image
# ---------------------------------------------------------------------------


def test_image_is_overridable_via_osprey_qmd_image():
    """The override variable is ``OSPREY_QMD_IMAGE``; the default tag is
    project-prefixed so two projects on one host never race to tag one image."""
    assert compose_service()["image"] == "${OSPREY_QMD_IMAGE:-demo-qmd:local}"


def test_explicit_service_image_replaces_the_built_default():
    service = compose_service(qmd={"image": "registry.example/osprey-qmd:2.5.3"})

    assert service["image"] == "${OSPREY_QMD_IMAGE:-registry.example/osprey-qmd:2.5.3}"


def test_build_context_is_repo_root_relative():
    """Every relative path resolves against the pinned compose project
    directory, which is the deployment repo root — not this file's subdir."""
    service = compose_service()

    assert service["build"]["context"] == "./build/services/qmd"
    assert service["build"]["dockerfile"] == "Dockerfile"


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------


def test_index_lives_on_a_named_volume_the_sidecar_owns():
    rendered = yaml.safe_load(render_compose())

    assert "qmd_index:/var/lib/qmd" in rendered["services"]["qmd"]["volumes"]
    assert "qmd_index" in rendered["volumes"]


def test_state_dir_environment_matches_the_volume_target():
    """The entrypoint reads ``OSPREY_QMD_STATE_DIR``; a volume mounted anywhere
    else means the index is written outside it and lost on recreate."""
    service = compose_service()
    state_dir = service["environment"]["OSPREY_QMD_STATE_DIR"]

    assert f"qmd_index:{state_dir}" in service["volumes"]


def test_rendered_index_config_is_mounted_read_only_where_the_entrypoint_looks():
    service = compose_service()
    index_config = service["environment"]["OSPREY_QMD_INDEX_CONFIG"]

    assert f"./build/services/qmd/index.yml:{index_config}:ro" in service["volumes"]


# ---------------------------------------------------------------------------
# Corpus mounts
# ---------------------------------------------------------------------------


def test_one_read_only_mount_per_configured_corpus():
    service = compose_service(**BOTH_CORPORA)

    corpus_mounts = [v for v in service["volumes"] if ":/corpus/" in v]
    assert corpus_mounts == [
        "./data/facility_knowledge:/corpus/okf:ro",
        "./data/ariel_mirror:/corpus/ariel:ro",
    ]


def test_okf_bundle_alone_mounts_only_its_own_corpus():
    service = compose_service(facility_knowledge={"bundle_path": "data/facility_knowledge"})

    assert [v for v in service["volumes"] if ":/corpus/" in v] == [
        "./data/facility_knowledge:/corpus/okf:ro"
    ]


def test_disabled_qmd_export_mounts_no_mirror():
    """A configured-but-disabled export writes nothing to mirror."""
    service = compose_service(
        ariel={
            "enhancement_modules": {
                "qmd_export": {"enabled": False, "mirror_path": "data/ariel_mirror"}
            }
        }
    )

    assert [v for v in service["volumes"] if ":/corpus/" in v] == []


def test_enabled_qmd_export_without_a_mirror_path_mounts_nothing():
    """There is no path to mount. The exporter refuses this config at runtime;
    the render must not invent a directory for it."""
    service = compose_service(ariel={"enhancement_modules": {"qmd_export": {"enabled": True}}})

    assert [v for v in service["volumes"] if ":/corpus/" in v] == []


def test_no_corpus_configured_mounts_none():
    service = compose_service()

    assert [v for v in service["volumes"] if ":/corpus/" in v] == []


# ---------------------------------------------------------------------------
# The collection config
# ---------------------------------------------------------------------------


def test_one_collection_per_mounted_corpus():
    index = yaml.safe_load(render_index(**BOTH_CORPORA))

    assert index["collections"] == {
        "okf": {"path": "/corpus/okf", "pattern": "**/*.md"},
        "ariel": {"path": "/corpus/ariel", "pattern": "**/*.md"},
    }


def test_collections_and_mounts_are_generated_from_one_list():
    """The agreement the whole design rests on, asserted directly.

    Every collection's ``path`` must be the container side of a corpus mount,
    and every corpus mount must have a collection. Either half alone renders a
    sidecar that starts, reports success, and finds nothing.
    """
    service = compose_service(**BOTH_CORPORA)
    collections = yaml.safe_load(render_index(**BOTH_CORPORA))["collections"]

    mount_targets = {v.split(":")[1] for v in service["volumes"] if ":/corpus/" in v}
    assert {c["path"] for c in collections.values()} == mount_targets
    # ...and each collection's mount target ends in its own name, so neither can
    # be renamed without the other.
    assert {name: c["path"] for name, c in collections.items()} == {
        name: f"/corpus/{name}" for name in collections
    }


def test_no_corpus_renders_an_empty_collection_mapping():
    """The honest render for a sidecar with nothing to search. The entrypoint's
    fail-closed gate then refuses to serve, which is the intended report."""
    assert yaml.safe_load(render_index()) == {"collections": {}}


def test_collection_config_declares_no_models_block():
    """The image pins the embedder, reranker and expansion models and bakes
    those exact files in. A second spelling here could disagree with what is on
    disk, and a disagreeing embedder costs a full rebuild."""
    assert "models" not in yaml.safe_load(render_index(**BOTH_CORPORA))


def test_collection_names_match_the_code_that_queries_them():
    """The names are a contract with the query code, not labels: a filtered
    query naming a collection the daemon does not have returns nothing, with no
    error anywhere. They are restated in ``compose_generator`` rather than
    imported, so that rendering a compose file does not drag the search services
    into the deployment import graph — which makes this test the only thing
    holding the two spellings together."""
    from osprey.deployment.compose_generator import QMD_ARIEL_COLLECTION, QMD_OKF_COLLECTION
    from osprey.services.ariel_search.search.qmd import ARIEL_COLLECTION
    from osprey.services.facility_knowledge.okf.bundle import OKF_COLLECTION

    assert QMD_OKF_COLLECTION == OKF_COLLECTION
    assert QMD_ARIEL_COLLECTION == ARIEL_COLLECTION


# ---------------------------------------------------------------------------
# Corpus path resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        pytest.param("data/facility_knowledge", "./data/facility_knowledge", id="repo-relative"),
        pytest.param("./data/kb", "./data/kb", id="already-dot-prefixed"),
        pytest.param("{repo}/data/kb", "./data/kb", id="absolute-inside-repo"),
        pytest.param("/srv/shared/kb", "/srv/shared/kb", id="absolute-outside-repo"),
    ],
)
def test_corpus_mount_sources_are_spelled_for_the_compose_project_directory(
    configured, expected, tmp_path
):
    """A bind source that is neither absolute nor dot-prefixed is not reliably
    read as a path by every runtime, and relative sources resolve against the
    repo root the deploy pins — so an absolute path INSIDE that root is spelled
    relative to it and one outside stays absolute."""
    context = _context(
        project_root=str(tmp_path),
        facility_knowledge={"bundle_path": configured.format(repo=tmp_path)},
    )
    service = yaml.safe_load(
        _packaged_template("services/qmd/docker-compose.yml.j2").render(context)
    )["services"]["qmd"]

    assert f"{expected}:/corpus/okf:ro" in service["volumes"]


def test_mirror_path_is_read_from_settings_when_present():
    """ARIEL's loader merges a module's extra keys with its ``settings:`` block,
    ``settings`` winning. The mount has to follow whichever the exporter will
    actually write to."""
    service = compose_service(
        ariel={
            "enhancement_modules": {
                "qmd_export": {
                    "enabled": True,
                    "mirror_path": "data/ignored",
                    "settings": {"mirror_path": "data/real_mirror"},
                }
            }
        }
    )

    assert "./data/real_mirror:/corpus/ariel:ro" in service["volumes"]


# ---------------------------------------------------------------------------
# The rest of the fragment
# ---------------------------------------------------------------------------


def test_sweep_interval_comes_from_config():
    service = compose_service(qmd={"interval": 900})

    assert service["environment"]["OSPREY_QMD_UPDATE_INTERVAL"] == "900"


def test_sweep_interval_default_tracks_the_schema_module():
    from osprey.deployment.qmd_service import DEFAULT_INTERVAL_SECONDS

    assert compose_service()["environment"]["OSPREY_QMD_UPDATE_INTERVAL"] == str(
        DEFAULT_INTERVAL_SECONDS
    )


def test_container_name_is_namespaced_per_project():
    """``container_name`` is a host-global docker identifier: two projects
    deploying a sidecar on one host must not collide on one name."""
    assert compose_service(project_name="other")["container_name"] == "other-qmd"


def test_health_probe_goes_through_the_forwarder():
    """Probing the daemon's internal port would report healthy for a container
    whose published path is dead."""
    service = compose_service()

    assert "http://127.0.0.1:8180/health" in service["healthcheck"]["test"][1]


def test_health_start_period_outlasts_a_first_boot_full_build():
    """The entrypoint refuses to open the port until the index is built, and a
    full build measured 41 minutes at ALS scale. A shorter grace period reports
    a container that is working correctly as unhealthy."""
    assert compose_service()["healthcheck"]["start_period"] == "3600s"


def test_joins_the_project_network_by_default():
    rendered = yaml.safe_load(render_compose())

    assert rendered["services"]["qmd"]["networks"] == ["osprey-network"]
    assert "osprey-network" in rendered["networks"]


def test_host_network_suppresses_ports_and_the_network_declaration():
    """The shared macro's contract, honoured mechanically: under ``host`` there
    is no port map to publish and no network for anything to join."""
    rendered = yaml.safe_load(render_compose(qmd={"network": "host"}))

    assert rendered["services"]["qmd"]["network_mode"] == "host"
    assert "ports" not in rendered["services"]["qmd"]
    assert "networks" not in rendered
    # The named volume is orthogonal to the network axis and must survive it.
    assert "qmd_index" in rendered["volumes"]


def test_fragment_is_valid_yaml_with_both_corpora():
    """The corpus mounts are emitted inside a Jinja loop between two macro
    calls; a whitespace slip there produces a file compose cannot parse."""
    rendered = yaml.safe_load(render_compose(**BOTH_CORPORA))

    assert set(rendered) == {"services", "volumes", "networks"}


# ---------------------------------------------------------------------------
# Build-time wiring
# ---------------------------------------------------------------------------


def test_render_service_templates_renders_siblings_and_skips_the_compose_template(
    tmp_path, monkeypatch
):
    """Templates are inputs: the build directory copy skips every ``.j2``, so a
    service whose container mounts a RENDERED config needs this step or the
    bind mount points at a file that was never produced.

    Run from *tmp_path*, because the render helpers resolve template paths
    against the working directory by contract (the build calls them from the
    repo root).
    """
    from osprey.deployment.compose_generator import render_service_templates

    source = tmp_path / "svc"
    (source / "sub").mkdir(parents=True)
    (source / "docker-compose.yml.j2").write_text("compose")
    (source / "index.yml.j2").write_text("hello {{ name }}")
    (source / "Dockerfile").write_text("FROM scratch")
    (source / "sub" / "kernel.json.j2").write_text("{}")
    out = tmp_path / "out"
    monkeypatch.chdir(tmp_path)

    rendered = render_service_templates("svc", {"name": "world"}, str(out))

    assert [os.path.basename(p) for p in rendered] == ["index.yml"]
    assert (out / "index.yml").read_text() == "hello world"
    assert not (out / "docker-compose.yml").exists()


def test_setup_build_dir_produces_both_of_the_sidecars_artifacts(tmp_path, monkeypatch):
    """End to end through the real build step: the compose fragment and the
    collection config it mounts are produced together, from one render."""
    from importlib import resources

    import osprey
    from osprey.deployment.compose_generator import setup_build_dir

    repo = tmp_path / "repo"
    packaged = resources.files(osprey).joinpath("templates", "services")
    shutil.copytree(str(packaged / "qmd"), repo / "services" / "qmd")
    # Every shared macro partial, by the same glob the build copies them with
    # (_copy_shared_service_partials) — naming one leaves the next to fail here
    # as a TemplateNotFound.
    for partial in sorted(Path(str(packaged)).glob("_*.j2")):
        shutil.copy2(partial, repo / "services" / partial.name)
    monkeypatch.chdir(repo)

    setup_build_dir(
        "services/qmd/docker-compose.yml.j2",
        {
            "project_name": "demo",
            "project_root": str(repo),
            "build_dir": "./build",
            "services": {"qmd": {}},
            "system": {"timezone": "UTC"},
            "facility_knowledge": {"bundle_path": "data/facility_knowledge"},
        },
        {},
    )

    out = repo / "build" / "services" / "qmd"
    compose = yaml.safe_load((out / "docker-compose.yml").read_text())
    index = yaml.safe_load((out / "index.yml").read_text())

    assert "./data/facility_knowledge:/corpus/okf:ro" in compose["services"]["qmd"]["volumes"]
    assert index["collections"]["okf"]["path"] == "/corpus/okf"
    # The templates themselves must never land in a build context.
    assert not list(out.glob("*.j2"))


# ---------------------------------------------------------------------------
# Pre-staged models
# ---------------------------------------------------------------------------

#: A host directory of pre-staged GGUF files, as an operator would configure it.
MODELS_DIR = "/srv/osprey/qmd-models"

#: Where the image looks for them. Spelled here and checked against the
#: Dockerfile's ``OSPREY_QMD_MODEL_DIR`` in the drift section below — compose
#: cannot expand the image's own ENV, so the fragment has to carry the literal.
MODELS_TARGET = "/opt/qmd/.cache/qmd/models"

#: Build arg that tells the image build the models arrive at runtime. Same
#: pairing: the name is a literal on both sides, drift-checked below.
MODELS_BUILD_ARG = "OSPREY_QMD_MODELS_MOUNTED"


def _build_args(service: dict) -> dict:
    return service["build"]["args"]


def _models_mounts(service: dict) -> list[str]:
    return [v for v in service["volumes"] if MODELS_TARGET in v]


def test_no_models_dir_renders_neither_the_build_arg_nor_the_mount():
    service = compose_service()

    assert MODELS_BUILD_ARG not in _build_args(service)
    assert _models_mounts(service) == []


def test_no_models_dir_leaves_the_fragment_exactly_as_it_was():
    """The unset state is the default, so it has to render the pre-change shape.

    Asserted as whole collections rather than as absences: an extra build arg or
    an extra volume that happened not to mention the model directory would slip
    past a negative check, and both are things a deployment acts on.
    """
    service = compose_service()

    assert _build_args(service) == {"OSPREY_PROJECT_NAME": "demo"}
    assert service["volumes"] == [
        "qmd_index:/var/lib/qmd",
        "./build/services/qmd/index.yml:/etc/qmd/index.yml:ro",
    ]


@pytest.mark.parametrize(
    ("qmd", "expected"),
    [({}, False), ({"models_dir": MODELS_DIR}, True)],
    ids=["unset", "set"],
)
def test_the_build_arg_and_the_mount_fire_together(qmd, expected):
    """One config key drives both sites, and neither may render without the other.

    The build arg alone builds an image with no models baked in and nothing
    mounted to supply them; the mount alone mounts three files over models the
    image already has. The first is the silent failure this pairing exists to
    prevent — the container starts, finds no model, and tries to download one on
    a host that was configured this way precisely because it cannot.
    """
    service = compose_service(qmd=qmd)

    fired = (MODELS_BUILD_ARG in _build_args(service), bool(_models_mounts(service)))
    assert fired == (expected, expected)


def test_models_dir_mounts_the_staged_directory_read_only():
    service = compose_service(qmd={"models_dir": MODELS_DIR})

    assert _models_mounts(service) == [f"{MODELS_DIR}:{MODELS_TARGET}:ro"]


def test_the_build_arg_is_truthy_only_and_carries_no_path():
    """The Dockerfile tests it with ``-n``. The host path is a HOST path — it
    means nothing inside the build, and passing it would bake an operator's
    directory layout into the image for no reader."""
    args = _build_args(compose_service(qmd={"models_dir": MODELS_DIR}))

    assert args[MODELS_BUILD_ARG] == "1"
    assert MODELS_DIR not in yaml.safe_dump(args)


def test_the_mount_source_is_trimmed_like_the_schema_trims_it():
    """The preflight checks the stripped path; a bind spec that kept the padding
    would name a different directory than the one that was checked."""
    service = compose_service(qmd={"models_dir": f"  {MODELS_DIR}  "})

    assert _models_mounts(service) == [f"{MODELS_DIR}:{MODELS_TARGET}:ro"]


def test_a_relative_models_dir_refuses_the_render():
    """Rendering resolves the block through the schema, so a relative path is
    refused here rather than mounted: the runtime would resolve it against the
    compose project directory, not against the directory the operator meant."""
    with pytest.raises(ValueError, match=r"services\.qmd\.models_dir"):
        compose_service(qmd={"models_dir": "qmd-models"})


def test_the_models_mount_precedes_the_corpus_mounts():
    """Position is not cosmetic: the corpus mounts are emitted by a Jinja loop,
    and a models mount rendered inside that loop would repeat per corpus."""
    volumes = compose_service(qmd={"models_dir": MODELS_DIR}, **BOTH_CORPORA)["volumes"]

    models = volumes.index(f"{MODELS_DIR}:{MODELS_TARGET}:ro")
    corpora = [i for i, v in enumerate(volumes) if ":/corpus/" in v]
    assert volumes.index("./build/services/qmd/index.yml:/etc/qmd/index.yml:ro") < models
    assert models < min(corpora)


def test_the_models_mount_survives_host_networking():
    """The network axis suppresses ports and the network stanza. It must not
    take an unrelated volume with it."""
    service = yaml.safe_load(render_compose(qmd={"network": "host", "models_dir": MODELS_DIR}))[
        "services"
    ]["qmd"]

    assert _models_mounts(service) == [f"{MODELS_DIR}:{MODELS_TARGET}:ro"]


# ---------------------------------------------------------------------------
# The fragment against the image it configures
# ---------------------------------------------------------------------------
# Two literal strings cross from the Dockerfile into the compose fragment — the
# directory the models are mounted at and the name of the build arg that gates
# the fetches. Neither can be expanded at render or at compose time: the ENV and
# the ARG exist only inside the image, while `${...}` in a compose file expands
# against the HOST environment, where both are unset. So they are spelled twice
# on purpose, and held together here.


def _dockerfile() -> str:
    return _packaged_text("services/qmd/Dockerfile")


def _entrypoint() -> str:
    return _packaged_text("services/qmd/entrypoint.sh")


def _dockerfile_env(name: str) -> str:
    """The value of one ``ENV NAME=value`` assignment.

    Matches the assignment at the start of a line (``ENV`` blocks continue with
    a backslash, so most assignments carry no keyword of their own) and so does
    not match a ``$NAME`` reference further along a ``RUN``.
    """
    match = re.search(rf"^[ \t]*(?:ENV[ \t]+)?{re.escape(name)}=(\S+)", _dockerfile(), re.MULTILINE)
    assert match is not None, f"the Dockerfile no longer sets ENV {name}"
    return match.group(1)


def _dockerfile_arg_default(name: str) -> str:
    match = re.search(rf"^ARG\s+{re.escape(name)}=(.*)$", _dockerfile(), re.MULTILINE)
    assert match is not None, f"the Dockerfile no longer declares ARG {name}"
    return match.group(1).strip().strip('"')


def _dockerfile_label_pins() -> dict[str, str]:
    """The ``com.osprey.qmd.<kind>_sha256`` label literals, by kind."""
    return dict(re.findall(r'com\.osprey\.qmd\.([a-z]+)_sha256="([0-9a-f]{64})"', _dockerfile()))


def test_the_mount_target_is_the_images_model_directory():
    assert MODELS_TARGET == _dockerfile_env("OSPREY_QMD_MODEL_DIR")


def test_the_build_arg_name_is_the_one_the_dockerfile_declares():
    assert _dockerfile_arg_default(MODELS_BUILD_ARG) == ""


def test_only_the_models_leaf_is_read_only_never_qmds_writable_cache():
    """qmd owns the cache tree above the models directory and writes into it.
    Mounting the tree read-only instead of the leaf would break the daemon on a
    write it makes for reasons that have nothing to do with models."""
    cache_home = _dockerfile_env("XDG_CACHE_HOME")
    service = compose_service(qmd={"models_dir": MODELS_DIR})

    assert MODELS_TARGET.startswith(f"{cache_home}/")
    assert MODELS_TARGET != cache_home
    assert [v for v in service["volumes"] if v.endswith(f":{cache_home}:ro")] == []


def test_the_writable_state_directory_is_outside_the_model_tree():
    """Everything the sidecar writes at runtime — the index, and the digest
    stamp the model check records — lands in the named volume. If the state
    directory sat under the model cache, the read-only mount would turn those
    writes into EROFS on a container that is otherwise configured correctly."""
    cache_home = _dockerfile_env("XDG_CACHE_HOME")
    state_dir = compose_service()["environment"]["OSPREY_QMD_STATE_DIR"]

    assert not state_dir.startswith(cache_home)
    assert 'MODEL_STAMP="$STATE_DIR/' in _entrypoint()


#: Every qmd subcommand the entrypoint is allowed to run. The exclusion that
#: matters is ``pull``: it HEAD-checks the model registry, treats an unreachable
#: registry as "stale", and DELETES the model files before failing to replace
#: them — which against a read-only mount cannot even be undone by a rebuild of
#: the index. The rest of the CLI resolves models from the cache and reads only.
READ_ONLY_QMD_SUBCOMMANDS = {"update", "embed", "mcp", "status", "collection"}


def test_the_entrypoint_runs_no_qmd_command_that_writes_to_the_model_cache():
    """The read-only mount is the last line of defence, not the first.

    A runtime write into the models leaf fails with EROFS and takes the sidecar
    down; the same command against a *baked* image silently deletes the models
    instead. So the check is on what the entrypoint invokes rather than on what
    the mount permits.

    Comment lines and quoted strings are stripped before the scan, because the
    file talks about qmd as well as running it: the paragraph explaining why
    ``qmd pull`` is never called must not read as a call, and neither must the
    log line "qmd daemon exited".
    """
    body = "\n".join(
        line for line in _entrypoint().splitlines() if not line.lstrip().startswith("#")
    )
    body = re.sub(r"\"[^\"]*\"|'[^']*'", " ", body)
    invoked = set(re.findall(r"\bqmd ([a-z-]+)", body))

    assert invoked <= READ_ONLY_QMD_SUBCOMMANDS, (
        f"the entrypoint invokes qmd subcommands outside the read-only set: "
        f"{sorted(invoked - READ_ONLY_QMD_SUBCOMMANDS)}. `qmd pull` in particular "
        f"deletes model files it then cannot re-download."
    )
    # Not vacuous: the three phases each have to still be there.
    assert {"update", "embed", "mcp"} <= invoked


def test_the_label_pins_are_the_arg_defaults_verbatim():
    """The labels are a deliberate second copy of the three SHA256 pins — CI
    greps them out of this file before any build, to key the layer cache — so
    they cannot reference the ARGs. This is what keeps the copy honest."""
    pins = _dockerfile_label_pins()

    assert set(pins) == {"embed", "rerank", "generate"}
    assert pins == {
        kind: _dockerfile_arg_default(f"OSPREY_QMD_{kind.upper()}_SHA256") for kind in pins
    }


def test_exactly_three_pin_labels_carry_a_bare_digest():
    """CI's cache-key step fails unless it greps exactly three 64-hex label
    literals out of the Dockerfile. A fourth — or a digest smuggled into
    ``model_delivery`` — reds that step long after the edit that caused it."""
    assert len(re.findall(r'com\.osprey\.qmd\.[a-z]+_sha256="[0-9a-f]{64}"', _dockerfile())) == 3
