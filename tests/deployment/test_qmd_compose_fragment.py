"""The qmd sidecar's two rendered artifacts, asserted on the rendered text.

The sidecar is a compose fragment plus a collection config, and the two only
work if they agree: the fragment mounts a corpus read-only at a path, and the
config declares a collection pointed at that same path. A mount with no
collection indexes nothing; a collection with no mount points at an empty
directory. Neither fails loudly — both surface as "search returns nothing" —
so the agreement is asserted here rather than assumed, and both renders go
through the production injection (``_inject_project_metadata``) so what is
tested is the derivation a deploy actually uses.

Three further properties carry the design and each has its own test:

* **The published port is 8180, never 8181.** qmd's own daemon hardcodes
  ``listen(port, "localhost")`` and binds ``[::1]`` only, so it is unreachable
  from any other container; the entrypoint fronts it with a forwarder that owns
  the routable port. Publishing 8181 would publish nothing.
* **The publish interface is project-wide.** ``deployment.bind_address``
  decides it, and a per-service ``bind_address`` is deliberately inert — the
  endpoint is unauthenticated, so it must not be possible to expose it on an
  interface the rest of the stack is not on.
* **The index survives a recreate.** It is a named volume, not a bind and not
  container-local: rebuilding it costs ~41 minutes at ALS scale.
"""

from __future__ import annotations

import os
import shutil

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

    qmd binds IPv6 loopback only and offers no ``--host`` flag, so a fragment
    that published 8181 would publish a port nothing outside the container's own
    namespace can reach.
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
    shutil.copy2(str(packaged / "_network_axis.j2"), repo / "services" / "_network_axis.j2")
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
