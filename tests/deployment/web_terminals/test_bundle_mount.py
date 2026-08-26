"""The deployment knowledge bundle, bound into every entitled web terminal.

One host directory, many containers. That shape is the whole feature and it is
what every assertion here is about:

* **The source is shared, the target is per service.** All entitled users mount
  the SAME host directory — a concept alice drafts is the same file bob's
  container and the qmd sidecar read, not a copy. The in-container path,
  though, is derived per persona from that persona's own
  ``container_project_dir``, because two personas built from different projects
  read their bundle at two different paths and one hardcoded target would mount
  it where only one of them looks.
* **The mount shadows the persona's baked copy.** An image may ship its own
  rendered bundle at that path; the bind covers it. Deliberate — the bundle is
  operational knowledge, edited between image builds — and asserted here so the
  behaviour cannot be quietly reversed.
* **Entitlement is per persona.** A persona whose project names no
  ``facility_knowledge.bundle_path`` has nothing that would read the directory
  and gets no mount, resolved the same way every other per-persona grant is.

The provisioning half — setgid, group-writable, group inherited never invented —
is what makes container A's writes readable by container B and indexable by the
sidecar, and has its own section at the bottom.
"""

from __future__ import annotations

import copy
import logging
import os
import pathlib
import stat

import pytest
import yaml

from osprey.deployment.web_terminals.render import render_web_terminals

from .test_golden_render import EXAMPLE_CONFIG

#: Where the reference config puts its bundle, and the source every entitled
#: user's mount must therefore name.
BUNDLE_PATH = "data/facility_knowledge"
BUNDLE_SOURCE = f"./{BUNDLE_PATH}"


def _config(*, bundle_path: str | None = BUNDLE_PATH, personas: bool = False) -> dict:
    """The reference roster, optionally persona-backed and bundle-configured."""
    config = copy.deepcopy(EXAMPLE_CONFIG)
    if bundle_path is not None:
        config["facility_knowledge"] = {"bundle_path": bundle_path}
    if personas:
        web_terminals = config["modules"]["web_terminals"]
        web_terminals["personas"] = {
            "operator": {"project": "dls-operator", "project_path": "../dls-operator"},
            "physicist": {"project": "dls-physicist", "project_path": "../dls-physicist"},
        }
        web_terminals["users"] = [
            {"name": "alice", "index": 0, "persona": "operator"},
            {"name": "bob", "index": 1, "persona": "physicist"},
        ]
    return config


def _volumes(config: dict, **kwargs) -> dict[str, list[str]]:
    """Per-user volume lists from a rendered overlay, keyed by service name."""
    compose = yaml.safe_load(render_web_terminals(config, **kwargs)["docker-compose.web.yml"])
    return {
        name: service.get("volumes", [])
        for name, service in compose["services"].items()
        if name != "nginx"
    }


def _bundle_mounts(config: dict, **kwargs) -> dict[str, list[str]]:
    """Only the bundle mounts, keyed by service name."""
    return {
        name: [
            v for v in volumes if v.startswith(f"{BUNDLE_SOURCE}:") or "/facility_knowledge" in v
        ]
        for name, volumes in _volumes(config, **kwargs).items()
    }


# ---------------------------------------------------------------------------
# The mount
# ---------------------------------------------------------------------------


def test_no_bundle_configured_mounts_nothing():
    """A deployment that names no bundle path has no directory to hand out, and
    must render exactly what it rendered before this feature existed."""
    assert _bundle_mounts(_config(bundle_path=None)) == {"web-alice": [], "web-bob": []}


def test_every_persona_less_user_gets_the_bundle():
    """The zero-migration roster: no persona catalog, so entitlement is answered
    from this same config, and both users read the one deployment bundle."""
    mounts = _bundle_mounts(_config())

    assert mounts == {
        "web-alice": [f"{BUNDLE_SOURCE}:/app/dls-assistant/{BUNDLE_PATH}"],
        "web-bob": [f"{BUNDLE_SOURCE}:/app/dls-assistant/{BUNDLE_PATH}"],
    }


def test_target_is_computed_per_persona_from_its_own_project_dir():
    """The property a shared hardcoded path would break.

    ``operator`` and ``physicist`` are built from different projects, so their
    ``container_project_dir`` values differ and the same configured
    ``data/facility_knowledge`` resolves to two different in-container paths.
    """
    mounts = _bundle_mounts(
        _config(personas=True), facility_bundle_personas={"operator", "physicist"}
    )

    assert mounts == {
        "web-alice": [f"{BUNDLE_SOURCE}:/app/dls-operator/{BUNDLE_PATH}"],
        "web-bob": [f"{BUNDLE_SOURCE}:/app/dls-physicist/{BUNDLE_PATH}"],
    }


def test_source_is_one_shared_directory_for_every_user():
    """Per-user copies would defeat the point: a concept drafted in one
    container has to be the file the next one reads."""
    mounts = _bundle_mounts(
        _config(personas=True), facility_bundle_personas={"operator", "physicist"}
    )
    sources = {mount.split(":")[0] for mounts_ in mounts.values() for mount in mounts_}

    assert sources == {BUNDLE_SOURCE}


def test_only_entitled_personas_get_the_mount():
    """A persona whose project names no bundle path has nothing that would read
    the directory."""
    mounts = _bundle_mounts(_config(personas=True), facility_bundle_personas={"operator"})

    assert mounts == {
        "web-alice": [f"{BUNDLE_SOURCE}:/app/dls-operator/{BUNDLE_PATH}"],
        "web-bob": [],
    }


def test_no_entitled_personas_mounts_nothing():
    """The default when the caller resolves no set — a bundle is never granted
    on a guess, the same rule the per-persona credentials follow."""
    assert _bundle_mounts(_config(personas=True)) == {"web-alice": [], "web-bob": []}


def test_mount_is_read_write():
    """The agent drafts concepts into the bundle; only the qmd sidecar's mount of
    the same directory is read-only."""
    mounts = _bundle_mounts(_config())["web-alice"]

    assert mounts == [f"{BUNDLE_SOURCE}:/app/dls-assistant/{BUNDLE_PATH}"]
    assert not mounts[0].endswith(":ro")


def test_absolute_bundle_path_is_not_re_anchored():
    """An absolute path names the same absolute path on both sides — the same
    distinction the agent-data mount makes, and the reason the join happens in
    Python rather than as a template concatenation."""
    mounts = _volumes(_config(bundle_path="/srv/shared/knowledge"))["web-alice"]

    assert "/srv/shared/knowledge:/srv/shared/knowledge" in mounts


def test_blank_bundle_path_is_treated_as_unset():
    assert _bundle_mounts(_config(bundle_path="   ")) == {"web-alice": [], "web-bob": []}


def test_overlay_still_parses_with_the_mount_present():
    """The mount is emitted inside a Jinja conditional between two other volume
    lines; a whitespace slip there produces YAML compose cannot read."""
    compose = yaml.safe_load(render_web_terminals(_config())["docker-compose.web.yml"])

    assert set(compose["services"]) == {"nginx", "web-alice", "web-bob"}


# ---------------------------------------------------------------------------
# Group membership — the half setgid cannot supply
# ---------------------------------------------------------------------------


def _services(config: dict, **kwargs) -> dict[str, dict]:
    compose = yaml.safe_load(render_web_terminals(config, **kwargs)["docker-compose.web.yml"])
    return {name: svc for name, svc in compose["services"].items() if name != "nginx"}


def test_entitled_services_join_the_bundles_group():
    """setgid decides which group OWNS a new file; it does not make any process
    a MEMBER of that group, and a container carries only the groups its image
    or its runtime grants it. `group_add` is the runtime grant to this
    container's INITIAL process -- the membership that carries access wherever
    the entrypoint's own group join never runs: a container started with
    `--user` takes the entrypoint's non-root branch, which skips both the join
    and the privilege drop, as does any image started without this framework's
    entrypoint at all. A framework-built image drops privileges with `gosu`,
    which re-derives supplementary groups from `/etc/group` for the target user
    and discards this grant; the membership that reaches the process actually
    SERVING requests there comes from the entrypoint's own `/etc/group` join,
    asserted separately in tests/deployment/test_entrypoint_script.py."""
    services = _services(_config(), facility_bundle_gid=20)

    assert services["web-alice"]["group_add"] == ["20"]
    assert services["web-bob"]["group_add"] == ["20"]


def test_unentitled_services_join_no_group():
    """A container with no bundle mounted has no corpus to reach, so handing it a
    host group would widen its access for nothing."""
    services = _services(
        _config(personas=True), facility_bundle_personas={"operator"}, facility_bundle_gid=20
    )

    assert services["web-alice"]["group_add"] == ["20"]
    assert "group_add" not in services["web-bob"]


def test_no_gid_emits_no_group_add():
    """An unprovisioned bundle directory has no group to join yet. Emitting a
    guessed one would be worse than emitting none: the mount still works
    wherever the container's own uid already has access."""
    services = _services(_config(), facility_bundle_gid=None)

    assert "group_add" not in services["web-alice"]


def test_gid_is_quoted_so_compose_reads_it_as_a_group():
    """Numeric group ids and group names share the field, so the value is
    rendered as a string rather than a bare YAML integer."""
    rendered = render_web_terminals(_config(), facility_bundle_gid=20)["docker-compose.web.yml"]

    assert '      - "20"' in rendered


def test_gid_is_read_off_the_provisioned_directory(tmp_path):
    """The gid is never invented: it is whatever group the bundle directory
    ended up with, which is what makes a single out-of-band ``chgrp`` enough to
    retarget sharing without a code change."""
    from osprey.deployment.compose_generator import ensure_shared_corpus_dir, shared_corpus_gid

    target = tmp_path / "kb"

    assert shared_corpus_gid(target) is None, "an unprovisioned directory answers nothing"
    provisioned = ensure_shared_corpus_dir(target)

    assert shared_corpus_gid(target) == provisioned


def test_shadowing_is_documented_in_the_rendered_artifact():
    """A persona-baked bundle disappearing under this mount is the most likely
    confusion the feature produces, so the reason must be readable in the file an
    operator on the deploy host actually opens — a Jinja comment would be
    stripped from the render and never reach them."""
    rendered = render_web_terminals(_config())["docker-compose.web.yml"]

    assert "SHADOWS the copy baked into the persona image" in rendered


def test_no_shadowing_note_without_a_mount_to_explain():
    """The note travels with the mount; a deployment that mounts nothing must
    render byte-identically to what it rendered before this feature."""
    rendered = render_web_terminals(_config(bundle_path=None))["docker-compose.web.yml"]

    assert "SHADOWS" not in rendered


def test_sharing_strategy_is_stated_in_the_rendered_artifact():
    """The deployment note requirement. A docstring is not a deployment note —
    the operator debugging "alice's concept is invisible to bob" is reading the
    rendered compose file on the deploy host, so both the mechanism and its
    umask limit have to be legible there."""
    rendered = render_web_terminals(_config(), facility_bundle_gid=20)["docker-compose.web.yml"]

    assert "SHARING:" in rendered
    assert "setgid" in rendered and "group_add" in rendered
    assert "LIMIT:" in rendered and "umask" in rendered


def test_sharing_strategy_is_stated_in_the_how_to_page():
    """Same requirement, for the reader who is not on the deploy host yet."""
    page = (
        pathlib.Path(__file__).resolve().parents[3]
        / "docs"
        / "source"
        / "how-to"
        / "facility-knowledge"
        / "okf-bundle.rst"
    ).read_text()

    assert "shared-bundle-multi-user" in page
    assert "setgid" in page and "group_add" in page and "umask" in page


# ---------------------------------------------------------------------------
# Entitlement resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param({"facility_knowledge": {"bundle_path": "data/kb"}}, True, id="configured"),
        pytest.param({"facility_knowledge": {"bundle_path": ""}}, False, id="empty"),
        pytest.param({"facility_knowledge": {"bundle_path": "  "}}, False, id="blank"),
        pytest.param({"facility_knowledge": {}}, False, id="section-without-path"),
        pytest.param({"facility_knowledge": None}, False, id="null-section"),
        pytest.param({}, False, id="absent"),
    ],
)
def test_config_needs_facility_bundle(config, expected):
    from osprey.deployment.web_terminals.personas import config_needs_facility_bundle

    assert config_needs_facility_bundle(config) is expected


def test_personas_needing_facility_bundle_reads_each_personas_config(tmp_path):
    """The same per-persona walk every other grant uses: a persona's own rendered
    config.yml decides, and an unrendered persona contributes nothing rather than
    being granted on a guess."""
    from osprey.deployment.web_terminals.personas import personas_needing_facility_bundle

    (tmp_path / "dls-operator").mkdir()
    (tmp_path / "dls-operator" / "config.yml").write_text(
        yaml.safe_dump({"facility_knowledge": {"bundle_path": "data/facility_knowledge"}})
    )
    (tmp_path / "dls-physicist").mkdir()
    (tmp_path / "dls-physicist" / "config.yml").write_text(yaml.safe_dump({"facility": {}}))
    config = _config(personas=True)
    catalog = config["modules"]["web_terminals"]["personas"]
    catalog["operator"]["project_path"] = "dls-operator"
    catalog["physicist"]["project_path"] = "dls-physicist"

    assert personas_needing_facility_bundle(config, tmp_path) == {"operator"}


def test_unrendered_persona_is_not_entitled(tmp_path):
    from osprey.deployment.web_terminals.personas import personas_needing_facility_bundle

    config = _config(personas=True)

    assert personas_needing_facility_bundle(config, tmp_path) == set()


# ---------------------------------------------------------------------------
# The entitlement/target split, caught by lint
# ---------------------------------------------------------------------------


def _lintable(tmp_path, persona_bundle_path: str | None) -> dict:
    """A local-mode roster with one rendered persona, linted from *tmp_path*.

    ``project_path`` resolves against the working directory in lint, which is
    why callers chdir; the roster is trimmed to one persona so the finding under
    test is the only one about bundles.
    """
    project = tmp_path / "dls-operator"
    project.mkdir()
    (project / "Dockerfile").write_text("FROM scratch")
    persona_config: dict = {"project_name": "dls-operator"}
    if persona_bundle_path is not None:
        persona_config["facility_knowledge"] = {"bundle_path": persona_bundle_path}
    (project / "config.yml").write_text(yaml.safe_dump(persona_config))

    config = _config(personas=True)
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["image_source"] = "local"
    web_terminals["personas"] = {
        "operator": {
            "project": "dls-operator",
            "project_path": "dls-operator",
            "build_profile": "personas/operator.yml",
        }
    }
    web_terminals["users"] = [{"name": "alice", "index": 0, "persona": "operator"}]
    return config


def _bundle_findings(config: dict) -> list:
    from osprey.deployment.web_terminals.lint import lint_web_terminals

    return [
        f
        for f in lint_web_terminals(config)
        if f.code == "web_terminals.persona_bundle_path_divergence"
    ]


def test_lint_flags_a_persona_that_relocated_its_bundle(tmp_path, monkeypatch):
    """The failure this catches has no other symptom: the persona IS entitled
    (its config names a bundle), so the deployment's bundle is bound at the
    deployment's path — while the persona's knowledge tools read its own path,
    which nothing mounted. The panel lists no concepts and every layer reports
    success."""
    monkeypatch.chdir(tmp_path)
    config = _lintable(tmp_path, "data/kb")

    findings = _bundle_findings(config)

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "data/kb" in findings[0].message
    assert BUNDLE_PATH in findings[0].message


def test_lint_is_quiet_when_the_paths_agree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert _bundle_findings(_lintable(tmp_path, BUNDLE_PATH)) == []


def test_lint_is_quiet_for_a_persona_that_configures_no_bundle(tmp_path, monkeypatch):
    """Not entitled, so nothing is mounted for it and there is no target to
    disagree with."""
    monkeypatch.chdir(tmp_path)

    assert _bundle_findings(_lintable(tmp_path, None)) == []


def test_lint_is_quiet_when_the_deployment_configures_no_bundle(tmp_path, monkeypatch):
    """No mount exists at all, so a persona's own path cannot be shadowed by a
    differing one."""
    monkeypatch.chdir(tmp_path)
    config = _lintable(tmp_path, "data/kb")
    del config["facility_knowledge"]

    assert _bundle_findings(config) == []


# ---------------------------------------------------------------------------
# Shared-corpus provisioning (the GID strategy)
# ---------------------------------------------------------------------------


def test_new_bundle_dir_gets_exactly_the_bits_sharing_needs(tmp_path):
    """A directory this call creates has no operator intent to preserve, so it
    starts at 2770 and nothing wider. Neither consumer needs the ``other``
    triad — the sidecar reads as root, the terminals come in through their
    group — so granting it would widen access to a facility's knowledge for
    nobody."""
    from osprey.deployment.compose_generator import ensure_shared_corpus_dir

    target = tmp_path / "data" / "facility_knowledge"

    gid = ensure_shared_corpus_dir(target)

    assert target.is_dir()
    assert stat.S_IMODE(target.stat().st_mode) == 0o2770
    assert gid == target.stat().st_gid


def test_new_files_inherit_the_directorys_group(tmp_path):
    """The property the whole strategy rests on, exercised rather than assumed:
    a file created LATER is group-owned by the directory, not by the writer's
    primary group. That is what lets the next container and the sidecar reach a
    draft this one wrote.

    Only the file is checked. Setgid propagation to new SUB-directories is a
    Linux behaviour that macOS does not reproduce, so asserting it would pin a
    platform rather than the property.
    """
    from osprey.deployment.compose_generator import ensure_shared_corpus_dir

    target = tmp_path / "kb"
    ensure_shared_corpus_dir(target)

    (target / "concept.md").write_text("drafted here")

    assert (target / "concept.md").stat().st_gid == target.stat().st_gid


def test_existing_bundle_dir_keeps_its_own_other_triad(tmp_path):
    """The needed bits are ADDED, never substituted for the operator's mode.

    A bundle deliberately kept at 0700 holds facility information someone chose
    not to publish; a deploy that silently widened it to world-readable would be
    unrequested permission widening on every ``osprey up``.
    """
    from osprey.deployment.compose_generator import ensure_shared_corpus_dir

    target = tmp_path / "kb"
    target.mkdir(mode=0o700)
    (target / "concept.md").write_text("kept")

    ensure_shared_corpus_dir(target)

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o2770, "sharing bits added, `other` left exactly as the operator had it"
    assert (target / "concept.md").read_text() == "kept"


def test_a_wider_existing_mode_is_not_narrowed_either(tmp_path):
    """Symmetric to the test above: the mode is a floor, not a target. An
    operator who published the corpus at 0755 keeps that choice too — deciding
    it for them in either direction is not this function's call."""
    from osprey.deployment.compose_generator import ensure_shared_corpus_dir

    target = tmp_path / "kb"
    target.mkdir(mode=0o755)

    ensure_shared_corpus_dir(target)

    assert stat.S_IMODE(target.stat().st_mode) == 0o2775


def test_no_chmod_when_the_mode_already_satisfies_sharing(tmp_path, caplog):
    """The INFO line reports a change to an operator's directory. A deploy that
    changed nothing must not claim it did — an every-``osprey up`` message about
    permissions nobody touched is noise that trains operators to ignore it."""
    from osprey.deployment.compose_generator import ensure_shared_corpus_dir

    target = tmp_path / "kb"
    target.mkdir()
    ensure_shared_corpus_dir(target)  # first call does the widening

    with caplog.at_level(logging.INFO, logger="deployment.compose"):
        ensure_shared_corpus_dir(target)  # second call has nothing to do

    assert "Shared corpus" not in caplog.text


def test_widening_an_existing_directory_is_reported_at_info(tmp_path, caplog):
    target = tmp_path / "data" / "kb"
    target.mkdir(mode=0o700, parents=True)

    with caplog.at_level(logging.INFO, logger="deployment.compose"):
        _widen(target, relative_to=tmp_path)

    assert "Shared corpus data/kb: 0700 -> 2770" in caplog.text


def test_the_info_line_names_no_absolute_path(tmp_path, caplog):
    """Output hygiene, guarded here as well as in the build's own view.

    ``osprey build``'s default view carries exactly ONE absolute path — the tree
    it just wrote — and ``tests/cli/test_build_output_quiet.py`` asserts that by
    equality, so a second line acquiring a path breaks it. That guard runs a
    whole build; this one states the same rule at the source, where a failure
    names the line that broke it.
    """
    target = tmp_path / "data" / "kb"
    target.mkdir(mode=0o700, parents=True)

    with caplog.at_level(logging.INFO, logger="deployment.compose"):
        _widen(target, relative_to=tmp_path)

    assert str(tmp_path) not in caplog.text
    # ...and the line still fits a normal terminal. Measured on the RECORD, not
    # on caplog.text, which prefixes each line with the logger name and lineno —
    # width the operator never sees.
    emitted = [r.getMessage() for r in caplog.records]
    assert emitted and all(len(m) <= 80 for m in emitted), emitted


def test_a_corpus_outside_the_root_is_still_named(tmp_path, caplog):
    """A bundle on another filesystem has no relative spelling; naming it
    absolutely is better than mangling it or saying nothing."""
    outside = tmp_path / "elsewhere" / "kb"
    outside.mkdir(mode=0o700, parents=True)

    with caplog.at_level(logging.INFO, logger="deployment.compose"):
        _widen(outside, relative_to=tmp_path / "repo")

    assert str(outside) in caplog.text


def _widen(target, *, relative_to):
    from osprey.deployment.compose_generator import ensure_shared_corpus_dir

    return ensure_shared_corpus_dir(target, relative_to=relative_to)


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses the directory permission this test relies on",
)
def test_provisioning_failure_is_not_fatal(tmp_path, caplog):
    """A corpus on a filesystem that refuses the change still deploys:
    single-user sharing works regardless, and the multi-user case fails visibly
    at read time rather than by refusing the whole deploy.

    The failure is forced with a genuinely unwritable parent rather than by
    patching ``os.chmod`` — a module-qualified stdlib patch takes effect
    process-wide, and this suite runs alongside tests that write files.
    """
    from osprey.deployment.compose_generator import ensure_shared_corpus_dir

    parent = tmp_path / "locked"
    parent.mkdir(mode=0o500)
    try:
        with caplog.at_level(logging.WARNING, logger="deployment.compose"):
            assert ensure_shared_corpus_dir(parent / "kb") is None
    finally:
        parent.chmod(0o700)  # so pytest's tmp_path cleanup can remove it

    assert "Could not provision shared corpus directory" in caplog.text
    assert "indexable by the qmd sidecar" in caplog.text


@pytest.mark.parametrize(
    ("bundle_path", "expected"),
    [
        pytest.param("data/kb", "data/kb", id="relative-anchors-on-repo-root"),
        pytest.param("/srv/kb", "/srv/kb", id="absolute-verbatim"),
    ],
)
def test_resolve_facility_bundle_dir(bundle_path, expected, tmp_path):
    """One reader for the key, so the directory that is provisioned, the one the
    sidecar indexes and the one bound into each terminal are provably the same."""
    from osprey.deployment.compose_generator import resolve_facility_bundle_dir

    resolved = resolve_facility_bundle_dir(
        {"facility_knowledge": {"bundle_path": bundle_path}}, tmp_path
    )

    assert resolved is not None
    assert str(resolved) == (expected if expected.startswith("/") else str(tmp_path / expected))


def test_resolve_facility_bundle_dir_is_none_when_unconfigured(tmp_path):
    from osprey.deployment.compose_generator import resolve_facility_bundle_dir

    assert resolve_facility_bundle_dir({}, tmp_path) is None


def test_services_build_path_provisions_the_bundle(tmp_path):
    """The single-user path mounts the same directory into the qmd sidecar —
    READ-ONLY, so the sidecar could never create it — and must not leave it to
    the container runtime either."""
    from osprey.deployment.compose_generator import _ensure_agent_data_structure

    _ensure_agent_data_structure(
        {
            "project_root": str(tmp_path),
            "facility_knowledge": {"bundle_path": BUNDLE_PATH},
        }
    )

    bundle = tmp_path / BUNDLE_PATH
    assert bundle.is_dir()
    assert stat.S_IMODE(bundle.stat().st_mode) & stat.S_ISGID


def test_web_deploy_provisions_the_bundle_before_any_compose_invocation(monkeypatch, tmp_path):
    """Ordering is the point. Left to the runtime, a rootful daemon creates the
    missing bind source owned by ROOT — after which neither the operator who
    authors the bundle nor a non-root container can write it."""
    from osprey.deployment.web_terminals import provision

    from .test_provision import _sidecar_config, _stub_web_stack

    _stub_web_stack(monkeypatch, tmp_path)
    bundle = tmp_path / BUNDLE_PATH
    # What the FIRST container-runtime invocation saw. The assertion is about
    # ordering, so it has to be sampled from inside the run, not after it.
    seen_at_first_compose: list[bool] = []

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(cmd, **kwargs):
        seen_at_first_compose.append(bundle.is_dir())
        return _Completed()

    monkeypatch.setattr(provision.subprocess, "run", _run)
    config = _sidecar_config("local")
    config["facility_knowledge"] = {"bundle_path": BUNDLE_PATH}

    provision.deploy_up_web_terminals(config, [], False, {}, [])

    assert seen_at_first_compose, "no compose invocation ran to order against"
    assert seen_at_first_compose[0] is True
    assert stat.S_IMODE(bundle.stat().st_mode) & stat.S_ISGID
