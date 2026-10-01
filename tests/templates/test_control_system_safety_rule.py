"""Render tests for ``control-system-safety.md.j2``.

Two contracts live here: the p4p (pvAccess) prohibition block, and the routing
section that tells the agent which tool answers which request shape.

The EPICS connector reads pvAccess channels through ``p4p``, so the shipped
safety rule has to name that library the same way it names ``pyepics`` --
otherwise an agent steered off ``epics.caput`` simply reaches for
``Context.put`` instead. Two details matter enough to pin here:

* Both client flavors are named. ``p4p`` ships parallel ``thread`` and
  ``asyncio`` client classes, and a rule that mentions only the threaded one
  leaves the async spelling looking permissible.
* ``rpc`` is called out as refused rather than merely discouraged. Approval
  cannot mediate an arbitrary rpc payload, so the rule has to say the call is
  not approvable instead of letting the agent burn a round trip discovering it.

The block belongs to the EPICS-family branch only: ``epics`` and
``virtual_accelerator`` (a plain ``EPICSConnector`` subclass) get it, every
other control-system type is left untouched.
"""

from pathlib import Path

import pytest
import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager


def _bundle_data_root(bundle: str = "control_assistant") -> Path:
    """The tree these fixtures hand the render as the profile's ``data:``.

    A build copies the tree its profile's ``data:`` key names, and that key is
    required — nothing falls back to a packaged tree any more. These fixtures
    render straight from a bundle rather than from a profile, so they name the
    tree that bundle packages, which is the content the render used to reach
    for on its own.
    """
    return Path(TemplateManager().template_root) / "apps" / bundle / "data"


def _create_project(manager: TemplateManager, **kwargs) -> Path:
    """``create_project`` plus the three steps a real build takes next.

    A build renders the framework template, overlays the resolved profile's
    ``config:`` block onto the result, stamps ``.osprey-manifest.json``, and
    regenerates ``.claude/`` from the finished config. The template carries
    only derived and profile-field-derived keys, so a fixture that stops after
    the render holds half a config — the declarative half is the preset's, and
    the artifacts rendered before it landed do not know about the deployment's
    control system, services or servers. These fixtures render from a bundle
    rather than from a profile, so they overlay the preset ``osprey init``
    pairs with that bundle.
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.utils.config_writer import config_update_fields

    bundle = kwargs.setdefault("data_bundle", "control_assistant")
    preset = bundle.replace("_", "-")
    kwargs.setdefault("data_root", _bundle_data_root(bundle))
    project = manager.create_project(**kwargs)
    profile, _preset_dir = resolve_build_profile(None, preset=preset)
    config_update_fields(project / "config.yml", profile.config)
    manager.generate_manifest(
        project, kwargs["project_name"], preset, {}, artifacts=kwargs.get("artifacts")
    )
    # The build's last render, and the one that ships: `create_project` wrote
    # `.claude/` from a config.yml that did not yet carry the preset's block.
    manager.regenerate_claude_code(project)
    return project


#: Lines the p4p block must contain, verbatim.
P4P_LINES = (
    "from p4p.client.thread import Context",
    "from p4p.client.asyncio import Context",
    "ctxt.get(",
    "ctxt.put(",
    "ctxt.rpc(",
)

#: Every p4p marker, including the refusal wording that separates rpc from the
#: merely-prohibited calls.
P4P_MARKERS = P4P_LINES + ("Not approvable — refused at runtime",)

#: The routing section's heading and the four protocol-neutral cases. The
#: multi-setting case is deployment-dependent and pinned separately.
ROUTING_HEADING = "## Choosing the Right Tool"
ROUTING_CASES = (
    "**One live value** — use `channel_read`",
    "**One live write the operator asked for by name** — use `channel_write`",
    "**Python analysis over data you already retrieved** — use `execute`",
    "**Python that must touch the control system** — use `execute`",
)

#: The one asymmetry the agent has to know: a write the machine did not confirm
#: reaches the two paths in different shapes.
WRITE_PATH_SENTENCE = (
    "`osprey.runtime.write_channel` raises, while `channel_write` reports it in "
    "the result's `outcome`."
)


def _render_template_directly(control_system_type: str, enabled_servers: set[str]) -> str:
    """Render the rule straight from Jinja with an explicit ``enabled_servers``.

    Which servers a scaffolded project enables is its preset's statement, so a
    project can only exercise whichever arm its preset happens to arm. Rendering
    the template directly is how both sides of the conditional are reached from
    one control-system type.
    """
    manager = TemplateManager()
    template = manager.jinja_env.get_template(
        "claude_code/claude/rules/control-system-safety.md.j2"
    )
    return template.render(control_system_type=control_system_type, enabled_servers=enabled_servers)


def _render_safety_rule(
    tmp_path,
    project_name: str,
    control_system_type: str | None,
    bundle: str = "control_assistant",
) -> str:
    """Scaffold a project, set ``control_system.type``, render the Claude Code
    integration files, and return the rendered safety-rule content."""
    manager = TemplateManager()
    project_dir = _create_project(
        manager,
        project_name=project_name,
        output_dir=tmp_path,
        data_bundle=bundle,
        context={"channel_finder_mode": "hierarchical"},
    )

    config = yaml.safe_load((project_dir / "config.yml").read_text())
    if control_system_type is not None:
        config.setdefault("control_system", {})["type"] = control_system_type
        (project_dir / "config.yml").write_text(yaml.dump(config))

    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )
    claude_code.create_claude_code_integration(
        manager.template_root, manager.jinja_env, project_dir, ctx
    )

    return (project_dir / ".claude" / "rules" / "control-system-safety.md").read_text()


def test_epics_rule_names_p4p(tmp_path):
    """The ``epics`` branch carries the full p4p prohibition block."""
    content = _render_safety_rule(tmp_path, "p4p-epics", "epics")

    for marker in P4P_MARKERS:
        assert marker in content, f"epics rule missing p4p marker: {marker!r}"


def test_virtual_accelerator_rule_names_p4p(tmp_path):
    """``virtual_accelerator`` shares the EPICS branch, so it shares the block."""
    content = _render_safety_rule(tmp_path, "p4p-va", "virtual_accelerator")

    for marker in P4P_MARKERS:
        assert marker in content, f"virtual_accelerator rule missing p4p marker: {marker!r}"


def test_both_client_flavors_and_rpc_carry_bypass_annotations(tmp_path):
    """Every p4p example line is annotated the way the pyepics examples are --
    a bare import list would not tell the agent what it is bypassing."""
    content = _render_safety_rule(tmp_path, "p4p-annot", "epics")

    annotated = {
        line.split("#", 1)[0].strip(): line.split("#", 1)[1].strip()
        for line in content.splitlines()
        if "#" in line and line.strip().startswith(("from p4p", "ctxt."))
    }

    assert annotated, "no annotated p4p example lines rendered"
    for stem, annotation in annotated.items():
        assert annotation, f"p4p example line has an empty annotation: {stem!r}"

    rpc_annotations = [a for stem, a in annotated.items() if stem.startswith("ctxt.rpc(")]
    assert rpc_annotations, "ctxt.rpc example line is not annotated"
    for annotation in rpc_annotations:
        assert "Not approvable" in annotation
        assert "refused at runtime" in annotation


def test_rpc_refusal_is_explained_honestly(tmp_path):
    """The prose behind the rpc line says approval cannot rescue the call, and
    names when the runtime refusal actually applies (readonly runs; limits
    checking) rather than overclaiming an unconditional block."""
    content = _render_safety_rule(tmp_path, "p4p-rpc-prose", "epics")

    prose = " ".join(content.split())
    assert "`ctxt.rpc(...)` is the one to remember" in prose
    assert "there is nothing for the approval workflow to check" in prose
    assert "refused at runtime in every readonly run" in prose
    assert "wherever limits checking is enabled" in prose
    assert "Use `write_channel` for the write you actually need." in prose


def test_epics_and_virtual_accelerator_prohibited_sections_still_match(tmp_path):
    """The block is added to the shared branch, not duplicated per type."""
    epics_content = _render_safety_rule(tmp_path / "epics", "p4p-epics", "epics")
    va_content = _render_safety_rule(tmp_path / "va", "p4p-va", "virtual_accelerator")

    def _prohibited_section(content: str) -> str:
        start = content.index("### Prohibited")
        end = content.index("### Why This Matters")
        return content[start:end]

    assert _prohibited_section(epics_content) == _prohibited_section(va_content)


def test_non_epics_branches_have_no_p4p_lines(tmp_path):
    """Tango, OPC-UA, LabVIEW and the generic branch are unchanged -- p4p is an
    EPICS-family library and naming it elsewhere would be noise."""
    for cs_type in ("tango", "opcua", "labview", "mock"):
        content = _render_safety_rule(tmp_path / cs_type, f"p4p-{cs_type}", cs_type)

        assert "p4p" not in content, f"{cs_type}: p4p leaked outside the EPICS branch"
        for marker in P4P_MARKERS:
            assert marker not in content, f"{cs_type}: unexpected p4p marker {marker!r}"


def test_existing_pyepics_prohibitions_survive(tmp_path):
    """Adding the p4p block must not displace the pyepics examples."""
    content = _render_safety_rule(tmp_path, "p4p-pyepics", "epics")

    assert "import epics" in content
    assert "epics.caget" in content
    assert "epics.caput" in content
    assert "Bypasses audit logging" in content
    assert RAW_PUT_ANNOTATION in content


#: The annotation every raw client put example carries in the branches whose
#: client libraries the executor and the notebook kernels refuse at runtime.
RAW_PUT_ANNOTATION = "# Refused at runtime: RAW_CLIENT_WRITE"

#: The raw-put example lines of each refusing branch, stem before the ``#``.
RAW_PUT_LINES = {
    "epics": (
        'epics.caput("SR:MAG:QF:01:CURRENT:SP", 150)',
        "pv.put(150)",
    ),
    "doocs": ('doocs4py.set("FACILITY/DEVICE/LOCATION/SETPOINT", 150.0)',),
    "tango": (
        'device.write_attribute("Current", 150)',
        'dev.write_attribute("Setpoint", 150)',
    ),
}


@pytest.mark.parametrize("cs_type", ["epics", "virtual_accelerator", "doocs", "tango"])
def test_raw_put_lines_say_refused_and_name_write_channel(cs_type):
    """Where the runtime refuses a raw client put, the rule says so -- a put
    marked merely as a bypass reads as a riskier route that approval could
    still let through -- and names the calls that do write."""
    content = _render_template_directly(cs_type, set())
    branch = "epics" if cs_type == "virtual_accelerator" else cs_type

    annotated = {
        line.split("#", 1)[0].strip(): "#" + line.split("#", 1)[1]
        for line in content.splitlines()
        if "#" in line
    }
    for stem in RAW_PUT_LINES[branch]:
        assert stem in annotated, f"{cs_type}: raw-put example missing: {stem!r}"
        assert annotated[stem].strip() == RAW_PUT_ANNOTATION, (
            f"{cs_type}: {stem!r} annotated {annotated[stem]!r}"
        )

    prose = " ".join(content.split())
    assert "refuse it at runtime with `RAW_CLIENT_WRITE`" in prose
    assert "in readwrite runs too" in prose
    assert "approving the run does not change that" in prose
    assert "`write_channel(address, value)`" in prose
    assert "`write_channels({address: value, ...})`" in prose
    assert "Bypasses limits + approval" not in content
    assert "Bypasses all safety layers" not in content


@pytest.mark.parametrize("cs_type", ["epics", "virtual_accelerator"])
def test_the_pva_put_is_named_as_the_one_exception(cs_type):
    """The connector does not write pvAccess yet, so the runtime limits-checks
    a raw PVA put instead of refusing it. The rule must not call it refused,
    and must say it is for a PVA channel only and still asks for approval."""
    content = _render_template_directly(cs_type, set())
    put_lines = [line for line in content.splitlines() if line.startswith("ctxt.put(")]

    assert len(put_lines) == 1, put_lines
    assert "RAW_CLIENT_WRITE" not in put_lines[0]
    assert "PVA channel only" in put_lines[0]
    prose = " ".join(content.split())
    assert "the one exception to that refusal, for writes only" in prose
    assert "`write_channel` does not write pvAccess channels yet" in prose
    assert "asks for approval" in prose
    assert "a Channel Access channel goes through `write_channel`" in prose


@pytest.mark.parametrize("cs_type", ["opcua", "labview", "mock"])
def test_non_refusing_branches_make_no_runtime_refusal_claim(cs_type):
    """The raw-put refusal covers the EPICS, DOOCS and Tango client libraries;
    naming it for any other branch would claim a guard that is not there."""
    content = _render_template_directly(cs_type, set())

    assert "RAW_CLIENT_WRITE" not in content


def test_rule_heading_contract_intact(tmp_path):
    """The rule's discovery contract -- its heading and section structure --
    is what the build keys on; the p4p block must not disturb it."""
    content = _render_safety_rule(tmp_path, "p4p-contract", "epics")

    assert content.lstrip().startswith("# Control System Safety — EPICS Channel Access")
    for heading in (
        "### Allowed",
        "### Prohibited",
        "### Why This Matters",
        "## Write Operations",
        ROUTING_HEADING,
    ):
        assert heading in content, f"missing section heading: {heading}"

    from osprey.services.build_artifacts.catalog import BuildArtifactCatalog

    artifact = BuildArtifactCatalog.default().get("rules/control-system-safety")
    assert artifact is not None
    assert artifact.template_path == "claude/rules/control-system-safety.md.j2"
    assert artifact.output_path == ".claude/rules/control-system-safety.md"
    assert artifact.description


def test_routing_section_renders_for_every_control_system_type(tmp_path):
    """The routing cases are about tools, not protocols, so every render gets
    them -- a mock deployment routes requests the same way an EPICS one does."""
    for cs_type in (
        "epics",
        "virtual_accelerator",
        "live_standin",
        "tango",
        "opcua",
        "labview",
        None,
    ):
        label = cs_type or "mock"
        content = _render_safety_rule(tmp_path / label, f"route-{label}", cs_type)

        assert ROUTING_HEADING in content, f"{label}: routing section missing"
        for case in ROUTING_CASES:
            assert case in content, f"{label}: missing routing case: {case!r}"


def test_routing_section_is_protocol_neutral(tmp_path):
    """The section must carry no protocol-specific text: it renders on every
    control-system type, and ``epics``/``virtual_accelerator`` are included
    because those are the only renders where the string ``EPICS`` exists
    elsewhere in the document -- they are where a leak into the routing
    section would actually be caught."""
    for cs_type in (
        "epics",
        "virtual_accelerator",
        "live_standin",
        "tango",
        "opcua",
        "labview",
        None,
    ):
        label = cs_type or "mock"
        content = _render_safety_rule(tmp_path / label, f"neutral-{label}", cs_type)

        start = content.index(ROUTING_HEADING)
        next_heading = content.find("\n## ", start + len(ROUTING_HEADING))
        end = next_heading if next_heading != -1 else len(content)
        section = content[start:end]
        for protocol_marker in ("EPICS", "epics", "Tango", "tango", "OPC-UA", "LabVIEW", "p4p"):
            assert protocol_marker not in section, (
                f"{label}: routing section names a protocol: {protocol_marker!r}"
            )


def test_routing_section_states_the_write_path_asymmetry(tmp_path):
    """The Python path raises; the tool reports. An agent that does not know
    the difference narrates a write the machine never confirmed as a success.

    The key the sentence sends the agent to is the one the tool emits -- the
    same parity ``safety.md``'s item 6 is pinned to."""
    from osprey.mcp_server.control_system.tools.channel_write import OUTCOME_KEY

    content = _render_safety_rule(tmp_path, "route-asym", "epics")

    prose = " ".join(content.split())
    assert WRITE_PATH_SENTENCE in prose
    assert f"`{OUTCOME_KEY}`" in WRITE_PATH_SENTENCE


def test_routing_section_without_bluesky_refuses_the_write_loop(tmp_path):
    """The scaffolded hello_world project runs only the ``controls`` server, so
    this is the shape a real queue-less deployment renders: the multi-setting
    case still appears, and it forbids substituting a loop of writes.

    hello-world rather than control-assistant: the latter's preset arms the
    Bluesky lane, so a project built from it has a queue and cannot reach this
    branch.
    """
    content = _render_safety_rule(tmp_path, "route-no-queue", "epics", bundle="hello_world")

    prose = " ".join(content.split())
    assert "**Multi-setting measurements** — no queue is configured in this deployment." in prose
    assert "Do not stand in for one with a loop of `channel_write` calls" in prose
    assert "an `execute` script that steps the setpoint" in prose
    assert "tell the operator what the measurement would require" in prose

    # Nothing may point at a queue this deployment does not have.
    assert "Bluesky" not in content
    assert "## Measurements Go Through the Queue" not in content


def test_routing_section_with_bluesky_points_at_the_queue():
    """With the server enabled the same bullet routes to the queue instead, and
    the existing queue sections still render behind it."""
    content = _render_template_directly("epics", {"controls", "bluesky"})

    prose = " ".join(content.split())
    assert "**Multi-setting measurements** — use the Bluesky queue." in prose
    assert "no queue is configured in this deployment" not in prose
    assert "## Measurements Go Through the Queue, Not Through Repeated Writes" in content
    assert "## Starting a Bluesky Queue" in content


def test_routing_section_renders_exactly_one_multi_setting_case():
    """Guard against both arms of the conditional escaping at once."""
    for enabled in ({"controls"}, {"controls", "bluesky"}):
        content = _render_template_directly("epics", enabled)
        assert content.count("**Multi-setting measurements**") == 1, enabled


def test_execution_mode_write_never_renders(tmp_path):
    """``write`` is not a recognised execution mode -- ``readonly`` and
    ``readwrite`` are the only two the executor accepts -- so the rendered rule
    must never spell it. Regression guard on wording that is already correct."""
    from osprey.mcp_server.python_executor.tools._execution_gates import VALID_EXECUTION_MODES

    assert "write" not in VALID_EXECUTION_MODES

    for cs_type in ("epics", "tango", None):
        label = cs_type or "mock"
        content = _render_safety_rule(tmp_path / label, f"mode-{label}", cs_type)

        assert 'execution_mode: "write"' not in content, f"{label}: rendered an invalid mode"
        assert 'execution_mode: "readwrite"' in content, f"{label}: lost the readwrite mode"

    for enabled in ({"controls"}, {"controls", "bluesky"}):
        content = _render_template_directly("epics", enabled)
        assert 'execution_mode: "write"' not in content, enabled


#: The pvaccess example lines, stem before the ``#``, with the annotation each
#: must carry. The split follows the provider: a Channel Access channel's put
#: is refused, a pvAccess channel's put is the PVA exception.
PVACCESS_LINES = {
    "import pvaccess": "# A raw client: reads skip audit",
    "ch.get()": "# DO NOT: bypasses audit logging; use read_channel",
    "ch.monitor(callback)": "# DO NOT: bypasses audit logging; use read_channel",
    "ch.putDouble(2.0)": "# PVA channel only: approval + limits check",
    "ca.put(150)": RAW_PUT_ANNOTATION,
    "ca.putDouble(150.0)": RAW_PUT_ANNOTATION,
    "pvaccess.MultiChannel(names).putAsDoubleArray(v)": RAW_PUT_ANNOTATION,
    'pvaccess.RpcClient("SR:SVC:ORBIT").invoke(req)': "# Not approvable — refused at runtime",
}


@pytest.mark.parametrize("cs_type", ["epics", "virtual_accelerator"])
def test_the_pvaccess_block_splits_on_the_provider(cs_type):
    """pvaPy speaks both protocols, so the rule names both channels: the CA
    one's puts refused like ``epics.caput``, the PVA one's under the PVA
    exception, MultiChannel writes refused, and the rpc not approvable."""
    content = _render_template_directly(cs_type, set())
    annotated = {
        line.split("#", 1)[0].strip(): "#" + line.split("#", 1)[1]
        for line in content.splitlines()
        if "#" in line
    }

    assert 'ca = pvaccess.Channel("SR:MAG:QF:01:CURRENT:SP", pvaccess.CA)' in content
    for stem, annotation in PVACCESS_LINES.items():
        assert stem in annotated, f"{cs_type}: pvaccess example missing: {stem!r}"
        assert annotated[stem].strip() == annotation, f"{cs_type}: {stem!r} -> {annotated[stem]!r}"
    prose = " ".join(content.split())
    assert "the split follows the channel, not the library" in prose
    assert "its puts are refused like `epics.caput`" in prose
    assert "A `MultiChannel` write is refused whichever protocol it speaks" in prose
    # The p4p block above is untouched: still exactly one ``ctxt.put(`` line.
    assert len([line for line in content.splitlines() if line.startswith("ctxt.put(")]) == 1


@pytest.mark.parametrize("cs_type", ["doocs", "tango", "opcua", "labview", "mock"])
def test_non_epics_branches_have_no_pvaccess_lines(cs_type):
    """pvaPy is an EPICS client; naming it elsewhere would describe a guard
    the branch's deployment has no use for."""
    content = _render_template_directly(cs_type, set())

    assert "pvaccess" not in content
    assert "RpcClient" not in content
