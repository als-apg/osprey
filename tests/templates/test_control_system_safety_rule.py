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

import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager

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

    The scaffolded projects the other helper builds never enable ``bluesky``,
    so the queue branch is unreachable through them. Rendering the template
    directly is the only way to exercise both sides of that conditional.
    """
    manager = TemplateManager()
    template = manager.jinja_env.get_template(
        "claude_code/claude/rules/control-system-safety.md.j2"
    )
    return template.render(control_system_type=control_system_type, enabled_servers=enabled_servers)


def _render_safety_rule(tmp_path, project_name: str, control_system_type: str | None) -> str:
    """Scaffold a project, set ``control_system.type``, render the Claude Code
    integration files, and return the rendered safety-rule content."""
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=project_name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
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
    for cs_type in ("tango", "opcua", "labview", None):
        label = cs_type or "mock"
        content = _render_safety_rule(tmp_path / label, f"p4p-{label}", cs_type)

        assert "p4p" not in content, f"{label}: p4p leaked outside the EPICS branch"
        for marker in P4P_MARKERS:
            assert marker not in content, f"{label}: unexpected p4p marker {marker!r}"


def test_existing_pyepics_prohibitions_survive(tmp_path):
    """Adding the p4p block must not displace the pyepics examples."""
    content = _render_safety_rule(tmp_path, "p4p-pyepics", "epics")

    assert "import epics" in content
    assert "epics.caget" in content
    assert "epics.caput" in content
    assert "Bypasses audit logging" in content
    assert "Bypasses limits + approval" in content
    assert "Bypasses all safety layers" in content


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
    """The scaffolded control_assistant project runs no ``bluesky`` server, so
    this is the shape a real queue-less deployment renders: the multi-setting
    case still appears, and it forbids substituting a loop of writes."""
    content = _render_safety_rule(tmp_path, "route-no-queue", "epics")

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
