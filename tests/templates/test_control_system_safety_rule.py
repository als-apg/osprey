"""Render tests for the p4p (pvAccess) block in ``control-system-safety.md.j2``.

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
    for heading in ("### Allowed", "### Prohibited", "### Why This Matters", "## Write Operations"):
        assert heading in content, f"missing section heading: {heading}"

    from osprey.services.build_artifacts.catalog import BuildArtifactCatalog

    artifact = BuildArtifactCatalog.default().get("rules/control-system-safety")
    assert artifact is not None
    assert artifact.template_path == "claude/rules/control-system-safety.md.j2"
    assert artifact.output_path == ".claude/rules/control-system-safety.md"
    assert artifact.description
