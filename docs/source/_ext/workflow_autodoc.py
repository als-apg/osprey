"""Sphinx extension for auto-documenting workflow files.

Similar to autodoc for Python code, this extension reads workflow markdown files
with YAML frontmatter and generates formatted RST documentation.

Usage in RST files:

    .. workflow-summary:: ../../workflows/pre-merge-cleanup.md

    .. workflow-list::
       :category: code-quality

"""

import re
from pathlib import Path
from typing import Any

import yaml
from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective


def format_use_when(applies_when: list[str]) -> str:
    """Convert applies_when list to a readable 'use when' sentence.

    Args:
        applies_when: List of workflow application contexts

    Returns:
        Human-readable sentence describing when to use the workflow
    """
    if not applies_when:
        return ""

    # Convert snake_case to readable text
    readable = [item.replace("_", " ") for item in applies_when]

    if len(readable) == 1:
        return readable[0].capitalize()
    elif len(readable) == 2:
        return f"{readable[0].capitalize()} or {readable[1]}"
    else:
        return f"{', '.join(readable[:-1]).capitalize()}, or {readable[-1]}"


def parse_workflow_file(filepath: Path) -> dict[str, Any]:
    """Extract YAML frontmatter and metadata from a workflow markdown file.

    Args:
        filepath: Path to the workflow markdown file

    Returns:
        Dictionary containing workflow metadata and content

    Example:
        >>> data = parse_workflow_file(Path("pre-merge-cleanup.md"))
        >>> data['metadata']['workflow']
        'pre-merge-cleanup'
    """
    if not filepath.exists():
        return {
            "metadata": {},
            "title": filepath.stem,
            "description": "",
            "content": "",
            "use_when": "",
        }

    content = filepath.read_text(encoding="utf-8")

    # Extract YAML frontmatter (between --- markers)
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)

    metadata = {}
    remaining_content = content

    if frontmatter_match:
        yaml_content = frontmatter_match.group(1)
        try:
            metadata = yaml.safe_load(yaml_content)
        except yaml.YAMLError:
            pass
        remaining_content = content[frontmatter_match.end() :]

    # Extract title (first # heading)
    title_match = re.search(r"^#\s+(.+)$", remaining_content, re.MULTILINE)
    title = title_match.group(1) if title_match else filepath.stem

    # Extract description (text between title and first ## heading or AI Quick Start)
    desc_pattern = r"^#\s+.+?\n\s*\n(.*?)(?=\n##|\n\n##|$)"
    desc_match = re.search(desc_pattern, remaining_content, re.DOTALL | re.MULTILINE)
    description = desc_match.group(1).strip() if desc_match else ""

    # Clean up description (remove ** markers, keep single line)
    description = re.sub(r"\*\*(.+?)\*\*", r"\1", description)
    description = " ".join(description.split())

    # Generate 'use when' text from applies_when metadata
    use_when = ""
    if "applies_when" in metadata and metadata["applies_when"]:
        use_when = format_use_when(metadata["applies_when"])

    return {
        "metadata": metadata or {},
        "title": title,
        "description": description,
        "content": remaining_content,
        "filepath": filepath,
        "use_when": use_when,
    }


class WorkflowSummaryDirective(SphinxDirective):
    """Directive to auto-document a single workflow file.

    Usage:
        .. workflow-summary:: ../../workflows/pre-merge-cleanup.md
           :show-metadata: true
           :show-use-when: true
    """

    required_arguments = 1  # workflow file path
    optional_arguments = 0
    option_spec = {
        "show-metadata": directives.flag,
        "compact": directives.flag,
        "show-use-when": directives.flag,
        "use-when-only": directives.flag,
    }
    has_content = False

    def run(self) -> list[nodes.Node]:
        """Generate RST nodes for the workflow summary."""
        # Resolve file path relative to current document
        env = self.state.document.settings.env
        source_dir = Path(env.srcdir)
        doc_dir = Path(env.doc2path(env.docname)).parent

        workflow_path_str = self.arguments[0]
        workflow_path = (doc_dir / workflow_path_str).resolve()

        # Parse workflow file
        data = parse_workflow_file(workflow_path)
        metadata = data["metadata"]

        # Build output nodes
        output = []

        # If use-when-only flag, just return the use_when text
        if "use-when-only" in self.options:
            if data["use_when"]:
                return [nodes.Text(data["use_when"])]
            return [nodes.Text("Various development scenarios")]

        # Title
        title_node = nodes.strong(text=data["title"])
        output.append(title_node)
        output.append(nodes.Text("\n\n"))

        # Use When (if requested)
        if "show-use-when" in self.options and data["use_when"]:
            use_when_para = nodes.paragraph()
            use_when_para += nodes.strong(text="Use when: ")
            use_when_para += nodes.Text(data["use_when"])
            output.append(use_when_para)

        # Description
        if data["description"]:
            desc_node = nodes.paragraph(text=data["description"])
            output.append(desc_node)

        # Metadata (if requested)
        if "show-metadata" in self.options and metadata:
            bullet_list = nodes.bullet_list()

            # Category
            if "category" in metadata:
                item = nodes.list_item()
                para = nodes.paragraph()
                para += nodes.strong(text="Category: ")
                para += nodes.Text(metadata["category"])
                item += para
                bullet_list += item

            # Estimated time
            if "estimated_time" in metadata:
                item = nodes.list_item()
                para = nodes.paragraph()
                para += nodes.strong(text="Estimated Time: ")
                para += nodes.Text(metadata["estimated_time"])
                item += para
                bullet_list += item

            # When to use
            if "applies_when" in metadata:
                item = nodes.list_item()
                para = nodes.paragraph()
                para += nodes.strong(text="Use When: ")
                applies = metadata["applies_when"]
                if isinstance(applies, list):
                    applies = ", ".join(applies)
                para += nodes.Text(str(applies))
                item += para
                bullet_list += item

            # AI Ready
            if "ai_ready" in metadata:
                item = nodes.list_item()
                para = nodes.paragraph()
                para += nodes.strong(text="AI Ready: ")
                para += nodes.Text("Yes ✓" if metadata["ai_ready"] else "No")
                item += para
                bullet_list += item

            output.append(bullet_list)

        # Link to workflow file (skip if link generation fails due to path issues)
        try:
            # Try to create a relative path from docs root to workflow file
            docs_root = source_dir.parent
            repo_root = docs_root.parent  # Go up from docs/ to repo root
            rel_path = workflow_path.relative_to(repo_root)

            link_node = nodes.paragraph()
            link_node += nodes.Text("📄 ")
            ref = nodes.reference(
                text=f"View workflow: {rel_path}",
                refuri=f"../../../{rel_path}",  # From docs/source/contributing/ up to repo root
            )
            link_node += ref
            output.append(link_node)
        except (ValueError, AttributeError):
            # Skip link if path calculation fails
            pass

        # Container for styling
        container = nodes.container()
        container["classes"].append("workflow-summary")
        container.extend(output)

        return [container]


class WorkflowListDirective(SphinxDirective):
    """Directive to list and summarize multiple workflows.

    Usage:
        .. workflow-list::
           :category: code-quality
           :compact:
    """

    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        "category": directives.unchanged,
        "compact": directives.flag,
    }
    has_content = False

    def run(self) -> list[nodes.Node]:
        """Generate a list of workflows."""
        env = self.state.document.settings.env
        source_dir = Path(env.srcdir)
        workflows_dir = source_dir.parent / "workflows"

        if not workflows_dir.exists():
            warning = nodes.warning()
            warning += nodes.paragraph(text=f"Workflows directory not found: {workflows_dir}")
            return [warning]

        # Find all workflow markdown files
        workflow_files = sorted(workflows_dir.glob("*.md"))
        workflow_files = [f for f in workflow_files if f.name != "README.md"]

        # Parse all workflows
        workflows = []
        for wf_path in workflow_files:
            data = parse_workflow_file(wf_path)
            # Filter by category if specified
            if "category" in self.options:
                if data["metadata"].get("category") != self.options["category"]:
                    continue
            workflows.append(data)

        # Build output
        output = []

        if "compact" in self.options:
            # Compact list format
            bullet_list = nodes.bullet_list()
            for wf in workflows:
                item = nodes.list_item()
                para = nodes.paragraph()

                # Workflow name in bold
                para += nodes.strong(text=wf["title"])
                para += nodes.Text(" — ")

                # Description
                para += nodes.Text(
                    wf["description"][:100] + ("..." if len(wf["description"]) > 100 else "")
                )

                item += para
                bullet_list += item

            output.append(bullet_list)
        else:
            # Detailed format with sections
            for wf in workflows:
                # Section for each workflow
                section = nodes.section()
                section["ids"].append(wf["metadata"].get("workflow", wf["title"]))

                # Title
                title = nodes.title(text=wf["title"])
                section += title

                # Description
                if wf["description"]:
                    section += nodes.paragraph(text=wf["description"])

                # Metadata
                meta = wf["metadata"]
                if meta:
                    field_list = nodes.field_list()

                    for key in ["category", "estimated_time", "ai_ready"]:
                        if key in meta:
                            field = nodes.field()
                            field_name = nodes.field_name(text=key.replace("_", " ").title())
                            field_body = nodes.field_body()
                            field_body += nodes.paragraph(text=str(meta[key]))
                            field += field_name
                            field += field_body
                            field_list += field

                    section += field_list

                output.append(section)

        return output


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the workflow autodoc extension with Sphinx."""
    app.add_directive("workflow-summary", WorkflowSummaryDirective)
    app.add_directive("workflow-list", WorkflowListDirective)

    # Add CSS for workflow summary boxes
    app.add_css_file("workflow-autodoc.css")

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
