"""Shared template rendering function."""

from pathlib import Path
from typing import Any

from jinja2 import Environment


def render_template(
    jinja_env: Environment, template_path: str, context: dict[str, Any], output_path: Path
):
    """Render a single template file.

    Args:
        jinja_env: Jinja2 environment for template rendering
        template_path: Relative path to template within templates directory
        context: Dictionary of variables for template rendering
        output_path: Path where rendered output should be written

    Raises:
        jinja2.TemplateNotFound: If template file doesn't exist
        IOError: If output file cannot be written
    """
    template = jinja_env.get_template(template_path)
    rendered = template.render(**context)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use UTF-8 encoding explicitly to support Unicode characters on Windows
    output_path.write_text(rendered, encoding="utf-8")
