"""Templates for project scaffolding.

This package contains Jinja2 templates for creating new projects from the
Osprey Framework. Templates are organized into:

- project/ : Base project structure (config, pyproject.toml, README, etc.)
- apps/ : Application code templates (minimal, hello_world_weather, wind_turbine)
- services/ : Docker/Podman service configurations (Jupyter, OpenWebUI, Pipelines)

Templates are bundled with the pip-installed framework package and used by
the CLI scaffolding commands (osprey init, osprey create-example, etc.).
"""

__all__ = []
