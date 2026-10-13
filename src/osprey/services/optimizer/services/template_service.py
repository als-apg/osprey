"""Service for managing configuration templates."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from osprey.services.optimizer.models.config import EnvironmentTemplate

logger = logging.getLogger(__name__)

MAX_TEMPLATE_SIZE = 65_536  # 64 KB — same cap as the YAML API endpoint


@dataclass
class TemplateStatus:
    """Status of a template file."""

    name: str
    valid: bool
    error: str | None = None
    content: EnvironmentTemplate | None = None


class TemplateService:
    """Service to scan and validate environment templates."""

    def __init__(self, template_dir: Path | str | None = None) -> None:
        """Initialize with optional template directory.

        Args:
            template_dir: Path to template directory. If None, checks defaults:
                          1. /app/templates (Docker)
                          2. ./templates (Local dev relative to CWD)
        """
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            docker_path = Path("/app/templates")
            local_path = Path("templates")

            if docker_path.exists():
                self.template_dir = docker_path
            else:
                self.template_dir = local_path.resolve()

    def scan_templates(self) -> list[TemplateStatus]:
        """Scan template directory for YAML files and validate them.

        Returns:
            List of TemplateStatus objects describing each found template.
        """
        if not self.template_dir.exists():
            logger.warning("Template directory not found: %s", self.template_dir)
            return []

        results = []
        files = list(self.template_dir.glob("*.yaml")) + list(self.template_dir.glob("*.yml"))

        for file_path in files:
            try:
                content = self._load_template(file_path)
                results.append(
                    TemplateStatus(
                        name=file_path.stem,
                        valid=True,
                        content=content,
                    )
                )
            except (ValidationError, ValueError) as e:
                results.append(
                    TemplateStatus(
                        name=file_path.stem,
                        valid=False,
                        error=str(e),
                    )
                )
            except Exception as e:
                logger.exception("Unexpected error loading template %s", file_path)
                results.append(
                    TemplateStatus(
                        name=file_path.stem,
                        valid=False,
                        error=f"Unexpected error: {e!s}",
                    )
                )

        return sorted(results, key=lambda x: x.name)

    def load_template_config(self, name: str) -> dict[str, Any]:
        """Load a specific template by name and return its config dictionary.

        Args:
            name: Name of the template (filename without extension)

        Returns:
            Dictionary suitable for merging into OptimizationConfig

        Raises:
            FileNotFoundError: If template not found
            ValueError: If template is invalid
        """
        yaml_path = self.template_dir / f"{name}.yaml"
        yml_path = self.template_dir / f"{name}.yml"

        if yaml_path.exists():
            path = yaml_path
        elif yml_path.exists():
            path = yml_path
        else:
            raise FileNotFoundError(f"Template '{name}' not found in {self.template_dir}")

        try:
            template = self._load_template(path)
            return template.model_dump(exclude_none=True)
        except ValidationError as e:
            raise ValueError(f"Invalid template '{name}': {e}") from e

    def _load_template(self, path: Path) -> EnvironmentTemplate:
        """Load and validate a single template file."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_size = path.stat().st_size
        if file_size > MAX_TEMPLATE_SIZE:
            raise ValueError(
                f"Template file too large ({file_size} bytes, max {MAX_TEMPLATE_SIZE}): {path.name}"
            )

        try:
            text = path.read_text()
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Template must be a dictionary")

        return EnvironmentTemplate(**data)
