"""
Example Middle Layer Facility Prompts

This package contains facility-specific prompts for the example middle layer accelerator.

For middle_layer pipeline:
- system.py: Facility description and terminology
- query_splitter.py: Stage 1 query splitting

Note: Unlike in-context pipeline, middle_layer doesn't use channel_matcher or correction prompts.
The React agent uses database query tools to explore the functional hierarchy and find channels.

Architecture:
  - Database (middle_layer.json): Contains ONLY DATA (functional hierarchy, channel addresses)
  - Prompts: Contains INSTRUCTIONS (query splitting and agent system prompt)
  - Agent Tools: Provides database exploration capabilities

This maintains clean separation: data vs prompts vs tools.
"""

from . import query_splitter
from .system import facility_description

__all__ = ["facility_description", "query_splitter"]
