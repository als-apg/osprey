"""
Example Hierarchical Facility Prompts

This package contains facility-specific prompts for the example hierarchical accelerator.

For hierarchical pipeline:
- system.py: Facility description and terminology
- query_splitter.py: Stage 1 query splitting
- hierarchical_context.py: Hierarchical navigation context and level instructions

Note: Unlike in-context pipeline, hierarchical doesn't use channel_matcher or correction prompts.
The hierarchical navigation system handles matching and validation during tree traversal.

Architecture:
  - Database (hierarchical_database.json): Contains ONLY DATA (tree structure, descriptions)
  - Prompts (hierarchical_context.py): Contains INSTRUCTIONS (hierarchical_context for LLM navigation)
  
This maintains clean separation: data vs prompts.
"""

from .system import facility_description
from . import query_splitter
from . import hierarchical_context

__all__ = [
    'facility_description', 
    'query_splitter', 
    'hierarchical_context'
]
