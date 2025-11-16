"""
Pydantic Models for Hierarchical Channel Finder

Defines structured outputs for each level of hierarchical channel selection.
Dynamic model generation constrains LLM to valid options at each level.
"""

from pydantic import BaseModel, Field, create_model
from typing import List
from enum import Enum


# Special marker for "nothing found"
NOTHING_FOUND_MARKER = "NOTHING_FOUND"


def create_selection_model(
    level_name: str,
    available_options: List[str],
    allow_multiple: bool = True
) -> type[BaseModel]:
    """
    Dynamically create a Pydantic model for a specific hierarchy level.
    
    This ensures the LLM can only select from valid options or indicate nothing was found.
    
    Args:
        level_name: Name of the hierarchy level (e.g., "system", "family")
        available_options: List of valid option names at this level
        allow_multiple: Whether to allow multiple selections (default: True)
        
    Returns:
        Dynamically created Pydantic model class
        
    Example:
        >>> options = ["BeamDiagnostics", "VacuumSystem"]
        >>> Model = create_selection_model("system", options)
        >>> result = Model(selections=["BeamDiagnostics"])
    """
    # Add NOTHING_FOUND to the options
    all_options = available_options + [NOTHING_FOUND_MARKER]
    
    # Create a dynamic Enum with all valid options
    # Enum values are the option names themselves
    enum_name = f"Option_{level_name}"
    OptionEnum = Enum(
        enum_name,
        {opt: opt for opt in all_options},
        type=str
    )
    
    # Create the model
    if allow_multiple:
        # Allow list of selections
        model_name = f"HierarchicalSelection_{level_name}"
        field_description = (
            f"Selected option(s) at the {level_name} level. "
            f"Can be multiple for wildcards or ambiguous queries. "
            f"Use '{NOTHING_FOUND_MARKER}' if no relevant options found."
        )
        
        return create_model(
            model_name,
            selections=(
                List[OptionEnum],
                Field(description=field_description)
            )
        )
    else:
        # Single selection only
        model_name = f"HierarchicalSelection_{level_name}_single"
        field_description = (
            f"Single selected option at the {level_name} level. "
            f"Use '{NOTHING_FOUND_MARKER}' if no relevant option found."
        )
        
        return create_model(
            model_name,
            selection=(
                OptionEnum,
                Field(description=field_description)
            )
        )
