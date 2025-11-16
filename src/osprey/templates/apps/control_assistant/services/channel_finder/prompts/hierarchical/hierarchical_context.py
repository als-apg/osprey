"""
Hierarchical Navigation Context for Example Accelerator

Provides high-level LLM instructions for navigating the hierarchical channel database.

Architecture:
  - Database (hierarchical_database.json): Contains DATA (tree structure, descriptions)
  - Prompts (this file): Contains INSTRUCTIONS (navigation guidance for LLM)

This maintains clean separation between data and prompt logic.
"""

from typing import Dict


# =============================================================================
# HIERARCHICAL NAVIGATION INSTRUCTIONS
# =============================================================================
# High-level guidance for LLM when navigating each hierarchy level.
# Specific descriptions come from the database options at runtime.

hierarchical_context = {
    "system": """
        Select the top-level system(s) that contain the equipment mentioned in the query.
        
        Use the detailed descriptions provided in the available options to understand 
        each system's purpose and scope.
        
        BRANCHING BEHAVIOR: If you select multiple systems, each will be explored 
        separately with its own set of families, devices, and fields. This ensures 
        correct results when different systems have different structures.
        
        Select multiple systems when the query spans subsystems or is ambiguous.
    """,
    
    "family": """
        Select the device family/families within the chosen system.
        
        Read the family descriptions carefully to understand the differences between 
        similar device types (e.g., different types of magnets or vacuum equipment).
        
        BRANCHING BEHAVIOR: If you select multiple families, each will be explored 
        separately with its own fields and subfields. This is ESSENTIAL because 
        different families have different available fields!
        
        IMPORTANT: Many queries require multiple families:
        - "correctors" or "corrector magnets" → select BOTH HCM and VCM families
        - "tune correction" or "tune quadrupoles" → select BOTH QF and QD families
        - "vacuum levels" or "vacuum pressure" → select BOTH ION-PUMP and GAUGE families
        - "RF system" → select ALL RF families (CAVITY, KLYSTRON, etc.)
        - General category queries → select ALL families in that category
        
        When in doubt, include more families rather than fewer. Each will be properly explored.
    """,
    
    "device": """
        Select specific device instance(s) based on the query.
        
        Guidelines:
        - Specific number/name mentioned → select that device exactly
        - "all" or no specific device → select all available devices
        - Range mentioned → select devices in that range
        - Location/sector mentioned → select devices in that location
        
        Note: All devices within a family have identical structure, so selecting 
        multiple devices does NOT cause branching - they're processed together efficiently.
        
        Device naming patterns are visible in the available options list.
    """,
    
    "field": """
        Select the physical quantity or parameter being referenced.
        
        Read the field descriptions carefully - they explain what each field represents
        for YOUR SPECIFIC path through the hierarchy (current system and family).
        
        BRANCHING BEHAVIOR: If you select multiple fields, each will be explored 
        separately to find its specific subfields. This is important because different 
        fields may have completely different subfield structures.
        
        Common query intents:
        - Measuring/monitoring → look for measurement fields (e.g., CURRENT, VOLTAGE, PRESSURE, POSITION)
        - Setting/controlling → look for control fields  
        - Status/health → look for STATUS fields
        - Reference/golden/saved values → these are typically SUBFIELD modifiers (like SP/RB/GOLDEN), 
          so select the underlying physical field (e.g., CURRENT for "golden current values")
        
        IMPORTANT: Even when the query emphasizes HOW to access a value (setpoint, readback, 
        golden, etc.), you still need to select WHAT field is being accessed. If not explicit, 
        infer from device type (e.g., magnets typically use CURRENT, cavities use POWER/VOLTAGE).
        
        Select multiple fields when the query could apply to several measurements/controls.
    """,
    
    "subfield": """
        Select the specific measurement/control type or channel function FOR THIS FIELD.
        
        IMPORTANT: You are now at a specific point in the hierarchy. The subfield 
        options you see are specific to the current system, family, and field. 
        Read the descriptions carefully as meanings are context-dependent!
        
        Common patterns (vary by context):
        - Setpoint (SP): for commanding/setting values
        - Readback (RB): for monitoring/reading values
        - Golden/Reference: for stored reference values
        - Status indicators (READY, ON, FAULT, etc.): for operational states
        - Axes (X/Y): for directional measurements
        
        Selection guidelines based on query intent:
        - "all PVs" or "all values" → select ALL subfields shown for this field
        - "all X" for specific device → select all relevant subfields for this field
        - Monitoring/reading queries → typically RB (readback) subfields
        - Control/setting queries → SP (setpoint) subfields  
        - "settings" or "configuration" → typically SP only (not both SP and RB)
        - "setting and reading" → both SP and RB
        - Status queries → select status/health subfields (READY, ON, FAULT, etc.)
        - "Golden values" or "reference values" → select GOLDEN subfields
        
        When in doubt about scope: if the query uses "all" or "everything", be inclusive.
        Otherwise, prefer precision - select only the subfields that match the query intent.
    """
}


def get_hierarchical_context() -> Dict[str, str]:
    """
    Get hierarchical navigation instructions.
    
    These instructions guide the LLM when selecting options at each level
    of the hierarchy. Specific descriptions come from the database at runtime.
    
    Returns:
        Dictionary mapping level names to high-level instruction strings
    """
    return hierarchical_context.copy()
