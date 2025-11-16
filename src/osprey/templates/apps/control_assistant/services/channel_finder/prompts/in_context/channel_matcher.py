"""Channel Matcher Prompt for Stage 2."""

import textwrap


def get_prompt(
    atomic_query: str, 
    channel_database: str,
    facility_name: str = "UCSB FEL",
    facility_description: str = None
) -> str:
    """Prompt for Stage 2: Channel matching.
    
    Args:
        atomic_query: The atomic query to match
        channel_database: Formatted channel database string
        facility_name: Name of the facility (ignored, always uses UCSB FEL)
        facility_description: Facility description (uses default if not provided)
        
    Returns:
        Formatted prompt string for channel matching
    """
    # Use provided description or fall back to default
    if facility_description is None:
        from .system import facility_description as default_desc
        facility_description = default_desc
    
    return textwrap.dedent(f"""
        You are a channel finder for the UCSB Free Electron Laser (FEL) facility.
        
        FACILITY CONTEXT:
        {facility_description}
        
        USER QUERY: "{atomic_query}"
        
        AVAILABLE CHANNELS:
        {channel_database}
        
        MATCHING RULES - Apply ALL constraints strictly:
        
        1. LOCATION SPECIFICITY:
           - When query specifies a location, match ONLY channels at that location
           - Different locations have different hardware - do not mix them
           - Use channel descriptions to determine location context
           - EXCLUDE all channels from other locations when location is specified
        
        2. DEVICE TYPE BOUNDARIES:
           - Each device type is distinct hardware - do NOT cross boundaries
           - Match device types exactly as specified in the query
           - Similar-function devices are still different types (e.g., steering coil ≠ dipole magnet)
           - Check both channel names AND descriptions for device type
        
        3. CHANNEL TYPE SPECIFICITY (Set/Control vs ReadBack/Status):
           - Pay attention to FACILITY CONTEXT conventions for control vs status channels
           - If query asks for "control", return ONLY control channels
           - If query asks for "status" or "readback", return ONLY readback channels
           - If ambiguous, include both types
        
        4. INSTANCE NUMBER PRECISION:
           - When instance number specified, match that EXACT instance only
           - "7" = instance 07 (NOT 17, 27, or numbers containing 7)
           - "first N" = N lowest-numbered instances; "last N" = N highest-numbered
        
        5. MEASUREMENT TYPE SPECIFICITY:
           - Use FULL phrase context, not isolated keywords
           - Avoid keyword matching - understand semantic meaning
           - When specific component named, return only that component type
        
        6. AXIS SPECIFICITY:
           - "horizontal"/"X" = X direction ONLY (exclude Y, Z)
           - "vertical"/"Y" = Y direction ONLY (exclude X, Z)
        
        7. PRECISION OVER RECALL:
           - Better to return 2 correct channels than 20 with false positives
           - Only include channels matching ALL constraints simultaneously
           - When in doubt, exclude rather than include
           - Do NOT add "possibly related" channels unless query says "all" or "any"
        
        INSTRUCTIONS:
        1. Identify ALL constraints in query (location, device type, instance, axis, measurement type)
        2. Return ONLY channels matching ALL constraints simultaneously
        3. Use exact channel names from the list above
        4. If no matches found, return empty list
        
        Return JSON with:
        - "channels_found": boolean
        - "channels": list of exact channel names
        """).strip()

