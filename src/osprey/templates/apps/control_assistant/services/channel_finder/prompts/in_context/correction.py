"""Correction Prompt for Stage 3."""

import textwrap
from typing import List, Dict


def get_prompt(
    original_queries: List[str], 
    validation_results: List[Dict[str, any]], 
    chunk_database: str,
    facility_name: str = "UCSB FEL"
) -> str:
    """Prompt for Stage 3: Channel correction.
    
    Args:
        original_queries: List of atomic queries that led to these channels
        validation_results: List of {channel, valid} dicts
        chunk_database: Formatted channel database for current chunk
        
    Returns:
        Formatted prompt string for channel correction
    """
    # Format validation results for prompt
    results_list = []
    for entry in validation_results:
        status = "✓ VALID" if entry['valid'] else "✗ INVALID (not in database)"
        results_list.append(f"- {entry['channel']} {status}")
    results_formatted = "\n".join(results_list)
    
    queries_formatted = "\n".join(f'- "{q}"' for q in original_queries)
    
    return textwrap.dedent(f"""
        Some channel names you returned don't exist in the database.
        
        ORIGINAL QUERIES:
        {queries_formatted}
        
        YOUR CHANNEL RESULTS (with validation):
        {results_formatted}
        
        AVAILABLE CHANNELS IN DATABASE:
        {chunk_database}
        
        INSTRUCTIONS:
        1. Review your complete channel list above
        2. For INVALID channels: either REMOVE (if hallucination) or CORRECT (if typo)
        3. Keep all VALID channels unchanged
        4. Ensure corrected names exist in the database above
        
        Return JSON with:
        - "corrected_channels": complete list of valid channel names only
        """).strip()

