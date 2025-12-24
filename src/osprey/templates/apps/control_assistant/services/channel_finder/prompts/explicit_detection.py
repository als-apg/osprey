"""Explicit Channel Detection Prompt (Shared across all pipelines)."""

import textwrap


def get_prompt(query: str) -> str:
    """
    Prompt for detecting explicit channel/PV addresses in user queries.

    This is a shared optimization used by all pipelines (hierarchical, in-context,
    middle-layer) to detect when users provide specific channel addresses directly,
    allowing pipelines to skip search/navigation for efficiency.

    Args:
        query: User's natural language query

    Returns:
        Formatted prompt string for explicit channel detection
    """
    return textwrap.dedent(
        f"""
        You are analyzing a user query to detect if it contains explicit channel or PV (process variable) addresses.

        Explicit channel addresses are specific technical identifiers like:
        - "SC:HCM1:SP" (colon-separated PV names)
        - "MAG:HCM:H01:CURRENT:SP" (hierarchical PV addresses)
        - "TerminalVoltageSetPoint" (specific parameter names)
        - Any identifier that looks like a specific technical address/name rather than a general description

        Examples of queries WITH explicit addresses:
        - "Set the SC:HCM1:SP pv to 4.6"
          → SC:HCM1:SP
          → needs_additional_search: false (complete address provided)

        - "What is the value of MAG:HCM:H01:CURRENT:RB?"
          → MAG:HCM:H01:CURRENT:RB
          → needs_additional_search: false (complete address provided)

        - "Read TerminalVoltageSetPoint"
          → TerminalVoltageSetPoint
          → needs_additional_search: false (complete address provided)

        - "Get MAG:VCM:V01:CURRENT:SP and MAG:HCM:H02:CURRENT:SP"
          → MAG:VCM:V01:CURRENT:SP, MAG:HCM:H02:CURRENT:SP
          → needs_additional_search: false (complete addresses provided)

        - "Get BR:HCM1 and all vertical corrector setpoints"
          → BR:HCM1
          → needs_additional_search: true (HCM1 is explicit but "all vertical corrector setpoints" needs search)

        Examples of queries WITHOUT explicit addresses (need search):
        - "Find the horizontal corrector magnet setpoint"
          → needs search
          → needs_additional_search: true

        - "Show me all beam diagnostics channels"
          → needs search
          → needs_additional_search: true

        - "What channels control the vacuum system?"
          → needs search
          → needs_additional_search: true

        Your task: Analyze the query and determine:
        1. Are there any explicit channel addresses? (has_explicit_addresses)
        2. Do we need additional channel finding beyond explicit addresses? (needs_additional_search)

        IMPORTANT:
        - If the query ONLY contains explicit addresses and nothing else → needs_additional_search: false
        - If the query contains partial/vague references or asks for "all" of something → needs_additional_search: true
        - If the query mixes explicit addresses with search terms → needs_additional_search: true

        User Query: "{query}"

        Return:
        - has_explicit_addresses: true if specific technical addresses are present, false otherwise
        - channel_addresses: list of detected addresses (empty if none)
        - needs_additional_search: true if channel finding pipeline should also be invoked, false if explicit addresses are sufficient
        - reasoning: brief explanation of what you detected and why additional search is/isn't needed
        """
    ).strip()
