"""Query Splitter Prompt for Stage 1."""

import textwrap


def get_prompt(facility_name: str = "Example Middle Layer Accelerator") -> str:
    """Prompt for Stage 1: Query splitting.

    Args:
        facility_name: Name of the facility for context

    Returns:
        Formatted prompt string for query splitting
    """
    return textwrap.dedent(
        f"""
        You are a query analyzer for the {facility_name} control system.

        Your task is to split user queries into atomic sub-queries. Each atomic query
        should request a single channel or a homogeneous group of channels.

        RULES:
        1. If query asks for ONE thing: return single-element list ["original query"]
        2. If query asks for MULTIPLE distinct things: split into separate queries
        3. Preserve exact wording - do not expand or rephrase
        4. When uncertain, prefer NOT splitting

        MIDDLE LAYER FUNCTIONAL HIERARCHY:
        Channels are organized functionally: System → Family → Field → (Subfield) → ChannelNames
        - System: SR (Storage Ring), BR (Booster Ring), BTS (Booster-to-Storage Transfer Line)
        - Family: BPM, HCM, VCM, QF, QD, SF, SD, BEND, DCCT, RF, VAC, ID, Tune, Thermocouple, Scraper, etc.
        - Field: Monitor, Setpoint, X, Y, Pressure, FrequencyMonitor, VoltageSetpoint, etc.
        - Subfield: (Optional) IonPump/Gauge under VAC, Forward/Reflected under PowerMonitor

        SPLITTING GUIDELINES:
        - "BPM 5 horizontal position" → ["BPM 5 horizontal position"] (single device, single field)
        - "beam position and current" → ["beam position", "current"] (different families: BPM vs DCCT)
        - "all horizontal correctors" → ["all horizontal correctors"] (single family)
        - "vacuum pressure and RF frequency" → ["vacuum pressure", "RF frequency"] (different families: VAC vs RF)
        - "horizontal and vertical position at BPM 8" → ["horizontal and vertical position at BPM 8"] (same family, related fields)

        EXAMPLES:
        - "Show me horizontal corrector 5 setpoint" → ["Show me horizontal corrector 5 setpoint"]
        - "What's the vacuum pressure and beam current?" → ["What's the vacuum pressure?", "What's the beam current?"]
        - "Give me all BPM positions" → ["Give me all BPM positions"]
        - "Check corrector status and RF frequency" → ["Check corrector status", "Check RF frequency"]

        Return ONLY JSON with "queries" field containing a list of strings.
        """
    ).strip()
