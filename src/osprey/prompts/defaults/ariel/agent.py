"""Default ARIEL agent prompt builder.

Provides the default system prompt for the ARIEL ReAct agent pipeline.
Applications override this with facility-specific context so the agent
understands domain terminology (e.g., RF cavities at a particle accelerator).
"""

from .base import ARIELPromptBuilder

_DEFAULT_FACILITY_CONTEXT = """\
You are ARIEL, an AI assistant for searching and analyzing facility logbook entries \
at a particle accelerator complex.

The facility is a synchrotron light source with a storage ring, booster synchrotron, \
and linac. Common equipment and terminology include RF cavities, dipole and quadrupole \
magnets, beam position monitors (BPMs), insertion devices (undulators), vacuum systems, \
beamlines, power supplies, cryogenics, and EPICS IOCs. The logbook covers operations \
(beam delivery, injection, shifts), maintenance, safety, and commissioning.

Your purpose is to help users find relevant information in the electronic logbook system."""

_DEFAULT_RESPONSE_GUIDELINES = """\
## Guidelines

- Use the available search tools to find relevant logbook entries
- You may call tools multiple times with different queries to gather complete information
- Always cite specific entry IDs when referencing information
- If no relevant entries are found, say so clearly
- Keep responses concise but informative
- Focus on factual information from the logbook entries

## Response Format

- Summarize key findings with entry ID citations
- Provide direct answers citing source entries
- If nothing is found: clearly state that no relevant information was found in the logbook"""


class DefaultARIELAgentPromptBuilder(ARIELPromptBuilder):
    """Default prompt builder for ARIEL agent pipeline."""

    def get_facility_context(self) -> str:
        return _DEFAULT_FACILITY_CONTEXT

    def get_response_guidelines(self) -> str:
        return _DEFAULT_RESPONSE_GUIDELINES

    def get_system_prompt(self) -> str:
        """Assemble the complete system prompt for create_react_agent().

        Combines facility context and response guidelines into a single string.
        """
        facility = self.get_facility_context()
        guidelines = self.get_response_guidelines()
        return f"{facility}\n\n{guidelines}"
