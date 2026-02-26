"""Base class for ARIEL prompt builders.

ARIEL prompt builders provide facility-specific context and response guidelines
that the ARIEL agent and RAG pipelines use to understand user queries in the
context of a specific scientific facility.

Each pipeline type (agent, rag) has its own prompt builder with appropriate
default content.
"""

from abc import abstractmethod

from osprey.prompts.base import FrameworkPromptBuilder


class ARIELPromptBuilder(FrameworkPromptBuilder):
    """Base class for ARIEL pipeline prompt builders.

    Subclasses provide facility_context and response_guidelines content
    that gets combined into prompts for the agent and RAG pipelines.

    Note: ARIEL prompt builders are specialized and don't use the
    standard get_role()/get_instructions() pattern. Instead they
    provide get_facility_context() and get_response_guidelines() which are
    assembled by subclass-specific methods (get_system_prompt, get_prompt_template).
    """

    def get_role(self) -> str:
        """Delegate to get_facility_context() for interface compliance."""
        return self.get_facility_context()

    def get_instructions(self) -> str:
        """Delegate to get_response_guidelines() for interface compliance."""
        return self.get_response_guidelines()

    @abstractmethod
    def get_facility_context(self) -> str:
        """Return facility-specific context describing the facility type and domain."""
        pass

    @abstractmethod
    def get_response_guidelines(self) -> str:
        """Return response formatting and behavior rules."""
        pass
