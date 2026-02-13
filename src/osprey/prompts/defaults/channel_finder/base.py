"""Base class for channel finder prompt builders.

Channel finder prompt builders provide facility-specific descriptions and
matching rules that the channel finder pipelines use to understand user queries
and match them to correct channels.

Each pipeline type (in_context, hierarchical, middle_layer) has its own
prompt builder with appropriate default content.
"""

from abc import abstractmethod

from osprey.prompts.base import FrameworkPromptBuilder


class ChannelFinderPromptBuilder(FrameworkPromptBuilder):
    """Base class for channel finder pipeline prompt builders.

    Subclasses provide facility_description and matching_rules content
    that gets combined into a single prompt for the pipeline.

    Note: Channel finder prompt builders are specialized and don't use the
    standard get_role_definition()/get_instructions() pattern. Instead they
    provide get_facility_description() and get_matching_rules() which are
    combined via get_combined_description().
    """

    def get_role_definition(self) -> str:
        """Not used for channel finder prompts. Use get_combined_description() instead."""
        return self.get_facility_description()

    def get_instructions(self) -> str:
        """Not used for channel finder prompts. Use get_combined_description() instead."""
        return self.get_matching_rules()

    @abstractmethod
    def get_facility_description(self) -> str:
        """Return facility-specific description text."""
        pass

    @abstractmethod
    def get_matching_rules(self) -> str:
        """Return channel matching rules and terminology conventions."""
        pass

    def get_combined_description(self) -> str:
        """Combine facility description and matching rules into complete prompt.

        This replaces the system.py assembler files from the service prompts.
        """
        facility = self.get_facility_description()
        rules = self.get_matching_rules()
        return f"{facility}\n\n{rules}".strip()
