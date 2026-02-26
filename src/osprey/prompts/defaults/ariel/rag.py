"""Default ARIEL RAG prompt builder.

Provides the default prompt template for the ARIEL RAG pipeline.
Applications override this with facility-specific context so the RAG
pipeline understands domain terminology when generating answers.
"""

from .base import ARIELPromptBuilder

_DEFAULT_FACILITY_CONTEXT = (
    "You are a helpful assistant answering questions about particle accelerator "
    "operations based on logbook entries. The facility is a synchrotron light source "
    "with a storage ring, booster, and linac. Equipment includes RF cavities, magnets, "
    "beam position monitors (BPMs), insertion devices, vacuum systems, and beamlines."
)

_DEFAULT_RESPONSE_GUIDELINES = """\
**Important:**
- Only answer based on the information provided in the entries
- If the entries don't contain relevant information, say "I don't have enough information to answer this question based on the available logbook entries"
- When referencing information, cite the entry ID in brackets with a hash, e.g., [#12345]
- Be concise but thorough"""


class DefaultARIELRAGPromptBuilder(ARIELPromptBuilder):
    """Default prompt builder for ARIEL RAG pipeline."""

    def get_facility_context(self) -> str:
        return _DEFAULT_FACILITY_CONTEXT

    def get_response_guidelines(self) -> str:
        return _DEFAULT_RESPONSE_GUIDELINES

    def get_prompt_template(self) -> str:
        """Assemble the complete RAG prompt template with format placeholders.

        Returns a string with {context} and {question} placeholders ready
        for .format() calls at query time.
        """
        facility = self.get_facility_context()
        guidelines = self.get_response_guidelines()
        return (
            f"{facility}\n\n"
            "Use the following logbook entries as context to answer the question. "
            "Each entry has an ID, timestamp, author, and content.\n\n"
            f"{guidelines}\n\n"
            "**Context (Logbook Entries):**\n"
            "{context}\n\n"
            "**Question:** {question}\n\n"
            "**Answer:**"
        )
