"""
Task Extraction Prompt Builder - Application-agnostic prompts for task extraction
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from osprey.base import BaseExample
from osprey.state import ChatHistoryFormatter, MessageUtils, UserMemories

from ..base import FrameworkPromptBuilder


@dataclass
class TaskExtractionExample(BaseExample):
    """Example for task extraction prompt."""

    def __init__(self, messages: list[BaseMessage], user_memory: UserMemories, expected_output: ExtractedTask):
        self.messages = messages
        self.user_memory = user_memory
        self.expected_output = expected_output

    def format_for_prompt(self) -> str:
        """Format this example for inclusion in prompts."""
        # Format chat history using native message formatter
        chat_formatted = ChatHistoryFormatter.format_for_llm(self.messages)

        # Format user memory
        memory_formatted = self.user_memory.format_for_prompt()

        return f"""
**Chat History:**
{chat_formatted}

**User Memory:**
{memory_formatted if memory_formatted else "No stored memories"}

**Expected Output:**
{self.expected_output.format_for_prompt()}
"""

class ExtractedTask(BaseModel):
    """Task extraction result."""
    task: str = Field(description="The actionable task extracted from the conversation")
    depends_on_chat_history: bool = Field(description="Whether the task depends on previous conversation context")
    depends_on_user_memory: bool = Field(description="Whether the task depends on stored user memory")

    def format_for_prompt(self) -> str:
        return f"""
Task: {self.task}
Depends on Chat History: {self.depends_on_chat_history}
Depends on User Memory: {self.depends_on_user_memory}
"""

class DefaultTaskExtractionPromptBuilder(FrameworkPromptBuilder):
    """Framework prompt builder for task extraction."""

    PROMPT_TYPE = "task_extraction"

    def __init__(self, include_default_examples: bool = True):
        """Initialize task extraction prompt builder.

        Args:
            include_default_examples: If True, loads framework default examples.
                                     Set to False to use only custom examples.
        """
        super().__init__()
        self.examples = []
        if include_default_examples:
            self._load_examples()

    def _load_examples(self):
        """Load task extraction examples with native LangGraph messages."""

        # Examples without memory first

        # Simple follow-up question requiring temporal reference resolution
        self.examples.append(TaskExtractionExample(
            messages=[
                MessageUtils.create_user_message("What's the current system status?"),
                MessageUtils.create_assistant_message("The current system status shows all services running normally. CPU usage is at 45%, checked at 14:23:05."),
                MessageUtils.create_user_message("What was it an hour ago?"),
            ],
            user_memory=UserMemories(entries=[]),
            expected_output=ExtractedTask(
                task="Retrieve historical system status and CPU usage data from approximately one hour ago (around 13:23:05)",
                depends_on_chat_history=True,
                depends_on_user_memory=False
            )
        ))

        # Complex multi-turn technical troubleshooting with progressive problem refinement
        self.examples.append(TaskExtractionExample(
            messages=[
                MessageUtils.create_user_message("I'm seeing some instability in our database cluster"),
                MessageUtils.create_assistant_message("I can help analyze the database cluster stability. Could you be more specific about what type of instability you're observing?"),
                MessageUtils.create_user_message("The response times are showing oscillations"),
                MessageUtils.create_assistant_message("I'll need to look at the performance metrics. Which specific nodes in the cluster are you concerned about?"),
                MessageUtils.create_user_message("Mainly nodes DB-01 and DB-02, the oscillations started around 2 hours ago"),
                MessageUtils.create_assistant_message("I found response time oscillations in DB-01 and DB-02 starting at 12:15. The pattern appears to be cyclical with 30-second intervals. Here's the analysis showing clear oscillatory behavior with response times varying between 50ms and 300ms."),
                MessageUtils.create_user_message("Can you check if there were any configuration changes to the load balancer around that time?"),
            ],
            user_memory=UserMemories(entries=[]),
            expected_output=ExtractedTask(
                task="Investigate load balancer configuration changes around 12:15 to correlate with observed database response time oscillations",
                depends_on_chat_history=True,
                depends_on_user_memory=False
            )
        ))

        # Reference resolution requiring extraction of specific values from previous analysis
        self.examples.append(TaskExtractionExample(
            messages=[
                MessageUtils.create_user_message("Show me the system uptime trend for the last 24 hours"),
                MessageUtils.create_assistant_message("Here's the system uptime data for the past 24 hours. The trend shows generally stable performance with 99.8% uptime, with a notable dip at 03:17 where it dropped to 95.2% before recovering by 04:30."),
                MessageUtils.create_user_message("That dip around 3 AM is concerning"),
                MessageUtils.create_assistant_message("Yes, I see the uptime dropped from 99.8% to 95.2% at 03:17. This represents a significant 4.6 percentage point decrease in system uptime. Would you like me to investigate potential causes?"),
                MessageUtils.create_user_message("Please do that, and also compare it to the same time period last week"),
            ],
            user_memory=UserMemories(entries=[]),
            expected_output=ExtractedTask(
                task="Investigate the system uptime drop from 99.8% to 95.2% that occurred at 03:17 today, and perform comparative analysis with the same time period from exactly one week ago",
                depends_on_chat_history=True,
                depends_on_user_memory=False
            )
        ))

        # Pure conversational query requiring no technical data or analysis
        self.examples.append(TaskExtractionExample(
            messages=[
                MessageUtils.create_user_message("Hi, what can you help me with?"),
                MessageUtils.create_assistant_message("I'm your workflow automation assistant! I can help with data analysis, system monitoring, process automation, and much more. I can retrieve historical data, analyze trends, troubleshoot issues, and provide insights about your systems."),
                MessageUtils.create_user_message("That's great. What's the difference between you and the other assistants?"),
            ],
            user_memory=UserMemories(entries=[]),
            expected_output=ExtractedTask(
                task="Explain the differences and unique capabilities of this workflow automation assistant compared to other available assistants in the system",
                depends_on_chat_history=False,
                depends_on_user_memory=False
            )
        ))

        # Fresh data request with no previous context needed
        self.examples.append(TaskExtractionExample(
            messages=[
                MessageUtils.create_user_message("Can you analyze the performance metrics for our web servers?"),
            ],
            user_memory=UserMemories(entries=[]),
            expected_output=ExtractedTask(
                task="Analyze performance metrics for web servers",
                depends_on_chat_history=False,
                depends_on_user_memory=False
            )
        ))

        # Fresh data request with no previous context needed
        self.examples.append(TaskExtractionExample(
            messages=[
                MessageUtils.create_user_message("What tools do you have?"),
            ],
            user_memory=UserMemories(entries=[]),
            expected_output=ExtractedTask(
                task="List all the tools you have",
                depends_on_chat_history=False,
                depends_on_user_memory=False
            )
        ))

        # Examples with memory

        # Memory-informed request referring to previously saved information
        self.examples.append(TaskExtractionExample(
            messages=[
                MessageUtils.create_user_message("Check if that problematic pattern I saved is happening again"),
            ],
            user_memory=UserMemories(entries=[
                "[2025-01-15 14:23] Database performance drops consistently at 3:15 AM every Tuesday - correlation with backup scheduling",
                "[2025-01-16 09:45] Web server response times spike when CPU usage exceeds 85%",
                "[2025-01-17 11:30] Important: Cache invalidation causes temporary performance degradation - need 30min recovery time"
            ]),
            expected_output=ExtractedTask(
                task="Monitor for database performance drops around 3:15 AM (Tuesday pattern), web server response time spikes when CPU usage > 85%, and performance degradation following cache invalidation",
                depends_on_chat_history=False,
                depends_on_user_memory=True
            )
        ))

        # Memory helps disambiguate vague reference
        self.examples.append(TaskExtractionExample(
            messages=[
                MessageUtils.create_user_message("How is that critical metric doing today?"),
            ],
            user_memory=UserMemories(entries=[
                "[2025-01-14 16:42] Critical metric to monitor: API response time - has been unstable lately",
                "[2025-01-15 08:15] Reminder: Weekly check needed for database connection pool utilization trends",
                "[2025-01-16 13:20] Follow up on load balancer configuration effectiveness"
            ]),
            expected_output=ExtractedTask(
                task="Check current status and recent behavior of API response time metric, focusing on stability assessment",
                depends_on_chat_history=False,
                depends_on_user_memory=True
            )
        ))

        # Memory provides context for comparative analysis request
        self.examples.append(TaskExtractionExample(
            messages=[
                MessageUtils.create_user_message("Compare today's performance with my baseline measurements"),
            ],
            user_memory=UserMemories(entries=[
                "[2025-01-10 10:30] Baseline established: System uptime 99.2±0.1%, response time 145±25 ms",
                "[2025-01-10 10:35] Baseline CPU usage: <75% average across all servers during normal operations",
                "[2025-01-12 14:15] Good performance day: uptime 99.7%, very stable response times, no scaling adjustments needed"
            ]),
            expected_output=ExtractedTask(
                task="Compare today's system uptime and response time performance against baseline of 99.2±0.1% and 145±25 ms, and assess CPU usage against <75% baseline from normal operations",
                depends_on_chat_history=False,
                depends_on_user_memory=True
            )
        ))

        # Memory helps resolve time-specific reference
        self.examples.append(TaskExtractionExample(
            messages=[
                MessageUtils.create_user_message("Is that maintenance window issue resolved?"),
            ],
            user_memory=UserMemories(entries=[
                "[2025-01-13 22:45] Next Tuesday 2AM maintenance: Database upgrade work on cluster 2, expect service interruption",
                "[2025-01-14 15:30] Maintenance concern: Last time database work caused connection pool instability in adjacent services",
                "[2025-01-15 08:00] Post-maintenance checklist: Verify database connections for services 1-3, check query performance consistency"
            ]),
            expected_output=ExtractedTask(
                task="Verify resolution of database upgrade maintenance issues from Tuesday 2AM work on cluster 2, specifically checking connection pool stability for services 1-3 and query performance consistency",
                depends_on_chat_history=False,
                depends_on_user_memory=True
            )
        ))

    def get_role_definition(self) -> str:
        """Get the role definition (generic for task extraction)."""
        return "Convert chat conversations into actionable task descriptions."

    def get_instructions(self) -> str:
        """Get the generic task extraction instructions."""
        return """
Core requirements:
• Create self-contained task descriptions executable without conversation context
• Resolve temporal references ("an hour ago", "yesterday") to specific times/values
• Extract specific details and parameters from previous responses
• Determine if task builds on previous conversation context
• Consider available data sources when interpreting requests
• Set depends_on_user_memory=true only when the task directly incorporates specific information from user memory
        """.strip()

    def get_system_instructions(self, messages: list[BaseMessage], retrieval_result=None) -> str:
        """Get system instructions for task extraction agent configuration.

        :param messages: Native LangGraph messages to extract task from
        :param retrieval_result: Optional data retrieval result
        :return: Complete prompt for task extraction
        """
        examples_text = "\n\n".join([
            f"## Example {i+1}:\n{example.format_for_prompt()}"
            for i, example in enumerate(self.examples)
        ])

        # Format the actual chat history using native message formatter
        chat_formatted = ChatHistoryFormatter.format_for_llm(messages)

        # Add data source context if available
        data_context = ""
        if retrieval_result and retrieval_result.has_data:
            # Get the actual retrieved content formatted for LLM consumption
            try:
                formatted_contexts = []
                for source_name, context in retrieval_result.context_data.items():
                    try:
                        formatted_content = context.format_for_prompt()
                        if formatted_content and formatted_content.strip():
                            formatted_contexts.append(f"**{source_name}:**\n{formatted_content}")
                    except Exception:
                        # Log error but continue with other sources
                        pass

                if formatted_contexts:
                    data_context = "\n\n**Retrieved Data:**\n" + "\n\n".join(formatted_contexts)
                else:
                    # Fallback to summary if no content could be formatted
                    data_context = f"\n\n**Available Data Sources:**\n{retrieval_result.get_summary()}"

            except Exception:
                # Fallback to summary on any error
                data_context = f"\n\n**Available Data Sources:**\n{retrieval_result.get_summary()}"

        final_prompt = f"""
You are a task extraction system that analyzes chat history and user memory to extract actionable tasks.

Your job is to:
1. Understand what the user is asking for
2. Extract a clear, actionable task
3. Determine if the task depends on chat history context
4. Determine if the task depends on user memory

## Guidelines:
- Extract the core task the user wants accomplished
- Set depends_on_chat_history=True if the task references previous messages or needs conversation context
- Set depends_on_user_memory=True if the task references stored user information or patterns
- Be specific and actionable in task descriptions
- Consider the full conversation context when determining dependencies

## Examples:
{examples_text}

## Current Chat History:
{chat_formatted}{data_context}

## User Memory:
No stored memories

Now extract the task from the provided chat history and user memory.
"""

        # Debug: Print prompt if enabled
        self.debug_print_prompt(final_prompt)

        return final_prompt
