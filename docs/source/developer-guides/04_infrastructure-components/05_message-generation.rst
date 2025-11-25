Message Generation
==================

.. currentmodule:: osprey.infrastructure.respond_node

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - How the framework generates responses for technical and conversational queries
   - Response mode detection and context integration
   - Clarification workflows for ambiguous requests
   - Prompt builder customization patterns

   **Prerequisites:** Understanding of :doc:`../03_core-framework-systems/02_context-management-system`

   **Time Investment:** 10 minutes for complete understanding

Core Capabilities
-----------------

The framework provides two message generation capabilities:

**RespondCapability**
   Generates final responses using execution context and conversation history

**ClarifyCapability**
   Asks targeted questions when user requests lack sufficient detail

Both capabilities adapt their behavior based on available context and use the framework's prompt builder system for consistent responses.

.. _respond-capability:

RespondCapability
-----------------

Generates responses by analyzing available context and using appropriate prompts:

.. code-block:: python

   @capability_node
   class RespondCapability(BaseCapability):
       name = "respond"
       description = "Generate responses for technical and conversational questions"
       provides = ["FINAL_RESPONSE"]
       requires = []  # Works with any context or none

       async def execute(self):
           # Gather response information
           response_context = _gather_information(self._state)

           # Build dynamic prompt
           prompt = _get_base_system_prompt(
               response_context.current_task, response_context
           )

           # Generate response
           response = await asyncio.to_thread(
               get_chat_completion,
               model_config=get_model_config("response"),
               message=prompt
           )

           return {"messages": [AIMessage(content=response)]}

**Response Mode Detection:**

.. code-block:: python

   def _determine_response_mode(state, current_step):
       has_step_inputs = current_step and current_step.get("inputs")
       has_capability_data = bool(state.get("capability_context_data", {}))

       if not has_step_inputs and not has_capability_data:
           return "conversational"      # General assistance
       elif has_step_inputs:
           return "specific_context"    # Technical response with step data
       else:
           return "general_context"     # Context-aware response

ResponseContext Structure
-------------------------

System aggregates information using structured context:

.. code-block:: python

   @dataclass
   class ResponseContext:
       current_task: str
       execution_history: List[Any]
       relevant_context: Dict[str, Any]
       is_killed: bool
       kill_reason: Optional[str]
       capabilities_overview: Optional[str]
       total_steps_executed: int
       reclassification_count: int
       current_date: str

**Context Assembly:**

.. code-block:: python

   def _gather_information(state: AgentState) -> ResponseContext:
       context_manager = ContextManager(state)
       current_step = StateManager.get_current_step(state)
       relevant_context = context_manager.get_summaries(current_step)

       response_mode = _determine_response_mode(state, current_step)

       # Adapt data based on response mode
       if response_mode == "conversational":
           execution_history = []
           capabilities_overview = _get_capabilities_overview()
       else:
           execution_history = _get_execution_history(state)
           capabilities_overview = None

       return ResponseContext(...)

.. _clarify-capability:

ClarifyCapability
-----------------

Generates targeted questions for ambiguous user requests:

.. code-block:: python

   @capability_node
   class ClarifyCapability(BaseCapability):
       name = "clarify"
       description = "Ask questions when queries are ambiguous or incomplete"
       provides = []  # Communication capability
       requires = []  # Works with any context

       async def execute(self):
           # Generate clarifying questions
           questions_response = await asyncio.to_thread(
               _generate_clarifying_questions,
               self._state,
               self.get_task_objective()
           )

           # Format for user interaction
           formatted_questions = _format_questions_for_user(questions_response)

           return {"messages": [AIMessage(content=formatted_questions)]}

**Structured Question Generation:**

.. code-block:: python

   class ClarifyingQuestionsResponse(BaseModel):
       reason: str = Field(description="Why clarification is needed")
       questions: List[str] = Field(
           description="Specific, targeted questions",
           min_items=1, max_items=4
       )
       missing_info: List[str] = Field(
           description="Types of missing information",
           default=[]
       )

**Question Generation Process:**

.. code-block:: python

   def _generate_clarifying_questions(state, task_objective):
       # Format conversation history
       messages = state.get("input_output", {}).get("messages", [])
       chat_history_str = ChatHistoryFormatter.format_for_llm(messages)

       # Get clarification prompt
       clarification_builder = prompt_provider.get_clarification_prompt_builder()
       system_instructions = clarification_builder.get_system_instructions()
       clarification_query = clarification_builder.build_clarification_query(
           chat_history_str, task_objective
       )

       # Generate structured questions
       return get_chat_completion(
           message=f"{system_instructions}\n\n{clarification_query}",
           model_config=get_model_config("response"),
           output_model=ClarifyingQuestionsResponse
       )

Prompt Builder Integration
--------------------------

Message generation uses the framework's prompt builder architecture:

.. code-block:: python

   def _get_base_system_prompt(current_task, info=None):
       prompt_provider = get_framework_prompts()
       response_builder = prompt_provider.get_response_generation_prompt_builder()

       return response_builder.get_system_instructions(
           current_task=current_task,
           info=info
       )

**Dynamic Context Building:**

.. code-block:: python

   class DefaultResponseGenerationPromptBuilder(FrameworkPromptBuilder):
       def _get_dynamic_context(self, current_task="", info=None, **kwargs):
           sections = []

           # Base role with current task
           sections.append(f"You are an expert assistant.\n\nCURRENT TASK: {current_task}")

           if info:
               # Show execution context if available
               if hasattr(info, 'execution_history') and info.execution_history:
                   sections.append(self._get_execution_section(info))

               # Show capabilities for conversational responses
               elif hasattr(info, 'capabilities_overview') and info.capabilities_overview:
                   sections.append(self._get_capabilities_section(info.capabilities_overview))

           return "\n\n".join(sections)

**Domain Customization:**

.. code-block:: python

   # Applications can override prompt builders
   class CustomResponsePromptBuilder(DefaultResponseGenerationPromptBuilder):
       def get_role_definition(self):
           return "You are a specialized domain expert assistant."

       def _get_dynamic_context(self, current_task="", info=None, **kwargs):
           # Customize response generation for specific domain
           return super()._get_dynamic_context(current_task, info, **kwargs)

Error Handling
--------------

Both capabilities include error classification for framework integration:

.. code-block:: python

   # RespondCapability error classification
   @staticmethod
   def classify_error(exc: Exception, context: dict):
       return ErrorClassification(
           severity=ErrorSeverity.CRITICAL,
           user_message=f"Failed to generate response: {str(exc)}"
       )

   # ClarifyCapability error classification
   @staticmethod
   def classify_error(exc: Exception, context: dict):
       return ErrorClassification(
           severity=ErrorSeverity.RETRIABLE,
           user_message=f"Failed to generate clarifying questions: {str(exc)}"
       )

Integration Patterns
--------------------

**Automatic Framework Integration:**
- RespondCapability typically planned as final execution step
- ClarifyCapability planned when queries lack detail
- Both use native LangGraph message patterns

**Usage in Execution Plans:**

.. code-block:: python

   # Normal execution plan ending
   PlannedStep(
       context_key="user_response",
       capability="respond",
       task_objective="Present results to user",
       inputs=["previous_step_data"]
   )

   # Clarification step
   PlannedStep(
       context_key="clarification",
       capability="clarify",
       task_objective="Ask for missing parameters",
       inputs=[]
   )

**Prompt Builder Registration:**

.. code-block:: python

   # Framework provides defaults, applications can override
   class ApplicationPrompts(FrameworkPrompts):
       def get_response_generation_prompt_builder(self):
           return CustomResponsePromptBuilder()

       def get_clarification_prompt_builder(self):
           return CustomClarificationPromptBuilder()

Key Features
------------

**Adaptive Responses:**
    - Automatically adjusts style based on available context
    - Technical responses for execution data, conversational for general queries

**Structured Clarification:**
    - Uses Pydantic models for consistent question format
    - Context-aware questions considering conversation history

**Domain Customization:**
    - Applications override prompt builders for specialized responses
    - Maintains consistent behavior while allowing domain adaptation

**Error Integration:**
    - Provides user-friendly error messages through framework error handling
    - Different severity levels for response vs. clarification failures

.. seealso::

   :doc:`../../api_reference/02_infrastructure/06_message-generation`
      API reference for response and clarification systems

   :doc:`../03_core-framework-systems/02_context-management-system`
       Response mode detection and context integration

   :doc:`../../api_reference/01_core_framework/05_prompt_management`
       Prompt builder customization patterns

   :doc:`06_error-handling-infrastructure`
       Error communication patterns

   :doc:`../03_core-framework-systems/02_context-management-system`
       Context integration details

   :doc:`../../api_reference/01_core_framework/05_prompt_management`
       Prompt builder customization

Message Generation provides the final user interaction layer, converting execution results and handling ambiguous requests through adaptive, context-aware response and clarification systems.