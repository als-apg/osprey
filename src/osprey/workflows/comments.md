---
workflow: inline-comments-guidelines
category: documentation
applies_when: [writing_code, code_review, complex_logic, business_rules]
estimated_time: reference as needed
ai_ready: true
related: [docstrings, update-documentation, pre-merge-cleanup]
---

# Inline Code Comments Guidelines for Professional Python Development

Comprehensive guidelines for writing clear, purposeful inline code comments that enhance code maintainability and team collaboration without cluttering the codebase.

## Philosophy: Comments as Strategic Documentation

Inline comments serve a fundamentally different purpose than docstrings. While docstrings document the public interface and overall purpose of code, inline comments provide contextual insights that cannot be easily expressed through code alone. The best inline comments explain the "why" behind decisions, not the "what" of implementation.

### Core Principles

**Comments must be stateless and timeless.** Write comments as if the current implementation always existed. Never reference old systems, migrations, or development history. New users should see purposeful, intentional code - not evolutionary artifacts.

**Comments should be strategic, not comprehensive.** Not every line needs a comment. Focus on areas where the reasoning behind the code is not immediately obvious from reading the implementation.

**Code clarity beats comment verbosity.** Before writing a comment, consider whether refactoring the code would make the comment unnecessary. Self-documenting code with meaningful variable names and clear structure often eliminates the need for explanatory comments.

**Comments must evolve with code.** Outdated comments are worse than no comments. Treat comment maintenance as seriously as code maintenance to prevent misleading documentation.

## When to Write Inline Comments

### Essential Comment Scenarios

**Business Logic and Domain Rules**
```python
# Apply 15% early bird discount for registrations more than 30 days in advance
if days_until_event > 30:
    price *= 0.85

# NASA requires 3-sigma confidence for mission-critical calculations
confidence_threshold = 0.9973
```

**Non-Obvious Algorithmic Decisions**
```python
# Use binary search since data is pre-sorted by timestamp
left, right = 0, len(measurements) - 1

# Fisher-Yates shuffle provides uniform random distribution
for i in range(len(items) - 1, 0, -1):
    j = random.randint(0, i)
    items[i], items[j] = items[j], items[i]
```

**Performance and Optimization Choices**
```python
# Cache expensive database queries for 5 minutes to reduce load
@lru_cache(maxsize=1000, ttl=300)
def get_user_permissions(user_id):
    return database.fetch_permissions(user_id)

# Batch API calls to avoid rate limiting (max 100 req/min)
for batch in chunked(requests, 50):
    process_batch(batch)
    time.sleep(1.2)  # Ensure we stay under rate limit
```

**External System Integration**
```python
# API requires ISO 8601 format with timezone for proper processing
timestamp = datetime.now(timezone.utc).isoformat()

# Legacy system expects data in specific order for compatibility
fields = ['id', 'name', 'created_at']  # Order required by v1.2 API
```

**Security and Compliance Considerations**
```python
# Hash passwords with salt to prevent rainbow table attacks
password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)

# GDPR requires explicit consent tracking for EU users
if user.region == 'EU':
    log_consent_action(user.id, 'data_processing', timestamp)
```

## When NOT to Write Comments

### Avoid These Comment Anti-Patterns

**Obvious Operations**
```python
# BAD: States the obvious
user_count = len(users)  # Get the length of users list
total += value           # Add value to total

# GOOD: No comment needed - code is self-explanatory
user_count = len(users)
total += value
```

**Redundant Type Information**
```python
# BAD: Type hints make this redundant
def calculate_tax(income: float) -> float:  # Takes float, returns float
    return income * 0.25

# GOOD: Focus on business logic instead
def calculate_tax(income: float) -> float:
    # Use standard 25% tax rate for simplified calculation
    return income * 0.25
```

**Parroting Code Structure**
```python
# BAD: Describes code structure rather than purpose
for item in items:        # Loop through items
    if item.valid:        # Check if item is valid
        process(item)     # Process the item

# GOOD: Explain the business logic when necessary
for item in items:
    if item.valid:
        # Only process validated items to maintain data integrity
        process(item)
```

## Strategic Comment Patterns

### Architecture and Design Decisions

**Framework Integration Comments**
```python
# Use streaming writer for real-time status updates
from langgraph.config import get_stream_writer
streaming = get_stream_writer()

# Centralized error handling ensures consistent behavior across capabilities
try:
    result = await execute_capability(state)
except Exception as exc:
    # Route all errors back to router for unified retry handling
    error_classification = classify_error(exc, context)
    return create_error_response(error_classification)
```

**State Management Clarity**
```python
# Update execution context with capability results and timing
state_updates = StateManager.handle_capability_completion(
    state, result, step, capability_name, execution_time
)

# Create fresh state while preserving user session data
fresh_state = StateManager.create_fresh_state(
    user_input=message_content,
    current_state=current_state  # Maintains persistent fields
)
```

### Future Work Comments

**Technical Debt and Planned Improvements**
```python
# TODO: Replace with async implementation for better performance
def sync_data_processor(data):
    return process_synchronously(data)

# FIXME: Memory leak in long-running processes - needs investigation
# Temporary workaround: restart worker every 1000 operations
if operation_count % 1000 == 0:
    restart_worker()
```

### Complex Logic Explanation

**Multi-Step Processes**
```python
def process_interrupt_flow(user_input, compiled_graph, config):
    # Step 1: Detect approval or rejection in user input
    approval_data = detect_approval_response(user_input)

    # Step 2: Extract business payload from interrupt state
    success, payload = extract_resume_payload(compiled_graph, config)

    # Step 3: Merge approval decision with business data for state injection
    if success and approval_data["approved"]:
        resume_command = Command(update={
            "approval_approved": True,
            "approved_payload": payload
        })
```

**Error Handling Strategy**
```python
# Classify errors by domain for appropriate retry strategies
try:
    result = await capability.execute(state)
except InfrastructureError as e:
    # Infrastructure errors: retry execution with backoff
    return schedule_retry(e, delay=exponential_backoff(attempt))
except BusinessLogicError as e:
    # Business errors: retry code generation instead
    return request_code_regeneration(e)
```

## Section Organization and Visual Structure

### Strategic Section Separation

Use comment blocks to organize code into logical sections, but only when the sections represent meaningful conceptual divisions:

```python
# ================================================================
# CORE EXECUTION PIPELINE
# ================================================================

async def execute_capability_pipeline(state, capability):
    # Validation and setup phase
    step = StateManager.get_current_step(state)
    start_time = time.time()

    # Main execution with streaming support
    from langgraph.config import get_stream_writer
    streaming = get_stream_writer()
    if streaming:
        streaming({"event_type": "status", "message": f"Executing {capability.name}"})

    # State management and result processing
    result = await capability.execute(state)
    execution_time = time.time() - start_time

    return StateManager.handle_completion(state, result, step, execution_time)

# ================================================================
# ERROR HANDLING AND RECOVERY
# ================================================================

def classify_capability_error(exc, context):
    # Error classification logic follows domain-specific patterns
    if isinstance(exc, ValidationError):
        return ErrorClassification(severity=ErrorSeverity.RETRIABLE)
```

### Avoid Over-Sectioning

Don't create sections for every small code block. Reserve section comments for major functional divisions:

```python
# BAD: Over-sectioning creates noise
# ================================================================
# VARIABLE INITIALIZATION
# ================================================================
user_id = get_current_user()
session_data = load_session(user_id)

# ================================================================
# DATA PROCESSING
# ================================================================
processed_data = transform(session_data)

# GOOD: Group related operations without excessive sectioning
# Initialize user session and process data
user_id = get_current_user()
session_data = load_session(user_id)
processed_data = transform(session_data)
```

## Comment Maintenance and Quality

### Keeping Comments Current

**Review Comments During Code Reviews**
- Verify that comments still accurately reflect the code's purpose
- Check for orphaned comments that no longer apply
- Ensure new functionality includes appropriate contextual comments

**Update Comments During Refactoring**
```python
# BEFORE refactoring
def process_payment(amount, currency):
    # Convert to USD for internal processing
    usd_amount = convert_currency(amount, currency, 'USD')
    return charge_card(usd_amount)

# AFTER refactoring with updated comment
def process_payment(amount, currency):
    # Process payments in original currency to avoid conversion fees
    return charge_card_native_currency(amount, currency)
```

### Comment Quality Checklist

**Before adding any comment, ask:**
- [ ] Does this explain WHY rather than WHAT?
- [ ] Would better variable names or function structure eliminate the need for this comment?
- [ ] Does this provide context that cannot be inferred from the code?
- [ ] Will this comment help someone understand business requirements or technical constraints?
- [ ] Is this comment likely to remain accurate as code evolves?

## Framework-Specific Guidelines

### LangGraph Integration Comments

**State Management**
```python
# LangGraph requires flat state structure for serialization
state_updates = {
    "execution_context": updated_context,
    "current_step_index": step.index + 1,
    "task_current_task": extracted_task
}

# Command pattern for interrupt resume with state injection
resume_command = Command(update={"approval_approved": True})
```

**Capability and Infrastructure Nodes**
```python
# Convention-based auto-discovery eliminates manual registration
@capability_node
class DataAnalysisCapability(BaseCapability):
    name = "data_analysis"  # Required for auto-discovery

    @staticmethod
    async def execute(state: AgentState, **kwargs):
        # Explicit logger retrieval follows professional patterns
        from configs.logger import get_logger
        logger = get_logger("data_analysis")
```

### Application-Specific Context

**Domain Knowledge Comments**
```python
# ALS beamline operations require specific timing constraints
if beam_current < MINIMUM_CURRENT_THRESHOLD:
    # Wait for beam stabilization before continuing measurements
    await wait_for_beam_stability(timeout=300)

# EPICS channel access follows specific naming conventions
pv_name = f"SR{sector:02d}:BPM{device_id}:X"  # Standard BPM naming
```

## Anti-Patterns to Avoid

### Historical and Transitional Comments

**CRITICAL: Never reference old systems, migrations, or development history in comments.**

Comments should be stateless and timeless - written as if the current implementation always existed. New users should never see the development journey or migration artifacts.

```python
# BAD: References development history
def create_capability_node(cls):
    # Migrated from PydanticAI to LangGraph - maintains backward compatibility
    # Auto-discovery patterns preserved from original decorator system
    capability_name = getattr(cls, 'name', None)

# BAD: References old system components
def handle_state_updates(state, result):
    # StateManager replaces Monitor responsibilities
    # Legacy Monitor used to handle step progression
    return StateManager.update_execution_context(state, result)

# BAD: Migration commentary
def process_message(user_input):
    # New Gateway pattern replaces old direct processing
    return gateway.process_message(user_input)

# GOOD: Describes current system purpose
def create_capability_node(cls):
    # Auto-discover capability components using naming conventions
    capability_name = getattr(cls, 'name', None)

# GOOD: Explains current functionality
def handle_state_updates(state, result):
    # Update execution context with capability results and timing
    return StateManager.update_execution_context(state, result)

# GOOD: Focuses on current behavior
def process_message(user_input):
    # Centralized message processing with interrupt handling
    return gateway.process_message(user_input)
```

**Why this matters for publication:**
- New users have no context about previous implementations
- Historical comments create confusion rather than clarity
- Published code should appear purposeful and intentional, not evolutionary
- Comments should serve current understanding, not document past decisions

### Commented-Out Code

**Never commit commented-out code to production:**
```python
# BAD: Clutters codebase
def process_data(data):
    # old_result = legacy_processor(data)
    # return old_result.transform()
    return new_processor(data).enhanced_transform()

# GOOD: Clean implementation
def process_data(data):
    return new_processor(data).enhanced_transform()
```

### Misleading Comments

**Worse than no comments:**
```python
# BAD: Comment contradicts implementation
def calculate_discount(price):
    # Apply 10% discount
    return price * 0.85  # Actually 15% discount!

# GOOD: Accurate comment or no comment
def calculate_discount(price):
    # Apply 15% standard customer discount
    return price * 0.85
```

### Journal Comments

**Avoid development diary entries:**
```python
# BAD: Personal notes don't belong in production
def optimize_query():
    # John tried this on 2024-01-15 but it was slow
    # Maria suggested using indexes on 2024-01-20
    # TODO: Ask database team about performance
    return execute_optimized_query()

# GOOD: Concise technical context
def optimize_query():
    # Use compound index for optimal performance on large datasets
    return execute_optimized_query()
```

## Examples from Framework Context

### Effective Framework Comments

```python
# Gateway centralizes all message processing and state management
class Gateway:
    async def process_message(self, user_input, compiled_graph, config):
        # Check for pending interrupts first (approval flow)
        if self._has_pending_interrupts(compiled_graph, config):
            return await self._handle_interrupt_flow(user_input, compiled_graph, config)

        # Process as new conversation turn with fresh state
        return await self._handle_new_message_flow(user_input, compiled_graph, config)

# Streamlined context storage for framework state management
def store_context(state, data_type, context_key, result):
    # Merge context data into flat state structure for framework compatibility
    context_data = state.get("execution_context", {}).get("context_data", {})
    context_data.setdefault(data_type, {})[context_key] = result
    return {"execution_context": {"context_data": context_data}}
```

### Error Handling Patterns

```python
# Consistent error handling across all framework components
try:
    result = await execute_func(state)
except Exception as exc:
    # All errors route back to router for unified retry handling
    error_classification = classify_error(exc, context)
    return create_router_command(error_classification)

# StateManager provides centralized state operations
# - Step progression: StateManager.get_current_step()
# - History management: StateManager.handle_completion()
# - Control flow: Router coordinates retry decisions
```

## Conclusion

Effective inline comments serve as bridges between code implementation and human understanding. They should provide context, explain decisions, and clarify intent without overwhelming the reader or duplicating information already expressed in the code.

**CRITICAL for publication:** Comments must be written for new users who have no knowledge of the development history. Every comment should serve current understanding, not document past decisions or evolutionary changes. Published code should appear purposeful and intentional from the first read.

Remember: **Comments are for humans, code is for computers.** Write comments that help your team (including your future self) understand not just what the code does, but why it exists and how it fits into the larger system architecture.

The best commenting strategy combines clean, self-documenting code with strategic comments that provide essential context. When in doubt, favor code clarity over comment verbosity, but don't hesitate to add comments when they genuinely enhance understanding.
