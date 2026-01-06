---
workflow: channel-finder-pipeline-selection
category: channel-finder
applies_when: [setting_up_channel_finder, choosing_pipeline, building_database]
estimated_time: 15-30 minutes
ai_ready: true
related: [channel-finder-database-builder]
---

# Channel Finder: Pipeline Selection Guide

**Purpose**: Help users select the most appropriate Channel Finder pipeline for their control system based on observable characteristics of their channel naming and organization.

**Target Audience**: Users setting up Channel Finder for the first time who need to choose between In-Context, Hierarchical, or Middle Layer pipelines.

**Important**: This workflow helps **assess** pipeline suitability based on evidence, not make definitive choices. When uncertain, start simple and evolve.

---

## 🤖 AI Quick Start

**Paste this prompt to your AI assistant:**

```
Following @docs/workflows/channel-finder-pipeline-selection.md, help me select the right Channel Finder pipeline.

Your approach:
1. DO NOT immediately recommend a pipeline
2. DO ask for concrete evidence (channel name examples, database files, etc.)
3. DO ask to see example databases and pipeline code to better understand requirements
4. DO recognize when you lack sufficient information
5. DO present multiple options when they're equally viable
6. DO suggest starting simple when uncertain

Start by asking:
1. For 10-15 representative channel names from my control system
2. If I've initialized my-control-assistant (so you can see example databases)
3. To read the database format files to understand schema requirements

Based on what I provide, analyze the structure and guide me through the decision.

Remember:
- Avoid false positives - better to ask for more info than guess
- Read relevant code in src/osprey/templates/.../channel_finder/ for implementation details
- Look at example databases in my-control-assistant/data/channel_databases/ if available
- When uncertain, suggest starting with In-Context (works for anything)
```

**Related workflows**: [channel-finder-database-builder.md](channel-finder-database-builder.md)

---

## 📊 Quick Reference: Three Pipeline Types

### In-Context Pipeline
**What it is**: Semantic search through a flat list of channels
**Best for**: Simple systems, arbitrary naming, when structure doesn't help
**Code**: `src/osprey/templates/.../pipelines/in_context/pipeline.py`

### Hierarchical Pipeline
**What it is**: Iterative navigation through a tree structure built from channel naming patterns
**Best for**: Channels with consistent hierarchical naming (e.g., `SYSTEM:SUBSYSTEM:DEVICE:FIELD`)
**Code**: `src/osprey/templates/.../pipelines/hierarchical/pipeline.py`

### Middle Layer Pipeline
**What it is**: Functional organization (System → Family → Field) where channel names are stored in database, not derived from patterns
**Best for**: Functional grouping independent of naming conventions; when channel addresses don't reflect logical organization
**Code**: `src/osprey/templates/.../pipelines/middle_layer/pipeline.py`

---

## 🎯 Decision Process Overview

```
┌─────────────────────────────────────┐
│  Gather Channel Name Samples        │
│  (10-15 representative examples)    │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Analyze Observable Patterns        │
│  - Naming consistency               │
│  - Hierarchy depth                  │
│  - Separator patterns               │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Assess Organizational Structure    │
│  - Derived from names? → Hierarchical│
│  - Functional grouping? → Middle Layer│
│  - No clear structure? → In-Context │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Validate Against Constraints       │
│  - Scale (channel count)            │
│  - Database availability            │
│  - Complexity tolerance             │
└─────────────────────────────────────┘
```

**Golden Rule**: When multiple pipelines seem viable, start with the simplest and evolve if needed.

---

## 📝 Phase 1: Essential Information Gathering

**Ask the user for concrete evidence** before making any recommendations.

### Required Information

1. **Channel Name Examples** (Most Important!)
   ```
   "Please provide 10-15 representative channel names from your system.
    Include variety - different devices, different types of signals."
   ```

2. **Scale**
   ```
   "Approximately how many channels total?"
   ```

3. **Existing Organization**
   ```
   "Do you have an existing database, configuration file, or
    organizational document for these channels?"
   ```

### What to Look For

**In the channel names:**
- ✅ Consistent separators (`:`, `_`, `-`)
- ✅ Repeated patterns across channels
- ✅ Clear hierarchy levels
- ✅ Functional keywords (Monitor, Setpoint, X, Y, Current, Voltage)
- ❌ Arbitrary/inconsistent naming
- ❌ No visible structure

**In their responses:**
- ✅ Can provide examples immediately → They know their system
- ✅ Mentions existing tools/databases → May have structure
- ❌ Very vague descriptions → Need concrete data first
- ❌ Focused on future/hypothetical → Focus on current state

---

## 🔍 Phase 2: Pattern Analysis

Once you have channel name samples, analyze them systematically.

### Pattern Recognition Checklist

**Run through these checks:**

1. **Naming Consistency**
   - [ ] All channels use same separator(s)?
   - [ ] Same number of segments across channels?
   - [ ] Predictable segment meanings?

2. **Hierarchy Depth**
   - [ ] How many levels? (Count segments)
   - [ ] Are levels semantic (BPM, HCM) or just identifiers (01, 02)?
   - [ ] Is there category nesting?

3. **Functional Organization**
   - [ ] Do channel names contain function keywords?
   - [ ] Or is function implicit/stored elsewhere?
   - [ ] Are there read/write pairs?

4. **Structural Patterns**
   - [ ] Device numbering patterns?
   - [ ] Sector/location identifiers?
   - [ ] Signal type indicators?

### Example Analysis

**Example Set 1**: Hierarchical Pattern Detected
```
SR01C:BPM1:X
SR01C:BPM1:Y
SR01C:BPM2:X
SR02C:BPM1:X
BR:DCCT:Current

Analysis:
✓ Consistent ':' separator
✓ 3 levels: SYSTEM:DEVICE:FIELD
✓ Clear semantic hierarchy
✓ Pattern repeats predictably
→ Hierarchical pipeline is a good fit
```

**Example Set 2**: Functional Organization Detected
```
SR:BPM:Monitor (actual PV: SR01C_BPM1_X_RB)
SR:BPM:Setpoint (actual PV: SR01C_BPM1_X_SP)
SR:HCM:Monitor (actual PV: SR_HCM_01_I_RB)
BR:DCCT:Current (actual PV: BR_DCCT_CURR)

Analysis:
✓ Logical organization (System/Family/Field)
✓ Actual PV addresses don't match logical structure
✓ Functional grouping (Monitor vs Setpoint)
✗ Can't derive PV from logical hierarchy alone
→ Middle Layer pipeline is appropriate
```

**Example Set 3**: No Clear Pattern
```
beam_current_main
bpm_horizontal_01
vertical_position_sector2
quad_power_supply_A
diagnostic_screen_intensity

Analysis:
✗ No consistent separator
✗ No predictable structure
✗ Arbitrary naming conventions
✓ Descriptive names (good for semantic search)
→ In-Context pipeline is the right choice
```

---

## 🎯 Phase 3: Pipeline Selection Logic

Use this decision tree based on your pattern analysis.

### Decision Tree

```
START: Do channel names follow a consistent hierarchical pattern?
│
├─ YES → Can you derive the full PV address from the hierarchy?
│   │
│   ├─ YES → Are there 100+ channels?
│   │   │
│   │   ├─ YES → Hierarchical Pipeline
│   │   │        Reason: Efficient navigation, pattern-based
│   │   │
│   │   └─ NO → In-Context Pipeline
│   │            Reason: Simpler for small sets, less overhead
│   │
│   └─ NO → Do you organize channels functionally?
│       │    (Monitor/Setpoint, System/Family, etc.)
│       │
│       ├─ YES → Middle Layer Pipeline
│       │        Reason: Functional organization + stored addresses
│       │
│       └─ NO → In-Context Pipeline
│                Reason: No exploitable structure
│
└─ NO → Is there any logical grouping at all?
    │
    ├─ YES → Could you define a hierarchy?
    │   │
    │   ├─ YES → Consider Hierarchical
    │   │        Note: Requires building hierarchy definition
    │   │
    │   └─ NO → In-Context Pipeline
    │            Reason: Functional groups but no clear tree
    │
    └─ NO → In-Context Pipeline
             Reason: Flat/arbitrary naming works best with semantic search
```

### Key Questions for Each Pipeline

**In-Context: "Should I use semantic search?"**
- ✅ Use when:
  - No consistent naming pattern
  - Small to medium scale (<1000 channels)
  - Channel names are descriptive
  - Want simplest setup

- ⚠️ Consider alternatives when:
  - Very large scale (>5000 channels) → chunking needed
  - Clear exploitable structure exists → inefficient use

**Hierarchical: "Should I use tree navigation?"**
- ✅ Use when:
  - Consistent hierarchical naming pattern
  - Can define levels (System → Subsystem → Device → Field)
  - Navigation reflects how users think about the system
  - Medium to large scale (100+ channels)

- ⚠️ Consider alternatives when:
  - Channel names don't reflect actual PV addresses → Middle Layer
  - Pattern is inconsistent or has many exceptions → In-Context
  - Very small scale (<50 channels) → In-Context is simpler

**Middle Layer: "Should I use functional organization?"**
- ✅ Use when:
  - Channels grouped by function (Monitor, Setpoint, etc.)
  - Actual PV addresses stored in database (not derived)
  - Logical organization ≠ naming pattern
  - Have or can build functional grouping structure

- ⚠️ Consider alternatives when:
  - No functional organization → In-Context
  - PV addresses follow naming pattern → Hierarchical
  - Very simple flat structure → In-Context

---

## 📋 Decision Matrix

Map observable features to pipeline suitability:

| Observable Feature | In-Context | Hierarchical | Middle Layer |
|-------------------|------------|--------------|--------------|
| **Scale** |
| < 100 channels | ✅ Good | ⚠️ Overkill | ⚠️ Overkill |
| 100-1000 channels | ✅ Good | ✅ Good | ✅ Good |
| 1000-5000 channels | ⚠️ Use chunking | ✅ Good | ✅ Good |
| > 5000 channels | ⚠️ Chunking required | ✅ Good | ✅ Good |
| **Naming Pattern** |
| Consistent hierarchy | ✅ Works | ✅ Best | ⚠️ Not needed |
| Functional keywords | ✅ Good | ✅ Can use | ✅ Best |
| Arbitrary/mixed | ✅ Best | ❌ Won't work | ❌ Won't work |
| **Organization** |
| Names = PV addresses | ✅ Simple | ✅ Pattern-based | ⚠️ Redundant |
| Names ≠ PV addresses | ✅ Works | ❌ Can't derive | ✅ Best |
| Functional grouping | ✅ Works | ⚠️ Possible | ✅ Designed for |
| No clear structure | ✅ Best | ❌ Won't work | ❌ Won't work |
| **Database** |
| Have existing DB | ✅ Easy | ⚠️ May need conversion | ✅ If functional |
| Can build hierarchy | ⚠️ Not needed | ✅ Required | ⚠️ Not needed |
| Flat list only | ✅ Perfect | ❌ Insufficient | ❌ Insufficient |
| **Descriptions** |
| Rich descriptions | ✅ Helps greatly | ✅ Helps | ✅ Helps |
| Minimal descriptions | ⚠️ Still works | ✅ Pattern helps | ✅ Structure helps |
| No descriptions | ⚠️ Names only | ✅ Pattern-based | ⚠️ Needs structure |

**Legend**: ✅ Good fit | ⚠️ Possible with caveats | ❌ Poor fit / won't work

---

## 🚨 Anti-Patterns: When to Ask for More Information

Recognize when you **don't have enough information** to recommend a pipeline.

### Red Flags

**User is being too vague:**
```
User: "I have a control system with channels"
❌ Don't recommend anything yet
✅ Ask: "Can you show me 10-15 example channel names?"
```

**User hasn't examined their data:**
```
User: "What's the best pipeline?"
❌ Don't ask theoretical questions
✅ Ask: "Let's look at your actual channel names. Can you provide examples?"
```

**User is guessing about future requirements:**
```
User: "We might add more channels later"
❌ Don't optimize for hypothetical future
✅ Focus: "Let's choose based on what you have now. You can migrate later."
```

**User has contradictory requirements:**
```
User: "My channels don't follow a pattern but I want hierarchical navigation"
❌ Don't force a pipeline to fit
✅ Clarify: "Hierarchical requires consistent patterns. Can you show examples?"
```

**Not enough examples provided:**
```
User provides 2-3 channel names
❌ Insufficient data for pattern analysis
✅ Request: "I need 10-15 examples covering different device types to see patterns"
```

### Validation Questions

Before recommending any pipeline, ensure you can answer:

- [ ] **Have I seen actual channel name examples?** (Not descriptions, actual names)
- [ ] **Do I understand how PV addresses relate to logical names?** (Same? Different?)
- [ ] **Can I articulate the observable pattern?** (Or lack thereof)
- [ ] **Have I considered if a simpler option would work?** (In-Context as baseline)
- [ ] **Can the user actually provide the required database structure?** (Don't recommend Hierarchical if they can't define hierarchy)

**If you answer NO to any of these → Ask for more information before recommending.**

---

## 💡 Practical Examples

Real-world scenarios showing the decision process.

### Example 1: Clear Hierarchical Pattern

**User provides:**
```
ALS:SR01C:BPM1:X
ALS:SR01C:BPM1:Y
ALS:SR01C:BPM2:X
ALS:SR02C:BPM1:X
ALS:BR:DCCT:Current
ALS:BTS:VCM1:Setpoint
```

**Analysis:**
- ✓ Consistent `:` separator
- ✓ Clear levels: FACILITY:SYSTEM:DEVICE:FIELD
- ✓ Pattern repeats across ~500 channels (user mentions)
- ✓ Can derive hierarchy from names

**Recommendation:**
```
Hierarchical Pipeline is a strong fit because:
1. Consistent 4-level naming pattern
2. Each level has semantic meaning
3. Pattern-based navigation matches user mental model
4. Scale (500 channels) benefits from structure

Next steps:
- Review hierarchical database format: .../databases/hierarchical.py
- Define hierarchy levels in config
- See example: data/channel_databases/hierarchical.json
```

---

### Example 2: Functional Organization with Stored Addresses

**User provides:**
```
"I have ~300 channels organized as:
 - Storage Ring (SR) → BPM family → Monitor field → [SR01C_BPM1_X_RB, SR01C_BPM2_X_RB, ...]
 - Storage Ring (SR) → BPM family → Setpoint field → [SR01C_BPM1_X_SP, ...]
 - Booster Ring (BR) → DCCT family → Monitor field → [BR_DCCT_I_RB]

The logical organization (System/Family/Field) doesn't match the actual EPICS PV naming."
```

**Analysis:**
- ✓ Functional organization (System → Family → Field)
- ✓ PV addresses stored separately (not derived from hierarchy)
- ✓ Clear Monitor/Setpoint distinction
- ✗ Can't build PV from logical path alone

**Recommendation:**
```
Middle Layer Pipeline is appropriate because:
1. Functional organization exists (System/Family/Field)
2. Actual PV addresses are stored, not derived
3. Logical grouping helps users navigate
4. Monitor/Setpoint separation is meaningful

This is similar to how MATLAB Middle Layer organizes accelerator channels,
but applicable to any system with functional organization independent of naming.

Next steps:
- Review middle layer database format: .../databases/middle_layer.py
- See example: data/channel_databases/middle_layer.json
- Map your existing organization to MML-style structure
```

---

### Example 3: No Exploitable Structure

**User provides:**
```
beam_current_main_ring
bpm_horizontal_position_01
bpm_vertical_position_01
quadrupole_power_supply_A
sextupole_magnet_01_current
diagnostic_screen_intensity
vacuum_gauge_sector_2
corrector_magnet_h_03
```

**Analysis:**
- ✗ No consistent separator
- ✗ No predictable segment structure
- ✗ Different naming styles mixed
- ✓ Names are descriptive (good for semantic search)
- ~150 channels (user mentions)

**Recommendation:**
```
In-Context Pipeline is the right choice because:
1. No consistent naming pattern to exploit
2. Arbitrary structure means hierarchical navigation wouldn't help
3. Descriptive names work well with semantic search
4. Scale (150 channels) is manageable for in-context approach

Alternative considered: Could you impose a hierarchy?
- If yes: Hierarchical becomes possible (but requires restructuring)
- If no: In-Context is correct choice

Next steps:
- Review in-context pipeline: .../pipelines/in_context/pipeline.py
- Create flat database: .../databases/flat.py
- See example: data/channel_databases/flat.json
```

---

### Example 4: Ambiguous Case - Multiple Options

**User provides:**
```
SR:BPM:01:X
SR:BPM:02:X
SR:BPM:01:Y
SR:HCM:01:Current
SR:HCM:02:Current
BR:DCCT:Current

~200 channels total
Has some hierarchy but not deeply nested
```

**Analysis:**
- ✓ Consistent pattern (SYSTEM:FAMILY:DEVICE:FIELD)
- ✓ Could work with Hierarchical
- ⚠️ Relatively simple structure
- ⚠️ Moderate scale (200 channels)

**Recommendation:**
```
Two viable options - start simple and evolve if needed:

OPTION 1 (Recommended): In-Context Pipeline
- Simpler setup for 200 channels
- Less configuration overhead
- Easy to migrate from if needs change
- Works immediately with flat list

OPTION 2: Hierarchical Pipeline
- Exploits the pattern structure
- More efficient if you frequently query by System/Family
- Better if you expect significant growth (>1000 channels)
- Requires building hierarchy definition

Suggested approach:
1. Start with In-Context (faster to set up)
2. If you find navigation tedious or scale grows → migrate to Hierarchical
3. Both pipelines can use the same underlying channel data

Which to choose?
- Prefer simplicity? → In-Context
- Value structured navigation? → Hierarchical
- Not sure? → Start with In-Context
```

---

### Example 5: Insufficient Information

**User asks:**
```
"What Channel Finder pipeline should I use for my beamline?"
```

**Response:**
```
I need to see your actual channel names to make a good recommendation.
Pattern analysis requires concrete examples, not descriptions.

Please provide:
1. 10-15 representative channel names from your system
   (Include variety - different devices, signal types)
2. Approximate total channel count
3. If you have an existing database or configuration file, mention it

Once I see the actual naming patterns, I can guide you to the right pipeline.

Example of what I'm looking for:
- BEAMLINE_01:VALVE:STATUS
- BEAMLINE_01:PRESSURE:READBACK
- etc.
```

---

## 🔄 Migration and Evolution

**Starting simple is safe** - you can always migrate between pipelines.

### Migration Paths

```
In-Context → Hierarchical
When: You discover a pattern worth exploiting
Effort: Moderate (build hierarchy definition from existing channels)
Risk: Low (same underlying data)

In-Context → Middle Layer
When: You add functional organization
Effort: Moderate (reorganize into System/Family/Field structure)
Risk: Low (PV addresses don't change)

Hierarchical → Middle Layer
When: You realize PV addresses don't follow the pattern you thought
Effort: Moderate (restructure to functional organization)
Risk: Medium (significant config changes)
```

### When to Migrate

**Don't migrate prematurely.** Migrate when you observe:
- Current pipeline is inefficient for common queries
- Scale has grown significantly (>1000 channels)
- User feedback indicates navigation is confusing
- Pattern you thought existed doesn't actually hold

**Do migrate** when:
- Clear improvement in user experience
- Existing pipeline doesn't support new requirements
- You have time to test thoroughly

---

## 🛠️ Code Reference Guide

**AI Assistants**: Always ask to read the relevant code files below to better understand database requirements and make more informed recommendations.

### Where to Look

**Database Format Classes** (READ THESE FIRST - contain schema documentation):
```
Flat (In-Context):     src/osprey/templates/.../databases/flat.py
Hierarchical:          src/osprey/templates/.../databases/hierarchical.py
Middle Layer:          src/osprey/templates/.../databases/middle_layer.py
```

These Python files contain:
- Detailed docstrings explaining the database schema
- Validation logic showing what's required vs optional
- Comments explaining design decisions
- Example snippets in docstrings

**Example Databases** (if user has initialized my-control-assistant):
```
my-control-assistant/data/channel_databases/in_context.json       (255 channels, UCSB FEL)
my-control-assistant/data/channel_databases/hierarchical.json     (1,050 channels, accelerator)
my-control-assistant/data/channel_databases/middle_layer.json     (300+ channels, MML-style)
```

**Pipeline Implementations** (to understand how queries are processed):
```
In-Context:    src/osprey/templates/.../pipelines/in_context/pipeline.py
Hierarchical:  src/osprey/templates/.../pipelines/hierarchical/pipeline.py
Middle Layer:  src/osprey/templates/.../pipelines/middle_layer/pipeline.py
```

**Configuration:**
```
config.yml (channel_finder section)
See: my-control-assistant/config.yml for complete example
```

### How AI Assistants Should Use Code

**When user asks about pipeline selection:**
1. Ask if they have `my-control-assistant` initialized
2. If yes, read example databases to show concrete patterns
3. Read database class files to understand schema requirements
4. Compare user's channel names against example patterns
5. Make evidence-based recommendations

**Example AI workflow:**
```
AI: "Do you have my-control-assistant initialized? If so, let me read the example databases
     to show you concrete patterns that match different pipelines."

User: "Yes, it's at ~/my-control-assistant"

AI: [Reads hierarchical.json]
    "I can see the hierarchical database uses a 5-level structure with rich descriptions
     at branching points. Looking at your channel names, you have a similar pattern..."
```

### Key Code Sections

**Understanding In-Context:**
- Pipeline logic: `in_context/pipeline.py` lines 143-194 (process_query method)
- Chunking: `flat.py` lines 100-137 (chunk_database, format_chunk_for_prompt)

**Understanding Hierarchical:**
- Pipeline logic: `hierarchical/pipeline.py` lines 130-183 (process_query method)
- Navigation: `hierarchical/pipeline.py` lines 211-468 (_navigate_recursive method)
- Database structure: `hierarchical.py` lines 1-150 (header comments explain format)

**Understanding Middle Layer:**
- Pipeline logic: `middle_layer/pipeline.py` lines 448-501 (process_query method)
- Tool creation: `middle_layer/pipeline.py` lines 141-324 (_create_tools method)
- Database structure: `middle_layer.py` lines 1-36 (header comments explain MML format)

---

## ✅ Decision Checklist

Use this checklist to validate your pipeline selection.

### Before Recommending

- [ ] I have seen 10+ actual channel name examples
- [ ] I understand the scale (total channel count)
- [ ] I can articulate what pattern exists (or doesn't exist)
- [ ] I know whether PV addresses match logical organization
- [ ] I've considered the simplest option first (In-Context)
- [ ] I can point to specific code/examples for the recommended pipeline
- [ ] I've validated the user can actually build the required database structure

### Red Flags to Watch For

- [ ] ❌ User hasn't shown actual channel names yet
- [ ] ❌ I'm making assumptions about their system
- [ ] ❌ Recommendation is based on theoretical requirements
- [ ] ❌ I haven't considered whether simpler option would work
- [ ] ❌ User seems confused by my questions → need to simplify

### Recommendation Template

When recommending a pipeline, structure your response:

```
Based on [SPECIFIC EVIDENCE FROM EXAMPLES]:

Recommended Pipeline: [PIPELINE NAME]

Reasoning:
1. [Observable pattern 1]
2. [Observable pattern 2]
3. [Why this pipeline matches those patterns]

Alternative Considered: [OTHER PIPELINE]
Why not chosen: [SPECIFIC REASON]

Next Steps:
1. Review code: [SPECIFIC FILE PATH]
2. See example database: [SPECIFIC EXAMPLE]
3. [Specific action to start building]

If uncertain: Start with In-Context (works for anything), migrate later if needed.
```

---

## 🎓 Understanding the Pipelines

Brief conceptual overview - point to code for implementation details.

### In-Context: Semantic Search

**Concept**: LLM reads entire channel list (or chunks) in context and selects matches based on semantic understanding.

**How it works**:
1. Load all channels into memory
2. Format channels as text (with descriptions)
3. Send to LLM with user query
4. LLM selects matching channels
5. Validate and correct any errors

**Advantages**:
- Works with any naming scheme
- No structure required
- Simple to set up
- Descriptions help matching

**Limitations**:
- Limited by context window (chunking needed for large datasets)
- Less efficient for very large scale (>5000 channels)
- Doesn't exploit structure if it exists

### Hierarchical: Tree Navigation

**Concept**: Navigate through hierarchy levels (System → Subsystem → Device → Field) by having LLM make selections at each level.

**How it works**:
1. Define hierarchy levels in database
2. At each level, present options to LLM
3. LLM selects relevant option(s) based on query
4. Navigate deeper until reaching channels
5. Build full channel names from path

**Advantages**:
- Efficient for large structured datasets
- Natural navigation for hierarchical systems
- Exploits naming patterns
- Scales well (>1000 channels)

**Limitations**:
- Requires consistent naming pattern
- Setup overhead (define hierarchy)
- Won't work for arbitrary naming
- Must be able to derive PV from hierarchy

### Middle Layer: Functional Organization

**Concept**: Organize channels by function (System → Family → Field) with actual PV addresses stored in database, not derived from pattern.

**How it works**:
1. Organize database into System/Family/Field structure
2. Store actual PV addresses at leaf nodes
3. LLM uses tools to explore the structure
4. Retrieve stored addresses (not built from pattern)
5. Support filtering by device/sector

**Advantages**:
- Functional organization independent of naming
- Flexible addressing (any PV format)
- Natural for control system operations (Monitor/Setpoint)
- Tools-based exploration

**Limitations**:
- Requires functional organization to exist or be built
- More complex database structure
- Setup overhead
- Not useful if PV names already reflect hierarchy

---

## 📚 Additional Resources

### Related Documentation

- **Channel Finder Service**: `src/osprey/templates/.../channel_finder/service.py`
- **Configuration Guide**: `config.yml` (channel_finder section)
- **Database Examples**: `data/channel_databases/` directory
- **Prompts**: `src/osprey/templates/.../channel_finder/prompts/`

### Related Workflows

- **Database Builder Workflow**: `channel-finder-database-builder.md` (coming next)
- **Testing Workflow**: `testing-workflow.md` (general testing guidance)
- **Pre-Merge Cleanup**: `pre-merge-cleanup.md` (before committing configs)

### External References

- **MATLAB Middle Layer**: For understanding functional organization concept (accelerator-specific but concept applies broadly)
- **EPICS Channel Access**: Standard for process variables in scientific control systems
- **Control System Patterns**: Many facilities use functional organization even without MML

---

## 🤝 When to Ask for Help

This workflow helps with standard cases. Consult the development team if:

- You have a novel database structure not covered here
- You need to support multiple facilities with different patterns
- You want to combine multiple pipeline approaches
- You're dealing with >10,000 channels and performance is critical
- Your system has unusual requirements not addressed here

**Remember**: It's better to ask questions early than to build the wrong thing. The pipelines are designed to be changed if initial choice doesn't work out.
